r"""
三位一体 v2: Population + NFSP + FP + 学习率调度 + 熵衰减 + Elo 评估

P0: 状态.copy() + GAE bootstrap 用最终观测
P1: cosine lr + 熵衰减 (0.03→0.005)
P2: Elo 评分替代纯胜率

用法:
  python src/train_full.py --rounds 30 --pop 3 --seed 0
"""

import os,sys,argparse,copy,random,math
from collections import deque
import numpy as np
import torch,torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from env import ControlEnv, STATE_DIM, N_ACTIONS

N_POP=3; device="cpu"

# 大种群支持: 通过命令行 --pop 覆盖


# ──────────────────────────────────────────────────────────────────────
# 策略网络
# ──────────────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(STATE_DIM,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.actor=nn.Linear(64,N_ACTIONS); self.critic=nn.Linear(64,1)
    def forward(self,x):
        h=self.f(x); return self.actor(h),self.critic(h).squeeze(-1)
    def evaluate(self,states,actions,legal_masks):
        logits,values=self.forward(states)
        mask=torch.full_like(logits,float("-inf"))
        mask[legal_masks]=0.0
        dist=Categorical(logits=logits+mask)
        log_probs=dist.log_prob(actions); entropy=dist.entropy().mean()
        return log_probs,values,entropy
    def act(self,x,legal,sample=True):
        logits,value=self.forward(x.unsqueeze(0)); logits=logits[0]
        mask=torch.full_like(logits,float("-inf"))
        for a in legal: mask[a]=0.0
        probs=torch.softmax(logits+mask,dim=0)
        if sample:
            dist=Categorical(probs); action=dist.sample()
            return action.item(),dist.log_prob(action).item(),value.item()
        else:
            return (logits+mask).argmax().item()


# ──────────────────────────────────────────────────────────────────────
# 对手模型 (NFSP)
# ──────────────────────────────────────────────────────────────────────
class OpponentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(STATE_DIM,64),nn.Tanh(),nn.Linear(64,N_ACTIONS))
    def forward(self,x): return self.net(x)
    def predict(self,x,legal):
        logits=self.forward(x.unsqueeze(0))[0]
        mask=torch.full_like(logits,float("-inf"))
        for a in legal: mask[a]=0.0
        probs=torch.softmax(logits+mask,dim=0)
        return Categorical(probs).sample().item()


# ──────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, ent_coef=0.03):
        self.policy=PolicyNet()
        self.opp_model=OpponentModel()
        self.opp_buffer=deque(maxlen=5000)
        self.ent_coef=ent_coef  # P1: 熵衰减, 每轮更新
        self.elo=1500           # P2: Elo 初始分

    def record_opp_behavior(self,state,opp_action):
        self.opp_buffer.append((state.copy(),opp_action))

    def train_opp_model(self,batch_size=128,epochs=3):
        if len(self.opp_buffer)<batch_size: return
        for _ in range(epochs):
            idx=random.sample(range(len(self.opp_buffer)),min(batch_size,len(self.opp_buffer)))
            states=torch.tensor(np.stack([self.opp_buffer[i][0] for i in idx])).float()
            targets=torch.tensor([self.opp_buffer[i][1] for i in idx]).long()
            logits=self.opp_model(states)
            loss=F.cross_entropy(logits,targets)
            # opp_model 没有自己的 optimizer——用 policy 的 optimizer 共享? 不对, 应该独立
            # 这里用一个简单 SGD (不占主要复杂度)
            for p in self.opp_model.parameters():
                p.grad=None
            loss.backward()
            with torch.no_grad():
                for p in self.opp_model.parameters():
                    if p.grad is not None:
                        p-=0.01*p.grad


# ──────────────────────────────────────────────────────────────────────
# PPO 更新
# ──────────────────────────────────────────────────────────────────────
def compute_gae(rewards,values,dones,lv,gamma=0.99,lam=0.95):
    adv=np.zeros(len(rewards),dtype=np.float32); gae=0.0; nv=lv
    for t in reversed(range(len(rewards))):
        nt=0.0 if dones[t] else 1.0
        delta=rewards[t]+gamma*nv*nt-values[t]; gae=delta+gamma*lam*nt*gae; adv[t]=gae; nv=values[t]
    return adv,adv+np.array(values,dtype=np.float32)


def ppo_update(policy,opt,S,A,LP,ADV,RET,legals,ent_coef=0.03,ue=5,bs=256):
    at=torch.tensor(ADV,dtype=torch.float32); at=(at-at.mean())/(at.std()+1e-8)
    rt=torch.tensor(RET,dtype=torch.float32)
    St=torch.tensor(np.stack(S)).float(); At=torch.tensor(A); LPt=torch.tensor(LP)
    n=len(S)
    for _ in range(ue):
        idx=torch.randperm(n)
        for s in range(0,n,bs):
            mb=idx[s:s+bs]
            leg_mask=torch.zeros(len(mb),N_ACTIONS,dtype=torch.bool)
            for j,i in enumerate(mb.tolist()):
                for a in legals[i]: leg_mask[j,a]=True
            nlp,vals,ent=policy.evaluate(St[mb],At[mb],leg_mask)
            ratio=torch.exp(nlp-LPt[mb])
            s1=ratio*at[mb]; s2=torch.clamp(ratio,1-0.2,1+0.2)*at[mb]
            loss=-torch.min(s1,s2).mean()+0.5*nn.MSELoss()(vals,rt[mb])-ent_coef*ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),0.5); opt.step()
    return ent.item()


def collect_and_train(agent,env,is_a,opponent,steps=1500,ent_coef=0.03):
    """P0修复: 状态.copy(), GAE bootstrap用最终观测, PPO mask。"""
    S,A,LP,R,V,D,legals=[],[],[],[],[],[],[]
    oa,ob=env.reset(); cnt=0
    while cnt<steps:
        la=env.legal_a() if is_a else env.legal_b()
        lb=env.legal_b() if is_a else env.legal_a()
        if not la or not lb: oa,ob=env.reset(); continue

        # P0: 存 step 前的 copy
        my_state=(oa.copy() if is_a else ob.copy())
        opp_state=(ob.copy() if is_a else oa.copy())

        my_t=torch.from_numpy(my_state).float()
        my_a,my_lp,my_v=agent.policy.act(my_t,la,sample=True)

        opp_t=torch.from_numpy(opp_state).float()
        opp_a=opponent.policy.act(opp_t,lb,sample=False)
        agent.record_opp_behavior(opp_state,opp_a)

        if is_a: oa,ra,da,ob,rb,db,_=env.step(my_a,opp_a)
        else: oa,ra,da,ob,rb,db,_=env.step(opp_a,my_a)
        my_r=ra if is_a else rb; my_done=da if is_a else db

        S.append(my_state); A.append(my_a); LP.append(my_lp); legals.append(la)
        R.append(my_r); V.append(my_v); D.append(my_done)
        cnt+=1
        if my_done: oa,ob=env.reset()

    if len(S)==0: return 0.0
    # P0: GAE bootstrap用循环结束后的最终观测
    final_obs=oa if is_a else ob
    with torch.no_grad():
        lst=torch.from_numpy(final_obs).float()
        _,lv=agent.policy(lst.unsqueeze(0))
    adv,ret=compute_gae(R,V,D,lv.item())
    ent=ppo_update(agent.policy,agent.policy_opt,S,A,LP,adv,ret,legals,ent_coef)
    agent.train_opp_model()
    return ent


# ──────────────────────────────────────────────────────────────────────
# P2: Elo 评分
# ──────────────────────────────────────────────────────────────────────
K_ELO=32
def update_elo(winner,loser,draw=False):
    if draw:
        e=1.0/(1+10**((loser.elo-winner.elo)/400))
        delta=K_ELO*(0.5-e)
        winner.elo+=delta; loser.elo-=delta
    else:
        e=1.0/(1+10**((loser.elo-winner.elo)/400))
        winner.elo+=K_ELO*(1-e); loser.elo-=K_ELO*e

def eval_match_elo(agent_a,agent_b,env,n_games=10):
    """对战并更新 Elo。返回(A胜, B胜, 平)。"""
    w,d=0,0
    for _ in range(n_games):
        oa,ob=env.reset()
        for _ in range(env.max_steps):
            la,lb=env.legal_a(),env.legal_b()
            if not la or not lb: break
            ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
            aa,_,_ = agent_a.policy.act(ta,la,sample=True)
            ab,_,_ = agent_b.policy.act(tb,lb,sample=True)
            oa,ra,da,ob,rb,db,_=env.step(aa,ab)
            if da or db: break
        if env.winner=="A": w+=1; update_elo(agent_a,agent_b)
        elif env.winner=="B": update_elo(agent_b,agent_a)
        else: d+=1; update_elo(agent_a,agent_b,draw=True)
    return w,n_games-w-d,d


# ──────────────────────────────────────────────────────────────────────
# 种群演进
# ──────────────────────────────────────────────────────────────────────
def evolve_population(pop_a,pop_b,env):
    scores_a=[0]*len(pop_a); scores_b=[0]*len(pop_b)
    for i,pa in enumerate(pop_a):
        for j,pb in enumerate(pop_b):
            aw,_,_=eval_match_elo(pa,pb,env,10)
            scores_a[i]+=aw; scores_b[j]+=10-aw
    # 淘汰+复制+变异
    worst_a=min(range(len(pop_a)),key=lambda i:scores_a[i])
    worst_b=min(range(len(pop_b)),key=lambda i:scores_b[i])
    best_a=max(range(len(pop_a)),key=lambda i:scores_a[i])
    best_b=max(range(len(pop_b)),key=lambda i:scores_b[i])
    pop_a[worst_a].policy.load_state_dict(pop_a[best_a].policy.state_dict())
    pop_b[worst_b].policy.load_state_dict(pop_b[best_b].policy.state_dict())
    for p in pop_a[worst_a].policy.parameters():
        p.data+=torch.randn_like(p)*0.01
    for p in pop_b[worst_b].policy.parameters():
        p.data+=torch.randn_like(p)*0.01
    return scores_a,scores_b


# ──────────────────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────────────────
def train(env,rounds=30,pop=N_POP,seed=0,ent_start=0.03,ent_end=0.005):
    if seed: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pop_a=[Agent(ent_start) for _ in range(pop)]
    pop_b=[Agent(ent_start) for _ in range(pop)]

    # P1: 每个体独立 optimizer + scheduler
    for agent in pop_a+pop_b:
        agent.policy_opt=torch.optim.Adam(agent.policy.parameters(),lr=3e-4)
        agent.policy_sched=torch.optim.lr_scheduler.CosineAnnealingLR(agent.policy_opt,T_max=rounds)

    print(f"[三位一体 v2] rounds={rounds} pop={pop} | P1:cosine_lr+ent_decay P2:Elo")

    for ro in range(1,rounds+1):
        # P1: 熵衰减
        ent_coef=ent_start-(ent_start-ent_end)*(ro/rounds)

        for agent in pop_a:
            agent.ent_coef=ent_coef
            opponent=random.choice(pop_b)
            collect_and_train(agent,env,True,opponent,steps=1500,ent_coef=ent_coef)
        for agent in pop_b:
            agent.ent_coef=ent_coef
            opponent=random.choice(pop_a)
            collect_and_train(agent,env,False,opponent,steps=1500,ent_coef=ent_coef)

        # P1: 每个体 lr 衰减一步
        for agent in pop_a+pop_b:
            agent.policy_sched.step()

        # 每5轮种群演进
        if ro%5==0 and ro>0:
            sa,sb=evolve_population(pop_a,pop_b,env)
            # P2: Elo 汇总
            elos_a=[f"{a.elo:.0f}" for a in pop_a]
            elos_b=[f"{b.elo:.0f}" for b in pop_b]

            # 最强个体对战评估
            best_a=pop_a[max(range(pop),key=lambda i:sa[i])]
            best_b=pop_b[max(range(pop),key=lambda i:sb[i])]
            w,d=0,0
            for _ in range(30):
                oa,ob=env.reset()
                for _ in range(env.max_steps):
                    la,lb=env.legal_a(),env.legal_b()
                    if not la or not lb: break
                    ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
                    aa,_,_ = best_a.policy.act(ta,la,sample=True)
                    ab,_,_ = best_b.policy.act(tb,lb,sample=True)
                    oa,ra,da,ob,rb,db,_=env.step(aa,ab)
                    if da or db: break
                if env.winner=="A": w+=1
                elif env.winner is None: d+=1
            bw=30-w-d
            print(f"[Evolve {ro:3d}/{rounds}] best: A{w} B{bw} D{d} "
                  f"ElosA={elos_a} ElosB={elos_b} ent={ent_coef:.3f}")

    return pop_a,pop_b


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--rounds",type=int,default=30)
    ap.add_argument("--pop",type=int,default=3)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    env=ControlEnv()
    pa,pb=train(env,rounds=args.rounds,pop=args.pop,seed=args.seed)
    d=os.path.join(os.path.dirname(__file__),"..","models"); os.makedirs(d,exist_ok=True)
    for i,p in enumerate(pa): torch.save(p.policy.state_dict(),os.path.join(d,f"full_a{i}.pt"))
    for i,p in enumerate(pb): torch.save(p.policy.state_dict(),os.path.join(d,f"full_b{i}.pt"))
    print("[保存] 种群模型")