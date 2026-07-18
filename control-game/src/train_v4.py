"""
POMDP 控制博弈 v4 训练: 部分可观测 + GRU 记忆。

关键差异: PolicyNet 加 GRUCell, 训练时传递 hidden state。
收集时 hidden 随步传递, episode 重置时归零。
更新时用冻结的 hidden 作为输入 (近似但常用)。

用法:
  python src/train_v4.py --rounds 200 --pop 3 --seed 0
"""

import os,sys,argparse,random
from collections import deque
import numpy as np
import torch,torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from env_v4 import ControlEnvV4, STATE_DIM, N_ACTIONS

N_POP=3
HIDDEN_SIZE=64


class PolicyNetRNN(nn.Module):
    """带 GRUCell 的策略网络。每步输入 (state, hidden) → (logits, value, new_hidden)。"""
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(STATE_DIM,128)
        self.gru=nn.GRUCell(128,HIDDEN_SIZE)
        self.actor=nn.Linear(HIDDEN_SIZE,N_ACTIONS)
        self.critic=nn.Linear(HIDDEN_SIZE,1)

    def forward_step(self,state,hidden):
        """单步前向: state(STATE_DIM,), hidden(HIDDEN_SIZE,) → logits, value, new_hidden。"""
        x=torch.tanh(self.fc(state))          # (128,)
        new_hidden=self.gru(x.unsqueeze(0),hidden.unsqueeze(0)).squeeze(0)  # (64,)
        logits=self.actor(new_hidden)         # (N_ACTIONS,)
        value=self.critic(new_hidden).squeeze(-1)
        return logits,value,new_hidden

    def evaluate(self,states,hiddens,actions,legal_masks):
        """批量前向: states(B,D), hiddens(B,H) → log_probs, values, entropy。"""
        x=torch.tanh(self.fc(states))         # (B,128)
        new_hidden=self.gru(x,hiddens)        # (B,64)
        logits=self.actor(new_hidden)         # (B,N_ACTIONS)
        values=self.critic(new_hidden).squeeze(-1)
        mask=torch.full_like(logits,float("-inf"))
        mask[legal_masks]=0.0
        dist=Categorical(logits=logits+mask)
        return dist.log_prob(actions),values,dist.entropy().mean()

    def get_action(self,state,hidden,legal,sample=True):
        with torch.no_grad():
            logits,value,new_hidden=self.forward_step(state,hidden)
            mask=torch.full_like(logits,float("-inf"))
            for a in legal: mask[a]=0.0
            probs=torch.softmax(logits+mask,dim=0)
            if sample:
                dist=Categorical(probs); action=dist.sample()
                return action.item(),dist.log_prob(action).item(),value.item(),new_hidden
            else:
                return (logits+mask).argmax().item(),0.0,0.0,new_hidden

    def act_greedy(self,state,hidden,legal):
        action,_,_,new_hidden=self.get_action(state,hidden,legal,sample=False)
        return action,new_hidden


class OpponentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(STATE_DIM,64),nn.Tanh(),nn.Linear(64,N_ACTIONS))
    def forward(self,x): return self.net(x)


class Agent:
    def __init__(self,ent_coef=0.03):
        self.policy=PolicyNetRNN()
        self.opp_model=OpponentModel()
        self.opp_buffer=deque(maxlen=5000)
        self.ent_coef=ent_coef
        self.elo=1500
    def record_opp_behavior(self,state,opp_action):
        self.opp_buffer.append((state.copy(),opp_action))
    def train_opp_model(self,batch_size=128,epochs=3):
        if len(self.opp_buffer)<batch_size: return
        for _ in range(epochs):
            idx=random.sample(range(len(self.opp_buffer)),min(batch_size,len(self.opp_buffer)))
            states=torch.tensor(np.stack([self.opp_buffer[i][0] for i in idx])).float()
            targets=torch.tensor([self.opp_buffer[i][1] for i in idx]).long()
            loss=F.cross_entropy(self.opp_model(states),targets)
            for p in self.opp_model.parameters(): p.grad=None
            loss.backward()
            with torch.no_grad():
                for p in self.opp_model.parameters():
                    if p.grad is not None: p-=0.01*p.grad


def compute_gae(rewards,values,dones,lv,gamma=0.99,lam=0.95):
    adv=np.zeros(len(rewards),dtype=np.float32); gae=0.0; nv=lv
    for t in reversed(range(len(rewards))):
        nt=0.0 if dones[t] else 1.0
        delta=rewards[t]+gamma*nv*nt-values[t]; gae=delta+gamma*lam*nt*gae; adv[t]=gae; nv=values[t]
    return adv,adv+np.array(values,dtype=np.float32)


def ppo_update(policy,opt,S,A,LP,ADV,RET,H,legals,ent_coef=0.03,ue=5,bs=256):
    """PPO 更新: 用冻结的 hidden state H 作为 GRU 输入。"""
    at=torch.tensor(ADV,dtype=torch.float32); at=(at-at.mean())/(at.std()+1e-8)
    rt=torch.tensor(RET,dtype=torch.float32)
    St=torch.tensor(np.stack(S)).float()
    Ht=torch.stack(H)  # (n, HIDDEN_SIZE)
    At=torch.tensor(A); LPt=torch.tensor(LP)
    n=len(S)
    for _ in range(ue):
        idx=torch.randperm(n)
        for s in range(0,n,bs):
            mb=idx[s:s+bs]
            leg_mask=torch.zeros(len(mb),N_ACTIONS,dtype=torch.bool)
            for j,i in enumerate(mb.tolist()):
                for a in legals[i]: leg_mask[j,a]=True
            nlp,vals,ent=policy.evaluate(St[mb],Ht[mb],At[mb],leg_mask)
            ratio=torch.exp(nlp-LPt[mb])
            s1=ratio*at[mb]; s2=torch.clamp(ratio,1-0.2,1+0.2)*at[mb]
            loss=-torch.min(s1,s2).mean()+0.5*nn.MSELoss()(vals,rt[mb])-ent_coef*ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),0.5); opt.step()
    return ent.item()


def collect_and_train(agent,env,is_a,opponent,steps=1500,ent_coef=0.03):
    """收集 rollout 时传递 hidden state, episode 重置时 hidden 归零。"""
    S,A,LP,R,V,D,H,legals=[],[],[],[],[],[],[],[]
    oa,ob=env.reset(); cnt=0
    h_a=torch.zeros(HIDDEN_SIZE)  # A 的 hidden
    h_b=torch.zeros(HIDDEN_SIZE)  # B 的 hidden (对手用)

    while cnt<steps:
        la=env.legal_a() if is_a else env.legal_b()
        lb=env.legal_b() if is_a else env.legal_a()
        if not la or not lb:
            oa,ob=env.reset()
            h_a=torch.zeros(HIDDEN_SIZE); h_b=torch.zeros(HIDDEN_SIZE)
            continue

        my_state=torch.from_numpy(oa.copy() if is_a else ob.copy()).float()
        opp_state=(ob.copy() if is_a else oa.copy())

        # 我方用 RNN 采样
        my_hidden=h_a if is_a else h_b
        my_a,my_lp,my_v,new_hidden=agent.policy.get_action(my_state,my_hidden,la,sample=True)

        # 对手用 RNN greedy
        opp_hidden=h_b if is_a else h_a
        opp_t=torch.from_numpy(opp_state).float()
        opp_a,opp_new_hidden=opponent.policy.act_greedy(opp_t,opp_hidden,lb)
        agent.record_opp_behavior(opp_state,opp_a)

        # 更新 hidden (detach 避免梯度回传过远)
        if is_a:
            h_a=new_hidden.detach(); h_b=opp_new_hidden.detach()
        else:
            h_b=new_hidden.detach(); h_a=opp_new_hidden.detach()

        if is_a: oa,ra,da,ob,rb,db,_=env.step(my_a,opp_a)
        else: oa,ra,da,ob,rb,db,_=env.step(opp_a,my_a)
        my_r=ra if is_a else rb; my_done=da if is_a else db

        S.append(my_state.numpy().copy());A.append(my_a);LP.append(my_lp)
        H.append(my_hidden.detach());legals.append(la)
        R.append(my_r);V.append(my_v);D.append(my_done)
        cnt+=1
        if my_done:
            oa,ob=env.reset()
            h_a=torch.zeros(HIDDEN_SIZE); h_b=torch.zeros(HIDDEN_SIZE)

    if len(S)==0: return 0.0
    final_obs=torch.from_numpy(oa.copy() if is_a else ob.copy()).float()
    final_h=(h_a if is_a else h_b).detach()
    with torch.no_grad():
        _,lv,_=agent.policy.forward_step(final_obs,final_h)
    adv,ret=compute_gae(R,V,D,lv.item())
    ent=ppo_update(agent.policy,agent.policy_opt,S,A,LP,adv,ret,H,legals,ent_coef)
    agent.train_opp_model()
    return ent


K_ELO=32
def update_elo(w,l,draw=False):
    e=1.0/(1+10**((l.elo-w.elo)/400))
    if draw: d=K_ELO*(0.5-e); w.elo+=d; l.elo-=d
    else: w.elo+=K_ELO*(1-e); l.elo-=K_ELO*e

def eval_match_elo(a,b,env,n=10):
    w,d=0,0
    for _ in range(n):
        oa,ob=env.reset()
        ha=torch.zeros(HIDDEN_SIZE); hb=torch.zeros(HIDDEN_SIZE)
        for _ in range(env.max_steps):
            la,lb=env.legal_a(),env.legal_b()
            if not la or not lb: break
            ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
            aa,_,_,ha=a.policy.get_action(ta,ha,la,sample=True)
            ab,_,_,hb=b.policy.get_action(tb,hb,lb,sample=True)
            oa,ra,da,ob,rb,db,_=env.step(aa,ab)
            if da or db: break
        if env.winner=="A": w+=1; update_elo(a,b)
        elif env.winner=="B": update_elo(b,a)
        else: d+=1; update_elo(a,b,draw=True)
    return w,n-w-d,d

def evolve(pa,pb,env):
    sa=[0]*len(pa); sb=[0]*len(pb)
    for i,a in enumerate(pa):
        for j,b in enumerate(pb):
            aw,_,_=eval_match_elo(a,b,env,10)
            sa[i]+=aw; sb[j]+=10-aw
    wa=min(range(len(pa)),key=lambda i:sa[i]); wb=min(range(len(pb)),key=lambda i:sb[i])
    ba=max(range(len(pa)),key=lambda i:sa[i]); bb=max(range(len(pb)),key=lambda i:sb[i])
    pa[wa].policy.load_state_dict(pa[ba].policy.state_dict())
    pb[wb].policy.load_state_dict(pb[bb].policy.state_dict())
    for p in pa[wa].policy.parameters(): p.data+=torch.randn_like(p)*0.01
    for p in pb[wb].policy.parameters(): p.data+=torch.randn_like(p)*0.01
    return sa,sb


def train(env,rounds=200,pop=N_POP,seed=0,ent_start=0.03,ent_end=0.005,print_every=10):
    if seed: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pa=[Agent(ent_start) for _ in range(pop)]; pb=[Agent(ent_start) for _ in range(pop)]
    for a in pa+pb:
        a.policy_opt=torch.optim.Adam(a.policy.parameters(),lr=3e-4)
        a.policy_sched=torch.optim.lr_scheduler.CosineAnnealingLR(a.policy_opt,T_max=rounds)
    print(f"[POMDP控制博弈v4] rounds={rounds} pop={pop} | state={STATE_DIM}D actions={N_ACTIONS} hidden={HIDDEN_SIZE}")
    for ro in range(1,rounds+1):
        ec=ent_start-(ent_start-ent_end)*(ro/rounds)
        for a in pa: a.ent_coef=ec; collect_and_train(a,env,True,random.choice(pb),1500,ec)
        for b in pb: b.ent_coef=ec; collect_and_train(b,env,False,random.choice(pa),1500,ec)
        for a in pa+pb: a.policy_sched.step()
        if ro%print_every==0 and ro>0:
            sa,sb=evolve(pa,pb,env)
            ea=[f"{a.elo:.0f}" for a in pa]; eb=[f"{b.elo:.0f}" for b in pb]
            ba=pa[max(range(pop),key=lambda i:sa[i])]; bb=pb[max(range(pop),key=lambda i:sb[i])]
            w,d=0,0
            for _ in range(30):
                oa,ob=env.reset()
                ha=torch.zeros(HIDDEN_SIZE); hb=torch.zeros(HIDDEN_SIZE)
                for _ in range(env.max_steps):
                    la,lb=env.legal_a(),env.legal_b()
                    if not la or not lb: break
                    ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
                    aa,_,_,ha=ba.policy.get_action(ta,ha,la,sample=True)
                    ab,_,_,hb=bb.policy.get_action(tb,hb,lb,sample=True)
                    oa,ra,da,ob,rb,db,_=env.step(aa,ab)
                    if da or db: break
                if env.winner=="A": w+=1
                elif env.winner is None: d+=1
            bw=30-w-d
            print(f"[Evolve {ro:3d}/{rounds}] best: A{w} B{bw} D{d} ElosA={ea} ElosB={eb} ent={ec:.3f}")
    return pa,pb


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--rounds",type=int,default=200)
    ap.add_argument("--pop",type=int,default=3)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    env=ControlEnvV4()
    pa,pb=train(env,rounds=args.rounds,pop=args.pop,seed=args.seed)
    d=os.path.join(os.path.dirname(__file__),"..","models"); os.makedirs(d,exist_ok=True)
    for i,p in enumerate(pa): torch.save(p.policy.state_dict(),os.path.join(d,f"v4_a{i}.pt"))
    for i,p in enumerate(pb): torch.save(p.policy.state_dict(),os.path.join(d,f"v4_b{i}.pt"))
    print("[保存] v4种群模型")
