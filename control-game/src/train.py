"""
网络控制博弈 + Fictitious Play 训练。

和 CTF FP 相同架构: 双方独立网络+优化器, 对手从全历史池随机抽。

用法:
  python src/train.py --rounds 100 --seed 0
"""

import os,sys,argparse,copy,random
import numpy as np
import torch,torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from env import ControlEnv, STATE_DIM, N_ACTIONS

class ActorCritic(nn.Module):
    def __init__(self,sd,na):
        super().__init__()
        self.feature=nn.Sequential(nn.Linear(sd,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.actor=nn.Linear(64,na); self.critic=nn.Linear(64,1)
    def forward(self,x):
        f=self.feature(x); return self.actor(f),self.critic(f).squeeze(-1)
    def evaluate(self,states,actions,legal_masks):
        logits,values=self.forward(states)
        mask=torch.full_like(logits,float("-inf"))
        mask[legal_masks]=0.0
        dist=Categorical(logits=logits+mask)
        log_probs=dist.log_prob(actions); entropy=dist.entropy().mean()
        return log_probs,values,entropy
    def get_action(self,x,legal):
        logits,value=self.forward(x.unsqueeze(0)); logits=logits[0]
        mask=torch.full_like(logits,float("-inf"))
        for a in legal: mask[a]=0.0
        probs=torch.softmax(logits+mask,dim=0); dist=Categorical(probs)
        action=dist.sample(); return action.item(),dist.log_prob(action).item(),value.item()
    def act_greedy(self,x,legal):
        logits,_=self.forward(x.unsqueeze(0)); logits=logits[0]
        mask=torch.full_like(logits,float("-inf"))
        for a in legal: mask[a]=0.0
        return (logits+mask).argmax().item()


def compute_gae(rewards,values,dones,lv,gamma=0.99,lam=0.95):
    adv=np.zeros(len(rewards),dtype=np.float32); gae=0.0; nv=lv
    for t in reversed(range(len(rewards))):
        nt=0.0 if dones[t] else 1.0
        delta=rewards[t]+gamma*nv*nt-values[t]; gae=delta+gamma*lam*nt*gae; adv[t]=gae; nv=values[t]
    return adv,adv+np.array(values,dtype=np.float32)


def ppo_update(ac,opt,S,A,LP,ADV,RET,legals,ue=10,bs=256,cr=0.2,vc=0.5,ec=0.02,mg=0.5,n_actions=18):
    device=next(ac.parameters()).device
    at=torch.tensor(ADV,dtype=torch.float32).to(device); at=(at-at.mean())/(at.std()+1e-8)
    rt=torch.tensor(RET,dtype=torch.float32).to(device)
    St=torch.tensor(np.stack(S)).float().to(device)
    At=torch.tensor(A).to(device); LPt=torch.tensor(LP).to(device)
    n=len(S)
    for _ in range(ue):
        idx=torch.randperm(n)
        for s in range(0,n,bs):
            mb=idx[s:s+bs]
            leg_mask=torch.zeros(len(mb),n_actions,dtype=torch.bool,device=device)
            for j,i in enumerate(mb.tolist()):
                for a in legals[i]: leg_mask[j,a]=True
            nlp,vals,ent=ac.evaluate(St[mb],At[mb],leg_mask)
            ratio=torch.exp(nlp-LPt[mb])
            s1=ratio*at[mb]; s2=torch.clamp(ratio,1-cr,1+cr)*at[mb]
            pl=-torch.min(s1,s2).mean(); vl=nn.MSELoss()(vals,rt[mb])
            loss=pl+vc*vl-ec*ent; opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(),mg); opt.step()
    return ent.item()


class HistoryPool:
    def __init__(self): self.snapshots=[]
    def add(self,sd): self.snapshots.append(copy.deepcopy(sd))
    def sample(self,device,state_dim,n_actions):
        if not self.snapshots: return None
        ac=ActorCritic(state_dim,n_actions).to(device)
        ac.load_state_dict(random.choice(self.snapshots))
        ac.eval(); return ac
    def __len__(self): return len(self.snapshots)


def train(env,rounds=100,rollout_steps=2000,lr=3e-4,seed=0,print_every=10):
    if seed is not None: np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac_a=ActorCritic(STATE_DIM,N_ACTIONS).to(device)
    ac_b=ActorCritic(STATE_DIM,N_ACTIONS).to(device)
    opt_a=torch.optim.Adam(ac_a.parameters(),lr=lr)
    opt_b=torch.optim.Adam(ac_b.parameters(),lr=lr)

    pool_a=HistoryPool(); pool_b=HistoryPool()
    pool_a.add(ac_a.state_dict()); pool_b.add(ac_b.state_dict())

    print(f"[控制博弈 FP] rounds={rounds} | state={STATE_DIM}D actions={N_ACTIONS}")

    for ro in range(1,rounds+1):
        # ── 训A (对手=B历史池) ──
        sA,aA,lpA,rA,vA,dA,legalsA=[],[],[],[],[],[],[]
        oa,ob=env.reset(); step=0
        while step<rollout_steps:
            la=env.legal_a(); lb=env.legal_b()
            if not la or not lb: oa,ob=env.reset(); continue
            ta=torch.from_numpy(oa).float().to(device); tb=torch.from_numpy(ob).float().to(device)
            with torch.no_grad():
                aa,lpa,va=ac_a.get_action(ta,la)
            opp_b=pool_b.sample(device,STATE_DIM,N_ACTIONS)
            if opp_b is not None: ab=opp_b.act_greedy(tb,lb)
            else: ab=ac_b.act_greedy(tb,lb)
            sA.append(oa.copy()); aA.append(aa); lpA.append(lpa); vA.append(va); legalsA.append(lA)
            oa,ra,da,ob,rb,db,_=env.step(aa,ab)
            rA.append(ra); dA.append(da)  # 奖励和done在step后收集
            step+=1
            if da or db: oa,ob=env.reset()

        ent_a=0.0
        if len(sA)>0:
            with torch.no_grad():
                st=torch.from_numpy(oa).float().to(device)  # oa是循环结束后的最终状态
                _,lv=ac_a(st.unsqueeze(0))
            advA,retA=compute_gae(rA,vA,dA,lv.item())
            ent_a=ppo_update(ac_a,opt_a,sA,aA,lpA,advA,retA,legalsA)
        pool_a.add(ac_a.state_dict())

        # ── 训B (对手=A历史池) ──
        sB,aB,lpB,rB,vB,dB,legalsB=[],[],[],[],[],[],[]
        oa,ob=env.reset(); step=0
        while step<rollout_steps:
            la=env.legal_a(); lb=env.legal_b()
            if not la or not lb: oa,ob=env.reset(); continue
            ta=torch.from_numpy(oa).float().to(device); tb=torch.from_numpy(ob).float().to(device)
            with torch.no_grad():
                ab,lpb,vb=ac_b.get_action(tb,lb)
            opp_a=pool_a.sample(device,STATE_DIM,N_ACTIONS)
            if opp_a is not None: aa=opp_a.act_greedy(ta,la)
            else: aa=ac_a.act_greedy(ta,la)
            sB.append(ob.copy()); aB.append(ab); lpB.append(lpb); vB.append(vb); legalsB.append(lB)
            oa,ra,da,ob,rb,db,_=env.step(aa,ab)
            rB.append(rb); dB.append(db)
            step+=1
            if da or db: oa,ob=env.reset()

        ent_b=0.0
        if len(sB)>0:
            with torch.no_grad():
                st=torch.from_numpy(ob).float().to(device)  # ob是循环结束后的最终状态
                _,lv=ac_b(st.unsqueeze(0))
            advB,retB=compute_gae(rB,vB,dB,lv.item())
            ent_b=ppo_update(ac_b,opt_b,sB,aB,lpB,advB,retB,legalsB)
        pool_b.add(ac_b.state_dict())

        if ro%print_every==0 or ro==1:
            wA,wB,wDraw=0,0,0
            for _ in range(30):
                oa,ob=env.reset()
                for _ in range(env.max_steps):
                    la=env.legal_a(); lb=env.legal_b()
                    if not la or not lb: break
                    ta=torch.from_numpy(oa).float().to(device)
                    tb=torch.from_numpy(ob).float().to(device)
                    aa=ac_a.act_greedy(ta,la); ab=ac_b.act_greedy(tb,lb)
                    oa,ra,da,ob,rb,db,_=env.step(aa,ab)
                    if da or db: break
                if env.winner=="A": wA+=1
                elif env.winner=="B": wB+=1
                else: wDraw+=1
            print(f"[Round {ro:3d}/{rounds}] A:{wA} B:{wB} draw:{wDraw} "
                  f"pool={len(pool_a.snapshots)} | A_avg={sum(rA)/len(rA):+.1f} B_avg={sum(rB)/len(rB):+.1f} "
                  f"entA={ent_a:.2f} entB={ent_b:.2f}")

    return ac_a,ac_b


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--rounds",type=int,default=100)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    env=ControlEnv()
    ac_a,ac_b=train(env,rounds=args.rounds,seed=args.seed)
    d=os.path.join(os.path.dirname(__file__),"..","models"); os.makedirs(d,exist_ok=True)
    torch.save({"state_dict":ac_a.state_dict()},"models/cg_a.pt")
    torch.save({"state_dict":ac_b.state_dict()},"models/cg_b.pt")
    print("[保存] cg_a.pt + cg_b.pt")