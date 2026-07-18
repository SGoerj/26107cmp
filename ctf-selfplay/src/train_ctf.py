"""
CTF self-play: 双方各用独立网络, 从同一预训练权重出发, 各自优化。

和之前的关键差异:
  - 双方各自一个 ActorCritic (ac_a, ac_b), 各自 optimizer
  - 都从同一 best_ppo.pt 加载, 但训练后自然分化
  - A 的数据只训 ac_a, B 的数据只训 ac_b
  - 更高学习率 (3e-4)+更高熵系数 (0.02) 允许更快分化

用法:
  python src/train_ctf.py --model models/best_ppo.pt --rounds 100 --seed 0
"""

import os,sys,argparse
import numpy as np
import torch,torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from ctf_env import CTFEnv, STATE_DIM, N_ACTIONS

class ActorCritic(nn.Module):
    def __init__(self,state_dim,n_actions):
        super().__init__()
        self.feature=nn.Sequential(nn.Linear(state_dim,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.actor=nn.Linear(64,n_actions); self.critic=nn.Linear(64,1)
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


def compute_gae(rewards,values,dones,last_value,gamma=0.99,lam=0.95):
    adv=np.zeros(len(rewards),dtype=np.float32); gae=0.0; nv=last_value
    for t in reversed(range(len(rewards))):
        nt=0.0 if dones[t] else 1.0
        delta=rewards[t]+gamma*nv*nt-values[t]; gae=delta+gamma*lam*nt*gae; adv[t]=gae; nv=values[t]
    return adv,adv+np.array(values,dtype=np.float32)


def ppo_update(ac,opt,S,A,LP,ADV,RET,legals,update_epochs=10,batch_size=256,clip_ratio=0.2,
               value_coef=0.5,entropy_coef=0.02,max_grad_norm=0.5,n_actions=11):
    device=next(ac.parameters()).device
    adv_t=torch.tensor(ADV,dtype=torch.float32).to(device)
    adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-8)
    ret_t=torch.tensor(RET,dtype=torch.float32).to(device)
    St=torch.tensor(np.stack(S)).float().to(device)
    At=torch.tensor(A).to(device); LPt=torch.tensor(LP).to(device)
    n=len(S)
    for _ in range(update_epochs):
        idx=torch.randperm(n)
        for s in range(0,n,batch_size):
            mb=idx[s:s+batch_size]
            leg_mask=torch.zeros(len(mb),n_actions,dtype=torch.bool,device=device)
            for j,i in enumerate(mb.tolist()):
                for a in legals[i]: leg_mask[j,a]=True
            nlp,vals,ent=ac.evaluate(St[mb],At[mb],leg_mask)
            ratio=torch.exp(nlp-LPt[mb])
            s1=ratio*adv_t[mb]; s2=torch.clamp(ratio,1-clip_ratio,1+clip_ratio)*adv_t[mb]
            policy_loss=-torch.min(s1,s2).mean()
            value_loss=nn.MSELoss()(vals,ret_t[mb])
            loss=policy_loss+value_coef*value_loss-entropy_coef*ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(),max_grad_norm); opt.step()
    return ent.item()


def train(env,rounds=100,rollout_steps=2000,lr=3e-4,seed=0,pretrained=None,print_every=10):
    if seed is not None: np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 两方各自网络, 各自优化器
    ac_a=ActorCritic(STATE_DIM,N_ACTIONS).to(device)
    ac_b=ActorCritic(STATE_DIM,N_ACTIONS).to(device)

    if pretrained and os.path.exists(pretrained):
        ckpt=torch.load(pretrained,map_location=device,weights_only=False)
        ac_a.load_state_dict(ckpt["state_dict"])
        ac_b.load_state_dict(ckpt["state_dict"])
        print(f"[加载预训练] {pretrained} -> ac_a + ac_b")
    else:
        print(f"[无预训练, 从零开始]")

    opt_a=torch.optim.Adam(ac_a.parameters(),lr=lr)
    opt_b=torch.optim.Adam(ac_b.parameters(),lr=lr)

    print(f"[CTF self-play 双网络] rounds={rounds} | state={STATE_DIM}D actions={N_ACTIONS} | lr={lr}")

    for ro in range(1,rounds+1):
        sA,aA,lpA,rA,vA,dB,legalsA=[],[],[],[],[],[],[]
        sB,aB,lpB,rB,vB,dA,legalsB=[],[],[],[],[],[],[]
        step,oA,oB=0,*env.reset()
        # 数据收集: A用ac_a, B用ac_b
        while step<rollout_steps:
            lA=env.legal_a(); lB=env.legal_b()
            if not lA or not lB: oA,oB=env.reset(); continue
            tA=torch.from_numpy(oA).float().to(device); tB=torch.from_numpy(oB).float().to(device)
            with torch.no_grad():
                aa,lpa,va=ac_a.get_action(tA,lA); ab,lpb,vb=ac_b.get_action(tB,lB)
            sA.append(oA.copy()); aA.append(aa); lpA.append(lpa); vA.append(va); legalsA.append(lA)
            sB.append(oB.copy()); aB.append(ab); lpB.append(lpb); vB.append(vb); legalsB.append(lB)
            oA,ra,doneA,oB,rb,doneB,_=env.step(aa,ab)
            rA.append(ra); dB.append(doneA)
            rB.append(rb); dA.append(doneB)
            step+=1
            if doneA or doneB: oA,oB=env.reset()

        # 分别更新
        ent_a=ent_b=0.0
        if len(sA)>0:
            with torch.no_grad():
                st=torch.from_numpy(oA).float().to(device)  # 最终观测
                _,lv=ac_a(st.unsqueeze(0))
            advA,retA=compute_gae(rA,vA,dB,lv.item())
            ent_a=ppo_update(ac_a,opt_a,sA,aA,lpA,advA,retA,legalsA)
        if len(sB)>0:
            with torch.no_grad():
                st=torch.from_numpy(oB).float().to(device)  # 最终观测
                _,lv=ac_b(st.unsqueeze(0))
            advB,retB=compute_gae(rB,vB,dA,lv.item())
            ent_b=ppo_update(ac_b,opt_b,sB,aB,lpB,advB,retB,legalsB)

        if ro%print_every==0 or ro==1:
            # 快速评估: 用各自网络 greedy 跑 30 局, 统计 winner
            wA,wB,wDraw=0,0,0
            for _ in range(30):
                oa,ob=env.reset()
                for _ in range(env.max_steps):
                    la=env.legal_a(); lb=env.legal_b()
                    if not la or not lb: break
                    ta=torch.from_numpy(oa).float().to(device)
                    tb=torch.from_numpy(ob).float().to(device)
                    aa,_,_=ac_a.get_action(ta,la); ab,_,_=ac_b.get_action(tb,lb)
                    oa,ra,da,ob,rb,db,_=env.step(aa,ab)
                    if da or db: break
                if env.winner=="A": wA+=1
                elif env.winner=="B": wB+=1
                else: wDraw+=1
            print(f"[Round {ro:3d}/{rounds}] A:{wA} B:{wB} draw:{wDraw} | "
                  f"A_avg={sum(rA)/len(rA):+.1f} B_avg={sum(rB)/len(rB):+.1f} "
                  f"entA={ent_a:.2f} entB={ent_b:.2f}")

    return ac_a,ac_b


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=None); ap.add_argument("--rounds",type=int,default=100)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    env=CTFEnv()
    ac_a,ac_b=train(env,rounds=args.rounds,seed=args.seed,pretrained=args.model)
    d=os.path.join(os.path.dirname(__file__),"..","models"); os.makedirs(d,exist_ok=True)
    torch.save({"state_dict":ac_a.state_dict()},os.path.join(d,"ctf_a.pt"))
    torch.save({"state_dict":ac_b.state_dict()},os.path.join(d,"ctf_b.pt"))
    print("[保存] ctf_a.pt + ctf_b.pt")