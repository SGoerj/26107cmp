"""
策略分析: 加载训练好的模型, 跑多局采样对弈, 同时统计双方策略模式。

一次对弈同时记录 A 和 B 的动作分布 + 胜负, 口径一致。

用法:
  python src/analyze.py --games 50
"""

import os,sys,argparse,collections
import numpy as np
import torch

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from env import ControlEnv, STATE_DIM, N_ACTIONS, N, ATTACK, SECURE, OVERLOAD
from train_full import PolicyNet

ACT_NAMES={ATTACK:"ATTACK",SECURE:"SECURE",OVERLOAD:"OVERLOAD"}

def load_agent(path,device):
    net=PolicyNet().to(device)
    sd=torch.load(path,map_location=device,weights_only=False)
    net.load_state_dict(sd); net.eval()
    return net

def new_stats():
    return {
        "act_type":collections.Counter(),
        "node_target":collections.Counter(),
        "overload_early":0,"overload_late":0,
        "wins":0,"losses":0,"draws":0,"win_steps":[],
    }

def record_action(stats,action,step,max_steps):
    t=action//N; n=action%N
    stats["act_type"][t]+=1
    stats["node_target"][n]+=1
    if t==OVERLOAD:
        if step<max_steps//2: stats["overload_early"]+=1
        else: stats["overload_late"]+=1

def analyze(a,b,env,games=50,device="cpu"):
    sa,sb=new_stats(),new_stats()
    for g in range(games):
        oa,ob=env.reset()
        steps=0
        for step in range(env.max_steps):
            la,lb=env.legal_a(),env.legal_b()
            if not la or not lb: break
            ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
            aa,_,_=a.act(ta,la,sample=True)
            ab,_,_=b.act(tb,lb,sample=True)
            record_action(sa,aa,step,env.max_steps)
            record_action(sb,ab,step,env.max_steps)
            oa,ra,da,ob,rb,db,_=env.step(aa,ab)
            steps+=1
            if da or db: break
        if env.winner=="A": sa["wins"]+=1; sb["losses"]+=1; sa["win_steps"].append(steps)
        elif env.winner=="B": sb["wins"]+=1; sa["losses"]+=1; sb["win_steps"].append(steps)
        else: sa["draws"]+=1; sb["draws"]+=1

    print("\n"+"="*70)
    print(f"  策略分析 ({games}局采样对弈, 双方口径一致)")
    print("="*70)

    for name,stats in [("A",sa),("B",sb)]:
        total=sum(stats["act_type"].values())
        print(f"\n--- {name} 方策略 ---")
        print(f"  胜负: {stats['wins']}胜 {stats['losses']}负 {stats['draws']}平")
        if stats["win_steps"]:
            print(f"  赢局平均步数: {np.mean(stats['win_steps']):.1f}")
        print(f"  动作类型分布:")
        for t in [ATTACK,SECURE,OVERLOAD]:
            cnt=stats["act_type"][t]
            print(f"    {ACT_NAMES[t]:<10} {cnt:5d} ({cnt/total*100:.0f}%)")
        print(f"  节点偏好:")
        for n in range(N):
            cnt=stats["node_target"][n]
            print(f"    节点{n}  {cnt:5d} ({cnt/total*100:.0f}%)")
        print(f"  OVERLOAD 时机: 前半{stats['overload_early']} 后半{stats['overload_late']}")

    print(f"\n--- 总计 ---")
    print(f"  A {sa['wins']}胜  B {sb['wins']}胜  {sa['draws']}平")
    print("="*70)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--games",type=int,default=50)
    ap.add_argument("--a",default="full_a0.pt")
    ap.add_argument("--b",default="full_b0.pt")
    args=ap.parse_args()
    d=os.path.join(os.path.dirname(__file__),"..","models")
    apath=os.path.join(d,args.a) if not os.path.isabs(args.a) else args.a
    bpath=os.path.join(d,args.b) if not os.path.isabs(args.b) else args.b
    if not os.path.exists(apath) or not os.path.exists(bpath):
        print(f"❌ 找不到模型"); sys.exit(1)
    device="cpu"
    a=load_agent(apath,device); b=load_agent(bpath,device)
    env=ControlEnv()
    analyze(a,b,env,games=args.games,device=device)
