"""
演示一局控制博弈对弈: 加载训练好的种群模型, 逐步打印双方动作和控制值变化。

用法:
  python src/demo.py --rounds 1 --sample
  python src/demo.py --rounds 1          # greedy
"""

import os,sys,argparse
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

def play_one(env,a,b,sample=False,device="cpu"):
    oa,ob=env.reset()
    print(f"\n{'='*70}")
    print(f"  控制博弈对弈演示 ({'采样' if sample else 'greedy'})")
    print(f"{'='*70}")
    print(f"  {'步':<4} {'A动作':<16} {'B动作':<16} {'控制值(6节点)'}")
    print(f"  {'':<4} {'':<16} {'':<16} {[f'{c:+.0f}' for c in env.control]}")
    print("-"*70)

    for step in range(env.max_steps):
        la=env.legal_a(); lb=env.legal_b()
        if not la or not lb:
            print(f"  {step:<4} [无合法动作]"); break
        ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
        if sample:
            aa,_,_=a.act(ta,la,sample=True); ab,_,_=b.act(tb,lb,sample=True)
        else:
            aa=a.act(ta,la,sample=False); ab=b.act(tb,lb,sample=False)

        aT,aN=aa//N,aa%N; bT,bN=ab//N,ab%N
        a_name=f"{ACT_NAMES.get(aT,'?')}_{aN}"
        b_name=f"{ACT_NAMES.get(bT,'?')}_{bN}"

        oa,ra,da,ob,rb,db,_=env.step(aa,ab)
        ctrl=[f'{c:+.0f}' for c in env.control]
        print(f"  {step:<4} {a_name:<16} {b_name:<16} {ctrl}")

        if da or db: break

    print("-"*70)
    a_ctrl=sum(1 for c in env.control if c>0)
    b_ctrl=sum(1 for c in env.control if c<0)
    print(f"  终局: A控制{a_ctrl}节点 B控制{b_ctrl}节点 | winner={env.winner or '平局'}")
    print(f"  控制值: {[f'{c:+.1f}' for c in env.control]}")
    print(f"{'='*70}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",default="full_a0.pt")
    ap.add_argument("--b",default="full_b0.pt")
    ap.add_argument("--sample",action="store_true",help="用采样而非greedy")
    args=ap.parse_args()

    d=os.path.join(os.path.dirname(__file__),"..","models")
    apath=os.path.join(d,args.a) if not os.path.isabs(args.a) else args.a
    bpath=os.path.join(d,args.b) if not os.path.isabs(args.b) else args.b

    if not os.path.exists(apath) or not os.path.exists(bpath):
        print(f"❌ 找不到模型 {apath} 或 {bpath}")
        print("   先运行 train_full.py 训练"); sys.exit(1)

    device="cpu"
    a=load_agent(apath,device); b=load_agent(bpath,device)
    env=ControlEnv()

    # 跑3局看策略稳定性
    for i in range(3):
        play_one(env,a,b,sample=args.sample,device=device)
