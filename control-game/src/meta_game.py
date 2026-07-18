"""
元博弈分析: 用已训练的种群模型构建收益矩阵, 计算纳什均衡。

对每个环境变体(v1-v4):
  1. 加载种群的 A/B 个体
  2. 个体两两对战, 构建 payoff 矩阵 (A_i vs B_j 的胜率)
  3. 求解矩阵博弈的纳什均衡 (混合策略)
  4. 计算可利用度 (exploitability) = 离均衡多远

用法:
  python src/meta_game.py
"""

import os,sys,argparse,itertools
os.environ["TORCHDYNAMO_DISABLE"]="1"
import numpy as np
import torch
try: torch._dynamo.config.disable=True
except: pass
from scipy.optimize import linprog

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from env import ControlEnv as EnvV1, N_ACTIONS as NA_V1
from env_v2 import ControlEnvV2 as EnvV2, N_ACTIONS as NA_V2
from env_v3 import ControlEnvV3 as EnvV3, N_ACTIONS as NA_V3
from env_v4 import ControlEnvV4 as EnvV4, N_ACTIONS as NA_V4

# v4 需要 GRU 网络, 单独 import
from train_full import PolicyNet as PolicyNetV1
from train_v2 import PolicyNet as PolicyNetV2
from train_v3 import PolicyNet as PolicyNetV3
from train_v4 import PolicyNetRNN, HIDDEN_SIZE as HS_V4

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","models")


def load_pop(tag,env_name,n_pop=3,net_type="mlp"):
    """加载一个种群的 A/B 个体。net_type: mlp-v1/mlp-v2/mlp-v3/gru。"""
    net_cls={"mlp-v1":PolicyNetV1,"mlp-v2":PolicyNetV2,"mlp-v3":PolicyNetV3,"gru":PolicyNetRNN}[net_type]
    agents_a,agents_b=[],[]
    for i in range(n_pop):
        apath=os.path.join(MODELS_DIR,f"{tag}_a{i}.pt")
        bpath=os.path.join(MODELS_DIR,f"{tag}_b{i}.pt")
        if not os.path.exists(apath): break
        a=net_cls(); b=net_cls()
        a.load_state_dict(torch.load(apath,map_location="cpu",weights_only=False))
        b.load_state_dict(torch.load(bpath,map_location="cpu",weights_only=False))
        a.eval(); b.eval()
        agents_a.append(a); agents_b.append(b)
    return agents_a,agents_b


def play_match(a_net,b_net,env,n_games=20,is_gru=False):
    """A 个体 vs B 个体, 跑 n_games 局采样, 返回 A 胜率。"""
    a_wins=0
    for _ in range(n_games):
        oa,ob=env.reset()
        if is_gru:
            ha=torch.zeros(HS_V4); hb=torch.zeros(HS_V4)
        for _ in range(env.max_steps):
            la,lb=env.legal_a(),env.legal_b()
            if not la or not lb: break
            ta=torch.from_numpy(oa).float(); tb=torch.from_numpy(ob).float()
            if is_gru:
                aa,_,_,ha=a_net.get_action(ta,ha,la,sample=True)
                ab,_,_,hb=b_net.get_action(tb,hb,lb,sample=True)
            else:
                aa,_,_=a_net.act(ta,la,sample=True)
                ab,_,_=b_net.act(tb,lb,sample=True)
            oa,ra,da,ob,rb,db,_=env.step(aa,ab)
            if da or db: break
        if env.winner=="A": a_wins+=1
    return a_wins/n_games


def build_payoff_matrix(agents_a,agents_b,env,is_gru=False,n_games=20):
    """构建 payoff 矩阵 M[i][j] = A_i vs B_j 的 A 胜率。"""
    na=len(agents_a); nb=len(agents_b)
    M=np.zeros((na,nb))
    for i in range(na):
        for j in range(nb):
            M[i][j]=play_match(agents_a[i],agents_b[j],env,n_games,is_gru)
    return M


def solve_nash(M):
    """求解零和矩阵博弈的纳什均衡。
    M[i][j] = A 选 i, B 选 j 时 A 的收益(胜率)。
    A 最大化, B 最小化。
    返回 (A混合策略, B混合策略, 均衡值)。
    """
    na,nb=M.shape
    # A 的线性规划: 最大化 v, s.t. sum_i x_i * M[i][j] >= v for all j, sum x_i = 1, x_i >= 0
    # 转化为: minimize -v
    # 变量: [x_0, ..., x_{na-1}, v]
    c=np.zeros(na+1); c[-1]=-1  # minimize -v
    # 约束: M^T @ x - v >= 0  →  -M^T @ x + v <= 0
    A_ub=np.zeros((nb,na+1))
    for j in range(nb):
        A_ub[j,:na]=-M[:,j]
        A_ub[j,-1]=1
    b_ub=np.zeros(nb)
    # 等式约束: sum x_i = 1
    A_eq=np.zeros((1,na+1)); A_eq[0,:na]=1; b_eq=np.array([1.0])
    # 边界: x_i >= 0, v 自由
    bounds=[(0,None)]*na+[(-1,1)]
    res=linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method="highs")
    if not res.success: return None,None,None
    x=res.x[:na]; v=res.x[-1]

    # B 的线性规划: 最小化 v, s.t. sum_j y_j * M[i][j] <= v for all i, sum y_j = 1
    c2=np.zeros(nb+1); c2[-1]=1  # minimize v
    A_ub2=np.zeros((na,nb+1))
    for i in range(na):
        A_ub2[i,:nb]=M[i,:]
        A_ub2[i,-1]=-1
    b_ub2=np.zeros(na)
    A_eq2=np.zeros((1,nb+1)); A_eq2[0,:nb]=1; b_eq2=np.array([1.0])
    bounds2=[(0,None)]*nb+[(-1,1)]
    res2=linprog(c2,A_ub=A_ub2,b_ub=b_ub2,A_eq=A_eq2,b_eq=b_eq2,bounds=bounds2,method="highs")
    if not res2.success: return x,None,v
    y=res2.x[:nb]
    return x,y,v


def exploitability(M,x,y):
    """计算可利用度: 如果双方偏离均衡, 对方能多赢多少。
    exploit_A = max_j sum_i x_i * M[i][j] - v (B 偏离时 A 能多赢)
    exploit_B = v - min_i sum_j y_j * M[i][j] (A 偏离时 B 能多赢)
    均衡时两者都为 0。
    """
    if x is None or y is None: return None
    v=x@M@y
    best_a=(M@y).max()  # A 对 y 的最优响应值
    best_b=(x@M).min()  # B 对 x 的最优响应值
    exploit=max(abs(best_a-v),abs(v-best_b))
    return exploit


def analyze_env(name,tag,env,net_type):
    print(f"\n{'='*70}")
    print(f"  {name} 元博弈分析")
    print(f"{'='*70}")
    agents_a,agents_b=load_pop(tag,name,net_type=net_type)
    if len(agents_a)==0:
        print(f"  ⚠️ 找不到 {tag} 模型, 跳过")
        return
    is_gru=(net_type=="gru")
    print(f"  加载 {len(agents_a)} 个A个体, {len(agents_b)} 个B个体")

    M=build_payoff_matrix(agents_a,agents_b,env,is_gru,n_games=20)
    print(f"\n  Payoff 矩阵 (A胜率):")
    print(f"  {'':>8}",end="")
    for j in range(len(agents_b)): print(f" B{j:>2}",end="")
    print()
    for i in range(len(agents_a)):
        print(f"  A{i:>2}    ",end="")
        for j in range(len(agents_b)):
            print(f" {M[i][j]:.2f}",end="")
        print()

    x,y,v=solve_nash(M)
    if v is not None:
        print(f"\n  纳什均衡值 (A期望胜率): {v:.3f}")
        if x is not None:
            print(f"  A 混合策略: {[f'{xi:.2f}' for xi in x]}")
        if y is not None:
            print(f"  B 混合策略: {[f'{yi:.2f}' for yi in y]}")
        expl=exploitability(M,x,y)
        if expl is not None:
            print(f"  可利用度 (离均衡距离): {expl:.4f}")
            if expl<0.01: print(f"  → 策略已收敛到均衡")
            elif expl<0.05: print(f"  → 接近均衡")
            else: print(f"  → 离均衡较远")
    else:
        print(f"  ⚠️ 纳什均衡求解失败")

    # 均势判断
    avg=M.mean()
    print(f"\n  平均 A 胜率: {avg:.3f}")
    if 0.4<avg<0.6: print(f"  → 双方势均力敌")
    elif avg>=0.6: print(f"  → A 占优")
    else: print(f"  → B 占优")


if __name__=="__main__":
    print("元博弈分析: 用种群模型构建收益矩阵, 求解纳什均衡")

    # v1 同质
    try: analyze_env("v1","full",EnvV1(),"mlp-v1")
    except Exception as e: print(f"  v1 分析失败: {e}")

    # v2 节点异构
    try: analyze_env("v2","v2",EnvV2(),"mlp-v2")
    except Exception as e: print(f"  v2 分析失败: {e}")

    # v3 能力异构
    try: analyze_env("v3","v3",EnvV3(),"mlp-v3")
    except Exception as e: print(f"  v3 分析失败: {e}")

    # v4 POMDP
    try: analyze_env("v4","v4",EnvV4(),"gru")
    except Exception as e: print(f"  v4 分析失败: {e}")
