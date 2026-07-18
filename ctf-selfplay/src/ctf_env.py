r"""
对称 CTF 环境: 每方只看对方 4 节点, 27D/11动作, 直接加载单智能体 PPO 权重。

与单智能体 env_v2 完全兼容: 同一 state 维度, 同一动作空间。
A 攻 B 的网络, B 攻 A 的网络。双方同时在对方网络中操作。
先 EXFIL 对方 flag 者胜。

之前训好的 best_ppo.pt / best_fitted.pt 可直接加载。
"""

import numpy as np

# 动作 (与 env_v2 一致)
RECON=0; VULN_SCAN=1; EXPLOIT_CVE=2; EXPLOIT_CREDS=3; EXPLOIT_SQLI=4
PRIVESC=5; EXFIL=6; MOVE_BASE=7
N_ACTIONS=11
STATE_DIM=27

N_PER_TEAM=4
SERVICE_MAP={"ssh":0,"http":1,"db":2}
VULN_MAP={"cve":0,"creds":1,"sqli":2}
_VULN_TO_EXPLOIT={"cve":EXPLOIT_CVE,"creds":EXPLOIT_CREDS,"sqli":EXPLOIT_SQLI}
_ACTION_TO_VULN={EXPLOIT_CVE:"cve",EXPLOIT_CREDS:"creds",EXPLOIT_SQLI:"sqli"}

# 单方网络模板 (双方镜像)
_NET_TEMPLATE=[
    {"name":"Gateway","services":{"ssh","http"},"vulns":{"creds"},"detection":0.05,"has_flag":False,"adj":[1,2]},
    {"name":"Web","services":{"http"},"vulns":{"cve"},"detection":0.70,"has_flag":False,"adj":[0,3]},
    {"name":"Jump","services":{"ssh"},"vulns":{"creds"},"detection":0.15,"has_flag":False,"adj":[0,3]},
    {"name":"DC","services":{"db"},"vulns":{"cve"},"detection":0.70,"has_flag":True,"adj":[1,2]},
]

ACTION_NAMES=["RECON","VULN_SCAN","EXPLOIT_CVE","EXPLOIT_CREDS","EXPLOIT_SQLI","PRIVESC","EXFIL",
              "MOVE_0","MOVE_1","MOVE_2","MOVE_3"]

class CTFEnv:
    def __init__(self,max_steps=40):
        self.max_steps=max_steps
        self.nodes_a=_NET_TEMPLATE   # A 攻 B 的网络 (B 的网络结构)
        self.nodes_b=_NET_TEMPLATE   # B 攻 A 的网络
        self.reset()

    def reset(self):
        self.steps=0
        self.a_pos=0; self.b_pos=0
        self.a_comp=[False]*4; self.a_comp[0]=True; self.a_root=[False]*4; self.a_root[0]=True
        self.a_recon=[False]*4; self.a_recon[0]=True; self.a_vuln=[False]*4; self.a_vuln[0]=True
        self.b_comp=[False]*4; self.b_comp[0]=True; self.b_root=[False]*4; self.b_root[0]=True
        self.b_recon=[False]*4; self.b_recon[0]=True; self.b_vuln=[False]*4; self.b_vuln[0]=True
        self.det_a=self.nodes_a[0]["detection"]; self.det_b=self.nodes_b[0]["detection"]
        self.det_hist_a=[]; self.det_hist_b=[]
        self.a_visited={0}; self.b_visited={0}
        self.winner=None
        return self._obs_a(),self._obs_b()

    def step(self,aa,ab):
        a_r,a_done=self._exec(aa,True)
        b_r,b_done=self._exec(ab,False)
        self.steps+=1
        if not a_done and not b_done and self.steps>=self.max_steps:
            a_done=b_done=True
        if self.winner=="A": b_done=True
        elif self.winner=="B": a_done=True
        return self._obs_a(),a_r,a_done,self._obs_b(),b_r,b_done,{}

    def _exec(self,action,is_a):
        pos=self.a_pos if is_a else self.b_pos
        comp=self.a_comp if is_a else self.b_comp
        root=self.a_root if is_a else self.b_root
        recon=self.a_recon if is_a else self.b_recon
        vuln=self.a_vuln if is_a else self.b_vuln
        visited=self.a_visited if is_a else self.b_visited
        nodes=self.nodes_a if is_a else self.nodes_b
        team="A" if is_a else "B"
        r=-0.3; done=False
        legal=self._legal(pos,comp,root,recon,vuln,nodes)
        if action not in legal: return r-3.0,False
        node=nodes[pos]
        if action==RECON: recon[pos]=True; r+=0.5
        elif action==VULN_SCAN: vuln[pos]=True; r+=0.5
        elif action in (EXPLOIT_CVE,EXPLOIT_CREDS,EXPLOIT_SQLI):
            vt=_ACTION_TO_VULN[action]
            if vt in node["vulns"]: comp[pos]=True; r+=3.0
            else: r-=2.0
        elif action==PRIVESC: root[pos]=True; r+=5.0
        elif action==EXFIL:
            if pos==3 and root[pos] and node["has_flag"]:
                if self.winner is None: self.winner=team   # 先到者赢, 不被后执行者覆盖
                r+=25.0; done=True
            else: r-=3.0
        elif action>=MOVE_BASE:
            t=action-MOVE_BASE
            if t in node["adj"]:
                if is_a: self.a_pos=t
                else: self.b_pos=t
                if t not in visited: r+=2.0; visited.add(t)
                else: r-=2.0
            else: r-=3.0
        # 检测 (简化: 用节点 base detection)
        m={RECON:0.1,VULN_SCAN:0.1,EXPLOIT_CVE:0.5,EXPLOIT_CREDS:0.5,EXPLOIT_SQLI:0.5,PRIVESC:1.0,EXFIL:1.0}.get(action,0.3)
        if np.random.random()<(node["detection"]*m): r-=5.0
        return r,done

    def _legal(self,pos,comp,root,recon,vuln,nodes):
        L=[]; n=nodes[pos]
        if not recon[pos]: L.append(RECON)
        if recon[pos] and not vuln[pos]: L.append(VULN_SCAN)
        if vuln[pos] and not comp[pos]:
            for v in n["vulns"]:
                if v in _VULN_TO_EXPLOIT: L.append(_VULN_TO_EXPLOIT[v])
        if comp[pos] and not root[pos]: L.append(PRIVESC)
        if root[pos] and n["has_flag"] and pos==3: L.append(EXFIL)
        if comp[pos] or root[pos]:
            for t in n["adj"]: L.append(MOVE_BASE+t)
        return L

    def legal_a(self): return self._legal(self.a_pos,self.a_comp,self.a_root,self.a_recon,self.a_vuln,self.nodes_a)
    def legal_b(self): return self._legal(self.b_pos,self.b_comp,self.b_root,self.b_recon,self.b_vuln,self.nodes_b)

    def _obs(self,pos,comp,root,recon,vuln,nodes,is_a):
        s=np.zeros(STATE_DIM,dtype=np.float32)
        s[pos]=1.0
        for i in range(4):
            s[4+i]=1.0 if comp[i] else 0.0
        for i in range(4):
            s[8+i]=1.0 if recon[i] else 0.0
        for i in range(4):
            s[12+i]=1.0 if vuln[i] else 0.0
        if recon[pos]:
            for svc in nodes[pos]["services"]:
                idx=SERVICE_MAP.get(svc,-1)
                if idx>=0:
                    s[16+idx]=1.0
        if vuln[pos]:
            for v in nodes[pos]["vulns"]:
                idx=VULN_MAP.get(v,-1)
                if idx>=0:
                    s[19+idx]=1.0
        s[22]=min(self.steps,self.max_steps)/self.max_steps
        visited=self.a_visited if is_a else self.b_visited
        for i in range(4):
            s[23+i]=1.0 if i in visited else 0.0
        return s

    def _obs_a(self): return self._obs(self.a_pos,self.a_comp,self.a_root,self.a_recon,self.a_vuln,self.nodes_a,True)
    def _obs_b(self): return self._obs(self.b_pos,self.b_comp,self.b_root,self.b_recon,self.b_vuln,self.nodes_b,False)


if __name__=="__main__":
    np.random.seed(0)
    e=CTFEnv()
    print(f"CTF 27D/11动, 和单智能体 PPO 兼容")
    for s in range(e.max_steps):
        al=e.legal_a(); bl=e.legal_b()
        if not al or not bl: break
        aa=np.random.choice(al); ab=np.random.choice(bl)
        _,ar,da,_,br,db,_=e.step(aa,ab)
        if da or db: break
    print(f"winner={e.winner}")