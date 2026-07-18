r"""
能力异构控制博弈 v3: A 快攻型, B 隐蔽型。

拓扑 (同 v1, 6节点):
  [0]----[1]----[2]
   |  \   |   /  |
  [3]----[4]----[5]

A 从节点0, B 从节点5 出发。

能力不对称:
  A (快攻): ATTACK+4, SECURE+1(锁2), OVERLOAD+7(冷却5), 检测罚-3
  B (隐蔽): ATTACK+2, SECURE+2(锁4), OVERLOAD+5(冷却3), 检测罚-1

终局: 控制节点多者胜 (同质节点, 同 v1)。
"""

import numpy as np

N=6
ADJ=[[1,3,4],[0,2,4],[1,4,5],[0,4],[0,1,2,3,5],[2,4]]

ATTACK=0; SECURE=1; OVERLOAD=2
N_ACT_TYPES=3; N_ACTIONS=N_ACT_TYPES*N; STATE_DIM=N*3+2

# 能力配置
A_CONF={"attack":4,"secure":1,"secure_lock":3,"overload":7,"overload_cd":6,"det_penalty":-3.0}
B_CONF={"attack":2,"secure":2,"secure_lock":5,"overload":5,"overload_cd":4,"det_penalty":-1.0}

class ControlEnvV3:
    def __init__(self,max_steps=60):
        self.max_steps=max_steps
        self.reset()

    def reset(self):
        self.control=[0.0]*N; self.control[0]=+6.0; self.control[5]=-6.0
        self.locks=[0]*N
        self.a_cooldown=0; self.b_cooldown=0
        self.steps=0
        self.winner=None
        return self._obs(True),self._obs(False)

    def step(self,aa,ab):
        aT,aN=aa//N,aa%N; bT,bN=ab//N,ab%N

        # A (快攻型)
        if aT==ATTACK and self.locks[aN]==0:
            self.control[aN]=np.clip(self.control[aN]+A_CONF["attack"],-6,6)
        elif aT==SECURE:
            self.control[aN]=np.clip(self.control[aN]+A_CONF["secure"],-6,6); self.locks[aN]=A_CONF["secure_lock"]
        elif aT==OVERLOAD and self.locks[aN]==0 and self.a_cooldown==0:
            self.control[aN]=np.clip(self.control[aN]+A_CONF["overload"],-6,6); self.a_cooldown=A_CONF["overload_cd"]

        # B (隐蔽型)
        if bT==ATTACK and self.locks[bN]==0:
            self.control[bN]=np.clip(self.control[bN]-B_CONF["attack"],-6,6)
        elif bT==SECURE:
            self.control[bN]=np.clip(self.control[bN]-B_CONF["secure"],-6,6); self.locks[bN]=B_CONF["secure_lock"]
        elif bT==OVERLOAD and self.locks[bN]==0 and self.b_cooldown==0:
            self.control[bN]=np.clip(self.control[bN]-B_CONF["overload"],-6,6); self.b_cooldown=B_CONF["overload_cd"]

        self.a_cooldown=max(0,self.a_cooldown-1)
        self.b_cooldown=max(0,self.b_cooldown-1)
        self.locks=[max(0,l-1) for l in self.locks]
        self.steps+=1

        a_ctrl=sum(1 for c in self.control if c>0)
        b_ctrl=sum(1 for c in self.control if c<0)
        done=(self.steps>=self.max_steps) or a_ctrl==6 or b_ctrl==6
        if a_ctrl>b_ctrl and done: self.winner="A"
        elif b_ctrl>a_ctrl and done: self.winner="B"

        a_r=self._r(True,a_ctrl,b_ctrl,done)
        b_r=self._r(False,b_ctrl,a_ctrl,done)
        return self._obs(True),a_r,done,self._obs(False),b_r,done,{}

    def _r(self,is_a,my_ctrl,opp_ctrl,done):
        conf=A_CONF if is_a else B_CONF
        r=my_ctrl*0.5-0.1
        # 检测惩罚: A重 B轻 (模拟攻击暴露度差异)
        # 简化: 每步有概率被检测, A罚重B罚轻
        if np.random.random()<0.08:
            r+=conf["det_penalty"]
        if done:
            if my_ctrl>opp_ctrl: r+=10.0
            elif my_ctrl<opp_ctrl: r-=10.0
        return r

    def _obs(self,is_a):
        s=np.zeros(STATE_DIM,dtype=np.float32)
        for i in range(N): s[i]=self.control[i]/6.0
        for i in range(N): s[N+i]=float(self.locks[i])
        cd=self.a_cooldown if is_a else self.b_cooldown
        for i in range(N): s[2*N+i]=float(cd>0)
        max_cd=A_CONF["overload_cd"] if is_a else B_CONF["overload_cd"]
        s[3*N]=cd/max_cd
        s[3*N+1]=self.steps/self.max_steps
        return s

    def legal_a(self):
        L=[]
        for t in range(N_ACT_TYPES):
            for n in range(N):
                if t in (ATTACK,OVERLOAD) and self.locks[n]>0: continue
                if t==OVERLOAD and self.a_cooldown>0: continue
                L.append(t*N+n)
        return L
    def legal_b(self):
        L=[]
        for t in range(N_ACT_TYPES):
            for n in range(N):
                if t in (ATTACK,OVERLOAD) and self.locks[n]>0: continue
                if t==OVERLOAD and self.b_cooldown>0: continue
                L.append(t*N+n)
        return L


if __name__=="__main__":
    np.random.seed(0)
    e=ControlEnvV3()
    print(f"能力异构博弈v3 {STATE_DIM}D/{N_ACTIONS}动 max_steps={e.max_steps}")
    print(f"A(快攻): ATK+{A_CONF['attack']} SEC+{A_CONF['secure']} OVL+{A_CONF['overload']} cd={A_CONF['overload_cd']} det={A_CONF['det_penalty']}")
    print(f"B(隐蔽): ATK+{B_CONF['attack']} SEC+{B_CONF['secure']} OVL+{B_CONF['overload']} cd={B_CONF['overload_cd']} det={B_CONF['det_penalty']}")
    for s in range(e.max_steps):
        al=e.legal_a(); bl=e.legal_b()
        if not al or not bl: break
        aa=np.random.choice(al); ab=np.random.choice(bl)
        _,ar,da,_,br,db,_=e.step(aa,ab)
        if da or db: break
    print(f"winner={e.winner} A_ctrl={sum(1 for c in e.control if c>0)} B_ctrl={sum(1 for c in e.control if c<0)}")
