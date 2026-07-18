r"""
异构控制博弈 v2: 8节点, 2高价值, 加权终局。

拓扑:
  [0]----[1]----[2]----[3H]
   |      |      |      |
  [4H]---[5]----[6]----[7]

A 从节点0出发, B 从节点7出发。
节点3和4是高价值(终局计2分, 普通计1分)。
节点3在B侧(B守), 节点4在A侧(A守)。

动作 (3类 × 8节点 = 24):
  ATTACK <n>:   控制值 ±3
  SECURE <n>:   控制值 ±1, 锁2步
  OVERLOAD <n>: 控制值 ±6, 冷却3步

终局: 控制价值高者胜 (高价值节点×2 + 普通节点×1)
"""

import numpy as np

N=8
ADJ=[
    [1,4],          # 0
    [0,2,5],        # 1
    [1,3,6],        # 2
    [2,7],          # 3 (高价值)
    [0,5],          # 4 (高价值)
    [1,4,6],        # 5
    [2,5,7],        # 6
    [3,6],          # 7
]

# 节点价值 (高价值=2, 普通=1)
NODE_VALUE=[1,1,1,2, 2,1,1,1]

ATTACK=0; SECURE=1; OVERLOAD=2
N_ACT_TYPES=3; N_ACTIONS=N_ACT_TYPES*N
STATE_DIM=N*3+2   # 8控制+8锁+8冷却标记+1冷却值+1步数=26

class ControlEnvV2:
    def __init__(self,max_steps=80):
        self.max_steps=max_steps
        self.reset()

    def reset(self):
        self.control=[0.0]*N; self.control[0]=+6.0; self.control[7]=-6.0
        self.locks=[0]*N
        self.a_cooldown=0; self.b_cooldown=0
        self.steps=0
        self.winner=None
        return self._obs(True),self._obs(False)

    def step(self,aa,ab):
        aT,aN=aa//N,aa%N; bT,bN=ab//N,ab%N

        # A
        if aT==ATTACK and self.locks[aN]==0:
            self.control[aN]=np.clip(self.control[aN]+3,-6,6)
        elif aT==SECURE:
            self.control[aN]=np.clip(self.control[aN]+1,-6,6); self.locks[aN]=3
        elif aT==OVERLOAD and self.locks[aN]==0 and self.a_cooldown==0:
            self.control[aN]=np.clip(self.control[aN]+6,-6,6); self.a_cooldown=4

        # B
        if bT==ATTACK and self.locks[bN]==0:
            self.control[bN]=np.clip(self.control[bN]-3,-6,6)
        elif bT==SECURE:
            self.control[bN]=np.clip(self.control[bN]-1,-6,6); self.locks[bN]=3
        elif bT==OVERLOAD and self.locks[bN]==0 and self.b_cooldown==0:
            self.control[bN]=np.clip(self.control[bN]-6,-6,6); self.b_cooldown=4

        self.a_cooldown=max(0,self.a_cooldown-1)
        self.b_cooldown=max(0,self.b_cooldown-1)
        self.locks=[max(0,l-1) for l in self.locks]
        self.steps+=1

        # 终局: 加权计分
        a_score=self._score("A"); b_score=self._score("B")
        done=(self.steps>=self.max_steps) or a_score>=9 or b_score>=9
        if a_score>b_score and done: self.winner="A"
        elif b_score>a_score and done: self.winner="B"

        a_r=self._r(True,a_score,b_score,done)
        b_r=self._r(False,b_score,a_score,done)
        return self._obs(True),a_r,done,self._obs(False),b_r,done,{}

    def _score(self,who):
        s=0
        for i in range(N):
            if who=="A" and self.control[i]>0: s+=NODE_VALUE[i]
            elif who=="B" and self.control[i]<0: s+=NODE_VALUE[i]
        return s

    def _r(self,is_a,my,opp,done):
        r=my*0.3-0.1
        if done:
            if my>opp: r+=10.0
            elif my<opp: r-=10.0
        return r

    def _obs(self,is_a):
        s=np.zeros(STATE_DIM,dtype=np.float32)
        for i in range(N): s[i]=self.control[i]/6.0
        for i in range(N): s[N+i]=float(self.locks[i])
        cd=self.a_cooldown if is_a else self.b_cooldown
        for i in range(N): s[2*N+i]=float(cd>0)
        s[3*N]=cd/4.0
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
    e=ControlEnvV2()
    print(f"异构控制博弈v2 {STATE_DIM}D/{N_ACTIONS}动 max_steps={e.max_steps}")
    print(f"高价值节点: 3(B侧), 4(A侧) 价值=2")
    for s in range(e.max_steps):
        al=e.legal_a(); bl=e.legal_b()
        if not al or not bl: break
        aa=np.random.choice(al); ab=np.random.choice(bl)
        _,ar,da,_,br,db,_=e.step(aa,ab)
        if da or db: break
    print(f"winner={e.winner} A_score={e._score('A')} B_score={e._score('B')}")
