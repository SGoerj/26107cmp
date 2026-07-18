r"""
网络控制博弈 v3: 加资源冷却 + 缩动作空间。

变化:
  - 去 MOVE, 3动作(ATTACK/SECURE/OVERLOAD) × 6节点 = 18
  - OVERLOAD 冷却 3步 (用后3步不能用)
  - 控制 ±6, ATTACK ±3, OVERLOAD ±6, SECURE ±1
  - 奖励: 0.5/节点/步, 全控制+15, 优势+8

设计理由: OVERLOAD冷却 = 不能无脑刷, 必须选择时机。
          去MOVE = 18动更容易收敛 (24动熵太高)。
"""

import numpy as np

N=6; ATTACK=0; SECURE=1; OVERLOAD=2
N_ACT_TYPES=3; N_ACTIONS=N_ACT_TYPES*N; STATE_DIM=N*3+2

class ControlEnv:
    def __init__(self,max_steps=60):
        self.max_steps=max_steps
        self.reset()

    def reset(self):
        self.control=[0.0]*N; self.control[0]=+6.0; self.control[5]=-6.0
        self.locks=[0]*N
        self.a_cooldown=0; self.b_cooldown=0
        self.steps=0
        self.a_det=0; self.b_det=0
        self.winner=None
        return self._obs(True),self._obs(False)

    def step(self,aa,ab):
        aT,aN=aa//N,aa%N; bT,bN=ab//N,ab%N

        # A
        if aT==ATTACK and self.locks[aN]==0:
            self.control[aN]=np.clip(self.control[aN]+3,-6,6)
        elif aT==SECURE:
            self.control[aN]=np.clip(self.control[aN]+1,-6,6); self.locks[aN]=2  # 锁2步
        elif aT==OVERLOAD and self.locks[aN]==0 and self.a_cooldown==0:
            self.control[aN]=np.clip(self.control[aN]+6,-6,6)
            self.a_cooldown=3

        # B
        if bT==ATTACK and self.locks[bN]==0:
            self.control[bN]=np.clip(self.control[bN]-3,-6,6)
        elif bT==SECURE:
            self.control[bN]=np.clip(self.control[bN]-1,-6,6); self.locks[bN]=2
        elif bT==OVERLOAD and self.locks[bN]==0 and self.b_cooldown==0:
            self.control[bN]=np.clip(self.control[bN]-6,-6,6)
            self.b_cooldown=3

        self.a_cooldown=max(0,self.a_cooldown-1)
        self.b_cooldown=max(0,self.b_cooldown-1)
        self.locks=[max(0,l-1) for l in self.locks]
        self.steps+=1

        a_ctrl=sum(1 for c in self.control if c>0)
        b_ctrl=sum(1 for c in self.control if c<0)
        done=(self.steps>=self.max_steps) or a_ctrl==6 or b_ctrl==6
        if a_ctrl==6: self.winner="A"
        elif b_ctrl==6: self.winner="B"
        elif done:
            if a_ctrl>b_ctrl: self.winner="A"
            elif b_ctrl>a_ctrl: self.winner="B"

        a_r=self._r(True,a_ctrl,b_ctrl,done)
        b_r=self._r(False,b_ctrl,a_ctrl,done)
        return self._obs(True),a_r,done,self._obs(False),b_r,done,{}

    def _r(self,is_a,my,opp,done):
        r=my*0.5 - 0.1
        if done:
            if my==6: r+=15.0
            elif my>opp: r+=8.0
            elif my<opp: r-=8.0
        return r

    def _obs(self,is_a):
        s=np.zeros(STATE_DIM,dtype=np.float32)
        for i in range(N): s[i]=self.control[i]/6.0
        for i in range(N): s[N+i]=float(self.locks[i])
        cd=self.a_cooldown if is_a else self.b_cooldown
        for i in range(N): s[2*N+i]=float(cd>0)  # 冷却标记(每节点复制,冗余但简单)
        s[3*N]=cd/3.0
        s[3*N+1]=self.steps/self.max_steps
        return s

    def _obs_a(self): return self._obs(True)
    def _obs_b(self): return self._obs(False)

    def legal_a(self):
        L=[]
        for t in range(N_ACT_TYPES):
            for n in range(N):
                # 被锁节点不能用 ATTACK/OVERLOAD (避免无效动作)
                if t in (ATTACK,OVERLOAD) and self.locks[n]>0: continue
                # 冷却中不能用 OVERLOAD
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
    e=ControlEnv()
    print(f"v3 {STATE_DIM}D/{N_ACTIONS}动")
    for s in range(e.max_steps):
        al=e.legal_a(); bl=e.legal_b()
        if not al or not bl: break
        aa=np.random.choice(al); ab=np.random.choice(bl)
        _,ar,da,_,br,db,_=e.step(aa,ab)
        if da or db: break
    print(f"winner={e.winner}")