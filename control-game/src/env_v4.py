r"""
部分可观测控制博弈 v4: 双方只看到已访问过的节点。

拓扑 (同 v1, 6节点):
  [0]----[1]----[2]
   |  \   |   /  |
  [3]----[4]----[5]

A 从节点0, B 从节点5 出发。
观测: 只能看到已访问节点的控制值, 未访问节点显示0。
      需要靠 RNN 记忆推断未访问节点的状态。

动作/能力: 同 v1 (同质, 对称)。
"""

import numpy as np

N=6
ADJ=[[1,3,4],[0,2,4],[1,4,5],[0,4],[0,1,2,3,5],[2,4]]

ATTACK=0; SECURE=1; OVERLOAD=2
N_ACT_TYPES=3; N_ACTIONS=N_ACT_TYPES*N; STATE_DIM=N*3+2

class ControlEnvV4:
    def __init__(self,max_steps=60):
        self.max_steps=max_steps
        self.reset()

    def reset(self):
        self.control=[0.0]*N; self.control[0]=+6.0; self.control[5]=-6.0
        self.locks=[0]*N
        self.a_cooldown=0; self.b_cooldown=0
        self.steps=0
        self.winner=None
        # POMDP: 各自的可见集合 (已访问节点)
        self.a_visible={0}; self.b_visible={5}
        return self._obs(True),self._obs(False)

    def step(self,aa,ab):
        aT,aN=aa//N,aa%N; bT,bN=ab//N,ab%N

        # A (同质, 同v1)
        if aT==ATTACK and self.locks[aN]==0:
            self.control[aN]=np.clip(self.control[aN]+3,-6,6)
        elif aT==SECURE:
            self.control[aN]=np.clip(self.control[aN]+1,-6,6); self.locks[aN]=3
        elif aT==OVERLOAD and self.locks[aN]==0 and self.a_cooldown==0:
            self.control[aN]=np.clip(self.control[aN]+6,-6,6); self.a_cooldown=4

        # B (同质, 同v1)
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

        # POMDP: 攻击/加固某节点 = 探索该节点, 加入可见集合
        self.a_visible.add(aN); self.b_visible.add(bN)

        a_ctrl=sum(1 for c in self.control if c>0)
        b_ctrl=sum(1 for c in self.control if c<0)
        done=(self.steps>=self.max_steps) or a_ctrl==6 or b_ctrl==6
        if a_ctrl>b_ctrl and done: self.winner="A"
        elif b_ctrl>a_ctrl and done: self.winner="B"

        a_r=self._r(True,a_ctrl,b_ctrl,done)
        b_r=self._r(False,b_ctrl,a_ctrl,done)
        return self._obs(True),a_r,done,self._obs(False),b_r,done,{}

    def _r(self,is_a,my_ctrl,opp_ctrl,done):
        r=my_ctrl*0.5-0.1
        if done:
            if my_ctrl>opp_ctrl: r+=10.0
            elif my_ctrl<opp_ctrl: r-=10.0
        return r

    def _obs(self,is_a):
        visible=self.a_visible if is_a else self.b_visible
        s=np.zeros(STATE_DIM,dtype=np.float32)
        # 只填已访问节点的控制值, 未访问=0
        for i in range(N):
            if i in visible:
                s[i]=self.control[i]/6.0
        # 锁状态: 只填已访问的
        for i in range(N):
            if i in visible:
                s[N+i]=float(self.locks[i])
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
    e=ControlEnvV4()
    print(f"POMDP控制博弈v4 {STATE_DIM}D/{N_ACTIONS}动 max_steps={e.max_steps}")
    print(f"A初始可见={e.a_visible} B初始可见={e.b_visible}")
    for s in range(e.max_steps):
        al=e.legal_a(); bl=e.legal_b()
        if not al or not bl: break
        aa=np.random.choice(al); ab=np.random.choice(bl)
        oa,ar,da,ob,br,db,_=e.step(aa,ab)
        if da or db: break
    print(f"winner={e.winner} A可见={e.a_visible} B可见={e.b_visible}")
