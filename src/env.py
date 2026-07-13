r"""
攻防对抗环境：模拟渗透测试 agent 在内网拓扑中的路径决策。

拓扑 (5 节点，最深 3 跳——保证随机探索能偶遇目标):

          [0] Internet (检测=0.05)      ← 入口
            /           \
     [1] Web (0.7)    [2] VPN (0.1)       ← 两条入口路由
            \           /   \
             \      [3] Jump (0.15)       ← 安全跳板
              \       /
           [4] DB-Master (0.9)   ★ flag  ← 目标

两条路线的期望奖励（数学期望）:
  🔴 快路径 (2步) : 0 → Web(0.7) → DB(0.9)
     期望: 10 - 2步 - 0.7x5 - 0.9x5 = +0.0

  🟢 安全路径 (3步) : 0 → VPN(0.1) → Jump(0.15) → DB(0.9)
     期望: 10 - 3步 - 0.1x5 - 0.15x5 - 0.9x5 = +1.25

核心决策: 在 Internet 节点选 Web 还是 VPN?
  → Web 一步到 DB, 但 Web 检测率 0.7 太高
  → VPN 多走一步 Jump 中转, 但全程低检测
  → DQN 必须学会"多走一步反而更划算"
"""

import numpy as np

ADJ = [
    [1, 2],        # 0: Internet  → Web / VPN
    [0, 4],        # 1: Web       → Internet / DB (直达!)
    [0, 3],        # 2: VPN       → Internet / Jump
    [2, 4],        # 3: Jump      → VPN / DB
    [1, 3],        # 4: DB-Master ★ flag
]

NODE_NAMES = [
    "Internet", "Web", "VPN", "Jump", "DB-Master",
]

DETECTION = [0.05, 0.7, 0.1, 0.15, 0.9]

TARGET_NODES = {4}


class NetworkEnv:
    """渗透路径决策环境。"""

    def __init__(self, max_steps: int = 20):
        self.n_nodes = len(ADJ)
        self.observation_space = self.n_nodes + 1
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> int:
        self.current = 0
        self.steps = 0
        self.detections = 0
        self._visited = set([0])
        return self.current

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if action not in ADJ[self.current]:
            reward = -5.0
            done = False
            info = {"invalid": True}
            return self.current, reward, done, info

        self.current = action
        self.steps += 1

        detected = np.random.random() < DETECTION[self.current]
        if detected:
            self.detections += 1

        if self.current in TARGET_NODES:
            reward = 10.0
            done = True
        else:
            reward = -1.0
            done = False

        if detected:
            reward -= 5.0

        # 探索奖励: 到达本轮还没踩过的节点 +0.5 (帮助 DQN 发现拓扑)
        if self.current not in self._visited:
            reward += 0.5
            self._visited.add(self.current)

        if self.steps >= self.max_steps:
            done = True

        info = {
            "node": self.current,
            "node_name": NODE_NAMES[self.current],
            "detected": detected,
            "total_detections": self.detections,
            "steps": self.steps,
            "invalid": False,
        }
        return self.current, reward, done, info

    def legal_actions(self) -> list[int]:
        return ADJ[self.current]

    def state_info(self) -> dict:
        return {
            "node": self.current,
            "node_name": NODE_NAMES[self.current],
            "legal_moves": [NODE_NAMES[a] for a in ADJ[self.current]],
            "detection_risk": DETECTION[self.current],
            "detections_so_far": self.detections,
        }