"""
DQN 智能体：用 Q-network 学习渗透路径的最优策略。

核心思路:
  - Q(s, a) 近似表示"从节点 s 执行动作 a 后的期望未来总奖励"
  - 用经验回放 (Replay Buffer) 打破样本关联
  - 用 Target Network 稳定训练
  - ε-greedy 做探索/利用平衡

输入特征: one-hot 5D (当前节点表征) + 1D (累计被检测次数) = 6D
输出: 5 个 Q 值，对应 5 个可能的目标节点；非邻接的动作会被 mask 掉
"""

import random
from collections import deque

import torch
import torch.nn as nn

from env import NODE_NAMES


# --- 网络结构 ---
class QNetwork(nn.Module):
    """一个小型 MLP: 6 维输入 → 128 → 64 → 5 维输出 (每个节点的 Q 值)"""

    def __init__(self, input_dim: int = 6, output_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- 经验回放 ---
Transition = tuple[int, int, float, int, bool]
#              state   action  reward  next_state  done


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# --- 特征编码 ---
def state_to_tensor(node: int, detections: int) -> torch.Tensor:
    """将 (当前节点索引, 检测次数) 编码为 6 维浮点向量。"""
    one_hot = torch.zeros(5)
    one_hot[node] = 1.0
    det_norm = min(detections, 5) / 5.0       # 归一化到 [0, 1]
    return torch.cat([one_hot, torch.tensor([det_norm])])


# --- ε-greedy 动作选择 ---
def select_action(
    q_net: QNetwork,
    state_tensor: torch.Tensor,
    legal_actions: list[int],
    epsilon: float,
) -> int:
    """ε 概率随机探索；否则选取 Q 值最高的合法动作。"""
    if random.random() < epsilon:
        return random.choice(legal_actions)
    with torch.no_grad():
        q_values = q_net(state_tensor.unsqueeze(0))[0]   # shape (7,)
        # Mask 非法动作（设为极低值，保证不会被选到）
        mask = torch.full_like(q_values, -1e9)
        for a in legal_actions:
            mask[a] = 0.0
        return (q_values + mask).argmax().item()


# --- 训练循环 ---
def train(
    env,                    # NetworkEnv 实例
    episodes: int = 800,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.998,
    target_update: int = 20,
    print_every: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用 {'GPU' if device.type == 'cuda' else 'CPU'} 训练")

    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer()

    epsilon = epsilon_start
    all_rewards: list[float] = []

    for episode in range(1, episodes + 1):
        node = env.reset()
        total_reward = 0.0
        done = False
        detections = 0

        while not done:
            state_tensor = state_to_tensor(node, detections).to(device)
            legal = env.legal_actions()
            action = select_action(policy_net, state_tensor, legal, epsilon)
            next_node, reward, done, info = env.step(action)

            # 记录经验
            replay.push(node, action, reward, next_node, done)
            total_reward += reward
            node = next_node
            if info.get("detected"):
                detections += 1

            # 从缓冲区采样训练
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                _train_step(policy_net, target_net, optimizer, batch, gamma, device)

        # ε 衰减
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        # 定期更新 target 网络
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % print_every == 0:
            avg100 = sum(all_rewards[-100:]) / min(len(all_rewards), 100)
            print(
                f"[Ep {episode:4d}/{episodes}] "
                f"reward={total_reward:7.2f}  avg100={avg100:7.2f}  "
                f"epsilon={epsilon:.3f}  buffer={len(replay)}"
            )

    return policy_net, all_rewards


def _train_step(policy_net, target_net, optimizer, batch, gamma, device):
    states = torch.stack([state_to_tensor(s, 0) for s, _, _, _, _ in batch]).to(device)
    actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long).to(device)
    rewards = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([state_to_tensor(ns, 0) for _, _, _, ns, _ in batch]).to(device)
    dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32).to(device)

    # Q(s, a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a')
    with torch.no_grad():
        max_next = target_net(next_states).max(1)[0]
        target = rewards + gamma * max_next * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()


# --- 演示：用训练好的 Q 网络走一局，打印决策路径 ---
def demonstrate(env, q_net: QNetwork):
    """走一局，打印完整渗透路径。"""
    device = next(q_net.parameters()).device
    node = env.reset()
    done = False
    detections = 0
    path = [NODE_NAMES[node]]
    total_reward = 0.0

    print("\n" + "=" * 60)
    print("🎯 渗透路径演示")
    print("=" * 60)

    while not done:
        legal = env.legal_actions()
        state_tensor = state_to_tensor(node, detections).to(device)
        action = select_action(q_net, state_tensor, legal, epsilon=0.0)   # 纯贪心
        next_node, reward, done, info = env.step(action)

        step_str = (
            f"  {NODE_NAMES[node]:20s} → {NODE_NAMES[next_node]:20s}"
            f"  reward={reward:+6.2f}"
        )
        if info.get("detected"):
            step_str += f"  ⚠️  DETECTED! (累计 {info.get('total_detections', 0)} 次)"
        print(step_str)

        path.append(NODE_NAMES[next_node])
        node = next_node
        total_reward += reward
        if info.get("detected"):
            detections += 1

    print("-" * 60)
    print(f"路径: {' → '.join(path)}")
    print(f"总奖励: {total_reward:.2f}  |  步数: {env.steps}  |  被检测: {detections} 次")
    print("=" * 60)


if __name__ == "__main__":
    from env import NetworkEnv
    env = NetworkEnv(max_steps=30)
    q_net, rewards = train(env, episodes=800)
    demonstrate(env, q_net)