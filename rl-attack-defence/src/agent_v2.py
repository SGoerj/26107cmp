"""
Double Dueling DQN 智能体 —— 自主攻击链决策

核心改进 (相对于 v1):
  1. Dueling Network: Value(s) + Advantage(s,a) 分离, 对多动作空间更友好
  2. Double DQN: 用 policy_net 选动作, target_net 估值, 消除 overestimation bias
  3. Action Masking: 非法动作 Q 值置为 -∞, 杜绝无效动作被选中的可能
  4. 攻击链可视化: 演示时展示完整的 RECON→EXPLOIT→MOVE→EXFIL 序列

输入: 27D 状态向量 (env_v2)
输出: 11 个动作的 Q 值 (RECON/VULN_SCAN/EXPLOIT×3/PRIVESC/EXFIL/MOVE×4)
"""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from env_v2 import (
    ACTION_NAMES,
    N_ACTIONS,
    N_NODES,
    NODE_NAMES,
    STATE_DIM,
    AttackChainEnv,
)


# --- Dueling 网络 ---
class DuelingNetwork(nn.Module):
    """Dueling Architecture: 分离 Value 和 Advantage 流。

    共享特征层 → Value 流 (标量) + Advantage 流 (N_ACTIONS 维)
    最终 Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, input_dim: int = STATE_DIM, output_dim: int = N_ACTIONS):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)           # (B, 1)
        advantage = self.advantage_stream(feat)   # (B, N_ACTIONS)
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# --- 经验回放 ---

Transition = tuple[np.ndarray, int, float, np.ndarray, bool]
#                  state(27D)   action  reward  next_state(27D)  done


class ReplayBuffer:
    """固定容量 FIFO 经验池。存完整 27D 状态向量。

    注: 曾尝试 PrioritizedReplayBuffer (按 |δ| 优先回放) 抗遗忘, 但在本 4 节点
    小环境实测反而更差 (失败经验 |δ| 大被集中回放, 加上固定 ε 持续制造失败
    经验, 策略持续恶化)。均匀回放 + best 快照兜底已验证最稳, 保留此版本。
    """

    def __init__(self, capacity: int = 20000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# --- 动作选择 (带 mask) ---
def select_action(
    q_net: DuelingNetwork,
    state_tensor: torch.Tensor,
    legal_actions: list[int],
    epsilon: float,
) -> int:
    """ε-greedy + action mask: 非法动作绝不被选中。"""
    if random.random() < epsilon:
        return random.choice(legal_actions)

    with torch.no_grad():
        q_values = q_net(state_tensor.unsqueeze(0))[0]   # (N_ACTIONS,)
        # Mask: 非法动作 → -inf
        mask = torch.full_like(q_values, float("-inf"))
        for a in legal_actions:
            mask[a] = 0.0
        return (q_values + mask).argmax().item()


# --- 训练循环 ---
def train(
    env: AttackChainEnv,
    episodes: int = 1500,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 5e-4,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.15,
    epsilon_decay: float = 0.9992,
    target_update: int = 15,
    print_every: int = 100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 维度从 env 取 (支持动态拓扑), 而非模块级常量 (那只在 import 时固定为默认配置)
    state_dim = env.state_dim
    n_actions = env.n_actions
    print(f"[设备] 使用 {'GPU' if device.type == 'cuda' else 'CPU'} 训练")
    print(f"[动作数] {n_actions}  |  [状态维度] {state_dim}D")
    print(f"[网络] Dueling Network: {state_dim}→128→64→(V:32→1 | A:32→{n_actions})")
    print(f"[算法] Double DQN + Action Masking")

    policy_net = DuelingNetwork(state_dim, n_actions).to(device)
    target_net = DuelingNetwork(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer()

    epsilon = epsilon_start
    all_rewards: list[float] = []
    all_flags: list[int] = []       # 每 episode 是否拿到 flag
    best_avg = float("-inf")
    best_state_dict = None          # avg100 最高时的权重快照
    best_episode = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        got_flag = 0
        detections = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().to(device)
            legal = env.legal_actions()

            if not legal:
                break

            action = select_action(policy_net, state_tensor, legal, epsilon)
            next_state, reward, done, info = env.step(action)

            # 记录经验 (存完整状态向量)
            replay.push(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            if info.get("detected"):
                detections += 1
            if info.get("flag_captured"):
                got_flag = 1

        # 每 episode 结束后训练 (公平对待长短 episode)
        if len(replay) >= batch_size:
            for _ in range(4):
                batch = replay.sample(batch_size)
                _train_double_dqn(
                    policy_net, target_net, optimizer, batch,
                    gamma, device,
                )

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)
        all_flags.append(got_flag)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 跟踪 avg100 最高的策略快照 (防止末期遗忘)。
        # 至少跑 100 个 episode 后才开始记录。
        # 注: 曾尝试加 "ε≤0.3 才记录" 的限制以抓低探索期真实策略, 但实验表明
        # 本环境 DQN 后期 avg100 一直为负 (越学越差), 严格限制会导致 best 为负。
        # 当前保留宽松记录: best 抓高探索期峰值, 演示成功但依赖快照运气——
        # 这是 DQN 在本环境的极限, 治本需换 on-policy 算法 (见 GUIDE 10.3)。
        if len(all_rewards) >= 100:
            avg100 = sum(all_rewards[-100:]) / 100
            if avg100 > best_avg:
                best_avg = avg100
                best_episode = episode
                best_state_dict = copy.deepcopy(policy_net.state_dict())

        if episode % print_every == 0:
            avg100 = sum(all_rewards[-100:]) / min(len(all_rewards), 100)
            flag_rate = sum(all_flags[-100:]) / min(len(all_flags), 100) * 100
            line = (
                f"[Ep {episode:5d}/{episodes}] "
                f"reward={total_reward:7.2f}  avg100={avg100:7.2f}  "
                f"ε={epsilon:.3f}  flag%={flag_rate:.0f}%  "
                f"det={detections}"
            )
            if avg100 >= best_avg and best_state_dict is not None:
                line += " ★"
            print(line)

    # 用 best 而非 final 做演示 (核心: 避免末期遗忘污染演示)
    if best_state_dict is not None:
        print(f"\n[Best] avg100 峰值={best_avg:.2f} @ Ep{best_episode}, 用它演示 (而非 final)")
        policy_net.load_state_dict(best_state_dict)
    else:
        print("\n[Best] 未达到 100 episode, 用 final 策略演示")

    return policy_net, all_rewards


# --- Double DQN 训练步 (带 PER 权重) ---
def _train_double_dqn(
    policy_net: DuelingNetwork,
    target_net: DuelingNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    gamma: float,
    device: torch.device,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Double DQN: a* = argmax_a Q_policy(s'), target = r + γ Q_target(s', a*)

    PER 模式下传入 importance-sampling weights, 用加权 MSE 替代均匀 MSE。
    返回 per-sample TD 误差供 PER 更新优先级。
    """
    states = torch.from_numpy(np.stack([s for s, _, _, _, _ in batch])).float().to(device)
    next_states = torch.from_numpy(np.stack([ns for _, _, _, ns, _ in batch])).float().to(device)
    actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long).to(device)
    rewards = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32).to(device)
    dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32).to(device)

    # Q(s, a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN target: a* = argmax Q_policy(s'), y = r + γ Q_target(s', a*)
    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        best_actions = next_q_policy.argmax(1)        # (B,)
        next_q_target = target_net(next_states)
        max_next = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * max_next * (1 - dones)

    td_error = q_values - target                     # (B,) 用于优先级
    if weights is not None:
        w = torch.from_numpy(weights).to(device)
        loss = (w * td_error.pow(2)).mean()          # 加权 MSE
    else:
        loss = td_error.pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 2.0)
    optimizer.step()

    return td_error.detach().cpu().numpy()


# --- 演示 ---
def demonstrate(env: AttackChainEnv, q_net: DuelingNetwork, max_steps: int = 40):
    """走一局完整攻击链, 并展示每一步的决策。"""
    device = next(q_net.parameters()).device
    state = env.reset()
    done = False
    total_reward = 0.0
    detections = 0
    step_num = 0

    print("\n" + "=" * 70)
    print("  🎯 攻击链演示 —— Double Dueling DQN 自主决策")
    print("=" * 70)
    print(f"  {'#':<4} {'动作':<16} {'所在节点':<14} {'结果':<18} {'奖励':>7}")
    print("-" * 70)

    while not done and step_num < max_steps:
        legal = env.legal_actions()
        if not legal:
            print("  [无合法动作，终止]")
            break

        state_tensor = torch.from_numpy(state).float().to(device)
        action = select_action(q_net, state_tensor, legal, epsilon=0.0)
        next_state, reward, done, info = env.step(action)

        node_name = info.get("node_name", "?")
        action_name = info.get("action_name", "?")
        detected = info.get("detected", False)
        flag_captured = info.get("flag_captured", False)
        compromised = info.get("compromised", False)

        # 构造结果描述
        if flag_captured:
            result = "🏁 FLAG!"
        elif compromised:
            result = "✅ 攻陷"
        elif action_name.startswith("MOVE"):
            result = f"→ {NODE_NAMES[action - 7]}" if action >= 7 else "移动"
        elif info.get("invalid"):
            result = "❌ 非法"
        elif detected:
            result = "⚠️ 被检测"
        else:
            result = "→"

        det_str = f" {'⚠️' if detected else '  '}"
        print(f"  {step_num:<4} {action_name:<16} {node_name:<14} {result:<18} {reward:+7.2f}{det_str}")

        total_reward += reward
        state = next_state
        step_num += 1
        if info.get("detected"):
            detections += 1

    print("-" * 70)
    print(f"  总奖励: {total_reward:+.2f}  |  步数: {step_num}")
    print(f"  被检测: {detections} 次  |  {'🏁 拿到 FLAG!' if env._flag_captured else '❌ 未拿到 flag'}")
    print("=" * 70)

    # 状态摘要
    print("\n  节点状态:")
    for i in range(N_NODES):
        c = "🟢" if env.compromised[i] else "⚪"
        r = "👑" if env.root[i] else "  "
        rec = "R" if env.recon_done[i] else "-"
        vuln = "V" if env.vuln_scan_done[i] else "-"
        flag = " 🏁" if env.root[i] else ""
        print(f"    {c} {r} [{rec}{vuln}] {NODE_NAMES[i]:<14}{flag}")


# --- 最佳攻击链提取 ---
def extract_attack_chain(env: AttackChainEnv):
    """演示后显示已执行的攻击链摘要。"""
    chain = []
    for i in range(N_NODES):
        actions = []
        if env.recon_done[i]:
            actions.append("RECON")
        if env.vuln_scan_done[i]:
            actions.append("VULN_SCAN")
        if env.compromised[i] and i > 0:
            actions.append("EXPLOIT")
        if env.root[i]:
            actions.append("PRIVESC")
        if env.root[i] and env.current == i and env.steps > 0:
            actions.append("EXFIL")
        if actions:
            chain.append(f"  {NODE_NAMES[i]}: {' → '.join(actions)}")
    return chain


if __name__ == "__main__":
    import sys
    import os
    import argparse

    # 确保从 src/ 目录或项目根目录运行都能找到 env_v2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 命令行选配置: --config configs/env_6node.yaml --episodes 3000
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="环境配置 YAML (默认 configs/env_default.yaml)")
    ap.add_argument("--episodes", type=int, default=2000)
    args = ap.parse_args()

    env = AttackChainEnv(config_path=args.config)
    print(f"[配置] {args.config or '默认 (4 节点)'}  | 节点={env.n_nodes} "
          f"state_dim={env.state_dim} n_actions={env.n_actions} max_steps={env.max_steps}")
    q_net, rewards = train(env, episodes=args.episodes)
    demonstrate(env, q_net)

    # 保存 best model (用 env 实际维度, 支持不同拓扑)
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    tag = "6node" if (args.config and "6node" in args.config) else "v2"
    model_path = os.path.join(model_dir, f"best_{tag}.pt")
    torch.save({
        "state_dict": q_net.state_dict(),
        "state_dim": env.state_dim,
        "n_actions": env.n_actions,
    }, model_path)
    print(f"\n[保存] best model → {os.path.abspath(model_path)}")