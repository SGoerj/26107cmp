"""
PPO (Proximal Policy Optimization) 智能体 —— 自主攻击链决策

为什么从 DQN 换 PPO:
  DQN (off-policy + 经验回放) 在本环境实测"学不深"——训练后期策略持续退化
  (avg100 从正跌到 -50), 之前所有"成功演示"都是 best 快照在高探索期碰巧抓到的。
  根因: 经验回放池里好策略被后期坏经验稀释 (灾难性遗忘)。

  PPO (on-policy) 治本:
    - 只用当前策略实时采样的数据, 用完即弃 → 天然不遗忘
    - 直接学策略概率 (而非 Q 值), greedy 评估稳定 → final 即 best
    - 熵正则替代 ε-greedy, 探索更有方向
    - clip 限制更新幅度, 训练稳定

核心组件:
  1. ActorCritic 网络: 共享底层 → actor(logits) + critic(V), 带 action mask
  2. GAE (Generalized Advantage Estimation): 算优势 A_t, 偏差-方差权衡
  3. PPO clip loss: 限制新旧策略比例, 防步子太大
  4. On-policy 训练: 收集 N 步 → 更新 K 轮 → 丢弃

环境不变 (复用 env_v2, 含 POMDP A 步的 27D state)。
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

# 禁用 torch dynamo (我们不用编译)。某些 HPC 计算节点上 torch dynamo 的
# lazy-import sympy 会因节点环境差异失败 (ModuleNotFoundError: sympy.assumptions.cnf),
# 登录节点却正常。禁用 dynamo 可绕过, 对 PPO 训练零影响。
try:
    torch._dynamo.config.disable = True          # type: ignore
except Exception:
    pass
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import AttackChainEnv, N_NODES, NODE_NAMES


# ──────────────────────────────────────────────────────────────────────
# Actor-Critic 网络
# ──────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """共享底层特征, 分出 actor (策略 logits) 和 critic (状态价值 V) 两个头。

    actor 输出原始 logits, 推理时对非法动作 logits 置 -inf 再 softmax,
    使非法动作概率为 0 且分布只在合法动作上归一化。
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden // 2, n_actions)
        self.critic = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor):
        feat = self.feature(x)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def evaluate(self, states, actions, legal_masks):
        """PPO 更新用: 在合法动作上算 log_prob 和 entropy。
        legal_masks: [B, n_actions] bool, True=合法。
        """
        logits, values = self.forward(states)
        mask = torch.full_like(logits, float("-inf"))
        mask[legal_masks] = 0.0
        masked = logits + mask
        dist = torch.distributions.Categorical(logits=masked)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, values, entropy

    def get_action(self, x: torch.Tensor, legal: list[int]):
        """采样一个动作 (带 mask)。返回 (action, log_prob, value)。"""
        logits, value = self.forward(x.unsqueeze(0))
        logits = logits[0]                   # (n_actions,)
        mask = torch.full_like(logits, float("-inf"))
        for a in legal:
            mask[a] = 0.0
        masked = logits + mask
        probs = torch.softmax(masked, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


# ──────────────────────────────────────────────────────────────────────
# 经验收集 (一个 rollout 内的轨迹)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Rollout:
    states: list           # list[np.ndarray]
    actions: list          # list[int]
    log_probs: list        # list[float]
    rewards: list          # list[float]
    values: list           # list[float]
    dones: list            # list[bool]
    legals: list           # list[list[int]]  每个状态的合法动作列表 (供evaluate用)


# ──────────────────────────────────────────────────────────────────────
# GAE: 算优势 A_t 和回报 R_t
# ──────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized Advantage Estimation。
    A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
    δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    R_t = A_t + V(s_t)  (critic 的回归目标)
    """
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ──────────────────────────────────────────────────────────────────────
# 训练循环
# ──────────────────────────────────────────────────────────────────────
def train(
    env: AttackChainEnv,
    rollouts: int = 60,            # 收集多少个 rollout
    rollout_steps: int = 2000,     # 每个 rollout 收集多少步
    update_epochs: int = 10,       # 一批数据重复用几轮
    batch_size: int = 256,         # mini-batch 大小
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    print_every: int = 5,
    seed: int | None = None,
):
    if seed is not None:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.state_dim
    n_actions = env.n_actions
    print(f"[设备] 使用 {'GPU' if device.type == 'cuda' else 'CPU'} 训练")
    print(f"[动作数] {n_actions}  |  [状态维度] {state_dim}D")
    print(f"[网络] ActorCritic: {state_dim}→128→64→(actor:{n_actions} | critic:1)")
    print(f"[算法] PPO + Action Masking (on-policy, 不遗忘)")

    ac = ActorCritic(state_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)

    all_rewards: list[float] = []   # 每个 episode 的总奖励
    all_flags: list[int] = []

    for rollout_idx in range(1, rollouts + 1):
        rb = Rollout([], [], [], [], [], [], [])
        step_count = 0
        ep_in_rollout = 0

        # ── 收集 rollout_steps 步 ──
        state = env.reset()
        ep_reward = 0.0
        ep_flag = 0
        while step_count < rollout_steps:
            legal = env.legal_actions()
            if not legal:
                state = env.reset()
                all_rewards.append(ep_reward)
                all_flags.append(ep_flag)
                ep_reward = 0.0
                ep_flag = 0
                ep_in_rollout += 1
                continue

            state_tensor = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                action, log_prob, value = ac.get_action(state_tensor, legal)

            next_state, reward, done, info = env.step(action)

            rb.states.append(state)
            rb.actions.append(action)
            rb.log_probs.append(log_prob)
            rb.rewards.append(reward)
            rb.values.append(value)
            rb.dones.append(done)
            rb.legals.append(legal)  # 记录合法动作 (供 PPO 更新时 mask)

            ep_reward += reward
            if info.get("flag_captured"):
                ep_flag = 1
            state = next_state
            step_count += 1

            if done:
                all_rewards.append(ep_reward)
                all_flags.append(ep_flag)
                ep_reward = 0.0
                ep_flag = 0
                ep_in_rollout += 1
                state = env.reset()

        # rollout 末尾的 bootstrap value
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(device)
            _, last_value = ac(state_tensor.unsqueeze(0))
            last_value = last_value.item()

        # ── 算 GAE ──
        advantages, returns = compute_gae(
            rb.rewards, rb.values, rb.dones, last_value, gamma, gae_lambda)
        # 优势标准化 (稳定训练)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32).to(device)

        states_t = torch.tensor(np.stack(rb.states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(rb.actions, dtype=torch.long).to(device)
        old_log_probs_t = torch.tensor(rb.log_probs, dtype=torch.float32).to(device)

        n = len(rb.states)

        # ── K 轮更新 ──
        for _ in range(update_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, batch_size):
                end = start + batch_size
                mb = idx[start:end]

                # 构造 batch legal mask
                mb_actions = actions_t[mb]
                batch_legals = [rb.legals[i] for i in mb.tolist()]
                leg_mask = torch.zeros(len(mb), n_actions, dtype=torch.bool, device=device)
                for j, leg in enumerate(batch_legals):
                    for a in leg:
                        leg_mask[j, a] = True

                new_log_probs, values, entropy = ac.evaluate(states_t[mb], mb_actions, leg_mask)

                # PPO clip surrogate
                ratio = torch.exp(new_log_probs - old_log_probs_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values, ret_t[mb])

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

        # ── 打印 ──
        if rollout_idx % print_every == 0 or rollout_idx == 1:
            recent = all_rewards[-200:] if len(all_rewards) >= 200 else all_rewards
            avg = sum(recent) / len(recent)
            rf = all_flags[-200:] if len(all_flags) >= 200 else all_flags
            flag_rate = sum(rf) / len(rf) * 100
            print(
                f"[Rollout {rollout_idx:3d}/{rollouts}] "
                f"steps={step_count} eps={ep_in_rollout}  "
                f"avg={avg:7.2f}  flag%={flag_rate:.0f}%  "
                f"policy_loss={policy_loss.item():+.3f}  "
                f"value_loss={value_loss.item():.3f}  "
                f"entropy={entropy.item():.3f}"
            )

    return ac, all_rewards


# ──────────────────────────────────────────────────────────────────────
# 演示 (greedy: 取 argmax, 不采样)
# ──────────────────────────────────────────────────────────────────────
def demonstrate(env: AttackChainEnv, ac: ActorCritic, max_steps: int = 50):
    device = next(ac.parameters()).device
    state = env.reset()
    done = False
    total_reward = 0.0
    detections = 0
    step_num = 0

    print("\n" + "=" * 70)
    print("  🎯 攻击链演示 —— PPO 自主决策 (greedy)")
    print("=" * 70)
    print(f"  {'#':<4} {'动作':<16} {'所在节点':<14} {'结果':<18} {'奖励':>7}")
    print("-" * 70)

    while not done and step_num < max_steps:
        legal = env.legal_actions()
        if not legal:
            print("  [无合法动作，终止]")
            break

        state_tensor = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            logits, _ = ac(state_tensor.unsqueeze(0))
            logits = logits[0]
            mask = torch.full_like(logits, float("-inf"))
            for a in legal:
                mask[a] = 0.0
            action = (logits + mask).argmax().item()   # greedy

        next_state, reward, done, info = env.step(action)
        action_name = env.action_names[action]
        node_name = info.get("node_name", "?")
        detected = info.get("detected", False)
        compromised = info.get("compromised", False)
        flag_captured = info.get("flag_captured", False)

        if flag_captured:
            result = "🏁 FLAG!"
        elif compromised:
            result = "✅ 攻陷"
        elif action_name.startswith("MOVE"):
            result = f"→ {NODE_NAMES[action - MOVE_BASE]}" if action - MOVE_BASE < N_NODES else "移动"
        elif info.get("invalid"):
            result = "❌ 非法"
        elif detected:
            result = "⚠️ 被检测"
        else:
            result = "→"

        det_str = " ⚠️" if detected else ""
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

    print("\n  节点状态:")
    for i in range(N_NODES):
        c = "🟢" if env.compromised[i] else "⚪"
        r = "👑" if env.root[i] else "  "
        rec = "R" if env.recon_done[i] else "-"
        vuln = "V" if env.vuln_scan_done[i] else "-"
        flag = " 🏁" if env.root[i] else ""
        print(f"    {c} {r} [{rec}{vuln}] {NODE_NAMES[i]:<14}{flag}")


# ──────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rollouts", type=int, default=60)
    args = ap.parse_args()

    env = AttackChainEnv(config_path=args.config)
    print(f"[配置] {args.config or '默认 (4 节点)'}  | 节点={env.n_nodes} "
          f"state_dim={env.state_dim} n_actions={env.n_actions} max_steps={env.max_steps}")
    ac, rewards = train(env, rollouts=args.rollouts)
    demonstrate(env, ac)

    # 保存 (PPO final 即 best, on-policy 不遗忘)
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    # tag 从配置文件名推导: 6node / default_dynamic / ppo(默认)
    if args.config:
        stem = os.path.splitext(os.path.basename(args.config))[0]  # env_default_dynamic
        tag = stem.replace("env_", "")                              # default_dynamic
    else:
        tag = "ppo"
    model_path = os.path.join(model_dir, f"best_{tag}.pt")
    torch.save({
        "state_dict": ac.state_dict(),
        "state_dim": env.state_dim,
        "n_actions": env.n_actions,
        "algo": "ppo",
    }, model_path)
    print(f"\n[保存] PPO model → {os.path.abspath(model_path)}")
