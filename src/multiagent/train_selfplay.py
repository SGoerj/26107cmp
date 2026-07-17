"""
兜底层 3: Self-play —— 红蓝双 PPO 交替训练 + 对手快照池

设计 (见对话设计4):
  每 round:
    步骤1: 固定蓝队 (从快照池随机抽), 训练红队 → R_k
    步骤2: 固定红队 (从快照池随机抽), 训练蓝队 → B_k
    双方新快照入池 (池大小有限, 满了替换最旧)
  重复。

对手快照池防策略循环: 训红队时对手不只来自最新蓝队, 而是池里随机抽,
让红队学"对各种防御都有效的通用攻击", 而非针对单一蓝队的特解。

收敛判断 (看博弈结果, 不看 avg):
  - 红队 flag% 稳定在 40-60% = 双方均衡 (理想)
  - 红队一直 100% = 蓝队学不动 (self-play 无价值, 止损)
  - 红队被压到 ~0% = 蓝队碾压 (红队学不动)

用法 (HPC):
  .venv/bin/python src/multiagent/train_selfplay.py --rounds 20
"""

import os
import sys
import copy
import argparse
import random as _r

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_ma import MultiAgentAttackEnv
from agent_ppo import ActorCritic, compute_gae


# ──────────────────────────────────────────────────────────────────────
# PPO 更新 (红/蓝通用, 各自网络/数据)
# ──────────────────────────────────────────────────────────────────────
def ppo_update(ac, optimizer, states, actions, old_log_probs, advs, rets, legals,
               update_epochs=10, batch_size=256, clip_ratio=0.2,
               value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
               n_actions=None):
    device = next(ac.parameters()).device
    S = torch.tensor(np.stack(states)).float().to(device)
    A = torch.tensor(actions).to(device)
    old_lp = torch.tensor(old_log_probs).to(device)
    adv_t = torch.tensor(advs).to(device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(rets).to(device)
    n = len(states)
    if n < batch_size:
        batch_size = n

    for _ in range(update_epochs):
        idx = torch.randperm(n)
        for s in range(0, n, batch_size):
            mb = idx[s:s+batch_size]
            leg_mask = torch.zeros(len(mb), n_actions, dtype=torch.bool, device=device)
            for j, i in enumerate(mb.tolist()):
                for a in legals[i]:
                    leg_mask[j, a] = True
            new_lp, vals, ent = ac.evaluate(S[mb], A[mb], leg_mask)
            ratio = torch.exp(new_lp - old_lp[mb])
            s1 = ratio * adv_t[mb]
            s2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_t[mb]
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = nn.MSELoss()(vals, ret_t[mb])
            loss = policy_loss + value_coef * value_loss - entropy_coef * ent
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            optimizer.step()
    return policy_loss.item(), value_loss.item(), ent.item()


def collect_rollout(env, ac, opponent_action_fn, is_red_training,
                    rollout_steps=2000, gamma=0.99, gae_lambda=0.95):
    """收集一轮数据。is_red_training=True 训红队(记录红队数据), False 训蓝队。
    opponent_action_fn: 对手动作函数 (传 env + legal, 返回动作)。
    """
    device = next(ac.parameters()).device
    states, actions, log_probs, rewards, values, dones, legals = [], [], [], [], [], [], []
    ep_rewards, ep_flags = [], []
    step_count, ep_r, ep_flag = 0, 0.0, 0
    rs, bs = env.reset()
    while step_count < rollout_steps:
        red_legal = env.red_legal_actions()
        if not red_legal:
            ep_rewards.append(ep_r); ep_flags.append(ep_flag)
            ep_r, ep_flag = 0.0, 0
            rs, bs = env.reset()
            continue

        if is_red_training:
            # 红队用 ac 采样, 蓝队用 opponent
            st = torch.from_numpy(rs).float().to(device)
            with torch.no_grad():
                red_a, red_lp, red_v = ac.get_action(st, red_legal)
            blue_a = opponent_action_fn(env, env.blue_legal_actions())
            states.append(rs.copy()); actions.append(red_a); log_probs.append(red_lp)
            values.append(red_v); legals.append(red_legal)
            rs, bs, rr, br, done, info = env.step(red_a, blue_a)
            rewards.append(rr); dones.append(done)
            ep_r += rr
            if info.get("flag_captured"):
                ep_flag = 1
        else:
            # 蓝队用 ac 采样, 红队用 opponent
            blue_legal = env.blue_legal_actions()
            st = torch.from_numpy(bs).float().to(device)
            with torch.no_grad():
                blue_a, blue_lp, blue_v = ac.get_action(st, blue_legal)
            red_a = opponent_action_fn(env, red_legal)
            states.append(bs.copy()); actions.append(blue_a); log_probs.append(blue_lp)
            values.append(blue_v); legals.append(blue_legal)
            rs, bs, rr, br, done, info = env.step(red_a, blue_a)
            rewards.append(br); dones.append(done)
            ep_r += br
            # 蓝队成功 = 没被偷 flag (终局判断)

        step_count += 1
        if done:
            ep_rewards.append(ep_r); ep_flags.append(ep_flag)
            ep_r, ep_flag = 0.0, 0
            rs, bs = env.reset()

    # bootstrap
    last_state = rs if is_red_training else bs
    with torch.no_grad():
        st = torch.from_numpy(last_state).float().to(device)
        _, last_v = ac(st.unsqueeze(0)); last_v = last_v.item()
    advs, rets = compute_gae(rewards, values, dones, last_v, gamma, gae_lambda)
    return states, actions, log_probs, advs, rets, legals, ep_rewards, ep_flags


# ──────────────────────────────────────────────────────────────────────
# 对手快照池
# ──────────────────────────────────────────────────────────────────────
class SnapshotPool:
    """存历史策略快照, 训练时随机抽一个当对手 (防策略循环)。"""
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.snapshots = []   # list of state_dict

    def add(self, state_dict):
        self.snapshots.append(copy.deepcopy(state_dict))
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)   # 丢最旧

    def sample_into(self, ac):
        """随机抽一个快照, 载入 ac。池空则不动 (用 ac 当前权重)。"""
        if not self.snapshots:
            return
        ac.load_state_dict(_r.choice(self.snapshots))


# ──────────────────────────────────────────────────────────────────────
# Self-play 主循环
# ──────────────────────────────────────────────────────────────────────
def train_selfplay(env, rounds=20, rollout_steps=2000, lr=3e-4,
                   pool_size=5, seed=None, print_every=1, entropy_coef=0.05):
    if seed is not None:
        _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red_ac = ActorCritic(env.red_state_dim, env.n_red_actions).to(device)
    blue_ac = ActorCritic(env.blue_state_dim, env.n_blue_actions).to(device)
    red_opt = torch.optim.Adam(red_ac.parameters(), lr=lr)
    blue_opt = torch.optim.Adam(blue_ac.parameters(), lr=lr)

    red_pool = SnapshotPool(pool_size)
    blue_pool = SnapshotPool(pool_size)
    # 初始快照: 存初始随机权重, 避免 Round 1 空池导致对手=随机弱网络
    red_pool.add(red_ac.state_dict())
    blue_pool.add(blue_ac.state_dict())

    # 对手动作函数: 用快照网络采样
    def make_opponent_fn(opponent_ac, is_opponent_red):
        def fn(e, legal):
            if not legal:
                return 0
            st = e._red_state() if is_opponent_red else e._blue_state()
            with torch.no_grad():
                logits, _ = opponent_ac(torch.from_numpy(st).float().to(device).unsqueeze(0))
                logits = logits[0]
                mask = torch.full_like(logits, float("-inf"))
                for a in legal:
                    mask[a] = 0.0
                return (logits + mask).argmax().item()   # 对手用 greedy
        return fn

    print(f"[Self-play] rounds={rounds} | 红队 {env.red_state_dim}D/{env.n_red_actions}动 | "
          f"蓝队 {env.blue_state_dim}D/{env.n_blue_actions}动 | 快照池={pool_size}")

    for ro in range(1, rounds + 1):
        # ── 步骤1: 训红队 (对手=蓝队快照池随机抽) ──
        opponent_blue = ActorCritic(env.blue_state_dim, env.n_blue_actions).to(device)
        blue_pool.sample_into(opponent_blue)
        opp_fn = make_opponent_fn(opponent_blue, is_opponent_red=False)
        S, A, LP, ADV, RET, LEG, ep_r, ep_f = collect_rollout(
            env, red_ac, opp_fn, is_red_training=True, rollout_steps=rollout_steps)
        if len(S) > 0:
            ppo_update(red_ac, red_opt, S, A, LP, ADV, RET, LEG, n_actions=env.n_red_actions,
                       entropy_coef=entropy_coef)
        red_flag_pct = sum(ep_f) / len(ep_f) * 100 if ep_f else 0
        red_avg = sum(ep_r) / len(ep_r) if ep_r else 0
        red_pool.add(red_ac.state_dict())

        # ── 步骤2: 训蓝队 (对手=红队快照池随机抽) ──
        opponent_red = ActorCritic(env.red_state_dim, env.n_red_actions).to(device)
        red_pool.sample_into(opponent_red)
        opp_fn = make_opponent_fn(opponent_red, is_opponent_red=True)
        S, A, LP, ADV, RET, LEG, ep_r, ep_f = collect_rollout(
            env, blue_ac, opp_fn, is_red_training=False, rollout_steps=rollout_steps)
        if len(S) > 0:
            ppo_update(blue_ac, blue_opt, S, A, LP, ADV, RET, LEG, n_actions=env.n_blue_actions,
                       entropy_coef=entropy_coef)
        blue_avg = sum(ep_r) / len(ep_r) if ep_r else 0
        blue_pool.add(blue_ac.state_dict())

        # 蓝队成功阻止率 = 1 - 红队 flag% (本轮红队数据里的)
        blue_block = 100 - red_flag_pct

        if ro % print_every == 0 or ro == 1:
            print(f"[Round {ro:3d}/{rounds}] "
                  f"红队 flag%={red_flag_pct:5.0f}% (avg={red_avg:+6.1f}) | "
                  f"蓝队 block%={blue_block:5.0f}% (avg={blue_avg:+7.1f}) | "
                  f"红池={len(red_pool.snapshots)} 蓝池={len(blue_pool.snapshots)}")

    return red_ac, blue_ac


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--rollout-steps", type=int, default=2000)
    ap.add_argument("--pool-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entropy", type=float, default=0.05)
    args = ap.parse_args()

    env = MultiAgentAttackEnv(config_path=args.config)
    red_ac, blue_ac = train_selfplay(env, rounds=args.rounds,
                                     rollout_steps=args.rollout_steps,
                                     pool_size=args.pool_size, seed=args.seed,
                                     entropy_coef=args.entropy)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save({"state_dict": red_ac.state_dict(),
                "state_dim": env.red_state_dim,
                "n_actions": env.n_red_actions},
               os.path.join(model_dir, "selfplay_red.pt"))
    torch.save({"state_dict": blue_ac.state_dict(),
                "state_dim": env.blue_state_dim,
                "n_actions": env.n_blue_actions},
               os.path.join(model_dir, "selfplay_blue.pt"))
    print(f"\n[保存] selfplay_red.pt + selfplay_blue.pt")
