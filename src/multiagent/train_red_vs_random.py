"""
兜底层 1: 固定随机蓝队 + 红队 PPO 训练

验证目标: 红队能不能在"有蓝队对抗"的环境下学会攻击。
蓝队纯随机选合法动作 (不学习), 隔离红队想去的节点、加固节点。
如果红队在这个对抗下还能学 (flag% 上升), 说明红队 PPO 在博弈环境可用,
再上 self-play (兜底层 2/3)。

用法 (HPC):
  .venv/bin/python src/multiagent/train_red_vs_random.py
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_ma import MultiAgentAttackEnv
from agent_ppo import ActorCritic, compute_gae


def random_blue_action(env, blue_legal):
    """随机蓝队策略: 偏向 MONITOR (50%), 其余随机选防御动作。"""
    if np.random.random() < 0.5:
        return 0  # MONITOR
    non_monitor = [a for a in blue_legal if a != 0]
    return np.random.choice(non_monitor) if non_monitor else 0


def train_red_vs_random(env, rollouts=60, rollout_steps=2000, update_epochs=10,
                        batch_size=256, clip_ratio=0.2, gamma=0.99,
                        gae_lambda=0.95, lr=3e-4, entropy_coef=0.01,
                        value_coef=0.5, max_grad_norm=0.5, seed=None,
                        print_every=5):
    if seed is not None:
        import random as _r
        _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red_dim, red_n = env.red_state_dim, env.n_red_actions
    print(f"[设备] {'GPU' if device.type=='cuda' else 'CPU'} | "
          f"红队 state={red_dim} actions={red_n} | 蓝队=随机")

    ac = ActorCritic(red_dim, red_n).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)

    all_rewards, all_flags = [], []

    for ro in range(1, rollouts + 1):
        # 收集 rollout
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        step_count, ep_in_ro, ep_r, ep_flag = 0, 0, 0.0, 0
        rs, bs = env.reset()
        while step_count < rollout_steps:
            red_legal = env.red_legal_actions()
            if not red_legal:
                rs, bs = env.reset()
                all_rewards.append(ep_r); all_flags.append(ep_flag)
                ep_r, ep_flag, ep_in_ro = 0.0, 0, ep_in_ro + 1
                continue
            st = torch.from_numpy(rs).float().to(device)
            with torch.no_grad():
                a, lp, v = ac.get_action(st, red_legal)
            blue_legal = env.blue_legal_actions()
            ba = random_blue_action(env, blue_legal)
            rs, bs, rr, br, done, info = env.step(a, ba)
            states.append(rs); actions.append(a); log_probs.append(lp)
            rewards.append(rr); values.append(v); dones.append(done)
            ep_r += rr
            if info.get("flag_captured"):
                ep_flag = 1
            step_count += 1
            if done:
                all_rewards.append(ep_r); all_flags.append(ep_flag)
                ep_r, ep_flag, ep_in_ro = 0.0, 0, ep_in_ro + 1
                rs, bs = env.reset()

        # bootstrap
        with torch.no_grad():
            st = torch.from_numpy(rs).float().to(device)
            _, last_v = ac(st.unsqueeze(0)); last_v = last_v.item()

        advs, rets = compute_gae(rewards, values, dones, last_v, gamma, gae_lambda)
        adv_t = torch.tensor(advs).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(rets).to(device)
        S = torch.tensor(np.stack(states)).float().to(device)
        A = torch.tensor(actions).to(device)
        old_lp = torch.tensor(log_probs).to(device)
        n = len(states)

        from torch.distributions import Categorical
        for _ in range(update_epochs):
            idx = torch.randperm(n)
            for s in range(0, n, batch_size):
                mb = idx[s:s+batch_size]
                logits, vals = ac(S[mb])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(A[mb])
                ent = dist.entropy().mean()
                ratio = torch.exp(new_lp - old_lp[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = torch.nn.MSELoss()(vals, ret_t[mb])
                loss = policy_loss + value_coef * value_loss - entropy_coef * ent
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

        if ro % print_every == 0 or ro == 1:
            recent = all_rewards[-200:] if len(all_rewards) >= 200 else all_rewards
            avg = sum(recent) / len(recent)
            rf = all_flags[-200:] if len(all_flags) >= 200 else all_flags
            fp = sum(rf) / len(rf) * 100
            print(f"[RO {ro:3d}/{rollouts}] eps={ep_in_ro} avg={avg:+7.2f} "
                  f"flag%={fp:.0f}% ploss={policy_loss.item():+.3f} "
                  f"vloss={value_loss.item():.2f} ent={ent.item():.2f}")

    return ac, all_rewards


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = MultiAgentAttackEnv(config_path=args.config)
    print(f"[环境] 红队 {env.red_state_dim}D/{env.n_red_actions}动 | "
          f"蓝队 {env.blue_state_dim}D/{env.n_blue_actions}动 | 蓝队=随机")
    ac, rewards = train_red_vs_random(env, rollouts=args.rollouts, seed=args.seed)

    # 保存红队模型
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "red_vs_random.pt")
    torch.save({"state_dict": ac.state_dict(),
                "state_dim": env.red_state_dim,
                "n_actions": env.n_red_actions}, path)
    print(f"\n[保存] 红队模型 → {os.path.abspath(path)}")
