"""
兜底层 2: 固定规则蓝队 + 红队 PPO 训练

规则蓝队 (比随机聪明):
  - 50% 概率 MONITOR (别乱动, 省预算)
  - 否则: 找告警最多的节点
      有告警 → ISOLATE 该节点 (隔离危险源)
      无告警 → 不动作 (避免误隔离/误加固, 规则蓝队保守)
  - 预算用完 → 只能 MONITOR

验证目标: 红队能不能对付"会反应"的蓝队。
随机蓝队 (兜底层1) 让红队 flag% 从 100% 降到 60%。
规则蓝队更聪明 (会隔离告警节点), 预期红队 flag% 更低 (40-50%)。
若红队仍能维持 40%+, 说明红队学会对抗反应型防御, 可上 self-play。
若红队被压到 ~0%, 说明红队不够强, 要先调红队。

用法 (HPC):
  .venv/bin/python src/multiagent/train_red_vs_rule.py --rollouts 60 --seed 0
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_ma import MultiAgentAttackEnv, BLUE_MONITOR, BLUE_HARDEN_BASE
from agent_ppo import ActorCritic, compute_gae


def rule_blue_action(env, blue_legal):
    """规则蓝队: 50% MONITOR; 否则隔离告警最多节点; 无告警则 MONITOR。

    注意: BLUE_ISOLATE_BASE 是 env 实例化时才设的全局值, 不能用 import 的
    (那是 None)。从 env 实例拿: 1 + n_nodes。
    """
    if env.blue_budget <= 0:
        return BLUE_MONITOR
    if np.random.random() < 0.5:
        return BLUE_MONITOR

    max_alert = max(env.alerts)
    if max_alert <= 0:
        return BLUE_MONITOR   # 无告警, 保守不动 (避免误隔离代价)

    target = env.alerts.index(max_alert)
    isolate_base = 1 + env.n_nodes   # BLUE_ISOLATE_BASE = 1 + n_nodes

    # 优先 ISOLATE (隔离危险源), 若已隔离则 HARDEN
    isolate_act = isolate_base + target
    if isolate_act in blue_legal and not env.isolated[target]:
        return isolate_act
    harden_act = BLUE_HARDEN_BASE + target
    if harden_act in blue_legal and not env.hardened[target]:
        return harden_act
    return BLUE_MONITOR


def train_red_vs_rule(env, rollouts=60, rollout_steps=2000, update_epochs=10,
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
          f"红队 state={red_dim} actions={red_n} | 蓝队=规则")

    ac = ActorCritic(red_dim, red_n).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)

    all_rewards, all_flags = [], []

    for ro in range(1, rollouts + 1):
        states, actions, log_probs, rewards, values, dones, legals = [], [], [], [], [], [], []
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
            ba = rule_blue_action(env, blue_legal)
            states.append(rs.copy()); actions.append(a); log_probs.append(lp); values.append(v)
            legals.append(red_legal)
            rs, bs, rr, br, done, info = env.step(a, ba)
            rewards.append(rr); dones.append(done)
            ep_r += rr
            if info.get("flag_captured"):
                ep_flag = 1
            step_count += 1
            if done:
                all_rewards.append(ep_r); all_flags.append(ep_flag)
                ep_r, ep_flag, ep_in_ro = 0.0, 0, ep_in_ro + 1
                rs, bs = env.reset()

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
        n_actions = env.n_red_actions
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
          f"蓝队 {env.blue_state_dim}D/{env.n_blue_actions}动 | 蓝队=规则")
    ac, rewards = train_red_vs_rule(env, rollouts=args.rollouts, seed=args.seed)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "red_vs_rule.pt")
    torch.save({"state_dict": ac.state_dict(),
                "state_dim": env.red_state_dim,
                "n_actions": env.n_red_actions}, path)
    print(f"\n[保存] 红队模型 → {os.path.abspath(path)}")
