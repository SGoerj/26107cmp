"""
敏感度分析 —— 参数波动下 PPO 策略是否稳健

回答的问题: 我们手编的参数 (检测率、利用率、动态α) 不精确怎么办?
如果策略对这些参数不敏感 (波动时 flag% 仍高), 参数不精确问题就缓解了;
如果某参数小波动导致 flag% 大跌, 说明该参数需要更精确估计。

做法: 以 env_default_dynamic_prob.yaml 为基线, 逐个扰动单个参数组,
      训练 PPO (短训练), 评估 final 策略 100 episode, 汇总指标。

用法: python src/sensitivity.py
"""

import os
import sys
import copy
import tempfile
from dataclasses import dataclass

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import AttackChainEnv, load_config, _build_from_config
from agent_ppo import ActorCritic, train as ppo_train


# ──────────────────────────────────────────────────────────────────────
# 基线配置 + 扰动
# ──────────────────────────────────────────────────────────────────────
BASELINE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "env_default_dynamic_prob.yaml")


def _perturb_detection(cfg: dict, factor: float) -> dict:
    """所有节点 detection 乘 factor。"""
    cfg = copy.deepcopy(cfg)
    for n in cfg["nodes"]:
        n["detection"] = round(min(0.99, max(0.01, n["detection"] * factor)), 3)
    return cfg


def _perturb_exploit_prob(cfg: dict, cve_prob: float) -> dict:
    """设 cve=cve_prob, creds/sqli 等比缩放保持相对关系。基线 cve=0.70。"""
    cfg = copy.deepcopy(cfg)
    base_cve = 0.70
    scale = cve_prob / base_cve
    p = cfg["exploit_success_prob"]
    p["cve"] = cve_prob
    p["creds"] = round(min(1.0, 0.85 * scale), 3)
    p["sqli"] = round(min(1.0, 0.60 * scale), 3)
    return cfg


def _perturb_alpha_attack(cfg: dict, alpha: float) -> dict:
    """设动态检测 alpha_attack。"""
    cfg = copy.deepcopy(cfg)
    cfg["dynamic_detection"]["alpha_attack"] = alpha
    return cfg


# 一组实验: (实验名, 扰动函数, 参数值, 短描述)
@dataclass
class Experiment:
    name: str
    perturb: callable
    value: float
    desc: str


EXPERIMENTS = [
    # 检测率缩放 (3 档)
    Experiment("det×0.8", _perturb_detection, 0.8, "所有节点检测率 ×0.8 (IDS 较弱)"),
    Experiment("det×1.0", _perturb_detection, 1.0, "基线检测率"),
    Experiment("det×1.2", _perturb_detection, 1.2, "所有节点检测率 ×1.2 (IDS 较强)"),
    # 利用成功率 (3 档, cve 为代表)
    Experiment("cve=0.5", _perturb_exploit_prob, 0.5, "利用率低 (利用更难成功)"),
    Experiment("cve=0.7", _perturb_exploit_prob, 0.7, "基线利用率"),
    Experiment("cve=0.9", _perturb_exploit_prob, 0.9, "利用率高 (利用易成功)"),
    # 动态 α_attack (3 档)
    Experiment("α=0.04", _perturb_alpha_attack, 0.04, "告警升级慢 (动态弱)"),
    Experiment("α=0.08", _perturb_alpha_attack, 0.08, "基线 α_attack"),
    Experiment("α=0.12", _perturb_alpha_attack, 0.12, "告警升级快 (动态强)"),
]


# ──────────────────────────────────────────────────────────────────────
# 评估: 用训练好的策略跑 N 个 episode, 统计指标
# ──────────────────────────────────────────────────────────────────────
def evaluate(ac: ActorCritic, env: AttackChainEnv, n_episodes: int = 100):
    """用训练好的策略 (greedy) 跑 n_episodes 个 episode, 统计指标。"""
    import torch
    device = next(ac.parameters()).device
    rewards, detections, steps = [], [], []
    flag_count = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_r = 0.0
        det = 0
        st = 0
        got = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            state_tensor = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                logits, _ = ac(state_tensor.unsqueeze(0))
                logits = logits[0]
                mask = torch.full_like(logits, float("-inf"))
                for a in legal:
                    mask[a] = 0.0
                action = (logits + mask).argmax().item()   # greedy 评估
            state, r, done, info = env.step(action)
            total_r += r
            if info.get("detected"):
                det += 1
            st += 1
            if info.get("flag_captured"):
                got = 1
        rewards.append(total_r)
        detections.append(det)
        steps.append(st)
        if got:
            flag_count += 1
    return {
        "avg_reward": float(np.mean(rewards)),
        "flag_pct": flag_count / n_episodes * 100,
        "avg_detections": float(np.mean(detections)),
        "avg_steps": float(np.mean(steps)),
    }


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────
def run_one(exp: Experiment, rollouts: int = 60, eval_episodes: int = 100,
            seeds: tuple = (0, 1, 2)):
    """对单个扰动: 写临时配置 → 多种子训练 PPO → 评估 → 取中位数。

    多种子取中位数, 消除短训练随机性 (单种子可能因运气不收敛)。
    """
    base_cfg = load_config(BASELINE_CONFIG)
    cfg = exp.perturb(base_cfg, exp.value)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir="/tmp", encoding="utf-8"
    ) as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        tmp_path = f.name

    per_seed = []
    try:
        for sd in seeds:
            env = AttackChainEnv(config_path=tmp_path)
            ac, _ = ppo_train(env, rollouts=rollouts, print_every=1000, seed=sd)
            per_seed.append(evaluate(ac, env, n_episodes=eval_episodes))
    finally:
        os.unlink(tmp_path)

    # 取中位数 (比均值抗异常种子)
    import statistics as st
    return {
        "avg_reward": st.median(m["avg_reward"] for m in per_seed),
        "flag_pct": st.median(m["flag_pct"] for m in per_seed),
        "avg_detections": st.median(m["avg_detections"] for m in per_seed),
        "avg_steps": st.median(m["avg_steps"] for m in per_seed),
        "seed_rewards": [m["avg_reward"] for m in per_seed],
        "seed_flags": [m["flag_pct"] for m in per_seed],
    }


if __name__ == "__main__":
    print("=" * 78)
    print("  敏感度分析: 参数波动下 PPO 策略稳健性")
    print("  基线: env_default_dynamic_prob.yaml (4节点+动态+概率利用)")
    print("  每组: 3 种子 × 训练60rollout × 评估100ep, 取中位数")
    print("=" * 78)

    results = []
    n_total = len(EXPERIMENTS)
    for i, exp in enumerate(EXPERIMENTS, 1):
        name, desc = exp.name, exp.desc
        print(f"\n[{i}/{n_total}] {name}  ({desc})")
        m = run_one(exp)
        m["name"] = name
        m["desc"] = desc
        results.append(m)
        ar, fp, ad, ast = m["avg_reward"], m["flag_pct"], m["avg_detections"], m["avg_steps"]
        sf = "/".join(f"{x:.0f}" for x in m["seed_flags"])
        print(f"  → 中位数: reward={ar:+7.2f} flag%={fp:.0f} det={ad:.2f} steps={ast:.1f}")
        print(f"    3种子 flag%: [{sf}]")

    # 汇总表
    print("\n" + "=" * 78)
    print("  汇总 (3 种子中位数)")
    print("=" * 78)
    print(f"  {'实验':<10} {'reward':>9} {'flag%':>7} {'det':>6} {'steps':>7}  {'3种子flag%':<16}")
    print("-" * 78)
    for r in results:
        sf = "/".join(f"{x:.0f}" for x in r["seed_flags"])
        print(f"  {r['name']:<10} {r['avg_reward']:>+9.2f} {r['flag_pct']:>6.0f}% "
              f"{r['avg_detections']:>6.2f} {r['avg_steps']:>7.1f}  [{sf}]")
    print("=" * 78)

    # 稳健性判断
    print("\n  稳健性判断:")
    flags = [r["flag_pct"] for r in results]
    print(f"    flag% 中位数范围: {min(flags):.0f}% ~ {max(flags):.0f}%")
    if min(flags) >= 90:
        print("    → 策略对参数波动 ROBUST (所有配置 flag% 中位数 ≥ 90%), 参数不精确问题可接受")
    elif min(flags) >= 70:
        print("    → 策略对参数波动中等敏感 (最低中位数 70-90%), 某些参数需更精确估计")
    else:
        print("    → 策略对参数波动敏感 (最低中位数 < 70%), 参数精确度对策略影响大")
