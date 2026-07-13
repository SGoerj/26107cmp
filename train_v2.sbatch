#!/bin/bash
#SBATCH -J rl-agent-v2
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 00:45:00
#SBATCH --cpus-per-task=4

# ── 如果你有 GPU 节点权限，取消下一行的注释 ──
# SBATCH --gres=gpu:1

set -euo pipefail

cd ~/projects/rl-attack-defense

# 环境变量选配置 (默认 4 节点):
#   SCENARIO=default  → configs/env_default.yaml
#   SCENARIO=6node    → configs/env_6node.yaml
# 算法选择:
#   ALGO=dqn (默认)   → src/agent_v2.py  (Double Dueling DQN)
#   ALGO=ppo          → src/agent_ppo.py (PPO, on-policy 不遗忘)
SCENARIO="${SCENARIO:-default}"
ALGO="${ALGO:-dqn}"
mkdir -p logs

echo "=========================================="
echo "  🔥 RL 自主攻击智能体  [场景: $SCENARIO | 算法: $ALGO]"
echo "=========================================="
echo "节点: $(hostname)"
echo "日期: $(date)"
echo "Python: $(.venv/bin/python --version)"
echo "=========================================="

# 先跑冒烟测试 (仅默认配置; 6节点路径不同, 跳过路径测试只验证加载)
echo ""
echo "[1/3] 冒烟测试..."
if [ "$SCENARIO" = "default" ]; then
  .venv/bin/python -c "
from src.test_local_v2 import *
tests = [test_safe_path, test_fast_path, test_invalid_actions,
         test_state_dimension, test_detection_contrast]
for t in tests:
    try:
        t()
    except Exception as e:
        print(f'FAIL: {e}')
        raise
"
  echo "  冒烟测试通过 ✅"
else
  # 非默认配置: 只验证能加载 + 维度自洽
  .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from env_v2 import AttackChainEnv
env = AttackChainEnv(config_path='configs/env_${SCENARIO}.yaml')
s = env.reset()
assert s.shape[0] == env.state_dim, 'state 维度不自洽'
print(f'  ✅ 配置加载 OK: 节点={env.n_nodes} state_dim={env.state_dim} n_actions={env.n_actions}')
print(f'  (路径测试跳过: ${SCENARIO} 拓扑与默认不同)')
"
fi

# 跑训练
echo ""
if [ "$SCENARIO" = "6node" ]; then
  EPISODES=3000
else
  EPISODES=2000
fi
echo "[2/3] 开始训练 (场景 $SCENARIO, 算法 $ALGO)..."

# RUN=sensitivity: 跑敏感度分析 (9 组扰动, 不跑常规训练)
if [ "${RUN:-}" = "sensitivity" ]; then
  echo "  敏感度分析模式: 9 组参数扰动 × 训练40rollout × 评估100ep"
  .venv/bin/python src/sensitivity.py
elif [ "$ALGO" = "ppo" ]; then
  # PPO rollouts 默认 60; 难场景 (6节点动态) 可通过 ROLLOUTS 环境变量加量
  ROLLOUTS="${ROLLOUTS:-60}"
  .venv/bin/python src/agent_ppo.py --config "configs/env_${SCENARIO}.yaml" --rollouts "$ROLLOUTS"
else
  .venv/bin/python src/agent_v2.py --config "configs/env_${SCENARIO}.yaml" --episodes "$EPISODES"
fi

echo ""
echo "[3/3] 完成。"
echo "=========================================="
echo "日志已保存到 logs/"
echo "=========================================="