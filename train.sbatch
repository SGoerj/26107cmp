#!/bin/bash
#SBATCH -J rl-attack       # 作业名
#SBATCH -p Students          # 分区
#SBATCH --qos=qos_stu_default # QOS
#SBATCH -o logs/%x-%j.out   # 标准输出
#SBATCH -e logs/%x-%j.err   # 错误日志
#SBATCH -t 00:30:00         # 最长 30 分钟
#SBATCH --cpus-per-task=2   # 2 个 CPU 核

# ── 如果你有 GPU 节点权限，取消下一行的注释 ──
# SBATCH --gres=gpu:1

set -euo pipefail

# 进入项目目录
cd ~/projects/rl-attack-defense

echo "=========================================="
echo "节点: $(hostname)"
echo "日期: $(date)"
echo "=========================================="

# 激活虚拟环境
source .venv/bin/activate
python src/agent.py

echo "=========================================="
echo "训练完成, 日志已保存到 logs/"
echo "=========================================="