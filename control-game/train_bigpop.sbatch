#!/bin/bash
#SBATCH -J cg-bigpop
#SBATCH -p Students
#SBATCH --account=stu
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd ~/projects/control-game
mkdir -p logs

echo "=========================================="
echo "  控制博弈 v1 大种群训练 (pop=8, 200轮)"
echo "  分区: cpu-8358p  QOS: qos_stu_small (8h)"
echo "=========================================="
echo "节点: $(hostname)"
echo "日期: $(date)"
echo "=========================================="

.venv/bin/python -B src/train_full.py --rounds 200 --pop 8 --seed 0

echo "完成。日志在 logs/"
