#!/bin/bash
#SBATCH -J cg-v3
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
echo "  能力异构控制博弈 v3 training"
echo "  A=快攻型 B=隐蔽型"
echo "=========================================="
echo "节点: $(hostname)"
echo "日期: $(date)"
echo "=========================================="

.venv/bin/python -B src/train_v3.py --rounds 200 --pop 3 --seed 0

echo "完成。日志在 logs/"
