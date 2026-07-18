#!/bin/bash
#SBATCH -J cg-train
#SBATCH -p Students
#SBATCH --account=stu
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd ~/projects/control-game
mkdir -p logs

echo "=========================================="
echo "  控制博弈 long training"
echo "=========================================="
echo "节点: $(hostname)"
echo "日期: $(date)"
echo "Python: $(.venv/bin/python --version 2>&1)"
echo "=========================================="

.venv/bin/python -B src/train_full.py --rounds 200 --pop 3 --seed 0

echo ""
echo "=========================================="
echo "完成。日志在 logs/"
echo "=========================================="
