#!/bin/bash
#SBATCH --job-name=s1_smoke
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Leakage check, unit tests, float32 / mixed-precision benchmark and a two-epoch smoke run.
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
source slurm/stage1_env.sh
TAG=${1:-a}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/train_stage1_scaled.py --run check --check-only
CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -p "test_stage1_scaled.py" 2>&1 | tail -4
$PY scripts/train_stage1_scaled.py --run bench16k_$TAG --benchmark --max-points 16000 --scope full --z-weight .33 --assign-weight .5
$PY scripts/train_stage1_scaled.py --run bench40k_$TAG --benchmark --max-points 40000 --scope full --z-weight .33 --assign-weight .5
$PY scripts/train_stage1_scaled.py --run smoke_fp32_$TAG --smoke --scope full --z-weight .33 --assign-weight .5 --thin-probability .5 --max-points 40000
$PY scripts/train_stage1_scaled.py --run smoke_amp_$TAG --smoke --amp --scope partial --thin-probability .5 || echo AMP_SMOKE_FAILED
echo SMOKE_DONE
