#!/bin/bash
#SBATCH --job-name=vf_smoke
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -p "test_vote_field.py" 2>&1 | tail -4
$PY scripts/train_vote_field.py --run athena_smoke --smoke --workers 14
echo SMOKE_DONE
