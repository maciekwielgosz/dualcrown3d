#!/bin/bash
#SBATCH --job-name=ss
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Fine-tune Model20 with the stem-seeded decoder (one query per stem foot).
# Usage: sbatch --job-name=ss_<run> slurm/train_stem_seeded.sbatch <run> [train_stem_seeded.py options]
# Always resumes from weights/last.pt, so the same command continues a run after the time limit.
# Requires stems.csv next to every training plot: $PY scripts/prepare_stem_feet.py (once, on the data copy).
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
RUN=$1; shift
echo "[$(date)] $(hostname) job $SLURM_JOB_ID run $RUN args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/train_stem_seeded.py --run "$RUN" --workers 6 --resume "$@"
echo "[$(date)] finished $RUN"
