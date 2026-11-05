#!/bin/bash
#SBATCH --job-name=vf
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Train one vote-field detector, then run the full validation sweep on its last checkpoint.
# Usage: sbatch --job-name=vf_<run> slurm/train_vote_field.sbatch <run> [train_vote_field.py options]
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
RUN=$1; shift
echo "[$(date)] $(hostname) job $SLURM_JOB_ID run $RUN args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/train_vote_field.py --run "$RUN" --workers 14 "$@"
$PY scripts/validate_vote_field.py --run "$RUN" --weights last.pt --tag full --jobs 14
echo "[$(date)] finished $RUN"
