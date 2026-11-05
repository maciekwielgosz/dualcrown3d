#!/bin/bash
#SBATCH --job-name=s1_val
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Vote-quality validation of one stage-1 checkpoint (validation plots only).
# Usage: sbatch --job-name=s1_val_<name> slurm/validate_stage1.sbatch <name> <checkpoint> [validate_stage1_votes.py options]
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
source slurm/stage1_env.sh
NAME=$1; CHECKPOINT=$2; shift 2
echo "[$(date)] $(hostname) job $SLURM_JOB_ID validate $NAME $CHECKPOINT args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/validate_stage1_votes.py --name "$NAME" --checkpoint "$CHECKPOINT" --jobs 14 "$@"
echo "[$(date)] finished $NAME"
