#!/bin/bash
#SBATCH --job-name=s1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Scaled stage-1 fine-tune of Model20, then the full vote-quality validation of its best and last checkpoints.
# Usage: sbatch --job-name=s1_<run> slurm/train_stage1.sbatch <run> [train_stage1_scaled.py options]
# An interrupted run continues with the same command plus --resume.
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
source slurm/stage1_env.sh
RUN=$1; shift
echo "[$(date)] $(hostname) job $SLURM_JOB_ID run $RUN args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/train_stage1_scaled.py --run "$RUN" --workers 14 --time-limit-hours "${TIME_LIMIT_HOURS:-10.5}" "$@"
ROOT=outputs/dualcrown3d_stage1_scaled_v1
for WHICH in best last; do
  if [ ! -e $ROOT/validation/${RUN}_$WHICH/summary.json ]; then
    $PY scripts/validate_stage1_votes.py --checkpoint $ROOT/runs/$RUN/weights/$WHICH.pt --name ${RUN}_$WHICH \
        --modes single ens4 --consensus --jobs 14
  fi
done
echo "[$(date)] finished $RUN"
