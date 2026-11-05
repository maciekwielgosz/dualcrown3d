#!/bin/bash
#SBATCH --job-name=vfw_ens
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plg4digitalforest-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Weiser-regime ensemble (trunk_reconstruction ROADMAP step A2): average the given detectors, train the
# centre verifier on the ensemble and run the validation sweep with selection (all validation sets,
# including the Weiser val plots and the 8 m reach configuration).
# Usage: sbatch [--dependency=afterok:<ids>] slurm/weiser_ensemble.sbatch <ensemble run> <run/weights> ...
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
RUN=$1; shift
echo "[$(date)] $(hostname) job $SLURM_JOB_ID ensemble $RUN of $*"
$PY scripts/build_vote_field_ensemble.py --run "$RUN" --members "$@"
$PY scripts/train_centre_verifier.py --run "$RUN" --jobs 14 --select
$PY scripts/validate_vote_field.py --run "$RUN" --weights last.pt --tag full --jobs 14 --select
echo "[$(date)] finished $RUN"
