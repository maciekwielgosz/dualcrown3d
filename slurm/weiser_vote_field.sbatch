#!/bin/bash
#SBATCH --job-name=vf_weiser
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
#SBATCH --error=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/logs/%x_%j.out
# Cache Model20's ensembled votes on the Weiser plots (real scans and simulated tiles), then train
# one vote-field detector with the Weiser regime added to the v7 mix (dense .3, thinned .3,
# HELIOS .15, Weiser .25) and validate it. Usage: sbatch slurm/weiser_vote_field.sbatch <run>
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
RUN=${1:-vf_weiser_a}
echo "[$(date)] $(hostname) job $SLURM_JOB_ID run $RUN"
$PY scripts/cache_stage1_ensemble.py --regimes weiser --splits val train
$PY scripts/train_vote_field.py --run "$RUN" --workers 14 --epochs 120 --samples 2000 --val-every 60 --mix .3 .3 .15 .25
$PY scripts/validate_vote_field.py --run "$RUN" --weights last.pt --tag full --jobs 14
echo "[$(date)] finished $RUN"
