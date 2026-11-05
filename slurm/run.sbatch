#!/bin/bash
#SBATCH --job-name=dc
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
# Run any repo script on one A100: sbatch --job-name=<name> slurm/run.sbatch scripts/<script>.py [options]
set -euo pipefail
source /net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas/segm/DL_model_version/slurm/env.sh
echo "[$(date)] $(hostname) job $SLURM_JOB_ID: $*"
$PY "$@"
echo "[$(date)] finished"
