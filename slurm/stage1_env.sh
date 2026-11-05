# Extra environment of the scaled stage-1 jobs (source after slurm/env.sh): caches on $SCRATCH, never in $HOME.
export STAGE1_CACHE=${SCRATCH:-/net/tscratch/people/$USER}/dualcrown_cache/stage1
mkdir -p "$STAGE1_CACHE"
export XDG_CACHE_HOME=$STAGE1_CACHE/xdg TORCH_HOME=$STAGE1_CACHE/torch MPLCONFIGDIR=$STAGE1_CACHE/matplotlib
export PYTHONDONTWRITEBYTECODE=1
