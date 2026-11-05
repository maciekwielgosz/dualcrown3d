# Common environment for DualCrown3D jobs on Athena (source this file).
export IDEAS=/net/pr2/projects/plgrid/plggwie/plgwielgosz/ideas
export DUALCROWN_PATH_MAP="/home/maciej.wielgosz/Projects/segm=$IDEAS/segm"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export PY=$IDEAS/envs/dualcrown/bin/python
cd "$IDEAS/segm/DL_model_version"
