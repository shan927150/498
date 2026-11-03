# Shared settings for the parallel 8-hour plan (one ICRN session, A100 20GB slice).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3          # 4 processes x 3 threads = 12 CPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CACHE=${CACHE:-$HOME/moe_cache}               # existing cache (layers 1/8/16/23)
export NEWCACHE=${NEWCACHE:-$HOME/moe_cache_new}     # new cache for layers 4/12/20
export DAMP=$(python -c "import json;print(json.load(open('configs/frozen_b4.json'))['gptq']['percdamp'])")
unset MOE_LOAD_LAYERS                                 # set per experiment, never globally
