#!/usr/bin/env bash
# One-time setup inside an ICRN JupyterLab terminal (PyTorch environment).
set -e
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install --user -r requirements.txt
nvidia-smi -L || true
echo "MEM_LIMIT=$MEM_LIMIT  CPU_LIMIT=$CPU_LIMIT"
python - <<'PY'
import torch
free, total = torch.cuda.mem_get_info()
print(f"GPU slice: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
PY
