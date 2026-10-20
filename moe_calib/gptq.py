"""Standard GPTQ solver (weight-only, per-group asymmetric), lazy-batch form.

Faithful to Frantar, Ashkboos, Hoefler & Alistarh, "GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers"
(ICLR 2023), Algorithm 1:
  * H = sum x x^T, 1% dampening of mean diag(H)
  * act-order (columns by descending diag(H))
  * upper Cholesky of H^{-1}; column-wise rounding with error feedback
  * LAZY BATCH updates (blocksize 128): errors inside a block are propagated
    eagerly within the block and to the remaining columns with one matmul
    per block — mathematically identical to the column-by-column update,
    several times faster on GPU (important on a MIG slice).
The solver is UNCHANGED across E1/E2/E3: the proposal only changes how
calibration evidence is acquired.
"""
from __future__ import annotations
import torch


def _qparams(Wg: torch.Tensor, maxq: int):
    wmax = Wg.max(dim=1).values.clamp(min=0)
    wmin = Wg.min(dim=1).values.clamp(max=0)
    scale = (wmax - wmin).clamp(min=1e-10) / maxq
    zero = torch.round(-wmin / scale)
    return scale, zero


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int = 4,
                  group_size: int = 128, percdamp: float = 0.01,
                  actorder: bool = True, blocksize: int = 128) -> torch.Tensor:
    """W: [rows, cols] (out, in); H: [cols, cols]. Returns dequantized fp32 W_q."""
    dev = H.device
    W = W.to(dev, torch.float32).clone()
    H = H.to(torch.float32).clone()
    rows, cols = W.shape
    maxq = 2 ** bits - 1

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]

    H += torch.eye(cols, device=dev) * (percdamp * torch.mean(torch.diag(H)))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),
                                 upper=True)

    Q = torch.zeros_like(W)
    scale = zero = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group_size == 0:
                # group start: columns [col, col+gs) of W1/W are up to date here
                # when group_size is a multiple of blocksize-aligned starts
                src = W1[:, i:i + group_size] if i + group_size <= i2 - i1 \
                    else W[:, col:col + group_size]
                scale, zero = _qparams(src, maxq)
            w = W1[:, i]
            q = torch.clamp(torch.round(w / scale + zero), 0, maxq)
            dq = scale * (q - zero)
            Q1[:, i] = dq
            err = (w - dq) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


@torch.no_grad()
def quantize_expert(weights: dict, H_gu: torch.Tensor, H_down: torch.Tensor,
                    bits: int = 4, group_size: int = 128, percdamp: float = 0.01) -> dict:
    """gate_proj/up_proj share H_gu; down_proj uses H_down. Never mutates the
    model: every call starts from the FP weights passed in."""
    from .model_utils import GPU
    Hg, Hd = H_gu.to(GPU), H_down.to(GPU)
    return {
        "gate_proj": gptq_quantize(weights["gate_proj"], Hg, bits, group_size, percdamp),
        "up_proj": gptq_quantize(weights["up_proj"], Hg, bits, group_size, percdamp),
        "down_proj": gptq_quantize(weights["down_proj"], Hd, bits, group_size, percdamp),
    }
