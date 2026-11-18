"""Opt-in GPTQ act-order policies, built ONLY on the existing moe_calib.gptq.gptq_quantize interface.

The production solver is not modified. Every policy is expressed through the solver's two existing switches:
  natural   actorder=True, actorder_override=None      -> argsort(diag H, descending) inside the solver (historical)
  fixed     actorder=True, actorder_override=ref_perm  -> the ORIGINAL configuration's natural permutation
  bucketed  actorder=True, actorder_override=bucketed_perm(own H)   (reference-free, deterministic)
  disabled  actorder=False                             -> original column order (desc_act=False)
Quantization groups always follow the solver's own semantics: consecutive group_size columns of the EXECUTION order
(no static groups), so for `disabled` the groups are consecutive original columns.

Bucketed key (frozen definition): the solver's diag(H) after its dead-column fix (diag == 0 -> 1.0), converted to
float64, written as m * 2**e with m in [0.5, 1) (frexp), truncated to `bits` significant binary bits:
key = floor(m * 2**bits) * 2**(e - bits). The boundaries are fixed powers-of-two subdivisions, independent of the
observed minimum / maximum. Columns are ordered by descending key; ties are broken by ascending original column index
(stable sort on the CPU, numpy kind="stable").
"""
from __future__ import annotations

import numpy as np
import torch

from .gptq import gptq_quantize

POLICIES = ("natural", "fixed", "bucketed", "disabled")


def solver_diag(H: torch.Tensor) -> torch.Tensor:
    """diag(H) exactly as the solver sees it when act-order is computed (after the dead-column fix)."""
    d = torch.diag(H).detach().to("cpu", torch.float32).clone()
    d[d == 0] = 1.0
    return d


def bucket_key(d: torch.Tensor, bits: int) -> torch.Tensor:
    m, e = torch.frexp(d.to(torch.float64))
    return torch.ldexp(torch.floor(m * (2.0 ** bits)), (e - bits).to(torch.float64))


def bucketed_perm(H: torch.Tensor, bits: int) -> torch.Tensor:
    """Descending bucketed diag(H); ties by ascending original column index. int64 CPU permutation."""
    k = bucket_key(solver_diag(H), bits).numpy()
    order = np.argsort(-k, kind="stable")                 # stable: equal keys keep ascending column index
    return torch.from_numpy(order.astype(np.int64))


def natural_perm_cpu(H: torch.Tensor) -> torch.Tensor:
    """The natural permutation, computed with the solver's own op on H's device (for the fixed-order reference)."""
    Hc = H.to(torch.float32).clone()
    dead = torch.diag(Hc) == 0
    Hc[dead, dead] = 1.0
    return torch.argsort(torch.diag(Hc), descending=True).cpu()


@torch.no_grad()
def solve(W: torch.Tensor, H: torch.Tensor, policy: str, bits: int, group_size: int, percdamp: float,
          ref_perm=None, bucket_bits: int | None = None, info: dict | None = None) -> torch.Tensor:
    """One projection under one policy. W: FP weights [out, in] (any dtype; .float() as quant_eval.fp_weights);
    H on the solve device. Returns the solver's FP32 dequantized W_q."""
    if policy == "natural":
        return gptq_quantize(W.float(), H, bits, group_size, percdamp, info=info)
    if policy == "fixed":
        if ref_perm is None:
            raise ValueError("fixed policy requires ref_perm")
        return gptq_quantize(W.float(), H, bits, group_size, percdamp, actorder_override=ref_perm, info=info)
    if policy == "bucketed":
        if bucket_bits is None:
            raise ValueError("bucketed policy requires bucket_bits")
        return gptq_quantize(W.float(), H, bits, group_size, percdamp,
                             actorder_override=bucketed_perm(H, bucket_bits), info=info)
    if policy == "disabled":
        return gptq_quantize(W.float(), H, bits, group_size, percdamp, actorder=False, info=info)
    raise ValueError(f"unknown policy {policy}")
