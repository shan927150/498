"""Quantize experts from Hessians and score them on held-out activations.

Metric, for expert (L,e) and held-out sequence s:
  relerr_s = sum_proj ||X_s (W - W_q)^T||^2 / sum_proj ||X_s W^T||^2
i.e. GPTQ's own objective on held-out inputs, in relative form so experts with
different activation scales are comparable. The PER-SEQUENCE values feed E1's
paired t-test (sampling unit = sequence, never token).
"""
from __future__ import annotations
import torch

from . import model_utils as MU
from .gptq import quantize_expert


def fp_weights(model, L, e):
    return {k: v.detach().float() for k, v in MU.get_expert_weights(model, L, e).items()}


@torch.no_grad()
def quantize_from_H(model, L, e, H_gu, H_down, bits=4, group_size=128):
    return quantize_expert(fp_weights(model, L, e), H_gu, H_down, bits, group_size)


@torch.no_grad()
def quantize_from_acc(model, acc, L, e, blocks=None, bits=4, group_size=128):
    b = blocks if blocks is not None else list(range(acc.K))
    return quantize_from_H(model, L, e, acc.H_sum((L, e, "gu"), b),
                           acc.H_sum((L, e, "down"), b), bits, group_size)


@torch.no_grad()
def per_seq_rel_errors(model, L, e, Wq: dict, holdout: dict) -> dict[int, float]:
    W = {k: v.to(MU.GPU) for k, v in fp_weights(model, L, e).items()}
    D = {k: W[k] - Wq[k].to(MU.GPU, torch.float32) for k in W}
    num: dict[int, float] = {}
    den: dict[int, float] = {}
    for proj, names in (("gu", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
        for sid, X in holdout.get((L, e, proj), []):
            Xf = X.to(MU.GPU, torch.float32)
            for nm in names:
                num[sid] = num.get(sid, 0.0) + float((Xf @ D[nm].T).pow(2).sum())
                den[sid] = den.get(sid, 0.0) + float((Xf @ W[nm].T).pow(2).sum())
    return {s: num[s] / (den.get(s, 0.0) + 1e-12) for s in num}
