"""Activation-weighted functional differences between quantized weight sets (pure functions, streaming accumulators).

Conventions: activations X are [tokens, in] (row = token), weights W are [out, in] (nn.Linear), outputs Y = X W^T.
All functional arithmetic is FP32 (TF32 disabled by the caller) on the INSTALLED BF16 weights and the captured BF16
activations, so differences reflect the weights, not BF16 compute rounding. Squared norms are accumulated in float64
over all tokens BEFORE any ratio is taken (no per-token ratios with near-zero denominators).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- accumulators
def acc_new():
    return dict(num=0.0, den=0.0, aa=0.0, bb=0.0, ab=0.0, n_el=0, n_tok=0, tok=[], tok_ref=[], tok_cos=[])


@torch.no_grad()
def acc_add(acc, D, Yb, Yref, keep_tokens=True):
    """D = Y_a - Y_b (or (W_a - W_b) X), Yb = variant-b output, Yref = reference output; all [T, out] fp32."""
    if D.shape[0] == 0:
        return acc
    Ya = Yb + D
    acc["num"] += float(D.double().pow(2).sum())
    acc["den"] += float(Yref.double().pow(2).sum())
    acc["aa"] += float(Ya.double().pow(2).sum())
    acc["bb"] += float(Yb.double().pow(2).sum())
    acc["ab"] += float((Ya.double() * Yb.double()).sum())
    acc["n_el"] += D.numel()
    acc["n_tok"] += D.shape[0]
    if keep_tokens:
        acc["tok"].append(D.norm(dim=1).cpu())
        acc["tok_ref"].append(Yref.norm(dim=1).cpu())
        acc["tok_cos"].append(F.cosine_similarity(Ya, Yb, dim=1, eps=1e-30).cpu())
    return acc


def acc_merge(accs):
    out = acc_new()
    for a in accs:
        for k in ("num", "den", "aa", "bb", "ab", "n_el", "n_tok"):
            out[k] += a[k]
        for k in ("tok", "tok_ref", "tok_cos"):
            out[k].extend(a[k])
    return out


def acc_final(acc, with_tokens=True):
    num, den = acc["num"], acc["den"]
    r = dict(E_func=math.sqrt(num / den) if den > 0 else float("nan"),
             rel_to_b=math.sqrt(num / acc["bb"]) if acc["bb"] > 0 else float("nan"),
             rmse=math.sqrt(num / acc["n_el"]) if acc["n_el"] else float("nan"),
             cosine=acc["ab"] / math.sqrt(acc["aa"] * acc["bb"]) if acc["aa"] > 0 and acc["bb"] > 0 else float("nan"),
             sumsq_diff=num, sumsq_ref=den, n_tokens=acc["n_tok"], n_elements=acc["n_el"])
    if with_tokens and acc["tok"]:
        t = torch.cat(acc["tok"]).double()
        tr = torch.cat(acc["tok_ref"]).double()
        tc = torch.cat(acc["tok_cos"]).double()
        rms_ref = float(tr.pow(2).mean().sqrt())
        q = torch.quantile(t, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64)) if t.numel() < 16_000_000 else None
        if q is not None:
            r.update(tok_diff_p50=float(q[0]), tok_diff_p95=float(q[1]), tok_diff_p99=float(q[2]),
                     tok_diff_max=float(t.max()),
                     tok_diff_p95_over_rms_ref=float(q[1]) / rms_ref if rms_ref > 0 else float("nan"),
                     tok_diff_p99_over_rms_ref=float(q[2]) / rms_ref if rms_ref > 0 else float("nan"),
                     tok_cos_median=float(tc.median()), tok_cos_p01=float(torch.quantile(tc.float(), 0.01)))
    return r


# ---------------------------------------------------------------- weight-level metrics
@torch.no_grad()
def weight_metrics(Wa, Wb, Wref):
    a, b, rf = Wa.float(), Wb.float(), Wref.float()
    d = a - b
    return dict(frac_bf16_changed=float((Wa != Wb).float().mean()),
                rel_frob=float(d.norm() / b.norm()), rel_frob_vs_ref=float(d.norm() / rf.norm()),
                n=Wa.numel())


# ---------------------------------------------------------------- MLP forward (Qwen2 / Qwen2-MoE MLP: SiLU-GLU)
@torch.no_grad()
def mlp_fp32(X, Wg, Wu, Wd):
    """down(silu(gate(x)) * up(x)) in FP32 from BF16 weights / activations."""
    x = X.float()
    return (F.silu(x @ Wg.float().T) * (x @ Wu.float().T)) @ Wd.float().T


# ---------------------------------------------------------------- curvature / importance bins
def quantile_bins(v: torch.Tensor, nb: int) -> torch.Tensor:
    """Bin index 0..nb-1 by rank of v (ascending); ties broken by column index (stable)."""
    order = torch.from_numpy(np.argsort(v.double().cpu().numpy(), kind="stable"))
    rank = torch.empty_like(order)
    rank[order] = torch.arange(v.numel())
    return (rank * nb) // v.numel()


@torch.no_grad()
def importance_profile(dW, flip, Hheld, bins, nb):
    """dW [out, in] fp32 weight difference (a - b), flip [out, in] bool code change, Hheld [in, in] float64 held-out
    X^T X, bins [in] bin id. Returns per-bin flip share / rate, diagonal-approximation energy share, block-exact
    energy share (within-bin full quadratic form, cross-bin terms excluded) and the totals used for additivity checks.
    Energies are ||dW X||^2-type quantities: tr(dW_S Hheld_SS dW_S^T)."""
    dev = Hheld.device
    dW = dW.to(dev, torch.float64)
    flip = flip.to(dev)
    bins = bins.to(dev)
    hd = torch.diag(Hheld)
    col_sq = dW.pow(2).sum(0)
    diag_e = col_sq * hd
    full = float(((dW @ Hheld) * dW).sum())
    out = dict(full_energy=full, diag_energy=float(diag_e.sum()), n_flips=int(flip.sum()), bins=[])
    tot_flip = max(int(flip.sum()), 1)
    tot_diag = float(diag_e.sum()) or 1.0
    blocks = 0.0
    for b in range(nb):
        S = (bins == b).nonzero().flatten()
        dWS = dW[:, S]
        e_block = float(((dWS @ Hheld[S][:, S]) * dWS).sum())
        blocks += e_block
        nf = int(flip[:, S].sum())
        out["bins"].append(dict(n_cols=int(S.numel()), flips=nf, flip_rate=nf / max(flip.shape[0] * S.numel(), 1),
                                flip_share=nf / tot_flip, diag_share=float(diag_e[S].sum()) / tot_diag,
                                block_share=e_block / full if full > 0 else float("nan"),
                                mean_hheld_diag=float(hd[S].mean())))
    out["sum_block_over_full"] = blocks / full if full > 0 else float("nan")
    out["diag_over_full"] = float(diag_e.sum()) / full if full > 0 else float("nan")
    return out


@torch.no_grad()
def flip_grid_split(dW, flip, Hheld):
    """Quadratic-form split of ||dW X||^2 into code-flip elements, same-code (grid-only) elements and their cross term."""
    dev = Hheld.device
    dW = dW.to(dev, torch.float64)
    m = flip.to(dev)
    A = torch.where(m, dW, torch.zeros_like(dW))
    B = dW - A
    qa = float(((A @ Hheld) * A).sum())
    qb = float(((B @ Hheld) * B).sum())
    qab = float(((A @ Hheld) * B).sum())
    tot = qa + qb + 2 * qab
    return dict(energy_flip=qa, energy_grid=qb, energy_cross=2 * qab, energy_total=tot,
                share_flip=qa / tot if tot > 0 else float("nan"), share_grid=qb / tot if tot > 0 else float("nan"),
                share_cross=2 * qab / tot if tot > 0 else float("nan"),
                frac_elements_flip=float(m.float().mean()), frac_elements_grid_changed=float(((B != 0)).float().mean()))
