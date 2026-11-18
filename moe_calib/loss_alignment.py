"""Loss-alignment diagnostics (MVP A + B): expert-importance-weighted reconstruction aggregates and
fixed-input, fixed-routing expert-output / router-weighted MoE-output errors.

Pure functions (no model loading) so every aggregation can be unit-tested on synthetic data.

Conventions
-----------
E[L, e]       stored per-expert E3 reconstruction error (`per_expert[...]["mean"]`: mean over held-out sequences of
              the per-sequence relative squared gate/up/down projection error). Kept as is, never re-defined.
weights       [n_layers, n_experts] non-negative, each row normalized to sum 1 (`normalize_rows`).
R(weights)    mean over layers of sum_e weights[L, e] * E[L, e]  (equal weight per layer).
R0            uniform weights 1/E  ->  mean_L mean_e E[L, e]  (= the project's macro metric).
R1            weights = routed-token frequency of a FIXED reference set (same for every strategy).
R2            weights ∝ sum_t 1[e selected] * w_{t,e}^2 with w the router weights actually applied (heuristic).

MoE-output metrics for one layer (`layer_output_errors`), with f_e(x) = down(SiLU(gate x) * up x):
  delta_e(x) = f_e^Q(x) - f_e^ref(x)
  B0_e  = sum_t ||delta_e||^2 / sum_t ||f_e^ref||^2                                  (per expert, token-pooled)
  U_t   = sum_{e in top-k(t)} w_{t,e}^2 ||delta_e(x_t)||^2                          (uncancelled energy)
  D_t   = || sum_{e in top-k(t)} w_{t,e} delta_e(x_t) ||^2                           (actual aggregate energy)
  M_t   = || sum_{e in top-k(t)} w_{t,e} f_e^ref(x_t) ||^2                           (routed reference energy)
  B1 = sum U / sum M,  B2 = sum D / sum M,  cancellation ratio B3 = sum D / sum U   (can be > 1 or < 1)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PROJS = ("gate_proj", "up_proj", "down_proj")


# ===================================================================== MVP A: aggregations
def normalize_rows(a, name="weights"):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name}: expected [layers, experts], got {a.shape}")
    if (a < 0).any() or not np.isfinite(a).all():
        raise ValueError(f"{name}: negative or non-finite entries")
    s = a.sum(1, keepdims=True)
    if (s <= 0).any():
        raise ValueError(f"{name}: a layer has zero total weight")
    w = a / s
    assert np.allclose(w.sum(1), 1.0, rtol=0, atol=1e-12)
    return w


def uniform_weights(n_layers, n_experts):
    return np.full((n_layers, n_experts), 1.0 / n_experts)


def weighted_reconstruction(E, weights):
    """mean_L sum_e weights[L, e] * E[L, e]; weights rows must sum to 1."""
    E = np.asarray(E, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if E.shape != w.shape:
        raise ValueError(f"shape mismatch {E.shape} vs {w.shape}")
    if not np.allclose(w.sum(1), 1.0, rtol=0, atol=1e-9):
        raise ValueError("weights not normalized per layer")
    if not np.isfinite(E).all():
        raise ValueError("non-finite expert errors")
    return float(np.mean((w * E).sum(1)))


def per_layer_weighted(E, weights):
    return (np.asarray(weights, dtype=np.float64) * np.asarray(E, dtype=np.float64)).sum(1)


def expert_matrix(per_expert, layers, n_experts, key="mean"):
    """List of {'layer','expert',key} -> [len(layers), n_experts]; every (L, e) must be present exactly once."""
    li = {L: i for i, L in enumerate(layers)}
    M = np.full((len(layers), n_experts), np.nan)
    for r in per_expert:
        i, e = li[int(r["layer"])], int(r["expert"])
        if not np.isnan(M[i, e]):
            raise ValueError(f"duplicate expert {(r['layer'], e)}")
        M[i, e] = float(r[key])
    if np.isnan(M).any():
        miss = [(layers[i], e) for i, e in zip(*np.where(np.isnan(M)))]
        raise ValueError(f"missing experts {miss[:5]} ...")
    return M


def routing_stats(idx, w, n_experts):
    """idx [..., k] selected expert ids, w [..., k] the router weights applied to them (same shape).
    Returns (counts [E], sum_w [E], sum_w2 [E]) as float64 numpy; counts = routed token-slots per expert."""
    idx = torch.as_tensor(idx).reshape(-1).long()
    w = torch.as_tensor(w).reshape(-1).double()
    if idx.numel() != w.numel():
        raise ValueError("idx / w shape mismatch")
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n_experts):
        raise ValueError("expert index out of range")
    cnt = torch.bincount(idx, minlength=n_experts).double()
    s1 = torch.zeros(n_experts, dtype=torch.float64).index_add_(0, idx, w)
    s2 = torch.zeros(n_experts, dtype=torch.float64).index_add_(0, idx, w * w)
    return cnt.numpy(), s1.numpy(), s2.numpy()


def router_topk(logits, k, norm_topk_prob=False, out_dtype=torch.bfloat16):
    """Exactly the HF Qwen2MoeSparseMoeBlock routing: FP32 softmax over all experts, top-k, optional renorm,
    then the cast to the hidden dtype. Returns (idx [T,k], w_fp32 [T,k], w_applied [T,k] in out_dtype)."""
    p = F.softmax(logits, dim=1, dtype=torch.float)
    w, idx = torch.topk(p, k, dim=-1)
    if norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    return idx, w, w.to(out_dtype)


def paired_delta(a, b):
    """Absolute and relative (a - b) / b."""
    return dict(abs=float(a - b), rel=float((a - b) / b))


# ===================================================================== MVP B: expert / MoE output errors
def mlp_fp32(X, W, act=F.silu):
    """FP32 expert MLP with [out, in] weights: down(act(gate x) * up x). Returns (y, h) with h the down input."""
    h = act(X @ W["gate_proj"].T) * (X @ W["up_proj"].T)
    return h @ W["down_proj"].T, h


def projection_errors(X, h_ref, W_ref, W_q):
    """Linear projection reconstruction on fixed inputs (E3's objective): numerator / denominator summed over the
    three projections, PER ROW (token). gate/up see X, down sees the reference down input h_ref."""
    num = torch.zeros(X.shape[0], dtype=torch.float64, device=X.device)
    den = torch.zeros_like(num)
    for nm, inp in (("gate_proj", X), ("up_proj", X), ("down_proj", h_ref)):
        num += (inp @ (W_ref[nm] - W_q[nm]).T).pow(2).sum(1).double()
        den += (inp @ W_ref[nm].T).pow(2).sum(1).double()
    return num, den


@torch.no_grad()
def layer_output_errors(X, idx, w, W_ref, W_q, seq_id=None, n_seq=None, act=F.silu, return_M=False, mlp=None):
    """Fixed-input, fixed-routing errors for one MoE layer.

    X       [T, d] layer MoE-block input (FP32 on the compute device).
    idx     [T, k] selected experts (the BF16 model's), w [T, k] the applied router weights (FP32 values).
    W_ref   list over experts of {proj: FP32 [out, in]} reference (BF16-original) weights; W_q same for the
            quantized weights. Experts never selected are skipped (and reported).
    seq_id  optional [T] long window index in [0, n_seq) for per-window numerators / denominators.
    mlp     expert forward (X, W, act) -> (y, h) returning FP32; default `mlp_fp32` (a BF16-native variant may be passed).
    Returns a dict of FP64 numpy arrays / floats (raw numerators and denominators, nothing pre-normalized)."""
    mlp = mlp or mlp_fp32
    T, k = idx.shape
    E = len(W_ref)
    dev = X.device
    d_out = W_ref[0]["down_proj"].shape[0]
    D = torch.zeros(T, d_out, dtype=torch.float32, device=dev)        # sum_e w delta_e
    M = torch.zeros(T, d_out, dtype=torch.float32, device=dev)        # sum_e w f_e^ref
    U = torch.zeros(T, dtype=torch.float64, device=dev)               # sum_e w^2 ||delta_e||^2
    V = torch.zeros(T, dtype=torch.float64, device=dev)               # sum_e w^2 ||f_e^ref||^2
    ex = dict(tokens=np.zeros(E), out_num=np.zeros(E), out_den=np.zeros(E), proj_num=np.zeros(E), proj_den=np.zeros(E),
              wout_num=np.zeros(E), wout_den=np.zeros(E))
    S = n_seq or 0
    ex_seq = dict(proj_num=np.zeros((E, S)), proj_den=np.zeros((E, S)), tokens=np.zeros((E, S)))
    for e in range(E):
        rows, slot = torch.where(idx == e)
        if rows.numel() == 0:
            continue
        we = w[rows, slot].to(torch.float32)
        Xe = X[rows]
        f_ref, h_ref = mlp(Xe, W_ref[e], act)
        f_q, _ = mlp(Xe, W_q[e], act)
        dl = f_q - f_ref
        dn = dl.pow(2).sum(1).double()
        fn = f_ref.pow(2).sum(1).double()
        D.index_add_(0, rows, we[:, None] * dl)
        M.index_add_(0, rows, we[:, None] * f_ref)
        w2 = we.double() ** 2
        U.index_add_(0, rows, w2 * dn)
        V.index_add_(0, rows, w2 * fn)
        pn, pd = projection_errors(Xe.float(), h_ref.float(), {k: v.float() for k, v in W_ref[e].items()},
                                   {k: v.float() for k, v in W_q[e].items()})
        ex["tokens"][e] = rows.numel()
        ex["out_num"][e] = float(dn.sum())
        ex["out_den"][e] = float(fn.sum())
        ex["wout_num"][e] = float((w2 * dn).sum())
        ex["wout_den"][e] = float((w2 * fn).sum())
        ex["proj_num"][e] = float(pn.sum())
        ex["proj_den"][e] = float(pd.sum())
        if seq_id is not None:
            sid = seq_id[rows]
            for key, val in (("proj_num", pn), ("proj_den", pd), ("tokens", torch.ones_like(pn))):
                ex_seq[key][e] += torch.zeros(S, dtype=torch.float64, device=dev).index_add_(0, sid, val).cpu().numpy()
    Dn = D.double().pow(2).sum(1)
    Mn = M.double().pow(2).sum(1)
    out = dict(experts=ex, D_sum=float(Dn.sum()), M_sum=float(Mn.sum()), U_sum=float(U.sum()), V_sum=float(V.sum()),
               n_tokens=T, n_unrouted_experts=int((ex["tokens"] == 0).sum()), finite=bool(
                   torch.isfinite(D).all() and torch.isfinite(M).all() and torch.isfinite(U).all()))
    if seq_id is not None:
        z = lambda v: torch.zeros(S, dtype=torch.float64, device=dev).index_add_(0, seq_id, v).cpu().numpy()
        out.update(seq_D=z(Dn), seq_M=z(Mn), seq_U=z(U), experts_seq=ex_seq)
    if return_M:
        out["M"] = M
    return out


def summarize_layer(r):
    """Normalized metrics from raw layer sums (all ratios of FP64 sums)."""
    ex = r["experts"]
    m = ex["tokens"] > 0
    return dict(B0_mean=float(np.mean(ex["out_num"][m] / ex["out_den"][m])),
                B0_pooled=float(ex["out_num"].sum() / ex["out_den"].sum()),
                P_pooled_mean=float(np.mean(ex["proj_num"][m] / ex["proj_den"][m])),
                B1=r["U_sum"] / r["M_sum"], B2=r["D_sum"] / r["M_sum"], B3=r["D_sum"] / r["U_sum"],
                B1_selfden=r["U_sum"] / r["V_sum"])


def per_seq_projection_mean(ex_seq):
    """E3-style per-expert error on fixed inputs: mean over windows (with >= 1 routed token) of the per-window
    relative projection error. Returns [E] (NaN for experts with no tokens)."""
    num, den, tok = ex_seq["proj_num"], ex_seq["proj_den"], ex_seq["tokens"]
    out = np.full(num.shape[0], np.nan)
    for e in range(num.shape[0]):
        s = tok[e] > 0
        if s.any():
            out[e] = float(np.mean(num[e, s] / (den[e, s] + 1e-12)))
    return out


def window_bootstrap_ratio_diff(numA, denA, numB, denB, n_boot=10000, seed=0):
    """Paired window bootstrap of (sum numA / sum denA) - (sum numB / sum denB); arrays are [n_windows]
    (layers already summed). Returns (point, lo, hi) of the relative difference (A - B) / B."""
    numA, denA, numB, denB = (np.asarray(x, dtype=np.float64) for x in (numA, denA, numB, denB))
    n = numA.shape[0]
    rng = np.random.default_rng(seed)
    pt = (numA.sum() / denA.sum()) / (numB.sum() / denB.sum()) - 1.0
    ix = rng.integers(0, n, size=(n_boot, n))
    a = numA[ix].sum(1) / denA[ix].sum(1)
    b = numB[ix].sum(1) / denB[ix].sum(1)
    r = a / b - 1.0
    return float(pt), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
