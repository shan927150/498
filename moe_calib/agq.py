"""Affinity-Guided Quantization (AGQ, MoEQuant arXiv 2505.03804 Sec. 4.3, Eq. 18/19, Algorithm 2) helpers.

AGQ changes ONE thing in GPTQ: the per-expert input Hessian becomes
    H_e^AGQ = sum_{i in I_e} c_{i,e} x_i x_i^T = X^T diag(c) X
where c_{i,e} is the router affinity of token i for routed expert e. The GPTQ solver is untouched.

For Qwen1.5-MoE-A2.7B (transformers Qwen2MoeSparseMoeBlock, norm_topk_prob=False):
    p = softmax(router_logits, dim=-1, dtype=float32); vals, sel = topk(p, 4)
    y = sum_k bf16(vals_k) * E_{sel_k}(h)  +  sigmoid(shared_gate(h)) * shared(h)
c = vals (fp32, pre-cast) - the quantity the official MoEQuant code records (save_gate_res_qwen); the model
multiplies by its BF16 rounding. The rows of expert e are dispatched slot-major:
    mask = one_hot(sel, E).permute(2, 1, 0); idx, top_x = where(mask[e]); rows = h[top_x]; c = vals[top_x, idx]
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

PROJ = ("gate_proj", "up_proj", "down_proj")


# ----------------------------------------------------------------------------- routing
@torch.no_grad()
def routing_coefficients(router_logits: torch.Tensor, k: int):
    """The model's exact op: fp32 softmax over all experts, then top-k. No renormalization (norm_topk_prob=False).
    Returns (vals [T, k] fp32, sel [T, k] long)."""
    probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    return torch.topk(probs, k, dim=-1)


@torch.no_grad()
def dispatch_rows(sel: torch.Tensor, vals: torch.Tensor, n_experts: int) -> dict:
    """{e: (top_x, slot, c)} in the model's dispatch row order (slot-major)."""
    mask = F.one_hot(sel, num_classes=n_experts).permute(2, 1, 0)
    out = {}
    for e in range(n_experts):
        idx, top_x = torch.where(mask[e])
        if top_x.numel():
            out[e] = (top_x, idx, vals[top_x, idx])
    return out


# ----------------------------------------------------------------------------- Hessian oracle
def weighted_hessian_reference(X, c) -> np.ndarray:
    """Independent float64 oracle: H = X^T diag(c) X (row-major X [t, d])."""
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    return X.T @ (c[:, None] * X)


def effective_sample_size(c) -> float:
    c = np.asarray(c, dtype=np.float64)
    s2 = float((c ** 2).sum())
    return float(c.sum() ** 2 / s2) if s2 > 0 else 0.0


def affinity_summary(c) -> dict:
    c = np.asarray(c, dtype=np.float64)
    if c.size == 0:
        return dict(n=0)
    q = np.quantile(c, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(n=int(c.size), mean=float(c.mean()), sd=float(c.std()), min=float(c.min()), max=float(c.max()),
                q05=float(q[0]), q25=float(q[1]), median=float(q[2]), q75=float(q[3]), q95=float(q[4]),
                cv=float(c.std() / c.mean()) if c.mean() > 0 else float("nan"), sum=float(c.sum()),
                n_eff=effective_sample_size(c), n_eff_frac=effective_sample_size(c) / c.size)


# ----------------------------------------------------------------------------- Hessian diagnostics
@torch.no_grad()
def hessian_diagnostics(Hs: torch.Tensor, Ha: torch.Tensor) -> dict:
    """Standard (Hs) vs AGQ (Ha) Hessian. Scale is separated from shape: the trace-normalized difference and the
    best-fit scalar alpha* = <Ha,Hs>/<Hs,Hs> residual measure SHAPE change only; GPTQ with relative damping is
    invariant to a global positive scale."""
    Hs, Ha = Hs.double(), Ha.double()
    fs = torch.linalg.norm(Hs)
    raw = float(torch.linalg.norm(Ha - Hs) / fs) if fs > 0 else float("nan")
    ts, ta = torch.trace(Hs), torch.trace(Ha)
    tn = float(torch.linalg.norm(Ha / ta - Hs / ts) / torch.linalg.norm(Hs / ts)) if ts > 0 and ta > 0 else float("nan")
    alpha = float((Ha * Hs).sum() / (Hs * Hs).sum()) if fs > 0 else float("nan")
    resid = float(torch.linalg.norm(Ha - alpha * Hs) / torch.linalg.norm(Ha)) if fs > 0 else float("nan")
    ds, da = torch.diag(Hs), torch.diag(Ha)
    dsn, dan = ds / ds.sum(), da / da.sum()
    corr = float(np.corrcoef(ds.cpu().numpy(), da.cpu().numpy())[0, 1]) if ds.std() > 0 and da.std() > 0 else float("nan")
    ps = torch.argsort(ds.float(), descending=True)            # act-order exactly as gptq_quantize computes it (fp32)
    pa = torch.argsort(da.float(), descending=True)
    rs = torch.empty_like(ps); rs[ps] = torch.arange(ps.numel(), device=ps.device)
    ra = torch.empty_like(pa); ra[pa] = torch.arange(pa.numel(), device=pa.device)
    spearman = float(np.corrcoef(rs.cpu().numpy(), ra.cpu().numpy())[0, 1])
    return dict(trace_ratio=float(ta / ts) if ts > 0 else float("nan"), rel_frob_raw=raw, rel_frob_trace_norm=tn,
                scale_alpha=alpha, shape_residual=resid, diag_corr=corr,
                diag_trace_norm_max_abs_diff=float((dan - dsn).abs().max() / dsn.abs().max()),
                actorder_positions_changed=int((ps != pa).sum()), actorder_frac_changed=float((ps != pa).float().mean()),
                actorder_rank_spearman=spearman)


@torch.no_grad()
def weight_diagnostics(Wq_s: torch.Tensor, Wq_a: torch.Tensor, W: torch.Tensor, info_s=None, info_a=None) -> dict:
    """Quantized weights of the two methods (same FP W). Codes are recovered from the solver's own scales/zeros."""
    Wq_s, Wq_a, W = Wq_s.float(), Wq_a.float(), W.float()
    d = Wq_a - Wq_s
    out = dict(frac_weights_changed=float((d != 0).float().mean()),
               rel_l2_diff=float(torch.linalg.norm(d) / torch.linalg.norm(Wq_s)),
               rel_l2_diff_vs_quant_error=float(torch.linalg.norm(d) / torch.linalg.norm(Wq_s - W)),
               quant_err_rel_std=float(torch.linalg.norm(Wq_s - W) / torch.linalg.norm(W)),
               quant_err_rel_agq=float(torch.linalg.norm(Wq_a - W) / torch.linalg.norm(W)))
    if info_s is not None and info_a is not None:
        out["actorder_changed"] = bool(not torch.equal(info_s["perm"], info_a["perm"]))
        out["frac_scales_changed"] = float((info_s["scale"] != info_a["scale"]).float().mean())
        out["frac_zeros_changed"] = float((info_s["zero"] != info_a["zero"]).float().mean())
        cs, ca = codes(Wq_s, info_s), codes(Wq_a, info_a)
        # codes compared per original column; groups are formed in permuted order, so compare codes column-wise
        out["frac_codes_changed"] = float((cs != ca).float().mean())
    return out


@torch.no_grad()
def codes(Wq: torch.Tensor, info: dict) -> torch.Tensor:
    """Integer codes of a dequantized GPTQ matrix: column j (permuted position p) belongs to group p // gs."""
    perm = info["perm"]
    G = info["scale"].shape[0]
    cols = Wq.shape[1]
    gs = math.ceil(cols / G)
    Wp = Wq[:, perm.to(Wq.device)]
    grp = torch.arange(cols, device=Wq.device) // gs
    sc = info["scale"].to(Wq.device)[grp].T                    # [rows, cols]
    zr = info["zero"].to(Wq.device)[grp].T
    q = torch.round(Wp / sc + zr)
    inv = torch.argsort(perm.to(Wq.device))
    return q[:, inv]


# ----------------------------------------------------------------------------- evaluation
def expert_fp32(X: torch.Tensor, W: dict) -> torch.Tensor:
    """Qwen2MoeMLP in fp32: down(silu(gate x) * up x)."""
    return (F.silu(X @ W["gate_proj"].T) * (X @ W["up_proj"].T)) @ W["down_proj"].T


def down_input_fp32(X: torch.Tensor, W: dict) -> torch.Tensor:
    return F.silu(X @ W["gate_proj"].T) * (X @ W["up_proj"].T)


class LayerEvalAccumulator:
    """Accumulates, for one MoE layer, on held-out tokens with the FROZEN BF16 routing (router + inputs unchanged):
      per expert e (all routed held-out tokens):
        unweighted full-expert   sum ||E(x)-Ê(x)||^2 , sum ||E(x)||^2
        affinity-weighted        sum c ||E(x)-Ê(x)||^2 , sum c ||E(x)||^2               (Eq. 18 on the expert output)
        projection-level (Eq.18) sum (c) ||(W-Ŵ)x_p||^2 , sum (c) ||W x_p||^2 for p in gate/up/down (BF16 inputs)
      layer:
        sum_x ||y_BF16(x) - y_quant(x)||^2 , sum_x ||y_BF16(x)||^2   (y includes the unquantized shared expert)
    all in fp32 math with per-sum float64 accumulation, for several quantized weight sets at once."""

    def __init__(self, methods, n_experts, d):
        self.methods, self.E = list(methods), n_experts
        z = lambda: np.zeros(n_experts)
        self.e = {m: dict(num=z(), den=z(), wnum=z(), wden=z(),
                          **{f"p_{k}_{p}": z() for p in PROJ for k in ("num", "den", "wnum", "wden")})
                  for m in self.methods}
        self.tokens, self.wsum = np.zeros(n_experts), np.zeros(n_experts)
        self.layer = {m: dict(num=0.0, den=0.0) for m in self.methods}
        self.layer_per_seq = {m: [] for m in self.methods}

    @torch.no_grad()
    def add_sequence(self, h, vals, sel, Wfp: dict, Wq: dict, shared_out):
        """h: [T, d] fp32 MoE-block input; vals/sel: routing; Wfp[e] / Wq[m][e]: fp32 weight dicts;
        shared_out: [T, d] fp32 sigmoid(shared_gate h) * shared(h)."""
        rows = dispatch_rows(sel, vals, self.E)
        y = shared_out.clone()
        err = {m: torch.zeros_like(h) for m in self.methods}
        for e, (top_x, slot, c) in rows.items():
            X = h[top_x]
            Wf = Wfp[e]
            xd = down_input_fp32(X, Wf)
            out = xd @ Wf["down_proj"].T
            y.index_add_(0, top_x, out * c[:, None])
            cn = c.double()
            self.tokens[e] += X.shape[0]
            self.wsum[e] += float(cn.sum())
            on2 = (out.double() ** 2).sum(1)
            inp = dict(gate_proj=X, up_proj=X, down_proj=xd)
            for m in self.methods:
                Wm = Wq[m][e]
                outq = expert_fp32(X, Wm)
                dlt = out - outq
                err[m].index_add_(0, top_x, dlt * c[:, None])
                dn2 = (dlt.double() ** 2).sum(1)
                A = self.e[m]
                A["num"][e] += float(dn2.sum()); A["den"][e] += float(on2.sum())
                A["wnum"][e] += float((cn * dn2).sum()); A["wden"][e] += float((cn * on2).sum())
                for p in PROJ:
                    xp = inp[p]
                    r2 = ((xp @ (Wf[p] - Wm[p]).T).double() ** 2).sum(1)
                    t2 = ((xp @ Wf[p].T).double() ** 2).sum(1)
                    A[f"p_num_{p}"][e] += float(r2.sum()); A[f"p_den_{p}"][e] += float(t2.sum())
                    A[f"p_wnum_{p}"][e] += float((cn * r2).sum()); A[f"p_wden_{p}"][e] += float((cn * t2).sum())
        yn2 = float((y.double() ** 2).sum())
        for m in self.methods:
            en2 = float((err[m].double() ** 2).sum())
            self.layer[m]["num"] += en2
            self.layer[m]["den"] += yn2
            self.layer_per_seq[m].append(en2 / (yn2 + 1e-30))
        return y

    def summary(self, eps=1e-30):
        out = {}
        for m in self.methods:
            A = self.e[m]
            seen = self.tokens > 0
            r = dict(
                moe_layer_output_rel_err=self.layer[m]["num"] / (self.layer[m]["den"] + eps),
                per_seq_moe_layer_output_rel_err=self.layer_per_seq[m],
                expert_unweighted=np.where(seen, A["num"] / (A["den"] + eps), np.nan).tolist(),
                expert_weighted=np.where(seen, A["wnum"] / (A["wden"] + eps), np.nan).tolist())
            for p in PROJ:
                r[f"proj_unweighted_{p}"] = np.where(seen, A[f"p_num_{p}"] / (A[f"p_den_{p}"] + eps), np.nan).tolist()
                r[f"proj_weighted_{p}"] = np.where(seen, A[f"p_wnum_{p}"] / (A[f"p_wden_{p}"] + eps), np.nan).tolist()
            r["proj_weighted_all"] = np.where(seen, sum(A[f"p_wnum_{p}"] for p in PROJ) /
                                              (sum(A[f"p_wden_{p}"] for p in PROJ) + eps), np.nan).tolist()
            r["proj_unweighted_all"] = np.where(seen, sum(A[f"p_num_{p}"] for p in PROJ) /
                                                (sum(A[f"p_den_{p}"] for p in PROJ) + eps), np.nan).tolist()
            out[m] = r
        out["tokens"] = self.tokens.tolist()
        out["affinity_sum"] = self.wsum.tolist()
        return out


# ----------------------------------------------------------------------------- comparisons / decision
def rel_diff(agq: float, base: float) -> float:
    """(AGQ - GPTQ) / GPTQ; negative = AGQ better for error metrics."""
    return (agq - base) / base


def classify(thr: dict, layer_rel: dict, weighted_rel: float, correctness_ok: bool, seeds_layer_rel: dict,
             primary_rel: float | None = None) -> dict:
    """Frozen primary rule (thresholds from protocol.json, fixed before quality results).
    layer_rel: {layer: rel diff of MoE-layer output error} for the PRIMARY seed;
    weighted_rel: rel diff of mean affinity-weighted expert error (primary seed);
    seeds_layer_rel: {seed: rel diff of the 4-layer mean MoE output error} for every measured seed;
    primary_rel: rel diff of the 4-layer MEAN error of the primary seed (protocol definition); if None, the mean of
    the per-layer rel diffs is used."""
    if not correctness_ok:
        return dict(verdict="AGQ-RUN-BLOCKED")
    tau, min_layers = thr["tau_rel"], thr["min_layers_improved"]
    mean_rel = float(np.mean(list(layer_rel.values()))) if primary_rel is None else float(primary_rel)
    n_impr = sum(v < 0 for v in layer_rel.values())
    seeds_all_improve = all(v < 0 for v in seeds_layer_rel.values())
    if mean_rel <= -tau and n_impr >= min_layers and seeds_all_improve:
        v = "AGQ-LAYER-QUALITY-GAIN"
    elif mean_rel >= tau:
        v = "AGQ-QUALITY-REGRESSION"
    elif weighted_rel <= -tau:
        v = "AGQ-SURROGATE-ONLY-GAIN"
    else:
        v = "AGQ-NO-MEASURABLE-GAIN"
    return dict(verdict=v, mean_layer_rel=mean_rel, layers_improved=int(n_impr), n_layers=len(layer_rel),
                seeds_all_improve=bool(seeds_all_improve), n_seeds=len(seeds_layer_rel),
                weighted_rel=float(weighted_rel))


def classify_secondary(rel: float, tau: float) -> str:
    return "gain" if rel <= -tau else ("regression" if rel >= tau else "no gain")


def classify_ppl(dppl: float, noise: float) -> str:
    return "gain" if dppl <= -noise else ("regression" if dppl >= noise else "inconclusive")


# ============================================================================= experimental c^q pilot (q in {0,1,2})
def effective_sample_size_q(c, q: int) -> float:
    """N_eff^(q) = (sum c^q)^2 / sum c^(2q); q = 0 gives N."""
    c = np.asarray(c, dtype=np.float64)
    w = c ** q
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def weighted_hessian_reference_q(X, c, q: int) -> np.ndarray:
    """float64 oracle H_q = X^T diag(c^q) X."""
    return weighted_hessian_reference(X, np.asarray(c, dtype=np.float64) ** q)


@torch.no_grad()
def slot_contributions(T: int, d: int, k: int, rows: dict, deltas: dict, device) -> torch.Tensor:
    """V[t, slot] = c_{t,e} * DeltaE_e(x_t) for the expert e selected in that slot (zeros if none); deltas[e] rows are in
    dispatch order. Sum over slots = Delta y."""
    V = torch.zeros(T, k, d, device=device, dtype=torch.float32)
    for e, (tx, slot, c) in rows.items():
        V[tx, slot] = deltas[e] * c[:, None]
    return V


@torch.no_grad()
def decompose(V: torch.Tensor) -> dict:
    """||sum_k V_k||^2 = D + C with D = sum_k ||V_k||^2 and C = 2 sum_{a<b} <V_a, V_b> (computed pairwise, independently)."""
    Vd = V.double()
    total = float((Vd.sum(1) ** 2).sum())
    D = float((Vd ** 2).sum())
    k = V.shape[1]
    C = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            C += 2.0 * float((Vd[:, a] * Vd[:, b]).sum())
    return dict(total=total, diag=D, cross=C)


def classify_c2(rules: dict, pooled_c2_rel_q2_vs_q1: float, layer_rel_q2_vs_q1: dict, correctness_ok: bool,
                cost_ok: bool, unstable: bool = False) -> dict:
    """Frozen c^2 pilot rule (protocol.json, frozen before any q = 2 quality result).
    pooled_c2_rel_q2_vs_q1: (E2_pooled(q2) - E2_pooled(q1)) / E2_pooled(q1) over the 12 experts;
    layer_rel_q2_vs_q1: {layer: rel diff of subset-replacement layer output error, condition C vs B}."""
    if not correctness_ok:
        return dict(verdict="C2-BLOCKED")
    sur = pooled_c2_rel_q2_vs_q1 <= -rules["surrogate_gain_rel"]
    any_impr = any(v < 0 for v in layer_rel_q2_vs_q1.values())
    worst = max(layer_rel_q2_vs_q1.values())
    no_reg = worst <= rules["max_layer_regression_rel"]
    if sur and any_impr and no_reg and cost_ok and not unstable:
        v = "C2-PILOT-PROMISING"
    elif unstable or worst > rules["max_layer_regression_rel"]:
        v = "C2-REGRESSION"
    elif sur:
        v = "C2-SURROGATE-ONLY"
    else:
        v = "C2-NO-GAIN"
    return dict(verdict=v, surrogate_criterion=bool(sur), any_layer_improved=bool(any_impr),
                no_layer_regression=bool(no_reg), cost_ok=bool(cost_ok), worst_layer_rel=float(worst),
                pooled_c2_rel=float(pooled_c2_rel_q2_vs_q1))
