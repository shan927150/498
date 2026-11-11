"""GPTQ act-order ablation after frozen-routing calibration (helpers).

Conditions per strategy c and grouping g (four layers, FR s0 / FC s0):
  OO    GPTQ(H_O)                       natural act-order P_O              (frozen reference)
  AO    GPTQ(H_AO,g)                    act-order recomputed from H_AO,g   (frozen-routing robustness study)
  AOF   GPTQ(H_AO,g, actorder=P_O)      act-order FIXED to the captured ORIGINAL permutation (new)
Fixing the permutation also fixes the column -> quantization-group membership (the solver groups permuted columns
in blocks of group_size), so AOF tests act-order and its induced group membership jointly. Scales, zero-points,
damping, Cholesky factor and error feedback are recomputed from H_AO,g.

Also: the precision-aware installation-integrity checks (defined after the installation audit) and the BF16
precision decomposition used by the audit. No module-level GPU state.
"""
from __future__ import annotations
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

PROJ = ("gate_proj", "up_proj", "down_proj")
PROJ_H = {"gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}
BF16_U = 2.0 ** -8           # unit roundoff of bfloat16 (8 significand bits): |bf16(x) - x| <= 2^-8 |x| (normal range)


# ---------------------------------------------------------------- permutations / groups
def group_of(perm, group_size=128):
    """Original column -> quantization group index under permutation `perm` (solver: permuted column j belongs to
    group j // group_size)."""
    perm = torch.as_tensor(perm).long()
    g = torch.empty_like(perm)
    g[perm] = torch.arange(perm.numel()) // group_size
    return g


def perm_compare(p, p_ref, group_size=128):
    p, p_ref = torch.as_tensor(p).long(), torch.as_tensor(p_ref).long()
    return dict(identical=bool(torch.equal(p, p_ref)), frac_positions_changed=float((p != p_ref).float().mean()),
                frac_columns_changed_group=float((group_of(p, group_size) != group_of(p_ref, group_size)).float().mean()))


def perm_sha(p):
    return hashlib.sha256(torch.as_tensor(p).to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- integrity (precision-aware, exact)
def cast_rounding_ok(W_solver, W_bf16):
    """I3: W_bf16 is a correctly rounded bfloat16 of W_solver elementwise: |W_bf16 - W| <= 2^-8 |W| (IEEE
    round-to-nearest, normal range); exact zeros stay zero. Returns (ok, max excess ratio)."""
    W = W_solver.float()
    d = (W_bf16.float() - W).abs()
    bound = BF16_U * W.abs()
    nz = W != 0
    ok = bool((d[~nz] == 0).all()) and bool((d[nz] <= bound[nz]).all())
    r = float((d[nz] / bound[nz]).max()) if bool(nz.any()) else 0.0
    return ok, r


def bf16_reference_mlp(X_bf16, W_bf16, act):
    """Independent BF16 reference of Qwen2MoeMLP.forward (down(act(gate x) * up x)) built from the given BF16
    weights with plain F.linear calls in the module's operation order and dtype."""
    g = F.linear(X_bf16, W_bf16["gate_proj"])
    u = F.linear(X_bf16, W_bf16["up_proj"])
    return F.linear(act(g) * u, W_bf16["down_proj"])


def mlp32(X, W, act):
    return F.linear(act(F.linear(X, W["gate_proj"])) * F.linear(X, W["up_proj"]), W["down_proj"])


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


@torch.no_grad()
def precision_decomposition(mod, Wq32, W_orig, X, dev):
    """Controlled decomposition on identical inputs X (fp32 holding exactly-BF16 values) for one expert module
    `mod` whose installed weights are being validated. Wq32: solver FP32 weights; W_orig: pristine BF16 weights.
    Returns per-projection and expert-level references (see installation audit)."""
    act = mod.act_fn
    Xf = X.to(dev, torch.float32)
    Xb = X.to(dev, torch.bfloat16)
    Wq = {k: Wq32[k].to(dev, torch.float32) for k in PROJ}
    Wb = {k: Wq32[k].to(torch.bfloat16).to(dev) for k in PROJ}                 # bf16 cast of the solver weights
    Wi = {k: getattr(mod, k).weight.detach().to(dev) for k in PROJ}            # what is actually installed
    Wo = {k: W_orig[k].to(dev) for k in PROJ}
    out = dict(n_tokens=int(X.shape[0]), inputs_exactly_bf16=bool(torch.equal(Xb.float(), Xf)),
               dtype={k: str(Wi[k].dtype) for k in PROJ}, shape={k: list(Wi[k].shape) for k in PROJ},
               stride={k: list(Wi[k].stride()) for k in PROJ}, device=str(dev),
               installed_bitwise_equal_cast={k: bool(torch.equal(Wi[k], Wb[k])) for k in PROJ},
               cast_rounding={k: dict(zip(("ok", "max_err_over_bound"), cast_rounding_ok(Wq[k], Wi[k]))) for k in PROJ})
    # per projection, identical inputs: gate/up see X; down sees the ACTUAL bf16 intermediate of the installed module
    gb, ub = F.linear(Xb, Wi["gate_proj"]), F.linear(Xb, Wi["up_proj"])
    zb = act(gb) * ub
    proj = {}
    for k, xin_b in (("gate_proj", Xb), ("up_proj", Xb), ("down_proj", zb)):
        xin = xin_b.float()
        y_solver = F.linear(xin, Wq[k])
        y_cast = F.linear(xin, Wi[k].float())
        y_bf16 = F.linear(xin_b, Wi[k]).float()
        proj[k] = dict(weight_repr_rel=rel(y_cast, y_solver), arith_rel=rel(y_bf16, y_cast), total_rel=rel(y_bf16, y_solver),
                       weight_repr_maxabs=float((y_cast - y_solver).abs().max()),
                       arith_maxabs=float((y_bf16 - y_cast).abs().max()), total_maxabs=float((y_bf16 - y_solver).abs().max()),
                       ref_norm=float(y_solver.norm()))
    out["per_projection"] = proj
    # expert level
    y_solver = mlp32(Xf, Wq, act)                                   # = old check's y_e3
    y_cast = mlp32(Xf, {k: Wi[k].float() for k in PROJ}, act)      # = old check's y_inst32
    m_dev = next(mod.parameters()).device
    mod.to(dev)
    try:
        y_mod = mod(Xb)
    finally:
        mod.to(m_dev)
    y_ref = bf16_reference_mlp(Xb, Wb, act)                         # independent: built from bf16(W_solver)
    y_fp = mlp32(Xf, {k: Wo[k].float() for k in PROJ}, act)
    y_orig_bf16 = bf16_reference_mlp(Xb, Wo, act)
    yb = y_mod.float()
    tok = ((yb - y_solver).norm(dim=1) / y_solver.norm(dim=1).clamp(min=1e-30))
    out["expert"] = dict(
        module_bitwise_equal_independent_reference=bool(torch.equal(y_mod, y_ref)),
        module_max_abs_diff_reference=float((y_mod.float() - y_ref.float()).abs().max()),
        weight_repr_rel=rel(y_cast, y_solver), arith_rel=rel(yb, y_cast), total_rel=rel(yb, y_solver),
        orig_weights_arith_rel=rel(y_orig_bf16.float(), y_fp), quant_effect_rel=rel(y_solver, y_fp),
        total_maxabs=float((yb - y_solver).abs().max()), per_token_rel_p50=float(tok.median()),
        per_token_rel_p99=float(torch.quantile(tok, .99)), per_token_rel_max=float(tok.max()),
        top5_token_share_of_sq_error=float(((yb - y_solver).pow(2).sum(1).sort(descending=True).values[:5].sum()
                                            / (yb - y_solver).pow(2).sum()).clamp(max=1)),
        finite=bool(torch.isfinite(yb).all()))
    e = out["expert"]
    # the frozen (old) check, recomputed from these tensors: tol = weight rounding + 1.5 x original-weight BF16 noise
    e["old_check"] = dict(installed_vs_e3_rel=e["total_rel"], weight_rounding_only_rel=e["weight_repr_rel"],
                          bf16_compute_noise_rel=e["orig_weights_arith_rel"],
                          tolerance=e["weight_repr_rel"] + 1.5 * e["orig_weights_arith_rel"])
    e["old_check"]["ratio"] = e["total_rel"] / e["old_check"]["tolerance"]
    e["arith_ratio_quantized_vs_original"] = e["arith_rel"] / max(e["orig_weights_arith_rel"], 1e-30)
    return out


def integrity_pass(dec):
    """New precision-aware integrity verdict for one checked expert (I1 restricted to it, I2, I3, I4)."""
    e = dec["expert"]
    return bool(dec["inputs_exactly_bf16"] and all(dec["installed_bitwise_equal_cast"].values())
                and all(v["ok"] for v in dec["cast_rounding"].values())
                and e["module_bitwise_equal_independent_reference"] and e["finite"])


INTEGRITY_PROTOCOL = {
    "name": "precision-aware installation integrity (actorder ablation); supersedes nothing: the frozen tolerance "
            "check of earlier studies is unchanged and still computed and reported for every configuration",
    "I1": "every one of the 720 installed BF16 projection tensors is bitwise equal to bfloat16(W_solver) of the FP32 "
          "solver output written back by E3 (checked at write-back for all 240 experts)",
    "I2": "for the 8 prespecified check experts (layers 1/8/16/23 x experts 0/30) the installed module's BF16 output on "
          "the frozen validation inputs is bitwise equal to an independent BF16 reference built from bfloat16(W_solver) "
          "with F.linear in the module's operation order (gate, up, act, mul, down)",
    "I3": "elementwise |W_bf16 - W_solver| <= 2^-8 |W_solver| (correct round-to-nearest; zeros stay zero) for all 720 tensors",
    "I4": "installed weights and checked outputs are finite",
    "not_a_criterion": "the relative FP32-vs-BF16 output deviation; it is reported (decomposed into weight "
                       "representation and arithmetic) but never gates, because it measures the BF16 execution "
                       "environment rather than installation correctness",
}


# ---------------------------------------------------------------- metrics
def sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def rng(x):
    x = np.asarray(x, float)
    return float(x.max() - x.min())


def ratio(a, b):
    return None if not np.isfinite(b) or abs(b) <= 1e-12 else float(a / b)


def weight_reduction(d_aof, d_ao):
    """1 - D_AOF,g / D_AO,g per grouping (aggregate relative L2 over all 720 tensors) and its mean."""
    r = [1 - a / b for a, b in zip(d_aof, d_ao)]
    return dict(per_grouping=r, mean=float(np.mean(r)), min=float(min(r)), max=float(max(r)))


def ppl_spread(ao, aof, oo):
    d_ao = [abs(x - oo) for x in ao]
    d_aof = [abs(x - oo) for x in aof]
    return dict(SD_AO=sd(ao), SD_AOF=sd(aof), SD_ratio=ratio(sd(aof), sd(ao)), Range_AO=rng(ao), Range_AOF=rng(aof),
                Range_ratio=ratio(rng(aof), rng(ao)), mean_AO=float(np.mean(ao)), mean_AOF=float(np.mean(aof)),
                mean_change_AOF_minus_AO=float(np.mean(aof) - np.mean(ao)),
                mean_dist_OO_AO=float(np.mean(d_ao)), mean_dist_OO_AOF=float(np.mean(d_aof)),
                n_AOF_closer_to_OO=int(sum(a < b for a, b in zip(d_aof, d_ao))))


def delta_stats(fr, fc, delta_oo):
    d = [b - a for a, b in zip(fr, fc)]
    return dict(Delta=d, mean=float(np.mean(d)), SD=sd(d), range=rng(d), n_positive=int(sum(x > 0 for x in d)),
                n_negative=int(sum(x < 0 for x in d)),
                n_preserved_vs_OO=int(sum(np.sign(x) == np.sign(delta_oo) and x != 0 for x in d)),
                n_reversals_vs_OO=int(sum(np.sign(x) != np.sign(delta_oo) for x in d)))


# ---------------------------------------------------------------- frozen decision rule
WEIGHT_MIN_REDUCTION = 0.25
SD_MAX_RATIO = 0.70             # "reduces SD by at least 30 %"
DECISION_RULE = {
    "metrics": {
        "WeightReduction[c]": "mean over the five groupings of 1 - D_AOF,g / D_AO,g, D = aggregate relative L2 of all "
                              "720 effective BF16 tensors vs OO (sqrt(sum ||W - W_OO||^2 / sum ||W_OO||^2))",
        "SD_ratio[c]": "std_g(PPL_AOF,g, ddof=1) / std_g(PPL_AO,g, ddof=1)",
        "SD_Delta_ratio": "std_g(Delta_AOF,g) / std_g(Delta_AO,g), Delta = PPL_FC - PPL_FR (same grouping)",
        "undefined_ratio": "denominator <= 1e-12 -> undefined; an undefined ratio never satisfies a criterion",
    },
    "criteria": {
        "W": f"WeightReduction[FR] >= {WEIGHT_MIN_REDUCTION} AND WeightReduction[FC] >= {WEIGHT_MIN_REDUCTION}",
        "P": f"SD_ratio[FR] <= {SD_MAX_RATIO} AND SD_ratio[FC] <= {SD_MAX_RATIO}",
        "D": f"SD_Delta_ratio <= {SD_MAX_RATIO}",
        "W_none": f"WeightReduction[FR] < {WEIGHT_MIN_REDUCTION} AND WeightReduction[FC] < {WEIGHT_MIN_REDUCTION}",
        "P_none": f"SD_ratio[FR] > {SD_MAX_RATIO} AND SD_ratio[FC] > {SD_MAX_RATIO} AND SD_Delta_ratio > {SD_MAX_RATIO}",
    },
    "precedence": [
        "RUN-BLOCKED: installation audit not AUDIT-CONFIRMED-BF16-REPRESENTATION, a solver regression fails, an AOF "
        "configuration is missing or fails a correctness / integrity / invariance check (Hessian == AO, permutation == "
        "ORIGINAL, routed-token counts == R_O)",
        "ACTORDER-CONTRIBUTES-TO-WEIGHT-AND-PPL-SENSITIVITY: W AND P AND D",
        "ACTORDER-CONTRIBUTES-TO-WEIGHT-SENSITIVITY-ONLY: W AND NOT (P AND D)",
        "ACTORDER-NOT-PRIMARY: W_none AND P_none",
        "ACTORDER-MIXED-EFFECT: otherwise (the satisfied / failed items are reported)",
    ],
    "note": "exploratory practical criteria, frozen (with this module's sha256) before any AOF quantization or PPL",
}


def criteria(wred, sp, dl):
    """wred[c] = weight_reduction(...), sp[c] = ppl_spread(...), dl = {'SD_Delta_ratio': ...}."""
    le = lambda r: r is not None and r <= SD_MAX_RATIO
    gt = lambda r: r is not None and r > SD_MAX_RATIO
    return dict(
        W=all(wred[c]["mean"] >= WEIGHT_MIN_REDUCTION for c in ("FR", "FC")),
        P=le(sp["FR"]["SD_ratio"]) and le(sp["FC"]["SD_ratio"]),
        D=le(dl["SD_Delta_ratio"]),
        W_none=all(wred[c]["mean"] < WEIGHT_MIN_REDUCTION for c in ("FR", "FC")),
        P_none=gt(sp["FR"]["SD_ratio"]) and gt(sp["FC"]["SD_ratio"]) and gt(dl["SD_Delta_ratio"]))


def decide(cr, blocked=False):
    if blocked:
        return "RUN-BLOCKED"
    if cr["W"] and cr["P"] and cr["D"]:
        return "ACTORDER-CONTRIBUTES-TO-WEIGHT-AND-PPL-SENSITIVITY"
    if cr["W"]:
        return "ACTORDER-CONTRIBUTES-TO-WEIGHT-SENSITIVITY-ONLY"
    if cr["W_none"] and cr["P_none"]:
        return "ACTORDER-NOT-PRIMARY"
    return "ACTORDER-MIXED-EFFECT"


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()
