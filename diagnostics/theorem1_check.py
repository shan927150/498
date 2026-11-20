"""Theorem-1 check of the GPTQ damping mechanism (diagnostic only; tuning split only).

Tests whether the idealized closed form of GPTQ's weight error (Chen & Yeh 2026, Theorem 1) predicts the
measured damping behaviour of runs_diag/damping_threshold_report.md and runs_diag/adaptive_damping.md.

In act-order-permuted, dead-column-fixed coordinates, with A = H + lambda I, lambda = nu * mean(diag(H))
(exactly moe_calib/gptq.py), GPTQ's per-row error is u_r = W_r - Q_r = M Delta_r with M unit lower
triangular, M = (diag(U)^-1 U)^T, U = upper Cholesky of A^-1 (= gptq.py's Hinv). Then
  R_ideal(nu) = tr(M^T S M) / tr(S)                         (predicted GPTQ/RTN ratio under R1+R2)
  R_w(nu)     = sum_t col_err2_t * (m_t^T S m_t)            (keeps R1 = martingale residuals, drops R2)
  e_meas(nu)  = sum_r u_r^T S u_r                           (actual quantized weights)
S = tuning-split second moment X^T X / n_tune (fp64, unweighted) of the same projection group.

  python diagnostics/theorem1_check.py protocol   # freeze runs_theorem1_check/protocol.json (refuses to overwrite)
  python diagnostics/theorem1_check.py grid       # Part B (B1 + B2), resumable per (seed, layer, n, nu)
  python diagnostics/theorem1_check.py sweep      # Part C controlled n/d sweep, resumable per (layer, expert, group)
  python diagnostics/report_theorem1_check.py     # report / figures

The solver moe_calib/gptq.py is frozen and only called, never modified.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_calib.gptq import gptq_quantize                            # noqa: E402

OUT = "runs_theorem1_check"
PROTOCOL = f"{OUT}/protocol.json"
RAW_GRID = f"{OUT}/raw_grid.jsonl"
RAW_SWEEP = f"{OUT}/raw_sweep.jsonl"
NU_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
NU_FINE = [float(x) for x in np.logspace(-3, 2, 28)]
NU_CURVE = [0.01, 0.1, 1.0]
NU_ANCHOR = [0.01, 1.0]
B2_RANKS = [0, 1, 2, 3, 28, 29, 30, 31, 56, 57, 58, 59]   # deviation D1 (see deviations.json)
B1_FALLBACK_EXTRA_RANKS = [9, 10, 19, 20, 39, 40, 49, 50, 14, 15, 44, 45]
SWEEP_RHO = [0.15, 0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 2.0, 2.5, 3.5, 5.0]
ANCHOR_RHO = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
SWEEP_DRAWS = 5
SWEEP_SALT = 9_100_001
GROUPS = {"gu": ("gate_proj", "up_proj"), "down": ("down_proj",)}
HK = {"gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}


# =================================================================== core math (fp64, device-agnostic)

def solver_coords(H: torch.Tensor, perm: torch.Tensor | None = None, actorder: bool = True):
    """Replicates gptq.py up to (not including) damping: dead-column fix on the given H, then act-order
    permutation argsort(diag(H), desc) computed on H's own dtype/device exactly as the solver does (or the
    solver's recorded `perm`). Returns (Hp fp64 permuted, perm, dead mask in ORIGINAL coordinates)."""
    Hs = H.clone()
    dead = torch.diag(Hs) == 0
    Hs[dead, dead] = 1.0
    if perm is None:
        perm = (torch.argsort(torch.diag(Hs), descending=True) if actorder
                else torch.arange(H.shape[0], device=H.device))
    perm = perm.to(H.device)
    Hp = Hs.double()[perm][:, perm]
    return Hp, perm, dead


def damped(Hp: torch.Tensor, nu: float) -> torch.Tensor:
    d = Hp.shape[0]
    return Hp + torch.eye(d, dtype=Hp.dtype, device=Hp.device) * (nu * torch.mean(torch.diag(Hp)))


def m_matrix(Hp: torch.Tensor, nu: float):
    """M = (diag(U)^-1 U)^T, U = upper Cholesky of A^-1 formed exactly like gptq.py
    (cholesky -> cholesky_inverse -> cholesky upper). fp64. Returns None if a Cholesky fails
    (no jitter is ever added)."""
    return m_from_A(damped(Hp, nu))


def m_from_A(A: torch.Tensor):
    """M from an already damped matrix A (same Cholesky chain as gptq.py); None on Cholesky failure."""
    L, info = torch.linalg.cholesky_ex(A)
    if int(info) != 0:
        return None
    U, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
    if int(info) != 0:
        return None
    return (U / torch.diag(U).unsqueeze(1)).T.contiguous()


def m_ridge(Hp: torch.Tensor, nu: float) -> torch.Tensor:
    """Nested-ridge form (test reference): M[t,t] = 1, M[>t,t] = -(A[>t,>t])^-1 Hp[>t,t]."""
    A = damped(Hp, nu)
    d = A.shape[0]
    M = torch.eye(d, dtype=A.dtype, device=A.device)
    for t in range(d - 1):
        M[t + 1:, t] = -torch.linalg.solve(A[t + 1:, t + 1:], Hp[t + 1:, t])
    return M


def col_quad(M: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """q_t = m_t^T S m_t for every column t of M (fp64): the per-column error weights."""
    return (M * (S @ M)).sum(0)


def r_ideal(M: torch.Tensor, S: torch.Tensor) -> float:
    return float(col_quad(M, S).sum() / torch.trace(S))


def quad_err(U: torch.Tensor, S: torch.Tensor) -> float:
    """sum_r u_r^T S u_r for a [rows, d] error matrix (fp64)."""
    U = U.double()
    return float(((U @ S) * U).sum())


def george_saab(n: float, d: int) -> float:
    return float(d / (n ** (1 / 3) * min(n, d) ** (2 / 3))) if n > 0 else float("nan")


def parabola_min(xs, ys, i):
    """Vertex of the parabola through (xs[i-1..i+1], ys[i-1..i+1]); None if not interior/convex."""
    if i <= 0 or i >= len(xs) - 1:
        return None
    x, y = np.asarray(xs[i - 1:i + 2], float), np.asarray(ys[i - 1:i + 2], float)
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return None
    v = -c[1] / (2 * c[0])
    return float(min(max(v, x[0]), x[2]))


def residuals_from_u(M: torch.Tensor, u_perm: torch.Tensor) -> torch.Tensor:
    """Delta = M^-1 u (rows of u_perm are u_r^T in permuted coordinates): unit lower triangular solve."""
    return torch.linalg.solve_triangular(M, u_perm.double().T, upper=False, unitriangular=True).T


# =================================================================== GPU helpers (real model)

def _lazy_imports():
    global MU, load_layer_cache, LD, DT, DS
    from moe_calib import model_utils as MU                        # noqa: F401
    from moe_calib.data_utils import load_layer_cache              # noqa: F401
    import low_data_diag as LD                                     # noqa: F401
    import damping_threshold as DT                                 # noqa: F401
    import damping_selection as DS                                 # noqa: F401


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def tune_sigma(T, g):
    """S = X^T X / n_tune over the expert's tuning activations of group g (fp64, unweighted, the same
    rows tune_err_groups uses). None if the expert has no tuning tokens."""
    if T[g] is None:
        return None, 0
    X = T[g][0].double()
    return (X.T @ X) / X.shape[0], int(X.shape[0])


def select_by_rank(tok32, ranks):
    order = sorted(range(len(tok32)), key=lambda e: (tok32[e], e))
    return [order[r] for r in ranks]


def layer_context(model, L, cache, sp, E):
    """Per-layer tuning-split context shared by Part B and runs_grid_ext: tuning activations, Sigma (fp64) per
    (expert, group), GPU weights, RTN weights / errors, seed-0 B32 routed tokens and the B2 expert set."""
    HO = LD.tune_holdout(model, L, cache, sp["tune"])
    Ws = {e: LD.gpu_weights(model, L, e) for e in range(E)}
    Ts = {e: DS.prep_tune(HO, L, e) for e in range(E)}
    Sig = {}
    for e in range(E):
        for g in GROUPS:
            Sig[(e, g)] = tune_sigma(Ts[e], g)[0]
    rtnQ = {e: LD.rtn(Ws[e]) for e in range(E)}
    rtn_err = {e: DT.tune_err_groups(Ws[e], rtnQ[e], Ts[e]) for e in range(E)}
    # B2 / B1 expert sets from seed-0 B32 tokens of THIS layer
    o0 = LD.nested_order(sp["pool"], 0)
    _, tok32 = LD.hessians(model, L, cache, o0[:32], K=1)
    tok32 = [tok32[e] for e in range(E)]
    b2 = select_by_rank(tok32, B2_RANKS)
    return dict(HO=HO, Ws=Ws, Ts=Ts, Sig=Sig, rtnQ=rtnQ, rtn_err=rtn_err, tok32=tok32, b2=b2)


def prefix_hessians(model, L, cache, order, E):
    """Yields (n, run, tok) for n in LD.PREFIXES: running sums of low_data_diag.hessians over the nested prefixes
    order[:n] (exactly damping_threshold.py). `run` is mutated in place between yields."""
    run, tok, lo = None, None, 0
    for n in LD.PREFIXES:
        Hc, tc = LD.hessians(model, L, cache, order[lo:n], K=1)
        if run is None:
            run, tok = Hc, tc
        else:
            for e in range(E):
                for p in GROUPS:
                    run[e][p] += Hc[e][p]
                tok[e] += tc[e]
            del Hc
        lo = n
        yield n, run, tok


@torch.no_grad()
def gptq_group(W, H, g, nu, gs=128):
    """Run the frozen solver on each projection of group g with info. Returns {proj: (Q, info)}."""
    out = {}
    for nm in GROUPS[g]:
        inf = {}
        Q = gptq_quantize(W[nm], H, 4, gs, nu, True, info=inf)
        out[nm] = (Q, inf)
    return out


@torch.no_grad()
def b2_cell(W, H, S, g, nu, maxq=15, gs=128):
    """Measured decomposition for one (expert, group, nu): R_ideal, R_w, e_meas, clamp fractions,
    in-situ check of u = M Delta. Returns (record dict, Q dict)."""
    res = gptq_group(W, H, g, nu, gs)
    perm = res[GROUPS[g][0]][1]["perm"].to(H.device)
    for nm in GROUPS[g]:
        assert torch.equal(res[nm][1]["perm"].to(H.device), perm)
    Hp, perm2, dead = solver_coords(H, perm)
    M = m_matrix(Hp, nu)
    if M is None:
        return dict(failed="cholesky"), {nm: res[nm][0] for nm in res}
    Sp = S[perm][:, perm]
    q = col_quad(M, Sp)
    rec = dict(R_ideal=float(q.sum() / torch.trace(Sp)), R_w=0.0, e_meas=0.0, e_true=0.0, sum_col_err2=0.0,
               n_cols=0, clamp_W=0, clamp_work=0, n_w=0, delta_recon_relerr=0.0, n_dead=int(dead.sum()))
    rec["trMSM"] = float(q.sum())
    rec["trS"] = float(torch.trace(Sp))
    dr_num = dr_den = 0.0
    for nm in GROUPS[g]:
        Q, inf = res[nm]
        ce = inf["col_err2"].to(H.device)
        rec["R_w"] += float((ce * q).sum())
        rec["sum_col_err2"] += float(ce.sum())
        rec["n_cols"] += int(ce.numel())
        Wfix = W[nm].clone()
        Wfix[:, dead] = 0.0
        u = (Wfix - Q.to(Wfix.device)).double()
        rec["e_meas"] += quad_err(u, S)
        rec["e_true"] += quad_err(W[nm] - Q.to(W[nm].device), S)
        # clamp fractions on the fitted per-group grid (scale/zero are [n_groups, rows] in permuted order)
        sc = inf["scale"].to(H.device).double().repeat_interleave(gs, 0)[:Hp.shape[0]].T   # [rows, d] permuted
        zr = inf["zero"].to(H.device).double().repeat_interleave(gs, 0)[:Hp.shape[0]].T
        Wp = Wfix.double()[:, perm]
        qW = torch.round(Wp / sc + zr)
        rec["clamp_W"] += int(((qW < 0) | (qW > maxq)).sum())
        Delta = residuals_from_u(M, u[:, perm])
        Qp = Q.to(Wfix.device).double()[:, perm]
        work = Qp + Delta
        qw = torch.round(work / sc + zr)
        rec["clamp_work"] += int(((qw < 0) | (qw > maxq)).sum())
        rec["n_w"] += Wp.numel()
        dr_num += float((Delta.pow(2).sum(0) - ce).abs().sum())
        dr_den += float(ce.sum())
    rec["delta_recon_relerr"] = dr_num / max(dr_den, 1e-300)
    rec["clamp_frac_W"] = rec.pop("clamp_W") / rec["n_w"]
    rec["clamp_frac_work"] = rec.pop("clamp_work") / rec.pop("n_w")
    rec["gap_R1"] = rec["e_meas"] / rec["R_w"] - 1 if rec["R_w"] > 0 else float("nan")
    # R2 baseline: per projection, mean(col_err2) * tr(M^T S M); summed over the group's projections
    # (gate and up share M and S, so this is (sum_col_err2 / d) * tr(M^T S M))
    base = rec["sum_col_err2"] / Hp.shape[0] * rec["trMSM"]
    rec["gap_R2"] = rec["R_w"] / base - 1 if base > 0 else float("nan")
    return rec, {nm: res[nm][0] for nm in res}


# =================================================================== protocol

def build_protocol():
    return dict(
        name="theorem1_check",
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        theory="Chen & Yeh 2026, 'How Much Should GPTQ Damp?', Theorem 1",
        solver="moe_calib/gptq.py gptq_quantize (FROZEN), 4-bit, g128, act-order, blocksize 128",
        model="Qwen/Qwen1.5-MoE-A2.7B", d_in=dict(gu=2048, down=1408),
        split="tuning split only (rows P..P+tune-1); test / confirmation splits never touched",
        sigma=("S per (layer, expert, group) = X^T X / n_tune in fp64, unweighted, X = the expert's tuning activations "
               "from prep_tune(tune_holdout(model, L, cache, sp['tune']), L, e) (first 128 routed tokens per sequence, "
               "the rows tune_err_groups uses)"),
        hessian=("H_hat = running sum of low_data_diag.hessians over nested_order(pool, seed) prefixes "
                 "(exactly damping_threshold.py), fp32 accumulator cast to fp64; dead-column fix and act-order perm "
                 "replicated from gptq.py (perm taken from info['perm'] whenever the solver was run)"),
        metrics=dict(
            R_ideal="tr(M^T S M) / tr(S), computed as sum(M * (S @ M)) in fp64, M = (diag(U)^-1 U)^T, "
                    "U = upper chol(cholesky_inverse(chol(H_hat + nu*mean(diag) I)))",
            R_w="sum over projections of the group of sum_t col_err2_t * m_t^T S m_t",
            e_meas="sum over projections of the group of sum_r u_r^T S u_r, u = W(dead cols zeroed as in the "
                   "solver) - Q; e_true (unfixed W) also recorded",
            gap_R1="e_meas / R_w - 1", gap_R2="R_w / (mean(col_err2) * tr(M^T S M)) - 1",
            clamp_frac="fraction of weights whose round(w/scale + zero) falls outside [0, 15] on the fitted grid "
                       "(info scale/zero): clamp_frac_W uses the original W; clamp_frac_work uses the working "
                       "weights Q + Delta with Delta = M^-1 u",
            measured_ratio="tune_err_groups(W, Q_gptq, T)[g] / tune_err_groups(W, Q_rtn, T)[g] (existing metric)",
            cholesky_failure="cell recorded as failed; no jitter ever added"),
        part_B=dict(
            seeds=[0, 1], layers=[1, 8, 16, 23], prefixes=[16, 32, 64, 128, 256], nu_grid=NU_GRID,
            B1=dict(experts="all 60 (fallback to 24 if > 6 GPU-hours estimated: B2's 12 + ranks "
                            f"{B1_FALLBACK_EXTRA_RANKS})", groups=["gu", "down"], record=["R_ideal(nu)", "tokens"]),
            B2=dict(experts=("12 per layer: sort experts by seed-0 B32 routed tokens ascending, ties to the lower id, "
                             f"take ranks {B2_RANKS[:2]}, {B2_RANKS[2:4]}, {B2_RANKS[4:]}"),
                    record=["R_ideal", "R_w", "e_meas", "e_meas RTN", "tune_err_groups GPTQ and RTN",
                            "clamp fractions", "delta_recon_relerr (in-situ u = M Delta check)"]),
            reproducibility=("on the first layer processed, recomputed tune_err_groups for the B2 experts must "
                             "match runs_diag/damping_threshold_raw.jsonl (per group, nu in its grid) and "
                             "runs_diag/damping_selection_raw.jsonl ('all', nu in {0.01,0.03,0.05,0.1}) to <= 1e-6 "
                             "relative; abort otherwise"),
            output=RAW_GRID),
        part_C=dict(
            seed=0, seed1="only if the time estimate allows (decided after seed 0, recorded)",
            experts="per layer the experts at ranks 0, 29, 59 of seed-0 B32 routed tokens (ascending, ties to lower id)",
            data=("per-sequence routed activations (all routed tokens, model dtype, selected experts only) over the "
                  "seed's 256 nested_order calibration ids"),
            invariant=("H_hat from these activations for the prefix-32 ids equals low_data_diag.hessians up to a "
                       "scalar (relative Frobenius diff <= 1e-5 after trace normalization), identical token counts; "
                       "asserted for 3 experts x 2 groups per layer"),
            subsets=("random subsets of whole sequences drawn without replacement from the 256 ids; for group g and "
                     f"target rho in {SWEEP_RHO}, k = clip(round(rho * d_g / mean tokens per sequence), 1, 256); "
                     f"{SWEEP_DRAWS} draws per size via random.Random(SWEEP_SALT + seed*1e6 + layer*1e4 + expert*10 + "
                     "group_index).sample; if k == 256 only one draw (all draws identical); realized n recorded"),
            nu_fine=NU_FINE, nu_curve=NU_CURVE,
            nu_star=("argmin of R_ideal over nu_fine, refined by the vertex of a parabola in log nu through the 3 points "
                     "around the argmin, R_ideal evaluated at the refined value; edge argmin flagged, not refined"),
            nu_GS="d / (n^(1/3) * min(n, d)^(2/3))",
            anchors=dict(group="down", rho=ANCHOR_RHO, nu=NU_ANCHOR,
                         record=["e_meas", "e_meas RTN", "tune_err_groups['down'] GPTQ and RTN"]),
            output=RAW_SWEEP),
        report=dict(
            T1=("per group (gu, down), cell = (seed, layer, n, expert) of B1 with at least one tuning sequence: "
                "Spearman over the nu grid between R_ideal(nu) and the measured GPTQ/RTN tune_err_groups ratio; "
                "argmin agreement = |index(argmin R_ideal) - index(argmin measured)| <= 1 on the grid. "
                "PRIMARY grid = the 10 nu values measured for all 60 experts in runs_diag/damping_threshold_raw.jsonl "
                "(0.03 and 0.05 have no per-group measurement there); SECONDARY = full 12-nu grid on B2 cells "
                "(measured here). Summaries by group x prefix."),
            verdict=dict(per_group_pass="frac argmin agreement >= 0.70 AND median Spearman >= 0.8 (primary T1)",
                         labels={"T1-PASS-BOTH": "both pass", "T1-PASS-GU-ONLY": "only gu",
                                 "T1-PASS-DOWN-ONLY": "only down", "T1-FAIL": "neither"}),
            T2="B2 medians / IQRs of gap_R1 and gap_R2 by group x nu, and clamp fractions (descriptive)",
            T3=("R_ideal at nu=0.01 binned by tokens/d_in with damping_threshold.RATIO_BINS (min 20 points per bin as "
                "in the original), side by side with the observed ratios of damping_threshold_report.md section 2; "
                "Part C per-expert peak n/d of R_ideal(0.01) (median, IQR) with down anchors overlaid (descriptive)"),
            T4=("Part C nu* vs n/d with nu_GS and adaptive_damping.md per-budget optima; OLS fit log nu* = a + b log(d/n) "
                "separately for n<d and n>d over refined interior nu*, 95% CI of b by bootstrap over experts "
                "(2000 resamples, seed 0) (descriptive)")),
        constraints=["fp64 for S, H_hat, M, R", "no edits to moe_calib/gptq.py, configs/frozen_b4.json or existing runs",
                     "no caching of full-layer activations for all experts x 256 sequences"],
    )


def cmd_protocol(a):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(PROTOCOL):
        raise SystemExit(f"{PROTOCOL} exists and is frozen; not overwriting")
    json.dump(build_protocol(), open(PROTOCOL, "w"), indent=1)
    print(f"froze {PROTOCOL}")


def load_protocol():
    if not os.path.exists(PROTOCOL):
        raise SystemExit("freeze the protocol first: python diagnostics/theorem1_check.py protocol")
    return json.load(open(PROTOCOL))


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# =================================================================== Part B

def stored_measurements():
    """(seed, layer, n) -> {nu: per-expert dict or float('all')} from the two existing raw files."""
    thr, sel = {}, {}
    for r in read_jsonl("runs_diag/damping_threshold_raw.jsonl"):
        thr.setdefault((r["seed"], r["layer"], r["n"]), {}).update({float(d): v for d, v in r["err"].items()})
    for r in read_jsonl("runs_diag/damping_selection_raw.jsonl"):
        sel.setdefault((r["seed"], r["layer"], r["n"]), {}).update({float(d): v for d, v in r["err"].items()})
    return thr, sel


def relclose(x, y, tol=1e-6):
    return abs(x - y) <= tol * max(abs(y), 1e-300)


def cmd_grid(a):
    _lazy_imports()
    P = load_protocol()
    nus = P["part_B"]["nu_grid"]
    meta, sp = LD.splits(a.cache_dir)
    os.makedirs(OUT, exist_ok=True)
    have = {(r["seed"], r["layer"], r["n"], float(r["nu"])) for r in read_jsonl(RAW_GRID) if "nu" in r}
    events = read_jsonl(f"{OUT}/grid_events.jsonl")
    b1_mode = next((e["b1_mode"] for e in events if "b1_mode" in e), None)
    repro_done = any(e.get("repro_check") == "pass" for e in events)
    thr, sel = stored_measurements()
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    t_start = time.time()
    n_units = len(a.seeds) * len(a.layers)
    units_done = 0
    for L in a.layers:
        todo = [s for s in a.seeds if any((s, L, n, float(nu)) not in have for n in LD.PREFIXES for nu in nus)]
        if not todo:
            units_done += len(a.seeds)
            continue
        cache = load_layer_cache(a.cache_dir, L)
        cx = layer_context(model, L, cache, sp, E)
        HO, Ws, Ts, Sig, rtnQ, rtn_err, tok32, b2 = (cx[k] for k in ("HO", "Ws", "Ts", "Sig", "rtnQ", "rtn_err",
                                                                      "tok32", "b2"))
        b1 = list(range(E)) if b1_mode != "24" else sorted(set(b2) | set(select_by_rank(tok32, B1_FALLBACK_EXTRA_RANKS)))
        append(f"{OUT}/grid_events.jsonl", dict(layer=L, b2_experts=b2, tok32_seed0=tok32))
        rtn_e = {(e, g): sum(quad_err(Ws[e][nm] - rtnQ[e][nm], Sig[(e, g)]) for nm in GROUPS[g])
                 if Sig[(e, g)] is not None else float("nan") for e in b2 for g in GROUPS}
        for seed in todo:
            t_unit = time.time()
            order = LD.nested_order(sp["pool"], seed)
            for n, run, tok in prefix_hessians(model, L, cache, order, E):
                for nu in nus:
                    if (seed, L, n, float(nu)) in have:
                        continue
                    t0 = time.time()
                    rec = dict(seed=seed, layer=L, n=n, nu=float(nu), tokens=[tok[e] for e in range(E)],
                               b1_experts=b1, b2_experts=b2, R_ideal={g: {} for g in GROUPS},
                               n_dead={g: {} for g in GROUPS}, b2={})
                    for e in b1:
                        for g in GROUPS:
                            S = Sig[(e, g)]
                            Hp, perm, dead = solver_coords(run[e][g])
                            rec["n_dead"][g][str(e)] = int(dead.sum())
                            if S is None:
                                rec["R_ideal"][g][str(e)] = None
                                continue
                            M = m_matrix(Hp, nu)
                            rec["R_ideal"][g][str(e)] = ("failed" if M is None else
                                                         r_ideal(M, S[perm][:, perm]))
                            del M, Hp
                    t1 = time.time()
                    for e in b2:
                        cell = dict(rtn_err=rtn_err[e])
                        Qall = {}
                        for g in GROUPS:
                            if Sig[(e, g)] is None:
                                cell[g] = dict(failed="no tuning tokens")
                                Qall.update({nm: q for nm, (q, _) in gptq_group(Ws[e], run[e][g], g, nu).items()})
                                continue
                            r, Qg = b2_cell(Ws[e], run[e][g], Sig[(e, g)], g, nu)
                            r["e_rtn"] = rtn_e[(e, g)]
                            cell[g] = r
                            Qall.update(Qg)
                        cell["gptq_err"] = DT.tune_err_groups(Ws[e], Qall, Ts[e])
                        rec["b2"][str(e)] = cell
                        if not repro_done:
                            chk = []
                            if float(nu) in thr.get((seed, L, n), {}):
                                ref = thr[(seed, L, n)][float(nu)][e]
                                for g in ("all", "gu", "down"):
                                    chk.append((f"thr:{g}", cell["gptq_err"][g], ref[g]))
                            if float(nu) in sel.get((seed, L, n), {}):
                                chk.append(("sel:all", cell["gptq_err"]["all"], sel[(seed, L, n)][float(nu)][e]))
                            for tag, x, y in chk:
                                if not relclose(x, y):
                                    append(f"{OUT}/grid_events.jsonl",
                                           dict(repro_check="FAIL", seed=seed, layer=L, n=n, nu=nu, expert=e,
                                                which=tag, recomputed=x, stored=y))
                                    raise SystemExit(f"REPRO FAIL seed={seed} L={L} n={n} nu={nu} e={e} {tag}: "
                                                     f"recomputed {x!r} vs stored {y!r}")
                            rec.setdefault("repro_checked", []).extend(
                                [dict(expert=e, which=t, rel=abs(x - y) / max(abs(y), 1e-300)) for t, x, y in chk])
                    rec["sec_b1"], rec["sec_b2"] = t1 - t0, time.time() - t1
                    append(RAW_GRID, rec)
                    have.add((seed, L, n, float(nu)))
                    b2v = [rec["b2"][str(e)]["down"].get("gap_R1", float("nan")) for e in b2]
                    print(f"[t1-grid] seed={seed} L={L} n={n} nu={nu}: R_ideal med gu="
                          f"{np.nanmedian([v for v in rec['R_ideal']['gu'].values() if isinstance(v, float)]):.4f} "
                          f"down={np.nanmedian([v for v in rec['R_ideal']['down'].values() if isinstance(v, float)]):.4f}"
                          f" | B2 down gapR1 med={np.nanmedian(b2v):+.3f} | b1 {t1 - t0:.1f}s b2 {rec['sec_b2']:.1f}s",
                          flush=True)
            if not repro_done:
                nchk = sum(len(r.get("repro_checked", [])) for r in read_jsonl(RAW_GRID)
                           if r["seed"] == seed and r["layer"] == L)
                if nchk == 0:
                    raise SystemExit("reproducibility invariant could not be checked (no overlapping cells)")
                append(f"{OUT}/grid_events.jsonl", dict(repro_check="pass", seed=seed, layer=L, n_checked=nchk))
                repro_done = True
                print(f"[t1-grid] reproducibility invariant PASSED ({nchk} comparisons, seed={seed} L={L})", flush=True)
            append(f"{OUT}/grid_events.jsonl", dict(checkpoint=True, seed=seed, layer=L, sec=time.time() - t_unit))
            units_done += 1
            if b1_mode is None:
                el = time.time() - t_unit
                b1_share = sum(r["sec_b1"] for r in read_jsonl(RAW_GRID) if r["seed"] == seed and r["layer"] == L)
                est = el * n_units
                est_b1 = b1_share * n_units
                b1_mode = "60" if est_b1 <= 6 * 3600 else "24"
                append(f"{OUT}/grid_events.jsonl", dict(b1_mode=b1_mode, est_total_sec=est, est_b1_sec=est_b1,
                                                       first_unit_sec=el))
                print(f"[t1-grid] first (seed, layer) took {el / 60:.1f} min -> estimated full Part B "
                      f"{est / 3600:.2f} h (B1 share {est_b1 / 3600:.2f} GPU-h) -> B1 experts: {b1_mode}", flush=True)
                if b1_mode == "24":
                    b1 = sorted(set(b2) | set(select_by_rank(tok32, B1_FALLBACK_EXTRA_RANKS)))
        del cache, cx, HO, Ws, Ts, Sig, rtnQ
        torch.cuda.empty_cache()
    print(f"[t1-grid] done in {(time.time() - t_start) / 3600:.2f} h", flush=True)


# =================================================================== Part C

class RoutedCollector:
    """Per-sequence routed activations (all routed tokens, model dtype) for a FEW experts of one layer,
    via forward-pre-hooks on gate_proj / down_proj (the exact rows the Profiler's Hessian hooks see)."""

    def __init__(self, model, L, experts):
        self.model, self.L, self.experts = model, L, list(experts)
        self.X = {(e, g): {} for e in self.experts for g in GROUPS}
        self.cur = None

    def _hook(self, e, g):
        def fn(module, args):
            X = args[0].reshape(-1, args[0].shape[-1])
            if X.shape[0] > 0:
                self.X[(e, g)][self.cur] = X.detach().to("cpu").clone()
        return fn

    @torch.no_grad()
    def run(self, cache, ids):
        blk = MU.moe_block(self.model, self.L)
        hooks = []
        for e in self.experts:
            hooks.append(blk.experts[e].gate_proj.register_forward_pre_hook(self._hook(e, "gu")))
            hooks.append(blk.experts[e].down_proj.register_forward_pre_hook(self._hook(e, "down")))
        try:
            with MU.layer_on_gpu(self.model, self.L):
                for sid in ids:
                    self.cur = sid
                    MU.run_layer(self.model, self.L, cache[sid:sid + 1].to(MU.GPU, non_blocking=True))
        finally:
            for h in hooks:
                h.remove()
        return self


def hessian_from(Xs, ids, device):
    """fp64 sum x x^T over the given sequences' rows; returns (H, n_tokens)."""
    d = next(iter(Xs.values())).shape[1]
    H = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    for s in ids:
        X = Xs.get(s)
        if X is None:
            continue
        Xd = X.to(device, torch.float64)
        H += Xd.T @ Xd
        n += X.shape[0]
    return H, n


def sweep_curve(Hp, Sp, n, d, nus=None, curve=None, gs=True):
    """All Part C quantities for one subset Hessian (already in solver coordinates). nus / curve / gs default to
    Part C's NU_FINE / NU_CURVE / George-Saab evaluation (runs_grid_ext passes its own grid, no curve, no GS)."""
    nus = NU_FINE if nus is None else nus
    curve = NU_CURVE if curve is None else curve
    trS = torch.trace(Sp)

    def R(nu):
        M = m_matrix(Hp, nu)
        return None if M is None else float(col_quad(M, Sp).sum() / trS)
    fine = [R(nu) for nu in nus]
    out = dict(R_fine=fine, R_curve={str(nu): R(nu) for nu in curve})
    ok = [i for i, v in enumerate(fine) if v is not None]
    if not ok:
        out["failed"] = "cholesky"
        return out
    i = min(ok, key=lambda j: fine[j])
    lx = np.log(nus)
    out["argmin_idx"] = int(i)
    out["edge"] = bool(i in (0, len(nus) - 1))
    v = None
    if not out["edge"] and all(fine[j] is not None for j in (i - 1, i, i + 1)):
        v = parabola_min(lx, fine, i)
    nu_star = float(math.exp(v)) if v is not None else nus[i]
    r_star = R(nu_star) if v is not None else fine[i]
    if r_star is None or r_star > fine[i]:                 # refinement may not beat the grid point
        nu_star, r_star = nus[i], fine[i]
    out.update(nu_star=nu_star, R_star=r_star, refined=v is not None)
    if gs:
        nu_gs = george_saab(n, d)
        out.update(nu_GS=nu_gs, R_GS=R(nu_gs))
    return out


def cmd_sweep(a):
    _lazy_imports()
    P = load_protocol()
    meta, sp = LD.splits(a.cache_dir)
    os.makedirs(OUT, exist_ok=True)
    done = {(r["seed"], r["layer"], r["expert"], r["group"], r.get("kind", "curve"))
            for r in read_jsonl(RAW_SWEEP) if "group" in r}
    inv_done = {(r["seed"], r["layer"]) for r in read_jsonl(RAW_SWEEP) if r.get("invariant")}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    D = dict(zip(("gu", "down"), MU.expert_dims(model)))
    t_all = time.time()
    for seed in a.seeds:
        order = LD.nested_order(sp["pool"], seed)
        for L in a.layers:
            cache = load_layer_cache(a.cache_dir, L)
            o0 = LD.nested_order(sp["pool"], 0)
            _, tok32 = LD.hessians(model, L, cache, o0[:32], K=1)
            exps = select_by_rank([tok32[e] for e in range(E)], [0, 29, 59])
            need = [(e, g, k) for e in exps for g in GROUPS for k in ("curve", "anchor")
                    if not (k == "anchor" and g != "down") and (seed, L, e, g, k) not in done]
            if not need and (seed, L) in inv_done:
                continue
            t_unit = time.time()
            col = RoutedCollector(model, L, exps).run(cache, order)
            if (seed, L) not in inv_done:
                Href, tref = LD.hessians(model, L, cache, order[:32], K=1)
                chk = []
                for e in exps:
                    for g in GROUPS:
                        H, n = hessian_from(col.X[(e, g)], order[:32], MU.GPU)
                        Hr = Href[e][g].to(MU.GPU, torch.float64)
                        rel = float(torch.linalg.norm(H / torch.trace(H) - Hr / torch.trace(Hr))
                                    / torch.linalg.norm(Hr / torch.trace(Hr)))
                        chk.append(dict(expert=e, group=g, rel_frob=rel, tokens=n, tokens_ref=tref[e]))
                        if rel > 1e-5 or n != tref[e]:
                            append(RAW_SWEEP, dict(invariant="FAIL", seed=seed, layer=L, checks=chk))
                            raise SystemExit(f"Part C invariant FAIL L={L} e={e} {g}: rel={rel:.2e} n={n} vs {tref[e]}")
                append(RAW_SWEEP, dict(invariant="pass", seed=seed, layer=L, experts=exps, checks=chk))
                del Href
                print(f"[t1-sweep] invariant pass L={L}: max rel {max(c['rel_frob'] for c in chk):.2e}", flush=True)
            HO = LD.tune_holdout(model, L, cache, sp["tune"])
            for e in exps:
                T = DS.prep_tune(HO, L, e)
                W = LD.gpu_weights(model, L, e)
                Qr = LD.rtn(W)
                rtn_err = DT.tune_err_groups(W, Qr, T)
                tokseq = {s: col.X[(e, "gu")][s].shape[0] for s in order if s in col.X[(e, "gu")]}
                mean_tps = sum(tokseq.values()) / len(order)
                for g in GROUPS:
                    S, nt = tune_sigma(T, g)
                    gi = list(GROUPS).index(g)
                    for kind in ("curve", "anchor"):
                        if (kind == "anchor" and g != "down") or (seed, L, e, g, kind) in done:
                            continue
                        rhos = SWEEP_RHO if kind == "curve" else ANCHOR_RHO
                        rec = dict(seed=seed, layer=L, expert=e, group=g, kind=kind, d=D[g], n_tune=nt,
                                   tokens_256=sum(tokseq.values()), mean_tok_per_seq=mean_tps, rtn_err=rtn_err,
                                   subsets=[])
                        e_rtn = quad_err(W[GROUPS[g][0]] - Qr[GROUPS[g][0]], S) if kind == "anchor" else None
                        for rho in rhos:
                            k = int(min(max(round(rho * D[g] / mean_tps), 1), len(order)))
                            ndraw = 1 if k == len(order) else SWEEP_DRAWS
                            rng = random.Random(SWEEP_SALT + seed * 1_000_000 + L * 10_000 + e * 10 + gi)
                            for j in range(ndraw):
                                ids = rng.sample(order, k)
                                H, n = hessian_from(col.X[(e, g)], ids, MU.GPU)
                                sub = dict(rho=rho, k=k, draw=j, n=n, ratio=n / D[g])
                                if kind == "curve":
                                    Hp, perm, dead = solver_coords(H.float())
                                    Hp = solver_coords(H, perm)[0]      # fp64 values, solver's fp32 perm
                                    sub["n_dead"] = int(dead.sum())
                                    sub.update(sweep_curve(Hp, S[perm][:, perm], n, D[g]))
                                else:
                                    for nu in NU_ANCHOR:
                                        r, Qg = b2_cell(W, H.float(), S, g, nu)
                                        Qd = {nm: W[nm] for nm in W}
                                        Qd.update(Qg)
                                        r["e_rtn"] = e_rtn
                                        r["gptq_err"] = DT.tune_err_groups(W, Qd, T)
                                        sub[str(nu)] = r
                                rec["subsets"].append(sub)
                        append(RAW_SWEEP, rec)
                        done.add((seed, L, e, g, kind))
                        if kind == "curve":
                            pk = max(rec["subsets"], key=lambda s: s["R_curve"]["0.01"] or -1)
                            print(f"[t1-sweep] s{seed} L={L} e={e} {g}: {len(rec['subsets'])} subsets, "
                                  f"peak R(0.01)={pk['R_curve']['0.01']:.3f} at n/d={pk['ratio']:.2f}", flush=True)
                        else:
                            print(f"[t1-sweep] s{seed} L={L} e={e} anchors: {len(rec['subsets'])} subsets", flush=True)
                del T, W, Qr
            print(f"[t1-sweep] seed={seed} L={L} took {(time.time() - t_unit) / 60:.1f} min", flush=True)
            del col, HO, cache
            torch.cuda.empty_cache()
    print(f"[t1-sweep] done in {(time.time() - t_all) / 3600:.2f} h", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "grid", "sweep"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args(argv)
    if a.seeds is None:
        a.seeds = [0, 1] if a.cmd == "grid" else [0]
    dict(protocol=cmd_protocol, grid=cmd_grid, sweep=cmd_sweep)[a.cmd](a)


if __name__ == "__main__":
    main()
