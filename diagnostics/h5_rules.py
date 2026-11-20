"""H5: deployable damping rules (calibration data only) vs the R_ideal oracle, plus Part K (tails vs spectrum).

Follow-up to update_theorem1_check.md (T1-PASS-BOTH) and update_damping_grid_extension.md (EXT-INTERIOR-CONFIRMED).
All math is imported from diagnostics/theorem1_check.py; per-cell R_ideal curves / nu* come from
runs_grid_ext/raw_e1.jsonl (never recomputed).

  python diagnostics/h5_rules.py protocol   # freeze runs_h5/protocol.json (refuses to overwrite)
  python diagnostics/h5_rules.py fit        # CPU: fit R1 / R1g / R2 / R2f / R3 / R3f on both folds -> runs_h5/fit.json
  python diagnostics/h5_rules.py eval       # GPU: R4 unit CV + measured GPTQ of all rules on the eval set (resumable)
  python diagnostics/h5_rules.py kappa      # GPU: Part K tail / spectrum statistics on tuning activations
  python diagnostics/report_h5_rules.py     # report / figures

Tuning split only. moe_calib/gptq.py is frozen and only called.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theorem1_check as T1                                         # noqa: E402

OUT = "runs_h5"
PROTOCOL = f"{OUT}/protocol.json"
FIT = f"{OUT}/fit.json"
RAW_EVAL = f"{OUT}/raw_eval.jsonl"
RAW_KAPPA = f"{OUT}/raw_kappa.jsonl"
EVENTS = f"{OUT}/events.jsonl"
RAW_E1 = "runs_grid_ext/raw_e1.jsonl"
GPTQ_SHA_PREFIX = "58e40d1c"

D_IN = {"gu": 2048, "down": 1408}
GROUPS = ("gu", "down")
PREFIXES = [16, 32, 64, 128, 256]
FOLDS = {"primary": dict(fit=dict(seed=0, layers=[1, 8]), eval=dict(seed=1, layers=[16, 23])),
         "swapped": dict(fit=dict(seed=1, layers=[16, 23]), eval=dict(seed=0, layers=[1, 8]))}
NU_E1 = None                                                        # filled from damping_grid_extension on import
NU_CV = [float(x) for x in np.logspace(-2, 2, 15)]
NU_CLIP = (1e-2, 1e2)
N_BLOCKS = 8
R4_CHUNK = 15
R4_FALLBACK_RANKS = T1.B2_RANKS + [9, 10, 19, 20, 39, 40, 49, 50, 14, 15, 44, 45]
R4_BUDGET_SEC = 4 * 3600
RULES = ["R0", "R1", "R1g", "R2", "R2f", "R3", "R3f", "R4", "R5"]
K_DIRS, K_TOP, KAPPA_SEED = 64, 16, 20261119
NBOOT = 2000
T1_PART_B_NU = T1.NU_GRID

from damping_grid_extension import NU_E1 as _NU_E1                 # noqa: E402
NU_E1 = _NU_E1
LX = np.log(NU_E1)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def clip_nu(nu):
    return np.clip(nu, *NU_CLIP)


# =================================================================== interpolation of R_ideal curves (vectorized)

class CurveInterp:
    """Monotone cubic (PCHIP) interpolation of each cell's R_ideal curve in log nu on the 42-point E1 grid;
    evaluates cell i at its own nu_i (vectorized). nu is clipped to [1e-2, 1e2]."""

    def __init__(self, R):
        from scipy.interpolate import PchipInterpolator
        R = np.asarray(R, float)
        self.n = R.shape[0]
        self.c = PchipInterpolator(LX, R, axis=1).c                 # (4, 41, ncell)
        self.cols = np.arange(self.n)

    def __call__(self, nu):
        x = np.log(clip_nu(np.broadcast_to(np.asarray(nu, float), (self.n,))))
        i = np.clip(np.searchsorted(LX, x, side="right") - 1, 0, len(LX) - 2)
        t = x - LX[i]
        c = self.c[:, i, self.cols]
        return ((c[0] * t + c[1]) * t + c[2]) * t + c[3]


# =================================================================== rule fitting (CPU)

def e1_cells(seed, layers):
    """Cells of raw_e1 for (seed, layers): per group arrays R [n, 42], R_star, nu_star, tokens, ntilde, meta."""
    out = {g: dict(R=[], R_star=[], nu_star=[], tokens=[], ntilde=[], edge=[], meta=[]) for g in GROUPS}
    for r in T1.read_jsonl(RAW_E1):
        if r["seed"] != seed or r["layer"] not in layers:
            continue
        nt = float(np.median(r["tokens"]))
        for k, c in r["cells"].items():
            e, g = k.split(":")
            if "R_fine" not in c or not all(isinstance(v, float) for v in c["R_fine"]):
                continue
            o = out[g]
            o["R"].append(c["R_fine"]), o["R_star"].append(c["R_star"]), o["nu_star"].append(c["nu_star"])
            o["tokens"].append(r["tokens"][int(e)]), o["ntilde"].append(nt), o["edge"].append(c["edge"])
            o["meta"].append((r["seed"], r["layer"], r["n"], int(e)))
    for g in GROUPS:
        for k in ("R", "R_star", "nu_star", "tokens", "ntilde"):
            out[g][k] = np.asarray(out[g][k], float)
        out[g]["edge"] = np.asarray(out[g]["edge"], bool)
        out[g]["interp"] = CurveInterp(out[g]["R"])
        out[g]["cluster"] = np.array([hash((m[1], m[3])) for m in out[g]["meta"]])
    return out


def law_x(cg, g, kind):
    n = cg["ntilde"] if kind == "budget" else cg["tokens"]
    with np.errstate(divide="ignore"):
        return np.log(D_IN[g] / n)


def law_nu(logc, b, x):
    return clip_nu(np.exp(logc + b * x))


def excess(cg, nu, w=None):
    ex = cg["interp"](nu) / cg["R_star"] - 1
    return float(np.average(ex, weights=w))


def fit_law(cg, g, kind, free, w=None, start=None):
    """Minimize the (weighted) mean fit-set excess R(nu_rule)/R(nu*) - 1 over log c (and b if free).
    Point fit: grid then refine; bootstrap fits (start given): local refine from the point estimate."""
    from scipy.optimize import minimize, minimize_scalar
    x = law_x(cg, g, kind)
    if not free:
        f = lambda lc: excess(cg, law_nu(lc, 2 / 3, x), w)
        if start is None:
            grid = np.linspace(math.log(1e-3), math.log(1e3), 241)
        else:
            grid = np.linspace(start[0] - 1.5, start[0] + 1.5, 61)
        v = [f(lc) for lc in grid]
        j = int(np.argmin(v))
        h = grid[1] - grid[0]
        r = minimize_scalar(f, bounds=(grid[j] - h, grid[j] + h), method="bounded", options=dict(xatol=1e-6))
        lc = float(r.x) if r.fun <= v[j] else float(grid[j])
        return dict(c=math.exp(lc), b=2 / 3, logc=lc, objective=f(lc))
    f = lambda p: excess(cg, law_nu(p[0], p[1], x), w)
    if start is None:
        best = None
        for lc in np.linspace(math.log(1e-3), math.log(1e3), 61):
            for b in np.linspace(0.0, 1.5, 31):
                v = f((lc, b))
                if best is None or v < best[0]:
                    best = (v, lc, b)
        p0 = best[1:]
    else:
        p0 = start
    r = minimize(f, p0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000))
    return dict(c=math.exp(r.x[0]), b=float(r.x[1]), logc=float(r.x[0]), objective=float(r.fun))


def cluster_weights(cl, rng):
    u, inv = np.unique(cl, return_inverse=True)
    cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
    return cnt[inv].astype(float)


def boot_law(cg, g, kind, free, point, rng, nboot=NBOOT):
    cs, bs = [], []
    for _ in range(nboot):
        w = cluster_weights(cg["cluster"], rng)
        if w.sum() == 0:
            continue
        r = fit_law(cg, g, kind, free, w=w, start=(point["logc"], point["b"]) if free else (point["logc"],))
        cs.append(r["c"]), bs.append(r["b"])
    out = dict(c_ci95=[float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))])
    if free:
        out["b_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def fit_r1g(cg, rng, nboot=NBOOT):
    """Per prefix: parabola-refined argmin of the fit-set expert-mean R_ideal curve (+ cluster bootstrap CI)."""
    from report_damping_grid_extension import vertex
    out = {}
    for n in PREFIXES:
        sel = np.array([m[2] == n for m in cg["meta"]])
        Rm = cg["R"][sel]
        cl = cg["cluster"][sel]
        nu = vertex(LX, Rm.mean(0))[0]
        bs = []
        for _ in range(nboot):
            w = cluster_weights(cl, rng)
            bs.append(vertex(LX, (w[:, None] * Rm).sum(0) / w.sum())[0])
        out[str(n)] = dict(nu=nu, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return out


def measured_all(seed, layers):
    """(layer, n) -> {nu: [per-expert 'all']} from damping_threshold_raw (10 nu) + damping_selection_raw (0.03, 0.05)."""
    out = {}
    for r in T1.read_jsonl("runs_diag/damping_threshold_raw.jsonl"):
        if r["seed"] == seed and r["layer"] in layers:
            for d, v in r["err"].items():
                out.setdefault((r["layer"], r["n"]), {})[float(d)] = [x["all"] for x in v]
    for r in T1.read_jsonl("runs_diag/damping_selection_raw.jsonl"):
        if r["seed"] == seed and r["layer"] in layers:
            for d, v in r["err"].items():
                out.setdefault((r["layer"], r["n"]), {}).setdefault(float(d), v)
    return out


def fit_r1(seed, layers, rng, nboot=NBOOT):
    """Per prefix: argmin over the 12 Part B nu of the fit-set expert-mean measured 'all' metric."""
    m = measured_all(seed, layers)
    out = {}
    for n in PREFIXES:
        rows, cl = [], []
        for L in layers:
            d = m[(L, n)]
            assert all(nu in d for nu in T1_PART_B_NU), (L, n, sorted(d))
            for e in range(60):
                v = [d[nu][e] for nu in T1_PART_B_NU]
                if all(isinstance(x, float) and math.isfinite(x) for x in v):
                    rows.append(v), cl.append(hash((L, e)))
        A = np.array(rows)
        cl = np.array(cl)
        mean = A.mean(0)
        nu = T1_PART_B_NU[int(np.argmin(mean))]
        bs = []
        for _ in range(nboot):
            w = cluster_weights(cl, rng)
            bs.append(T1_PART_B_NU[int(np.argmin((w[:, None] * A).sum(0)))])
        out[str(n)] = dict(nu=nu, experts=len(rows), mean_curve=mean.tolist(),
                           boot_p2_5=float(np.percentile(bs, 2.5)), boot_p97_5=float(np.percentile(bs, 97.5)),
                           boot_mode=float(max(set(bs), key=bs.count)))
    return out


def cmd_fit(a):
    load_protocol()
    rng = np.random.default_rng(0)
    res = {}
    for fold, spec in FOLDS.items():
        f = spec["fit"]
        cells = e1_cells(f["seed"], f["layers"])
        r = dict(fit_set=f, n_cells={g: int(len(cells[g]["R_star"])) for g in GROUPS}, R1g={}, laws={})
        if fold == "primary":
            r["R1"] = fit_r1(f["seed"], f["layers"], rng)
        for g in GROUPS:
            cg = cells[g]
            r["R1g"][g] = fit_r1g(cg, rng)
            r["laws"][g] = {}
            for rule, kind, free in (("R2", "budget", False), ("R2f", "budget", True),
                                     ("R3", "expert", False), ("R3f", "expert", True)):
                t0 = time.time()
                p = fit_law(cg, g, kind, free)
                p.update(boot_law(cg, g, kind, free, p, rng))
                p["oracle_floor_check"] = excess(cg, cg["nu_star"])   # ~0 by construction (interp vs refined)
                r["laws"][g][rule] = p
                print(f"[h5-fit] {fold} {g} {rule}: c={p['c']:.4g} {p['c_ci95']} b={p['b']:.4g} "
                      f"{p.get('b_ci95', '')} excess={100 * p['objective']:.3f}% ({time.time() - t0:.0f}s)", flush=True)
        res[fold] = r
    json.dump(res, open(FIT, "w"), indent=1)
    print(f"wrote {FIT}")


# =================================================================== R4: unit cross-validation (Chen & Yeh eq. 2)

def cv_curve(blocks, perm, nus):
    """blocks: list of (G_b fp64 [d, d], n_b) for the NON-EMPTY blocks; perm: the full-prefix act-order permutation.
    Hhat = G/n in solver coordinates (dead-column fix of solver_coords), lambda = nu * mean(diag Hhat) from the
    FULL prefix; fold f: A_-f = Hhat - G_f/n + lambda I (= ((G - G_f) + n lambda I)/n; M is scale invariant);
    CV(nu) = (1/n) sum_f tr(M_-f^T G_f M_-f). None for a nu where any fold's Cholesky fails."""
    n = sum(nb for _, nb in blocks)
    G = sum(Gb for Gb, _ in blocks)
    Hp, _, dead = T1.solver_coords(G / n, perm)
    Fs = [(Gb / n)[perm][:, perm] for Gb, _ in blocks]
    lam0 = torch.mean(torch.diag(Hp))
    eye = torch.eye(Hp.shape[0], dtype=Hp.dtype, device=Hp.device)

    def cv(nu):
        tot = 0.0
        for F in Fs:
            M = T1.m_from_A(Hp - F + eye * (nu * lam0))
            if M is None:
                return None
            tot += float(T1.col_quad(M, F).sum())
        return tot

    vals = [cv(nu) for nu in nus]
    out = dict(cv=vals)
    ok = [i for i, v in enumerate(vals) if v is not None]
    if not ok:
        out["failed"] = "cholesky"
        return out
    i = min(ok, key=lambda j: vals[j])
    out["argmin_idx"] = int(i)
    out["edge"] = bool(i in (0, len(nus) - 1))
    v = None
    if not out["edge"] and all(vals[j] is not None for j in (i - 1, i, i + 1)):
        v = T1.parabola_min(np.log(nus), vals, i)
    nu_star = float(math.exp(v)) if v is not None else nus[i]
    c_star = cv(nu_star) if v is not None else vals[i]
    if c_star is None or c_star > vals[i]:
        nu_star, c_star = nus[i], vals[i]
    out.update(nu=nu_star, cv_star=c_star, refined=v is not None, n_dead=int(dead.sum()))
    return out


def block_ids(order, n):
    k = n // N_BLOCKS
    return [order[b * k:(b + 1) * k] for b in range(N_BLOCKS)]


def gram(Xs, ids, device):
    """fp64 sum x x^T over the given sequences' routed rows (None, 0 if no rows)."""
    rows = [Xs[s] for s in ids if s in Xs]
    if not rows:
        return None, 0
    return T1.hessian_from({s: Xs[s] for s in ids if s in Xs}, ids, device)


# =================================================================== quantization with per-group damping

@torch.no_grad()
def quant_group(W, H, g, nu):
    return {nm: T1.gptq_quantize(W[nm], H[g], 4, 128, nu, True) for nm in T1.GROUPS[g]}


# =================================================================== protocol

def build_protocol():
    return dict(
        name="h5_rules",
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        follows=["update_theorem1_check.md (T1-PASS-BOTH)", "update_damping_grid_extension.md (EXT-INTERIOR-CONFIRMED)",
                 "runs_diag/adaptive_damping.md"],
        solver=f"moe_calib/gptq.py gptq_quantize (FROZEN, sha256 prefix {GPTQ_SHA_PREFIX}), 4-bit g128 act-order; "
               "per-group damping: gate_proj/up_proj with nu_gu on H_gu, down_proj with nu_down on H_down, one "
               "gptq_quantize call per projection with its own percdamp (asserted bitwise equal to "
               "low_data_diag.quant when nu_gu = nu_down on 3 experts of the first eval unit)",
        split=dict(
            fit_primary=FOLDS["primary"]["fit"], eval_primary=FOLDS["primary"]["eval"],
            fit_swapped=FOLDS["swapped"]["fit"], eval_swapped=FOLDS["swapped"]["eval"],
            note="tuning split only; all 60 experts; prefixes 16-256; no rule sees eval-set errors, eval Sigma or eval "
                 "nu* except the oracle R5; the swapped fold is scored with R_ideal only (no GPTQ, no R4)"),
        data=dict(
            e1="per-cell R_ideal curves on the 42-point grid, R_star and nu_star from runs_grid_ext/raw_e1.jsonl "
               "(not recomputed); n_e,g = routed calibration tokens of the expert at that (seed, layer, prefix) "
               "(raw_e1 'tokens', identical for gu and down); ntilde = median of n_e over the layer's 60 experts",
            measured_fit="runs_diag/damping_threshold_raw.jsonl (10 nu) + damping_selection_raw.jsonl (0.03, 0.05), "
                         "'all' metric"),
        rules={
            "R0": "nu_gu = nu_down = 1.0",
            "R1": "per prefix one shared nu: argmin over the 12 Part B nu of the fit-set expert-mean measured 'all' "
                  "(experts finite at all 12 nu); no refinement",
            "R1g": "per (prefix, group): parabola-refined argmin (log nu) of the fit-set expert-mean R_ideal curve "
                   "(report_damping_grid_extension.vertex)",
            "R2": "nu = c_g (d_g / ntilde)^(2/3)", "R2f": "nu = c_g (d_g / ntilde)^b_g",
            "R3": "nu = c'_g (d_g / n_e)^(2/3)", "R3f": "nu = c'_g (d_g / n_e)^b'_g",
            "R4": ("per (expert, group, prefix) unit CV: the prefix's nested-order sequences split into 8 contiguous "
                   "blocks of n/8 sequences; G_f = fp64 Gram of the expert's routed calibration inputs in block f "
                   "(theorem1_check.RoutedCollector, bf16 rows, all routed tokens, experts collected in chunks of "
                   f"{R4_CHUNK}); A_-f = (G - G_f) + n lambda I with lambda = nu mean(diag(G/n)) of the FULL prefix, "
                   "in the act-order permutation (from the solver's fp32 H) and dead-column handling of the full "
                   "prefix; M_-f via theorem1_check.m_from_A; CV(nu) = (1/n) sum_f tr(M_-f^T G_f M_-f); folds with 0 "
                   "routed tokens skipped and counted; selection = argmin over logspace(-2, 2, 15) refined by the "
                   "parabola vertex in log nu (CV re-evaluated, kept only if not above the grid minimum); sum of block "
                   "tokens asserted equal to the Profiler's token count"),
            "R5": "oracle: per (expert, group) nu_star of raw_e1 (eval-set Sigma, 42-point grid, parabola refined)",
            "clip": "every rule's nu clipped to [1e-2, 1e2]"},
        fitting=dict(
            objective=("R2/R2f/R3/R3f minimize the mean over fit-set cells (seed, layer, prefix, expert) of "
                       "R_ideal(nu_rule)/R_ideal(nu*) - 1, R_ideal(nu) by PCHIP interpolation of the cell's 42-point "
                       "curve in log nu, R_ideal(nu*) = raw_e1 R_star; fitted per group"),
            optimizer=("fixed exponent: grid of 241 log c in [log 1e-3, log 1e3] then bounded Brent refine; free "
                       "exponent: grid 61 log c x 31 b in [0, 1.5] then Nelder-Mead"),
            ci=("cluster bootstrap over (layer, expert), 2000 resamples (multiplicity weights), numpy default_rng(0), "
                "local refit from the point estimate; R1: bootstrap percentiles of the argmin; R1g: of the vertex")),
        r4_cost=("estimate printed after the first (layer, prefix); if the projected R4 time over the 10 eval units "
                 f"exceeds 4 GPU-h, R4 is restricted for the remaining units to 24 experts per layer (ranks "
                 f"{R4_FALLBACK_RANKS} of seed-0 B32 tokens) and all rules are also reported on that subset"),
        evaluation=dict(
            measured=("every eval (layer, prefix, expert): quantize with each rule's (nu_gu, nu_down) - GPTQ run once "
                      "per distinct nu per group - and record tune_err_groups (all, gu, down) + RTN"),
            E_rule="expert-mean measured 'all' over the eval set per prefix (experts finite for every rule)",
            gain="(E_R0 - E_rule) / E_R0 in %, paired cluster bootstrap over (layer, expert), 2000 resamples, "
                 "default_rng(0) (same resamples for all rules and prefixes)",
            capture="per prefix (E_R0 - E_rule)/(E_R0 - E_R5); overall = sum_prefix (E_R0 - E_rule) / "
                    "sum_prefix (E_R0 - E_R5); CI from the same bootstrap",
            per_group="same with the 'gu' and 'down' metrics",
            r4_minus_r3="(E_R4 - E_R3)/E_R0 per prefix and overall, paired bootstrap CI",
            invariants=["R0 measured tune_err_groups equals runs_diag/damping_threshold_raw.jsonl at nu=1.0 for every "
                        "eval expert (<= 1e-6 relative)",
                        "per-group quantizer with nu_gu = nu_down = 1.0 equals low_data_diag.quant bitwise (3 experts, "
                        "first unit)",
                        "first unit: R_ideal at the 12 Part B nu for all 60 experts x 2 groups equals "
                        "runs_theorem1_check/raw_grid.jsonl bitwise (refactor check of m_from_A)"]),
        verdict=dict(
            order=["R2", "R3", "R4"],
            pass_rule="overall oracle capture (measured 'all') >= 0.80 AND gain vs R0 95% CI lower bound > 0 at both "
                      "B16 and B32",
            labels={"H5-PASS-R2": "R2 passes", "H5-PASS-R3": "R2 fails, R3 passes",
                    "H5-PASS-R4": "R2 and R3 fail, R4 passes", "H5-FAIL": "none pass"}),
        secondary=dict(
            swapped=("fit on seed 1 layers {16, 23}, score on seed 0 layers {1, 8} with interpolated R_ideal: "
                     "E_rule = mean R_ideal(nu_rule) per (prefix, group); capture as above, per group and pooled "
                     "(mean over both groups' cells); rules R0, R1g, R2, R2f, R3, R3f, R5. The same R_ideal-scored "
                     "table is reported for the primary fold (all rules) for comparison"),
            exponents="b_g, b'_g 95% CI vs 2/3", transfer="c_g, c'_g of both folds with CIs"),
        part_K=dict(
            data=("tuning-split activations X = prep_tune(tune_holdout(...))[g] rows (fp16 -> fp64), per (layer, "
                  "expert, group), layers {1, 8, 16, 23}, all 60 experts; cells with < 64 tuning rows dropped"),
            kappa_coord="median over coordinates with E[x_j^2] > 0 of E[x_j^4] / E[x_j^2]^2",
            kappa_dir=(f"mean over {K_DIRS} fixed random unit directions (torch.Generator seed {KAPPA_SEED}, standard "
                       "normal normalized, one set per d) of E[(x.v)^4] / E[(x.v)^2]^2"),
            kappa_top=(f"eigenvalue-weighted mean over the top-{K_TOP} eigenvectors of Sigma = X^T X / n of "
                       "E[(x.v)^4] / E[(x.v)^2]^2"),
            spectrum="erank(Sigma)/d (exp entropy) and participation ratio/d via damping_threshold.spectrum_stats",
            regression=("y = log nu* per raw_e1 cell (all seeds, layers, prefixes, both groups; edge cells and "
                        "tokens = 0 dropped); OLS; M0: log(d/n_e) + 1[down]; M1: M0 + log kappa (each variant); "
                        "M2: M0 + log(erank/d); M3: M0 + log kappa_best + log(erank/d); kappa_best = variant with the "
                        "highest overall R^2 in M1; report the 1[down] coefficient with cluster-bootstrap 95% CI "
                        "(2000, clusters (layer, expert), default_rng(0)), overall R^2, within-down R^2 and within-gu "
                        "R^2 (1 - SSE_g / SST_g of the joint fit)"),
            verdict={"K-TAILS": "M1 (kappa_best) shrinks |group coef| by >= 50% vs M0 AND raises within-down R^2 by "
                                ">= 0.10",
                     "K-SPECTRUM": "same two conditions for M2",
                     "K-BOTH": "both", "K-NEITHER": "neither"}),
        constraints=["fp64 Gram / H_hat / Sigma / M / R / CV; Cholesky failure recorded, never jittered",
                     "no edits to moe_calib/gptq.py, configs/frozen_b4.json or existing outputs; writes only runs_h5/",
                     "resumable, checkpoint per (layer, prefix)"],
    )


def cmd_protocol(a):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(PROTOCOL):
        raise SystemExit(f"{PROTOCOL} exists and is frozen; not overwriting")
    json.dump(build_protocol(), open(PROTOCOL, "w"), indent=1)
    print(f"froze {PROTOCOL}")


def load_protocol():
    if not os.path.exists(PROTOCOL):
        raise SystemExit("freeze the protocol first: python diagnostics/h5_rules.py protocol")
    return json.load(open(PROTOCOL))


# =================================================================== eval (GPU)

def rule_nus(fit, L, n, e, tokens, ntilde, r4, r5):
    """rule -> (nu_gu, nu_down) for one eval expert."""
    P = fit["primary"]
    out = {"R0": (1.0, 1.0), "R1": (P["R1"][str(n)]["nu"],) * 2}
    out["R1g"] = tuple(P["R1g"][g][str(n)]["nu"] for g in GROUPS)
    for rule, kind in (("R2", "budget"), ("R2f", "budget"), ("R3", "expert"), ("R3f", "expert")):
        pr = [P["laws"][g][rule] for g in GROUPS]
        nn = ntilde if kind == "budget" else tokens
        out[rule] = tuple(float(clip_nu(p["c"] * (D_IN[g] / nn) ** p["b"])) if nn > 0 else NU_CLIP[1]
                          for p, g in zip(pr, GROUPS))
    out["R4"] = tuple(r4.get(g, float("nan")) for g in GROUPS)
    out["R5"] = tuple(float(clip_nu(r5[g])) for g in GROUPS)
    return out


def cmd_eval(a):
    if not sha256("moe_calib/gptq.py").startswith(GPTQ_SHA_PREFIX):
        raise SystemExit("moe_calib/gptq.py hash changed")
    load_protocol()
    fit = json.load(open(FIT))
    T1._lazy_imports()
    LD, MU, DT = T1.LD, T1.MU, T1.DT
    meta, sp = LD.splits(a.cache_dir)
    os.makedirs(OUT, exist_ok=True)
    seed = FOLDS["primary"]["eval"]["seed"]
    layers = FOLDS["primary"]["eval"]["layers"]
    done = {(r["layer"], r["n"]) for r in T1.read_jsonl(RAW_EVAL)}
    ev = T1.read_jsonl(EVENTS)
    r4_mode = next((e["r4_mode"] for e in ev if "r4_mode" in e), None)
    inv_done = any(e.get("partb_bitwise") == "pass" for e in ev)
    e1 = {(r["seed"], r["layer"], r["n"]): r for r in T1.read_jsonl(RAW_E1)}
    pb = {(r["seed"], r["layer"], r["n"], float(r["nu"])): r for r in T1.read_jsonl(T1.RAW_GRID)}
    thr, _ = T1.stored_measurements()
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    t_all = time.time()
    n_units = len(layers) * len(PREFIXES)
    for L in layers:
        if all((L, n) in done for n in PREFIXES):
            continue
        cache = T1.load_layer_cache(a.cache_dir, L)
        cx = T1.layer_context(model, L, cache, sp, E)
        Ws, Ts, rtnQ, rtn_err, tok32 = (cx[k] for k in ("Ws", "Ts", "rtnQ", "rtn_err", "tok32"))
        r4_all = list(range(E))
        r4_sub = sorted(T1.select_by_rank(tok32, R4_FALLBACK_RANKS))
        order = LD.nested_order(sp["pool"], seed)
        for n, run, tok in T1.prefix_hessians(model, L, cache, order, E):
            if (L, n) in done:
                continue
            t_unit = time.time()
            er = e1[(seed, L, n)]
            assert [float(tok[e]) for e in range(E)] == er["tokens"], "token counts differ from raw_e1"
            ntilde = float(np.median([tok[e] for e in range(E)]))
            rec = dict(seed=seed, layer=L, n=n, tokens=[float(tok[e]) for e in range(E)], ntilde=ntilde,
                       r4_experts=r4_all if r4_mode != "24" else r4_sub, r4_sub=r4_sub, experts={})
            # ---- refactor invariant (first unit): R_ideal at the Part B nu, bitwise
            if not inv_done:
                worst = 0.0
                for nu in T1_PART_B_NU:
                    ref = pb[(seed, L, n, nu)]["R_ideal"]
                    for e in range(E):
                        for g in GROUPS:
                            S = cx["Sig"][(e, g)]
                            Hp, perm, _ = T1.solver_coords(run[e][g])
                            x = T1.r_ideal(T1.m_matrix(Hp, nu), S[perm][:, perm])
                            if x != ref[g][str(e)]:
                                worst = max(worst, abs(x - ref[g][str(e)]) / abs(ref[g][str(e)]))
                T1.append(EVENTS, dict(partb_bitwise="pass" if worst == 0.0 else "FAIL", layer=L, n=n, max_rel=worst))
                if worst != 0.0:
                    raise SystemExit(f"Part B bitwise reproduction FAILED (max rel {worst:.2e})")
                print("[h5-eval] Part B R_ideal reproduced bitwise after the m_from_A refactor", flush=True)
                inv_done = True
            # ---- R4 unit CV
            t0 = time.time()
            r4 = {e: {} for e in range(E)}
            r4_detail = {}
            blocks = block_ids(order, n)
            r4e = rec["r4_experts"]
            for c0 in range(0, len(r4e), R4_CHUNK):
                chunk = r4e[c0:c0 + R4_CHUNK]
                col = T1.RoutedCollector(model, L, chunk).run(cache, order[:n])
                for e in chunk:
                    for g in GROUPS:
                        Xs = col.X[(e, g)]
                        bl = [gram(Xs, ids, MU.GPU) for ids in blocks]
                        ntot = sum(nb for _, nb in bl)
                        if ntot != tok[e]:
                            raise SystemExit(f"R4 token mismatch L{L} n{n} e{e} {g}: {ntot} vs {tok[e]}")
                        nonempty = [(G, nb) for G, nb in bl if nb > 0]
                        _, perm, _ = T1.solver_coords(run[e][g])
                        res = cv_curve(nonempty, perm.to(MU.GPU), NU_CV) if nonempty else dict(failed="no tokens")
                        res["skipped_folds"] = N_BLOCKS - len(nonempty)
                        res["block_tokens"] = [nb for _, nb in bl]
                        r4_detail[f"{e}:{g}"] = res
                        if "nu" in res:
                            r4[e][g] = res["nu"]
                        del bl, nonempty
                del col
                torch.cuda.empty_cache()
            sec_r4 = time.time() - t0
            # ---- measured evaluation of every rule
            t0 = time.time()
            for e in range(E):
                c = {g: er["cells"][f"{e}:{g}"] for g in GROUPS}
                r5 = {g: c[g]["nu_star"] for g in GROUPS}
                nus = rule_nus(fit, L, n, e, tok[e], ntilde, r4[e], r5)
                if e not in r4e:
                    nus.pop("R4")
                Q = {g: {} for g in GROUPS}
                for rule, pair in nus.items():
                    for g, nu in zip(GROUPS, pair):
                        if math.isfinite(nu) and nu not in Q[g]:
                            Q[g][nu] = quant_group(Ws[e], run[e], g, nu)
                errs = {}
                for rule, pair in nus.items():
                    if not all(math.isfinite(v) for v in pair):
                        continue
                    Qd = dict(Q["gu"][pair[0]])
                    Qd.update(Q["down"][pair[1]])
                    errs[rule] = DT.tune_err_groups(Ws[e], Qd, Ts[e])
                # invariants
                ref = thr[(seed, L, n)][1.0][e]
                for k in ("all", "gu", "down"):
                    x, y = errs["R0"][k], ref[k]
                    if not (math.isnan(x) and math.isnan(y)) and not T1.relclose(x, y):
                        raise SystemExit(f"R0 reproduction FAIL L{L} n{n} e{e} {k}: {x} vs {y}")
                if not any(e_.get("ldquant_bitwise") for e_ in T1.read_jsonl(EVENTS)) and e < 3:
                    Qref = LD.quant(Ws[e], run[e], 1.0)
                    Qd = dict(Q["gu"][1.0])
                    Qd.update(Q["down"][1.0])
                    if not all(torch.equal(Qd[nm], Qref[nm]) for nm in Qref):
                        raise SystemExit(f"per-group quantizer != low_data_diag.quant (e{e})")
                    if e == 2:
                        T1.append(EVENTS, dict(ldquant_bitwise="pass", layer=L, n=n, experts=[0, 1, 2]))
                rec["experts"][str(e)] = dict(nus={k: list(v) for k, v in nus.items()}, err=errs, rtn=rtn_err[e],
                                              n_quant={g: len(Q[g]) for g in GROUPS})
                del Q
            sec_q = time.time() - t0
            rec.update(r4_detail=r4_detail, sec_r4=sec_r4, sec_quant=sec_q, sec=time.time() - t_unit)
            T1.append(RAW_EVAL, rec)
            done.add((L, n))
            T1.append(EVENTS, dict(checkpoint=True, layer=L, n=n, sec=rec["sec"], sec_r4=sec_r4, sec_quant=sec_q))
            m = {r: np.nanmean([x["err"][r]["all"] for x in rec["experts"].values() if r in x["err"]]) for r in RULES}
            print(f"[h5-eval] L{L} n{n}: " + " ".join(f"{r}={m[r]:.5f}" for r in RULES) +
                  f" | R4 {sec_r4:.0f}s quant {sec_q:.0f}s", flush=True)
            if r4_mode is None:
                est_r4 = sec_r4 * n_units
                r4_mode = "60" if est_r4 <= R4_BUDGET_SEC else "24"
                T1.append(EVENTS, dict(r4_mode=r4_mode, est_total_sec=rec["sec"] * n_units, est_r4_sec=est_r4,
                                       first_unit_sec=rec["sec"]))
                print(f"[h5-eval] first (layer, prefix) took {rec['sec'] / 60:.1f} min -> estimated eval "
                      f"{rec['sec'] * n_units / 3600:.2f} h (R4 {est_r4 / 3600:.2f} GPU-h) -> R4 experts: {r4_mode}",
                      flush=True)
        del cache, cx, Ws, Ts, rtnQ
        torch.cuda.empty_cache()
    print(f"[h5-eval] done in {(time.time() - t_all) / 3600:.2f} h", flush=True)


# =================================================================== Part K (GPU)

@torch.no_grad()
def kurt_along(X, V):
    """E[(x.v)^4] / E[(x.v)^2]^2 for each column v of V (fp64)."""
    Y = X @ V
    m2 = Y.pow(2).mean(0)
    m4 = Y.pow(4).mean(0)
    return m4 / m2.pow(2)


@torch.no_grad()
def kappa_stats(X, dirs):
    """X: [n, d] fp64. Returns kappa_coord, kappa_dir, kappa_top, erank/d, pr/d."""
    n, d = X.shape
    m2 = X.pow(2).mean(0)
    m4 = X.pow(4).mean(0)
    ok = m2 > 0
    k_coord = float(torch.median(m4[ok] / m2[ok].pow(2)))
    k_dir = float(kurt_along(X, dirs).mean())
    S = X.T @ X / n
    lam, V = torch.linalg.eigh(S)
    top = torch.argsort(lam, descending=True)[:K_TOP]
    kt = kurt_along(X, V[:, top])
    w = lam[top].clamp(min=0)
    k_top = float((w * kt).sum() / w.sum())
    er, pr = T1.DT.spectrum_stats(S)
    return dict(kappa_coord=k_coord, kappa_dir=k_dir, kappa_top=k_top, erank_d=er / d, pr_d=pr / d, n_tune=n)


def cmd_kappa(a):
    load_protocol()
    T1._lazy_imports()
    LD, MU, DS = T1.LD, T1.MU, T1.DS
    meta, sp = LD.splits(a.cache_dir)
    done = {r["layer"] for r in T1.read_jsonl(RAW_KAPPA)}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    dirs = {}
    for g, d in D_IN.items():
        gen = torch.Generator().manual_seed(KAPPA_SEED)
        v = torch.randn(d, K_DIRS, generator=gen, dtype=torch.float64)
        dirs[g] = (v / v.norm(dim=0, keepdim=True)).to(MU.GPU)
    for L in [1, 8, 16, 23]:
        if L in done:
            continue
        t0 = time.time()
        cache = T1.load_layer_cache(a.cache_dir, L)
        HO = LD.tune_holdout(model, L, cache, sp["tune"])
        rec = dict(layer=L, cells={})
        for e in range(E):
            T = DS.prep_tune(HO, L, e)
            for g in GROUPS:
                if T[g] is None or T[g][0].shape[0] < 64:
                    rec["cells"][f"{e}:{g}"] = dict(dropped=True, n_tune=0 if T[g] is None else int(T[g][0].shape[0]))
                    continue
                rec["cells"][f"{e}:{g}"] = kappa_stats(T[g][0].double(), dirs[g])
            del T
        T1.append(RAW_KAPPA, rec)
        k = [c["kappa_dir"] for kk, c in rec["cells"].items() if kk.endswith("down") and "kappa_dir" in c]
        print(f"[h5-kappa] L{L}: median kappa_dir down {np.median(k):.2f} ({time.time() - t0:.0f}s)", flush=True)
        del cache, HO
        torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "fit", "eval", "kappa"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    a = ap.parse_args(argv)
    dict(protocol=cmd_protocol, fit=cmd_fit, eval=cmd_eval, kappa=cmd_kappa)[a.cmd](a)


if __name__ == "__main__":
    main()
