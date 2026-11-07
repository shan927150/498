"""Exposure-only allocation utilities for E3 (P2c M1 / Dlb / M4 -> controller).

P2c (diagnostics/p2c_layer_exposure.py) validated, on characterization data
only (candidate pool + tuning split), that the future relative GPTQ benefit of
doubling an expert's calibration data,
    y = future_rel_delta = (err(n0) - err(2 n0)) / err(n0),
depends on routed-token exposure n, layer and calibration regime n0 (the
number of calibration SEQUENCES). This module

  * owns the P2c design matrices (`design`; the diagnostic calls it too, so
    there is exactly one implementation),
  * fits M1 / Dlb / M4 on the full characterization table and exports them
    as a controller artifact (`fit_export`; standardization from the training
    rows only, coefficients + column schema + support ranges + provenance),
  * applies an exported model at an E3 checkpoint without refitting
    (`predict_marginal_benefit`) and turns it into expert utilities
    (`compute_expert_utility`).

Two quantities are kept distinct everywhere:
  m    number of calibration sequences selected so far (= the regime n0)
  n_e  routed-token exposure of expert e (the count feature)

Intermediate regimes (prespecified, never tuned on E3 results): for
b_lo < m < b_hi (neighbouring characterized budgets) the prediction is
linear in log budget between the two characterized regimes, at the SAME n_e:
    f(n, L, m) = (1 - w) f(n, L, b_lo) + w f(n, L, b_hi),
    w = log(m / b_lo) / log(b_hi / b_lo).
m outside [min budget, max budget] -> no prediction (never extrapolated).

Count support: expert e of layer L is predicted only if
    lo(L, b_lo) <= n_e <= hi(L, b_hi)
(the observed routed-count hull of that layer over the bracketing regimes;
b_lo = b_hi = m at a characterized budget). Outside it the model is not
applied: below -> exploration target (as in E3's existing rule), above -> no
utility; zero exposure -> exploration. Every fallback is counted and logged.
"""
from __future__ import annotations
import csv
import hashlib
import json

import numpy as np

# E3 strategy -> P2c model
STRATEGY_MODEL = {"global_exposure": "M1", "layer_budget": "Dlb", "layer_aware": "M4"}
LP_UTILITY_STRATEGIES = ("coverage_lp", *STRATEGY_MODEL)
# constrained variants: strategy -> the utility it shares (moe_calib/repr_constraint.py)
CONSTRAINED_LP_STRATEGIES = {"coverage_lp_repr": "coverage_lp"}
# aggregate-shift-constrained variant (moe_calib/aggregate_constraint.py)
AGGREGATE_LP_STRATEGIES = {"coverage_lp_aggregate": "coverage_lp"}
ARTIFACT_KIND = "e3_allocation_models"
ARTIFACT_VERSION = 1


# --------------------------------------------------------------- design

def design(name, lc, budget, layer, budgets, layers):
    """P2c ladder design matrix and column names. lc = standardized log count
    (train mu/sd); budgets / layers = the sorted TRAINING levels (first level
    is the reference). Column order is the P2c order."""
    cols, names = [np.ones_like(lc), lc], ["1", "lc"]
    if name == "M0":
        return np.stack(cols, 1), names
    B = [(budget == b).astype(float) for b in budgets[1:]]
    bn = [f"b{b:g}" for b in budgets[1:]]
    cols += [lc ** 2, lc ** 3] + B + [x * lc for x in B]
    names += ["lc^2", "lc^3"] + bn + [f"{b}*lc" for b in bn]
    if name == "M1":
        return np.stack(cols, 1), names
    Lc = [(layer == l).astype(float) for l in layers[1:]]
    ln = [f"L{l:g}" for l in layers[1:]]
    cols += Lc
    names += ln
    if name in ("Dlb", "DlbS"):
        cols += [x * y for x in Lc for y in B]
        names += [f"{a}*{b}" for a in ln for b in bn]
    powers = {"M3": 1, "M4": 2, "M4c": 3, "DlbS": 1}.get(name, 0)
    for p in range(1, powers + 1):
        cols += [x * lc ** p for x in Lc]
        names += [f"{a}*lc^{p}" for a in ln]
    return np.stack(cols, 1), names


# ----------------------------------------------------------- fit/export

def load_table(path):
    """The P2 observation table (e2_signals_b*_d*.csv), exposure columns only."""
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    d = dict(seed=col("characterization_seed"), layer=col("layer"), budget=col("starting_budget"),
             expert=col("expert"), count=col("routed_count"), y=col("future_rel_delta"))
    ok = np.isfinite(d["y"]) & (d["count"] > 0)
    return {k: v[ok] for k, v in d.items()}, int((~ok).sum())


def fit_model(d, name):
    """Least squares exactly as P2c's fit(): mu/sd of log count from `d`."""
    lc = np.log(d["count"])
    mu, sd = float(lc.mean()), float(lc.std() + 1e-12)
    budgets = sorted(set(d["budget"].tolist()))
    layers = sorted(set(d["layer"].tolist()))
    X, names = design(name, (lc - mu) / sd, d["budget"], d["layer"], budgets, layers)
    beta, *_ = np.linalg.lstsq(X, d["y"], rcond=None)
    r = d["y"] - X @ beta
    return dict(model=name, columns=names, beta=beta.tolist(), lc_mean=mu, lc_std=sd,
                budgets=budgets, layers=layers, n_obs=int(d["y"].size),
                rank=int(np.linalg.matrix_rank(X)), train_mse=float(np.mean(r ** 2)))


def support_table(d):
    """Observed routed-count range per (layer, characterized budget)."""
    out = {}
    for L in sorted(set(d["layer"].tolist())):
        out[f"{L:g}"] = {}
        for b in sorted(set(d["budget"].tolist())):
            c = d["count"][(d["layer"] == L) & (d["budget"] == b)]
            out[f"{L:g}"][f"{b:g}"] = [float(c.min()), float(c.max())]
    return out


def file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def artifact_fingerprint(art: dict) -> str:
    body = {k: v for k, v in art.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def fit_export(d, provenance: dict, models=tuple(STRATEGY_MODEL.values())) -> dict:
    art = dict(kind=ARTIFACT_KIND, version=ARTIFACT_VERSION,
               target="future_rel_delta (E1, n0 -> 2 n0, tuning split)",
               strategy_model=dict(STRATEGY_MODEL),
               models={m: fit_model(d, m) for m in models},
               count_support=support_table(d),
               budgets=sorted(set(d["budget"].tolist())),
               layers=sorted(set(d["layer"].tolist())),
               characterization_seeds=sorted(set(d["seed"].tolist())),
               n_obs=int(d["y"].size),
               interpolation="linear in log(m) between neighbouring characterized budgets "
                             "at fixed routed count; no extrapolation outside [min, max] budget",
               support_rule="lo(L, b_lo) <= n <= hi(L, b_hi); below -> exploration, "
                            "above -> no utility, n = 0 -> exploration",
               **provenance)
    art["fingerprint"] = artifact_fingerprint(art)
    return art


def check_artifact(art: dict, cfg: dict, layers, strategy: str) -> None:
    """Reject an artifact that is corrupted or does not match the frozen run."""
    if art.get("kind") != ARTIFACT_KIND or art.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"not an allocation-model artifact (kind={art.get('kind')!r})")
    if artifact_fingerprint(art) != art.get("fingerprint"):
        raise ValueError("allocation-model artifact fingerprint mismatch (edited or corrupted)")
    want = dict(model=cfg["model"], gptq=cfg["gptq"], split=cfg["split"],
                layers=list(cfg["layers"]))
    got = {k: art.get("frozen", {}).get(k) for k in want}
    if got != want:
        raise ValueError(f"allocation models were fitted under a different configuration:\n"
                         f"artifact={got}\nfrozen={want}")
    missing = [L for L in layers if float(L) not in art["layers"]]
    if missing:
        raise ValueError(f"layers {missing} were not characterized in the artifact")
    if strategy in STRATEGY_MODEL and STRATEGY_MODEL[strategy] not in art["models"]:
        raise ValueError(f"artifact has no model {STRATEGY_MODEL[strategy]} for {strategy}")


# ------------------------------------------------------------ predict

def predict_at_budget(entry: dict, layer: float, count, budget: float) -> np.ndarray:
    """Model prediction at a CHARACTERIZED budget (no support check)."""
    if float(budget) not in entry["budgets"]:
        raise ValueError(f"budget {budget} is not characterized {entry['budgets']}")
    if float(layer) not in entry["layers"]:
        raise ValueError(f"layer {layer} is not characterized {entry['layers']}")
    count = np.asarray(count, float)
    lc = (np.log(count) - entry["lc_mean"]) / entry["lc_std"]
    X, names = design(entry["model"], lc, np.full(lc.shape, float(budget)),
                      np.full(lc.shape, float(layer)), entry["budgets"], entry["layers"])
    assert names == entry["columns"], "feature schema changed since export"
    return X @ np.asarray(entry["beta"])


def bracket(m: float, budgets):
    """(b_lo, b_hi, w) for regime m, or None outside the characterized range."""
    budgets = sorted(float(b) for b in budgets)
    if not budgets[0] <= m <= budgets[-1]:
        return None
    if m in budgets:
        return m, m, 0.0
    lo = max(b for b in budgets if b < m)
    hi = min(b for b in budgets if b > m)
    return lo, hi, float(np.log(m / lo) / np.log(hi / lo))


def predict_marginal_benefit(art: dict, model: str, layer, count, m):
    """Predicted future_rel_delta of the experts of one layer with routed
    counts `count` after m selected sequences. Returns (pred, status) where
    status is 'ok' | 'zero' | 'invalid' | 'below' | 'above' | 'budget_out_of_range'
    per expert; pred is NaN wherever status != 'ok'."""
    entry = art["models"][model]
    count = np.asarray(count, float)
    pred = np.full(count.shape, np.nan)
    status = np.full(count.shape, "ok", dtype=object)
    status[count == 0] = "zero"
    status[~np.isfinite(count) | (count < 0)] = "invalid"
    br = bracket(float(m), art["budgets"])
    if br is None:
        status[status == "ok"] = "budget_out_of_range"
        return pred, status
    b_lo, b_hi, w = br
    sup = art["count_support"][f"{float(layer):g}"]
    lo, hi = sup[f"{b_lo:g}"][0], sup[f"{b_hi:g}"][1]
    ok = status == "ok"
    status[ok & (count < lo)] = "below"
    status[ok & (count > hi)] = "above"
    ok = status == "ok"
    if ok.any():
        f = predict_at_budget(entry, layer, count[ok], b_lo)
        if w > 0:
            f = (1 - w) * f + w * predict_at_budget(entry, layer, count[ok], b_hi)
        pred[ok] = f
    return pred, status


STATUSES = ("ok", "zero", "invalid", "below", "above", "budget_out_of_range")


def compute_expert_utility(strategy: str, n, layers, E: int, m, art: dict | None):
    """delta_hat over the layer-major expert vector (j = li * E + e) for one of
    LP_UTILITY_STRATEGIES, plus the per-expert status. The LP value is then
    max(delta_hat, 0) / n_e (utility.allocation_terms), so
      coverage_lp      delta_hat = 1          -> u_e = 1 / n_e
      learned          delta_hat = f(n_e, L, m) -> u_e = max(f, 0) / n_e
    NaN delta_hat => no utility; 'zero'/'below' experts become exploration
    targets in E3 (existing rule), 'above' / out-of-range get nothing."""
    n = np.asarray(n, float)
    dh = np.full(n.shape, np.nan)
    status = np.full(n.shape, "ok", dtype=object)
    if strategy == "coverage_lp":
        status[n == 0] = "zero"
        dh[n > 0] = 1.0
        return dh, status
    if strategy not in STRATEGY_MODEL:
        raise ValueError(f"unknown utility strategy {strategy}")
    if art is None:
        raise ValueError(f"{strategy} needs an allocation-model artifact")
    for li, L in enumerate(layers):
        sl = slice(li * E, (li + 1) * E)
        dh[sl], status[sl] = predict_marginal_benefit(art, STRATEGY_MODEL[strategy], L, n[sl], m)
    return dh, status


def status_summary(status, layers, E):
    """Fraction of experts per status, overall and per layer."""
    out = {"all": {s: float(np.mean(status == s)) for s in STATUSES}}
    for li, L in enumerate(layers):
        st = status[li * E:(li + 1) * E]
        out[str(L)] = {s: float(np.mean(st == s)) for s in STATUSES}
    return out
