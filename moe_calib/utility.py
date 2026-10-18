"""Expert utility = predicted FUTURE held-out GPTQ benefit (the E2 -> E3 link).

Sufficiency is defined by E1's target, rel_delta: the relative held-out
reconstruction-error improvement from doubling an expert's calibration data
(n -> 2n). Statistics such as drift or relSE are only candidate predictors.

E2 fits, per candidate signal (orientation o: -1 for margin, else +1),
    delta_hat = b0 + b_count * log(n) + b_sig * (o * s - mu) / sd
("count" drops the signal term = exposure-only predictor), selects one by
a deterministic out-of-sample criterion on TUNING-split labels, and writes
the predictors to e2_model_b{bits}.json. E3 turns delta_hat into the LP
terms in `allocation_terms`.
"""
from __future__ import annotations
import numpy as np

SIGNALS = ("count", "drift", "relse", "margin", "entropy")
ORIENT = {"count": 1.0, "drift": 1.0, "relse": 1.0, "margin": -1.0, "entropy": 1.0}


# ----------------------------------------------------------------- fitting

def _design(signal, count, value, norm):
    lc = np.log(np.where(np.asarray(count, float) > 0, count, np.nan))
    cols = [np.ones_like(lc), lc]
    if signal != "count":
        v = ORIENT[signal] * np.asarray(value, float)
        cols.append((v - norm["mean"]) / norm["std"])
    return np.stack(cols, axis=1)


def fit_predictor(signal, count, value, target) -> dict:
    count = np.asarray(count, float)
    norm = None
    if signal != "count":
        v = ORIENT[signal] * np.asarray(value, float)
        norm = dict(mean=float(v.mean()), std=float(v.std() + 1e-12))
    X = _design(signal, count, value, norm)
    beta, *_ = np.linalg.lstsq(X, np.asarray(target, float), rcond=None)
    coef = dict(beta0=float(beta[0]), beta_count=float(beta[1]))
    if signal != "count":
        coef["beta_signal"] = float(beta[2])
    return dict(signal=signal, orientation=ORIENT[signal], coef=coef, norm=norm,
                n_obs=int(count.size),
                count_range=[float(count.min()), float(count.max())])


def predict(entry: dict, count, value=None) -> np.ndarray:
    """delta_hat; NaN where count <= 0 or the signal is undefined."""
    c = entry["coef"]
    count = np.asarray(count, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = c["beta0"] + c["beta_count"] * np.log(np.where(count > 0, count, np.nan))
        if entry["signal"] != "count":
            v = entry["orientation"] * np.asarray(value, float)
            out = out + c["beta_signal"] * (v - entry["norm"]["mean"]) / entry["norm"]["std"]
    return out


def _mse(entry, count, value, target):
    r = predict(entry, count, value) - target
    return float(np.mean(r ** 2))


def select_signal(obs: dict, seeds: np.ndarray, layers: np.ndarray):
    """obs: arrays count, target and one per signal, restricted to rows where
    every signal is finite (so all candidates are scored on the same rows).
    Criterion (lower is better; ties -> earlier in SIGNALS, i.e. count):
      >=2 characterization seeds: leave-one-seed-out MSE   (validated)
      else >=2 layers:            leave-one-layer-out MSE  (preliminary)
      else:                       -adjusted in-sample R^2   (preliminary)
    """
    if len(np.unique(seeds)) >= 2:
        groups, crit, prelim = seeds, "leave_one_char_seed_out_mse", False
    elif len(np.unique(layers)) >= 2:
        groups, crit, prelim = layers, "leave_one_layer_out_mse", True
    else:
        groups, crit, prelim = None, "neg_adjusted_in_sample_r2", True
    y, n = obs["target"], obs["target"].size
    scores = {}
    for s in SIGNALS:
        v = obs.get(s) if s != "count" else None
        if groups is None:
            ent = fit_predictor(s, obs["count"], v, y)
            k = 2 if s == "count" else 3
            r2 = 1 - _mse(ent, obs["count"], v, y) / (float(np.var(y)) + 1e-12)
            scores[s] = -(1 - (1 - r2) * (n - 1) / max(n - k, 1))
        else:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                ent = fit_predictor(s, obs["count"][tr], None if v is None else v[tr], y[tr])
                r = predict(ent, obs["count"][te], None if v is None else v[te]) - y[te]
                errs.append(r ** 2)
            scores[s] = float(np.mean(np.concatenate(errs)))
    selected = min(SIGNALS, key=lambda s: (scores[s], SIGNALS.index(s)))
    return selected, crit, prelim, scores


# ---------------------------------------------------------------- E3 side

def predicted_delta(model: dict, signal: str, stats: dict):
    """delta_hat per expert from the artifact's predictor for `signal`, ONLY
    inside the fitted routed-count support [lo, hi]. Outside it the log-count
    model is never extrapolated: delta_hat = NaN and the expert is flagged
    below/above support. In-support experts whose signal is undefined (e.g.
    relSE with < 2 non-empty blocks) use the artifact's count-only predictor.
    Returns (delta_hat, info) with boolean masks in info."""
    n = np.asarray(stats["count"], float)
    lo, hi = model["predictors"][signal]["count_range"]
    below = (n > 0) & (n < lo)
    above = n > hi
    inside = (n > 0) & ~below & ~above
    dh = np.full(n.shape, np.nan)
    v = None if signal == "count" else np.asarray(stats[signal], float)[inside]
    dh[inside] = predict(model["predictors"][signal], n[inside], v)
    undefined = inside & ~np.isfinite(dh)
    if undefined.any():
        dh[undefined] = predict(model["predictors"]["count"], n[undefined])
    return dh, dict(below=below, above=above, signal_undefined=undefined)


def allocation_terms(n, delta_hat, w):
    """Per-expert LP terms. capacity = n (E1 measured n -> 2n);
    value = w * max(delta_hat, 0) / max(n, 1)  [rel. improvement per token].
    Experts without a defined delta_hat (zero exposure, out of support) get
    value 0 and capacity 0: they are handled by exploration, not by a
    made-up benefit."""
    n = np.asarray(n, float)
    defined = np.isfinite(delta_hat) & (n > 0)
    benefit = np.where(defined, np.maximum(np.nan_to_num(delta_hat), 0.0), 0.0)
    cap = np.where(defined, n, 0.0)
    value = np.asarray(w, float) * benefit / np.maximum(n, 1.0)
    return value, cap


def exploration_cover(Ccand: np.ndarray, targets: np.ndarray, budget: int):
    """Exploration (NOT a sufficiency estimate) for experts whose benefit is
    undefined: greedily pick the candidate row reaching the most still-
    uncovered targets, never more than `budget` rows. Ties -> lowest row
    index (deterministic). Returns (picks, reached_mask)."""
    hit = Ccand > 0
    uncovered = targets & hit.any(axis=0)
    reached = np.zeros_like(targets)
    picks: list[int] = []
    while uncovered.any() and len(picks) < budget:
        gain = hit[:, uncovered].sum(axis=1)
        gain[picks] = -1
        i = int(np.argmax(gain))
        if gain[i] <= 0:
            break
        picks.append(i)
        reached |= targets & hit[i]
        uncovered &= ~hit[i]
    assert len(picks) <= budget
    return picks, reached
