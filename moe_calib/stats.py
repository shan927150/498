"""Statistical machinery for E1/E2.

E1: paired one-sided t-test over held-out SEQUENCES, in the spirit of
Korattikara, Chen & Welling, "Austerity in MCMC Land" (ICML 2014), who use a
(sequential) t-test to decide whether the data already seen suffices for a
decision.  Here the decision is: "does doubling this expert's calibration
data significantly reduce its held-out reconstruction error?"

Multiple testing: ~24 layers x 60 experts worth of hypotheses -> control FDR
with Benjamini-Hochberg (1995).  Effect sizes (mean paired difference) are
reported alongside q-values; significance alone != importance.

E2: predictive-value metrics — AUROC (Mann-Whitney form), Spearman, and the
count-controlled incremental R^2 (does the signal add information BEYOND
log routed-token count?), which is the paper's central question.  The
"estimate beforehand whether non-uniform selection will pay off" framing
follows Katharopoulos & Fleuret, "Not All Samples Are Created Equal"
(ICML 2018), who gate importance sampling on a predicted variance-reduction
estimate.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sps


def paired_t_one_sided(d: np.ndarray):
    """H1: mean(d) > 0 (more data helped). Returns (t, p, mean, n)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return np.nan, np.nan, np.nan, n
    t, p_two = sps.ttest_1samp(d, 0.0)
    p = p_two / 2 if t > 0 else 1 - p_two / 2
    return float(t), float(p), float(d.mean()), n


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    ps = p[ok]
    m = ps.size
    order = np.argsort(ps)
    q = ps[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(q, 0, 1)
    out[ok] = res
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC: P(score_pos > score_neg)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = sps.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(sps.spearmanr(x[ok], y[ok]).statistic)


def binned_spearman(signal, target, counts, n_bins: int = 5) -> float:
    """Mean within-quintile Spearman after binning by routed-token count:
    a non-parametric version of 'information beyond exposure'."""
    s, t, c = map(lambda a: np.asarray(a, float), (signal, target, counts))
    ok = np.isfinite(s) & np.isfinite(t) & np.isfinite(c)
    s, t, c = s[ok], t[ok], c[ok]
    if s.size < n_bins * 4:
        return np.nan
    edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    vals = []
    for i in range(n_bins):
        m = (c >= edges[i]) & (c <= edges[i + 1] if i == n_bins - 1 else c < edges[i + 1])
        if m.sum() >= 4:
            v = spearman(s[m], t[m])
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / (ss_tot + 1e-12)


def incremental_r2(signal, target, counts):
    """R^2 of  target ~ 1 + log(count)            (exposure only)
       vs      target ~ 1 + log(count) + signal   (exposure + signal).
    Returns (r2_base, r2_full, delta)."""
    s, t, c = map(lambda a: np.asarray(a, float), (signal, target, counts))
    ok = np.isfinite(s) & np.isfinite(t) & np.isfinite(c) & (c > 0)
    s, t, c = s[ok], t[ok], c[ok]
    if s.size < 10:
        return np.nan, np.nan, np.nan
    s = (s - s.mean()) / (s.std() + 1e-12)
    lc = np.log(c)
    X0 = np.stack([np.ones_like(lc), lc], axis=1)
    X1 = np.stack([np.ones_like(lc), lc, s], axis=1)
    r0, r1 = _ols_r2(X0, t), _ols_r2(X1, t)
    return r0, r1, r1 - r0
