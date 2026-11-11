"""Multi-grouping robustness of the FR-vs-FC GPTQ PPL comparison: pure helpers (no model, no I/O).

Five prespecified randomized BF16 calibration batch groupings of the SAME 217 calibration
sequences (sorted FR s0 u FC s0). Each grouping: numpy default_rng(seed).permutation of the sorted
union, consecutive batches of 8 (27 x 8 + 1). The same grouping is shared by FR and FC (paired
design); only the selected subsets differ. No evaluation / held-out rows enter the groupings.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

GROUPING_SEEDS = (20261111, 20261112, 20261113, 20261114, 20261115)
BATCH = 8
N_UNION = 217
EXPECTED_SIZES = [8] * 27 + [1]
DELTA_ORIGINAL = 9.277908190612935 - 9.267851290916612        # frozen partial-PPL FC s0 - FR s0 (+0.010057)


def gname(seed):
    return f"G{int(seed)}"


# ---------------------------------------------------------------- groupings
def grouping_batches(union, seed, B=BATCH):
    """sorted union -> default_rng(seed).permutation -> consecutive batches of B (last may be short).
    Batch membership AND within-batch order are those of the permuted sequence."""
    u = sorted(int(x) for x in union)
    perm = np.random.default_rng(int(seed)).permutation(len(u))
    s = [u[i] for i in perm]
    return [s[i:i + B] for i in range(0, len(s), B)]


def validate_grouping(batches, union, forbidden=(), B=BATCH):
    """Every structural requirement of a grouping; returns a dict of named booleans."""
    flat = [r for b in batches for r in b]
    sizes = [len(b) for b in batches]
    n = len(set(union))
    exp = [B] * (n // B) + ([n % B] if n % B else [])
    return dict(exact_union=sorted(flat) == sorted(set(union)), no_duplicates=len(flat) == len(set(flat)),
                n_rows=len(flat) == n, batch_sizes=sizes == exp, last_batch_singleton=sizes[-1] == (n % B or B),
                no_forbidden_rows=not (set(flat) & set(forbidden)))


def manifest(seed, batches, union, strategies):
    """Frozen per-grouping manifest. `strategies`: {name: chosen ids} -> per-batch selected members (the
    SAME batches for both strategies: only which members each strategy's Hessian uses differs)."""
    body = dict(grouping=gname(seed), seed=int(seed), batch_size=BATCH,
                construction="sorted(FR s0 u FC s0) -> numpy.random.default_rng(seed).permutation -> consecutive "
                             "batches of 8; batch membership and within-batch order preserved; executed in list order",
                n_batches=len(batches), batch_size_pattern=[len(b) for b in batches],
                batches=[list(map(int, b)) for b in batches],
                union_sorted=sorted(map(int, union)),
                selected_members_per_batch={k: [[r for r in b if r in set(v)] for b in batches]
                                            for k, v in strategies.items()})
    body["manifest_sha256"] = obj_sha(body)
    return body


def obj_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strip_sha(m):
    return {k: v for k, v in m.items() if k != "manifest_sha256"}


def remap(ids, rows):
    """Original sequence ids -> row index in a cache stored in `rows` order."""
    r = {int(s): i for i, s in enumerate(rows)}
    return [r[int(s)] for s in ids]


# ---------------------------------------------------------------- statistics
def grouping_stats(ppl, groupings, delta_original=DELTA_ORIGINAL, tau=None):
    """ppl[(strategy, grouping)] -> per-grouping Delta_g = PPL(FC,g) - PPL(FR,g) and descriptive statistics
    over the given groupings (the primary randomized sample). tau: deterministic PPL tolerance."""
    gs = list(groupings)
    d = np.array([ppl[("FC", g)] - ppl[("FR", g)] for g in gs], float)
    inter = d - delta_original
    s0 = np.sign(delta_original)
    rng = {s: float(max(ppl[(s, g)] for g in gs) - min(ppl[(s, g)] for g in gs)) for s in ("FR", "FC")}
    out = dict(groupings=gs, delta={g: float(x) for g, x in zip(gs, d)},
               interaction={g: float(x) for g, x in zip(gs, inter)},
               mean_delta=float(d.mean()), median_delta=float(np.median(d)),
               sd_delta=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
               min_delta=float(d.min()), max_delta=float(d.max()), delta_range=float(d.max() - d.min()),
               n_fc_worse=int((d > 0).sum()), n_fc_better=int((d < 0).sum()), n_tie=int((d == 0).sum()),
               n_sign_reversals=int((np.sign(d) != s0).sum()),
               strategy_range=rng,
               strategy_mean={s: float(np.mean([ppl[(s, g)] for g in gs])) for s in ("FR", "FC")},
               strategy_sd={s: float(np.std([ppl[(s, g)] for g in gs], ddof=1)) for s in ("FR", "FC")},
               max_abs_interaction=float(np.abs(inter).max()),
               delta_original=float(delta_original),
               sd_over_abs_delta_original=float((d.std(ddof=1) if len(d) > 1 else 0.0) / abs(delta_original)))
    if tau is not None:
        out["n_sign_reversals_beyond_tau"] = int(((np.sign(d) != s0) & (np.abs(d) > tau)).sum())
        out["tau"] = float(tau)
    return out


DECISION_RULE = {
    "quantities": "over the 5 new groupings g only: Delta_g = PPL(FC,g) - PPL(FR,g); Delta_O = +0.010057 (frozen "
                  "partial-PPL ORIGINAL); I_g = Delta_g - Delta_O; Range_s = max_g PPL(s,g) - min_g PPL(s,g); "
                  "SD = sample SD (ddof=1) of Delta_g; tau = 1e-4 x PPL_BF16 (the established deterministic PPL "
                  "tolerance of the earlier protocols; NOT re-derived from strategy effects); "
                  "reversal_g = sign(Delta_g) != sign(Delta_O) AND |Delta_g| > tau",
    "precedence": "INVALID-COMPARISON > RUN-BLOCKED > GROUPING-DEPENDENT-RANKING > RANKING-STABLE-BUT-SENSITIVE > "
                  "RANKING-ROBUST-WITHIN-TESTED-GROUPINGS",
    "INVALID-COMPARISON": "any grouping differs from ORIGINAL in token ids, layer kwargs (positions / mask / rotary), "
                          "model weights, dtype, attention implementation or eval mode, or a configuration fails its "
                          "installation checks (240 experts, 720 projections changed, no RTN fallback, untouched "
                          "parameters unchanged, finite weights)",
    "RUN-BLOCKED": "a configuration is missing, or a determinism control fails (ORIGINAL FR/FC not reproduced "
                   "bitwise, repeated PPL evaluation differs, reversed execution order changes the activations)",
    "GROUPING-DEPENDENT-RANKING": "any reversal_g, OR max_g |I_g| >= 0.5 |Delta_O|, OR SD >= 0.5 |Delta_O|",
    "RANKING-STABLE-BUT-SENSITIVE": "no reversal and both comparability conditions fail, but max_g |I_g| > "
                                    "max(tau, 0.1 |Delta_O|) or max_s Range_s > max(tau, 0.1 |Delta_O|)",
    "RANKING-ROBUST-WITHIN-TESTED-GROUPINGS": "otherwise (direction preserved in 5/5; I and Range within "
                                              "max(tau, 0.1 |Delta_O|))",
}


def decide(ppl, groupings, bf16, valid=True, blocked=False, delta_original=DELTA_ORIGINAL):
    if not valid:
        return "INVALID-COMPARISON"
    if blocked:
        return "RUN-BLOCKED"
    tau = 1e-4 * bf16
    s = grouping_stats(ppl, groupings, delta_original, tau)
    d0 = abs(delta_original)
    if s["n_sign_reversals_beyond_tau"] > 0 or s["max_abs_interaction"] >= 0.5 * d0 or s["sd_delta"] >= 0.5 * d0:
        return "GROUPING-DEPENDENT-RANKING"
    floor = max(tau, 0.1 * d0)
    if s["max_abs_interaction"] > floor or max(s["strategy_range"].values()) > floor:
        return "RANKING-STABLE-BUT-SENSITIVE"
    return "RANKING-ROBUST-WITHIN-TESTED-GROUPINGS"


BOOTSTRAP_NOTE = ("paired-window bootstrap: windows resampled jointly for FC and FR of ONE grouping; the interval is "
                  "evaluation-text sampling uncertainty CONDITIONAL on those two quantized models. It does not describe "
                  "calibration-batching uncertainty, and an interval excluding zero in one grouping does not make the "
                  "strategy difference robust across groupings.")


def bootstrap_summary(cis, point):
    """cis: {g: [lo, hi]}; point {g: d}. Counts per-grouping 'conditional' conclusions, which can disagree."""
    excl_pos = [g for g, c in cis.items() if c[0] > 0]
    excl_neg = [g for g, c in cis.items() if c[1] < 0]
    return dict(n_ci_excluding_zero_positive=len(excl_pos), n_ci_excluding_zero_negative=len(excl_neg),
                n_ci_including_zero=len(cis) - len(excl_pos) - len(excl_neg),
                contradictory_conditional_conclusions=bool(excl_pos and excl_neg), note=BOOTSTRAP_NOTE)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
