"""C4-prefix diversity-rescue study (modified EBSS, not a MoEQuant reproduction): pure helpers.

Prefix extraction, token accounting, region-level diversity / repetition metrics and the frozen decision rules.
Everything here is CPU-only and deterministic so it can be unit-tested without the model.
"""
from __future__ import annotations

import numpy as np

from . import ebss_official_data as ED

P_LEN, G_LEN, SEQ = 64, 448, 512
REGIONS = {"prefix": (0, P_LEN), "continuation": (P_LEN, SEQ), "full": (0, SEQ)}
HALVES = {"first_half": (P_LEN, P_LEN + G_LEN // 2), "second_half": (P_LEN + G_LEN // 2, SEQ)}
LOOP_NGRAM = 16          # loop onset: first continuation position whose next 16 tokens copy an earlier 16-gram
COVER_NGRAM = 8          # coverage: continuation positions inside an 8-gram occurring >= 2x in the sequence
LONG_SPAN = 64           # "repeated span exceeding 64 tokens"
LONG_RUN = 16            # "long single-token loop": identical-token run >= 16
DIV_D2 = 0.05            # frozen diversity materiality: continuation distinct-2 +0.05 absolute ...
DIV_REP = 0.10           # ... AND continuation repetition rate -0.10 absolute (10 percentage points)
QUAL_MARGIN_PCT = 0.5    # reported "material" flag for reconstruction reductions (label rule itself is strict <)


def token_hash(ids) -> str:
    return ED.window_hash(list(map(int, ids)))


def split_windows(windows, p_len=P_LEN):
    """512-token parent windows -> (prefixes [:p_len], natural continuations [p_len:])."""
    W = [list(map(int, w)) for w in windows]
    assert all(len(w) == SEQ for w in W)
    return [w[:p_len] for w in W], [w[p_len:] for w in W]


def prefix_checks(prefixes, parents, heldout_windows=None, leak_fn=None):
    pre = [list(map(int, x)) for x in prefixes]
    par = [list(map(int, x)) for x in parents]
    out = dict(n=len(pre), all_length_64=all(len(x) == P_LEN for x in pre),
               all_distinct=len({tuple(x) for x in pre}) == len(pre),
               parents_512=all(len(x) == SEQ for x in par),
               prefix_is_parent_head=all(a == b[:P_LEN] for a, b in zip(pre, par)),
               continuation_preserved=all(len(b[P_LEN:]) == G_LEN for b in par),
               no_padding=True)
    if leak_fn is not None:
        out["leakage"] = leak_fn(par, heldout_windows)
        out["no_leakage"] = out["leakage"]["windows_with_32gram_match"] == 0
    out["valid"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


def accounting(n, p_len, seqlen, w, vocab):
    """Token denominators of prefix-conditioned generation (cf. ebss_pilot.token_accounting for P = 1)."""
    steps = seqlen - p_len
    return dict(prefix_tokens=n * p_len, emitted_continuation_tokens=n * steps, final_calibration_tokens=n * seqlen,
                prefill_token_positions=n * p_len, decode_forward_rows=n * steps * w,
                scored_candidates=n * (vocab + (steps - 1) * w * vocab), forward_calls=1 + steps)


# ------------------------------------------------------------------ text metrics (prior-study definitions)
distinct_n, longest_run, zlib_ratio, set_distinct_n = ED.distinct_n, ED.longest_run, ED.zlib_ratio, ED.set_distinct_n
near_duplicates = ED.near_duplicate_stats          # 13-gram Jaccard (ED.NEAR_NGRAM), as in earlier EBSS studies


def longest_repeated_ngram(ids):
    """largest n such that some n-gram occurs at least twice (overlaps allowed); = diagnostics_ebss_pilot."""
    ids = list(ids)
    lo, hi = 0, max(len(ids) - 1, 0)

    def has(n):
        seen = set()
        for i in range(len(ids) - n + 1):
            t = tuple(ids[i:i + n])
            if t in seen:
                return True
            seen.add(t)
        return False
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if has(mid) else (lo, mid - 1)
    return lo


def region_text(W, tok_nll=None, region=(0, SEQ)):
    """Diversity of tokens[a:b] of every sequence. tok_nll [n, 511]: NLL of target t at index t-1; the region's
    NLL averages the targets inside [max(a,1), b)."""
    a, b = region
    R = [list(map(int, w[a:b])) for w in W]
    d1 = np.array([distinct_n(r, 1) for r in R]); d2 = np.array([distinct_n(r, 2) for r in R])
    d3 = np.array([distinct_n(r, 3) for r in R])
    runs = np.array([longest_run(r) for r in R]); lr = np.array([longest_repeated_ngram(r) for r in R])
    zr = np.array([zlib_ratio(r) for r in R])
    out = dict(region=[a, b], n_tokens_per_seq=b - a,
               per_sequence_mean=dict(distinct1=float(d1.mean()), distinct2=float(d2.mean()), distinct3=float(d3.mean()),
                                      repetition_rate=float(1 - d1.mean()), zlib_ratio=float(zr.mean()),
                                      longest_identical_run=float(runs.mean()), longest_repeated_ngram=float(lr.mean())),
               corpus=dict(distinct1=set_distinct_n(R, 1), distinct2=set_distinct_n(R, 2), distinct3=set_distinct_n(R, 3),
                           n_unique_types=len({t for r in R for t in r})),
               longest_identical_run_max=int(runs.max()), longest_repeated_ngram_max=int(lr.max()),
               longest_repeated_ngram_per_seq=lr.tolist(), distinct2_per_seq=d2.tolist(),
               exact_duplicate_pairs=int(sum(R[i] == R[j] for i in range(len(R)) for j in range(i + 1, len(R)))),
               near_duplicates=near_duplicates(R))
    if tok_nll is not None:
        x = np.asarray(tok_nll, float)[:, max(a, 1) - 1:b - 1]
        out["nll"] = dict(mean_per_token=float(x.mean()), per_sequence=x.mean(1).tolist(), ppl=float(np.exp(x.mean())))
    return out


def loop_onset(ids, start=P_LEN, n=LOOP_NGRAM):
    """Offset (from `start`) of the first position t >= start whose n-gram ids[t:t+n] already occurred at some
    s < t (copy of earlier text, prefix included). None if no such position."""
    ids = list(map(int, ids))
    first = {}
    for s in range(len(ids) - n + 1):
        g = tuple(ids[s:s + n])
        if g in first and s >= start:
            return s - start
        first.setdefault(g, s)
    return None


def repeat_coverage(ids, a=P_LEN, b=SEQ, n=COVER_NGRAM):
    """Fraction of positions in [a, b) covered by an n-gram (fully inside [0, b)) that occurs >= 2x in ids[:b]."""
    ids = list(map(int, ids[:b]))
    cnt = {}
    for s in range(len(ids) - n + 1):
        g = tuple(ids[s:s + n]); cnt[g] = cnt.get(g, 0) + 1
    cov = np.zeros(len(ids), bool)
    for s in range(len(ids) - n + 1):
        if cnt[tuple(ids[s:s + n])] >= 2:
            cov[s:s + n] = True
    return float(cov[a:b].mean())


def repetition_profile(W, start=P_LEN):
    W = [list(map(int, w)) for w in W]
    ons = [loop_onset(w, start) for w in W]
    lr = [longest_repeated_ngram(w[start:]) for w in W]
    runs = [longest_run(w[start:]) for w in W]
    halves = {}
    for h, (a, b) in HALVES.items():
        R = [w[a:b] for w in W]
        halves[h] = dict(distinct1=float(np.mean([distinct_n(r, 1) for r in R])),
                         distinct2=float(np.mean([distinct_n(r, 2) for r in R])),
                         repetition_rate=float(1 - np.mean([distinct_n(r, 1) for r in R])),
                         repeat_coverage_8gram=float(np.mean([repeat_coverage(w, a, b) for w in W])))
    on = [x for x in ons if x is not None]
    return dict(loop_onset_offset=ons, n_with_loop=len(on), loop_onset_median=float(np.median(on)) if on else None,
                continuation_longest_repeated_ngram=lr,
                n_repeated_span_gt_64=int(sum(x > LONG_SPAN for x in lr)),
                continuation_longest_identical_run=runs, n_single_token_loops_ge_16=int(sum(x >= LONG_RUN for x in runs)),
                repeat_coverage_8gram=[repeat_coverage(w, start) for w in W],
                repeat_coverage_8gram_mean=float(np.mean([repeat_coverage(w, start) for w in W])), halves=halves)


def repetition_by_position(ids, start=P_LEN, n=COVER_NGRAM, bins=14):
    """Per 32-token bin of the continuation: fraction of positions covered by repeated 8-grams (Fig. 2)."""
    ids = list(map(int, ids))
    cnt = {}
    for s in range(len(ids) - n + 1):
        g = tuple(ids[s:s + n]); cnt[g] = cnt.get(g, 0) + 1
    cov = np.zeros(len(ids), bool)
    for s in range(len(ids) - n + 1):
        if cnt[tuple(ids[s:s + n])] >= 2:
            cov[s:s + n] = True
    c = cov[start:]
    return [float(x.mean()) for x in np.array_split(c, bins)]


def boundary_decomposition(W, n=2, p_len=P_LEN):
    """Distinct n-grams of the full sequence split into prefix-internal, continuation-internal and
    cross-boundary n-grams (per sequence, averaged)."""
    rows = []
    for w in W:
        w = list(map(int, w))
        g = lambda a, b: {tuple(w[i:i + n]) for i in range(a, b - n + 1)}
        pre, con = g(0, p_len), g(p_len, len(w))
        cross = {tuple(w[i:i + n]) for i in range(p_len - n + 1, p_len)}
        full = g(0, len(w))
        rows.append(dict(full=len(full), prefix=len(pre), continuation=len(con), cross_only=len(cross - pre - con),
                         shared_prefix_continuation=len(pre & con), total_ngrams=len(w) - n + 1))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ------------------------------------------------------------------ exposure decomposition
def region_counts(sel, a, b, E=60):
    """sel [n, T, L, k] (B1 routing) -> set counts [L, E] over token positions [a, b)."""
    s = np.asarray(sel)[:, a:b]
    n, T, L, k = s.shape
    out = np.zeros((L, E), np.int64)
    for l in range(L):
        out[l] = np.bincount(s[:, :, l, :].reshape(-1).astype(np.int64), minlength=E)
    return out


def rate_stats(counts, n_tokens, k=4):
    """Length-normalised per-layer exposure: relative load r_e = count_e / (n_tokens * k / E)."""
    c = np.asarray(counts, float)
    L, E = c.shape
    r = c / (n_tokens * k / E)
    cv = r.std(1) / r.mean(1)
    return dict(n_tokens=int(n_tokens), cv_mean=float(cv.mean()), rel_min_mean=float(r.min(1).mean()),
                rel_p10_mean=float(np.quantile(r, .1, axis=1).mean()), n_zero_mean=float((c == 0).sum(1).mean()),
                cv_per_layer=cv.tolist())


# ------------------------------------------------------------------ frozen decisions
def diversity_label(cont, hist_cont, full, hist_full, d2=DIV_D2, rep=DIV_REP):
    """cont / hist_cont / full / hist_full: {method: per-sequence-mean dict} for the continuation [64:512] of the
    prefix condition, the same slice of the historical random-start set, and the full [0:512] sequences."""
    per = {}
    for m in cont:
        dd2 = cont[m]["distinct2"] - hist_cont[m]["distinct2"]
        drep = cont[m]["repetition_rate"] - hist_cont[m]["repetition_rate"]
        fd2 = full[m]["distinct2"] - hist_full[m]["distinct2"]
        per[m] = dict(d_cont_distinct2=dd2, d_cont_repetition=drep, d_full_distinct2=fd2,
                      cont_distinct2_pass=dd2 >= d2, cont_repetition_pass=drep <= -rep,
                      continuation_improved=bool(dd2 >= d2 and drep <= -rep), full_improved=bool(fd2 >= d2))
    allc = all(v["continuation_improved"] for v in per.values())
    anyc = any(v["cont_distinct2_pass"] or v["cont_repetition_pass"] for v in per.values())
    anyf = any(v["full_improved"] for v in per.values())
    label = ("CONTINUATION-DIVERSITY-IMPROVED" if allc else
             "PREFIX-CONTENT-ONLY" if anyf else "NO-DIVERSITY-IMPROVEMENT")
    return dict(label=label, mixed=bool(anyc and not allc), per_method=per,
                rule=dict(CONTINUATION_DIVERSITY_IMPROVED=f"both PO and EBSS: continuation distinct-2 >= +{d2} AND "
                          f"continuation repetition rate <= -{rep} vs the same [64:512] slice of their random-start sets",
                          PREFIX_CONTENT_ONLY=f"not the above, but full-sequence distinct-2 >= +{d2} for some method",
                          NO_DIVERSITY_IMPROVEMENT="neither", mixed="some but not all continuation criteria met"))


def primary_label(recon, valid=True, margin=QUAL_MARGIN_PCT):
    """recon: dict with C4R_s0..2, RS_PO, RS_EBSS, PX_PO, PX_EBSS (mean held-out reconstruction)."""
    if not valid or any(recon.get(k) is None for k in ("C4R_s0", "C4R_s1", "C4R_s2", "RS_PO", "RS_EBSS", "PX_PO", "PX_EBSS")):
        return dict(label="PREFIX-RESCUE-RUN-BLOCKED")
    c4 = [recon[f"C4R_s{s}"] for s in range(3)]
    beats = recon["PX_EBSS"] < min(c4)
    imp = {m: recon[f"PX_{m}"] < recon[f"RS_{m}"] for m in ("PO", "EBSS")}
    rel = {m: (recon[f"PX_{m}"] / recon[f"RS_{m}"] - 1) * 100 for m in ("PO", "EBSS")}
    label = ("PREFIX-RESCUE-BEATS-C4" if beats else "PREFIX-RESCUE-PARTIAL" if any(imp.values())
             else "PREFIX-RESCUE-NO-GAIN")
    return dict(label=label, ebss_below_all_c4=beats, po_below_all_c4=recon["PX_PO"] < min(c4),
                improved_vs_random_start=imp, rel_vs_random_start_pct=rel,
                material_vs_random_start={m: rel[m] <= -margin for m in rel})


def gap_recovery(e_old, e_new, e_c4_mean):
    d = e_old - e_c4_mean
    return float((e_old - e_new) / d) if d != 0 else None
