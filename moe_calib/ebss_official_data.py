"""Official MoEQuant EBSS calibration artifact: loading, audit, frozen window extraction and
calibration diagnostics (pure helpers: no model, no GPU).

The artifact is `code/EBSS_data/EBSS_qwen.jsonl` from https://github.com/chenzx921020/MoEQuant.
It is used AS RELEASED: records are never edited, regenerated, padded, repeated or concatenated.

Frozen extraction rule (EXTRACTION_POLICY), fixed before any quantization quality was measured:
  1. tokenize each record's "text" with the Qwen1.5-MoE-A2.7B tokenizer (add_special_tokens=False,
     which for this tokenizer adds nothing anyway);
  2. a record is ELIGIBLE iff it has >= 512 tokens; its window is tokens [0, 512) (offset 0 = the
     record's own start, as the official loader consumes records from their start);
  3. walk eligible records in FILE ORDER; skip a window that is token-identical to an earlier kept
     window, or that exactly leaks into the frozen evaluation data (any shared 32-gram);
  4. keep the first 128. Fewer than 128 -> DATA-BLOCKED.
No quality score, model statistic or held-out result is used.
"""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np

SEQLEN = 512
B = 128
LEAK_NGRAM = 32          # exact-leakage n-gram (token ids)
NEAR_NGRAM = 13          # near-duplicate / contamination n-gram (reported only)
EXPOSURE_ABS_THRESHOLDS = (32, 128, 512)          # routed calibration tokens per expert
EXPOSURE_REL_THRESHOLDS = (0.1, 0.25, 0.5)        # x the uniform mean B*T*k/E
TAU_REL_GAIN = 0.5                                # % mean relative reconstruction improvement

EXTRACTION_POLICY = dict(
    unit="one official JSONL record = one independent parent example",
    tokenizer="Qwen/Qwen1.5-MoE-A2.7B (snapshot 1a758c50...), add_special_tokens=False",
    eligible="records with >= 512 tokens",
    window="tokens [0, 512) of the record (offset 0); no padding, no repetition, no concatenation, "
           "never crossing a record boundary",
    order="file order (line number)",
    exclusions=["a window token-identical to an earlier kept window (exact duplicate)",
                f"a window sharing any {LEAK_NGRAM}-gram with the frozen held-out C4 rows (tune + test, "
                "2000-2255) or the 128 frozen WikiText-2 evaluation windows (exact leakage)"],
    budget=f"first {B} kept windows; fewer -> DATA-BLOCKED",
    subdivision="one EBSS set only; no further subsets are defined",
    not_used="no model score, likelihood, routing statistic, typicality, reconstruction or PPL is used",
)


# ---------------------------------------------------------------- loading
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_jsonl(raw: bytes):
    """Strict parse: every non-empty line must be a JSON object with a string "text"."""
    recs, bad = [], []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            bad.append((i, "empty line"))
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError as e:
            bad.append((i, f"json: {e}"))
            continue
        if not isinstance(o, dict) or not isinstance(o.get("text"), str):
            bad.append((i, "no string 'text'"))
            continue
        recs.append(o)
    return recs, bad


def window_hash(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def ids_hash(seq_of_windows) -> str:
    h = hashlib.sha256()
    for w in seq_of_windows:
        h.update(np.asarray(w, dtype=np.int64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- n-grams / leakage
def ngram_hashes(ids, n):
    """Set of n-gram fingerprints (python hash of the id tuple; collisions irrelevant at this scale,
    exact matches are re-verified by `exact_ngram_overlap`)."""
    ids = list(map(int, ids))
    return {hash(tuple(ids[i:i + n])) for i in range(len(ids) - n + 1)}


def build_ngram_index(windows, n):
    idx = set()
    for w in windows:
        idx |= ngram_hashes(w, n)
    return idx


def shares_ngram(ids, index, n) -> bool:
    ids = list(map(int, ids))
    return any(hash(tuple(ids[i:i + n])) in index for i in range(len(ids) - n + 1))


def exact_ngram_overlap(a, b, n) -> bool:
    """Verified (collision-free) check that token sequences a and b share an n-gram."""
    sa = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    return any(tuple(b[i:i + n]) in sa for i in range(len(b) - n + 1))


# ---------------------------------------------------------------- frozen extraction
def extract_windows(token_lists, leak_index=None, seqlen=SEQLEN, budget=B, leak_n=LEAK_NGRAM):
    """Apply EXTRACTION_POLICY. token_lists: per record (file order) token ids.
    Returns dict(selected=[(record_id, offset)], windows=[ids], status, skipped=...)."""
    kept, wins, seen = [], [], set()
    skipped_dup, skipped_leak, short = [], [], []
    for rid, ids in enumerate(token_lists):
        if len(ids) < seqlen:
            short.append(rid)
            continue
        w = [int(x) for x in ids[:seqlen]]
        h = window_hash(w)
        if h in seen:
            skipped_dup.append(rid)
            continue
        if leak_index is not None and shares_ngram(w, leak_index, leak_n):
            skipped_leak.append(rid)
            continue
        seen.add(h)
        kept.append((rid, 0))
        wins.append(w)
        if len(kept) == budget:
            break
    status = "OK" if len(kept) == budget else "DATA-BLOCKED"
    return dict(selected=kept, windows=wins, status=status, n_short=len(short),
                skipped_duplicate=skipped_dup, skipped_leakage=skipped_leak,
                last_record_used=kept[-1][0] if kept else None)


# ---------------------------------------------------------------- text diagnostics
def distinct_n(ids, n):
    ids = list(map(int, ids))
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def set_distinct_n(windows, n):
    """Corpus-level distinct-n over a set of windows (n-grams never cross windows)."""
    tot, uniq = 0, set()
    for w in windows:
        w = list(map(int, w))
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        tot += len(g)
        uniq |= set(g)
    return len(uniq) / max(tot, 1)


def longest_run(ids):
    best = cur = 1
    for a, b in zip(ids[:-1], ids[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def zlib_ratio(ids):
    b = np.asarray(ids, dtype=np.int32).tobytes()
    return len(zlib.compress(b, 9)) / len(b)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def near_duplicate_stats(windows, n=NEAR_NGRAM, thresholds=(0.5, 0.8)):
    """Pairwise Jaccard of n-gram sets inside one set of windows."""
    S = [set(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) for w in map(list, windows)]
    m = len(S)
    J = []
    for i in range(m):
        for j in range(i + 1, m):
            J.append(jaccard(S[i], S[j]))
    J = np.array(J) if J else np.zeros(1)
    return dict(n_pairs=int(m * (m - 1) // 2), max_jaccard=float(J.max()), mean_jaccard=float(J.mean()),
                **{f"pairs_jaccard_ge_{t}": int((J >= t).sum()) for t in thresholds},
                **{f"windows_in_a_pair_ge_{t}": int(sum(
                    any(jaccard(S[i], S[j]) >= t for j in range(m) if j != i) for i in range(m))) for t in thresholds})


def text_diagnostics(windows):
    W = [list(map(int, w)) for w in windows]
    d1 = np.array([distinct_n(w, 1) for w in W])
    d2 = np.array([distinct_n(w, 2) for w in W])
    runs = np.array([longest_run(w) for w in W])
    zr = np.array([zlib_ratio(w) for w in W])
    hashes = [window_hash(w) for w in W]
    q = lambda a: {k: float(np.quantile(a, v)) for k, v in (("min", 0), ("p10", .1), ("median", .5),
                                                             ("p90", .9), ("max", 1))}
    return dict(n_windows=len(W), exact_duplicate_windows=len(W) - len(set(hashes)),
                distinct1_mean=float(d1.mean()), distinct2_mean=float(d2.mean()),
                distinct1_quantiles=q(d1), distinct2_quantiles=q(d2),
                frac_windows_distinct1_below_0_1=float((d1 < 0.1).mean()),
                set_distinct1=set_distinct_n(W, 1), set_distinct2=set_distinct_n(W, 2),
                n_unique_token_types=len(set(t for w in W for t in w)),
                repetition_rate_mean=float(1 - d1.mean()),
                longest_identical_token_run=q(runs), zlib_ratio=q(zr))


# ---------------------------------------------------------------- exposure / routing
def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def exposure_stats(counts, uniform_mean=None):
    """counts: routed calibration tokens per expert of ONE layer (E,)."""
    c = np.asarray(counts, float)
    mu = float(c.mean()) if uniform_mean is None else float(uniform_mean)
    out = dict(mean=float(c.mean()), min=float(c.min()), p5=float(np.quantile(c, .05)),
               p10=float(np.quantile(c, .10)), max=float(c.max()), sd=float(c.std()),
               cv=float(c.std() / c.mean()) if c.mean() > 0 else float("nan"), gini=gini(c),
               n_zero=int((c == 0).sum()))
    for t in EXPOSURE_ABS_THRESHOLDS:
        out[f"n_below_{t}"] = int((c < t).sum())
    for r in EXPOSURE_REL_THRESHOLDS:
        out[f"n_below_{r}x_uniform"] = int((c < r * mu).sum())
    return out


def routing_l1(counts, ref):
    c = np.asarray(counts, float)
    return float(np.abs(c / c.sum() - np.asarray(ref, float)).sum())


# ---------------------------------------------------------------- comparisons / decision
def relative_penalty(err, base):
    """(Error_EBSS / Error_Baseline - 1) x 100 %. Negative = EBSS better."""
    return (float(err) / float(base) - 1.0) * 100.0


PROVENANCE_CLASSES = ("MATCHED-DOCUMENTED", "DIFFERENT-CHECKPOINT", "UNDOCUMENTED")

DECISION_RULE = {
    "precedence": "1 data availability -> 2 numerical / quantization correctness -> 3 source-checkpoint "
                  "provenance -> 4 reconstruction and PPL criteria",
    "DATA-BLOCKED": "the official file cannot be acquired / parsed, or the frozen extraction policy yields "
                    "fewer than 128 valid 512-token windows",
    "RUN-BLOCKED": "a primary configuration (FR/TR/FC s0-2, EBSS) is missing or invalid (install check, untouched "
                   "parameters, non-finite or unchanged projections, repeat-evaluation PPL difference != 0, RTN "
                   "fallback in a C4 baseline), the quantization-pipeline regression (FR s0 on the historical "
                   "cache) does not reproduce the partial-PPL weights/PPL/reconstruction exactly, the extraction "
                   "implementation regression or the batch-1 repeatability control fails, or the BF16 PPL differs "
                   "from 9.237090",
    "OFFICIAL-EBSS-TRANSFER-LIMITED": "provenance class != MATCHED-DOCUMENTED (generating checkpoint differs or "
                                      "cannot be established). Takes precedence over the quality classes; the "
                                      "quality class that would otherwise apply is still reported",
    "quality criteria": {
        "R (reconstruction gain)": "EBSS mean expert reconstruction < FR_s and < TR_s for s = 0, 1, 2 AND "
                                   "mean_s RelativePenalty(EBSS, FR_s) <= -0.5 % AND mean_s RelativePenalty(EBSS, "
                                   "TR_s) <= -0.5 %",
        "P (PPL gain)": "EBSS PPL < FR_s and < TR_s PPL for s = 0, 1, 2 (matched protocol)",
        "OFFICIAL-EBSS-QUALITY-GAIN": "R and P",
        "OFFICIAL-EBSS-RECONSTRUCTION-GAIN-ONLY": "R and not P",
        "OFFICIAL-EBSS-PPL-GAIN-ONLY": "P and not R",
        "OFFICIAL-EBSS-NO-CONSISTENT-GAIN": "neither",
    },
    "notes": "FC is a negative control and is not part of R or P. One EBSS set: no EBSS SD is computed; experts "
             "are not treated as independent replications.",
}


def quality_class(ebss_recon, fr_recon, tr_recon, ebss_ppl, fr_ppl, tr_ppl, tau=TAU_REL_GAIN):
    """fr_*/tr_*: lists over seeds 0,1,2."""
    rp_fr = [relative_penalty(ebss_recon, b) for b in fr_recon]
    rp_tr = [relative_penalty(ebss_recon, b) for b in tr_recon]
    R = (all(ebss_recon < b for b in fr_recon) and all(ebss_recon < b for b in tr_recon)
         and np.mean(rp_fr) <= -tau and np.mean(rp_tr) <= -tau)
    P = all(ebss_ppl < b for b in fr_ppl) and all(ebss_ppl < b for b in tr_ppl)
    cls = ("OFFICIAL-EBSS-QUALITY-GAIN" if R and P else "OFFICIAL-EBSS-RECONSTRUCTION-GAIN-ONLY" if R
           else "OFFICIAL-EBSS-PPL-GAIN-ONLY" if P else "OFFICIAL-EBSS-NO-CONSISTENT-GAIN")
    return dict(R=bool(R), P=bool(P), quality_class=cls, mean_rel_penalty_vs_FR=float(np.mean(rp_fr)),
                mean_rel_penalty_vs_TR=float(np.mean(rp_tr)), rel_penalty_vs_FR=rp_fr, rel_penalty_vs_TR=rp_tr)


def decide(data_ok, correctness_ok, provenance, q):
    if not data_ok:
        return "DATA-BLOCKED"
    if not correctness_ok:
        return "RUN-BLOCKED"
    assert provenance in PROVENANCE_CLASSES
    if provenance != "MATCHED-DOCUMENTED":
        return "OFFICIAL-EBSS-TRANSFER-LIMITED"
    return q["quality_class"]


# ---------------------------------------------------------------- timing
def measured_time(t_tok, t_act, t_hess, t_gptq):
    """T_measured = T_tokenization + T_activation + T_Hessian + T_GPTQ (each counted once; generation
    cost of the pre-generated EBSS data is NOT included because it is unknown)."""
    parts = dict(tokenization_s=t_tok, activation_s=t_act, hessian_s=t_hess, gptq_s=t_gptq)
    for k, v in parts.items():
        assert v is not None and v >= 0, f"{k} missing"
    return dict(parts, T_measured_s=float(sum(parts.values())))
