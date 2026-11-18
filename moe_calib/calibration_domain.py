"""C4 vs WikiText-2-train calibration-domain ablation: pure helpers (no model, no GPU, no I/O beyond arguments).

Everything here re-states an EXISTING project convention so the WikiText-train arm is built exactly like the C4 arm:
  * windows:      eval_partial_ppl.stage_protocol packing ("\\n\\n".join(text), one tokenizer call, non-overlapping
                  512-token windows i*512..(i+1)*512, tail discarded) -- applied to the TRAIN split;
  * warm start:   diagnostics/typicality_pools.py + typicality_warm_starts.py
                  (typical_half = 1000 lowest routing-L1 d(s); random_half = random.Random(20261106).sample(ids, 1000);
                   W_s = random.Random(s).sample(sorted(typical_half & random_half), 32));
  * FR:           e3_allocation.main --strategy random with a fixed warm start
                  (rng = random.Random(seed); W + rng.sample([pool ids not in W], B - |W|));
  * groupings:    moe_calib.multigroup_robustness.grouping_batches (sorted union -> default_rng(g).permutation -> 8s).
The unit tests check that the warm-start and FR rules reproduce the stored C4 selections bit for bit.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

from . import typicality as TY

TYPICAL_HALF = 1000
RANDOM_HALF_SEED = 20261106          # runs_typicality/pools.json pool_seed
WARM = 32


# ---------------------------------------------------------------- windows / pool
def partition_windows(n_tokens: int, seqlen: int) -> int:
    """Number of complete non-overlapping windows (tail discarded)."""
    return n_tokens // seqlen


def sample_pool(n_windows: int, pool_size: int, seed: int) -> list[int]:
    """Window indices of the candidate pool: sorted(random.Random(seed).sample(range(n_windows), pool_size)).
    Raises (never silently shrinks) if fewer windows exist."""
    if n_windows < pool_size:
        raise ValueError(f"only {n_windows} complete windows < pool size {pool_size}")
    return sorted(random.Random(seed).sample(range(n_windows), pool_size))


def windows_from_ids(enc, idx, seqlen):
    """enc: 1-D token tensor/array; returns stacked [len(idx), seqlen] windows (torch or numpy, same type)."""
    import torch
    if torch.is_tensor(enc):
        return torch.stack([enc[i * seqlen:(i + 1) * seqlen] for i in idx])
    return np.stack([enc[i * seqlen:(i + 1) * seqlen] for i in idx])


def token_sha(arr) -> str:
    """sha256 of int64 token ids (same convention as eval_partial_ppl.tensor_fp)."""
    a = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
    return hashlib.sha256(np.ascontiguousarray(a).astype(np.int64).tobytes()).hexdigest()


def ngram_overlap(pool_tokens, test_tokens, n: int = 32) -> dict:
    """Leakage check between two sets of windows ([N, T] int arrays). Exact-window matches, and the fraction of test
    n-grams (stride 1, within windows) that also occur in the pool."""
    P = np.asarray(pool_tokens, dtype=np.int64)
    Q = np.asarray(test_tokens, dtype=np.int64)
    pw = {r.tobytes() for r in P}
    exact = sum(r.tobytes() in pw for r in Q)

    def grams(X):
        out = set()
        for r in X:
            v = np.lib.stride_tricks.sliding_window_view(r, n)
            out.update(hash(x.tobytes()) for x in v)
        return out
    pg = grams(P)
    tg_list = [hash(x.tobytes()) for r in Q for x in np.lib.stride_tricks.sliding_window_view(r, n)]
    hit = sum(h in pg for h in tg_list)
    return dict(n=n, exact_window_matches=int(exact), test_ngrams=len(tg_list), test_ngrams_in_pool=int(hit),
                fraction=hit / max(len(tg_list), 1))


# ---------------------------------------------------------------- warm start / FR (existing conventions)
def typicality_halves(C, ids, n_layers: int, E: int, half: int = TYPICAL_HALF, pool_seed: int = RANDOM_HALF_SEED):
    ref = TY.pool_reference(C, n_layers, E)
    sc = TY.sequence_scores(C, n_layers, E, ref)
    return TY.typical_half(ids, sc["d"], half), TY.random_half(ids, half, pool_seed), sc


def warm_start(typical, rand_half, seed: int, warm: int = WARM) -> dict:
    elig = sorted(set(typical) & set(rand_half))
    if len(elig) < warm:
        raise ValueError(f"typical & random half has {len(elig)} < {warm} sequences")
    ids = random.Random(seed).sample(elig, warm)
    return dict(ids=ids, eligible=elig)


def fr_selection(pool_ids, warm_ids, B: int, seed: int) -> list[int]:
    ws = set(warm_ids)
    rng = random.Random(seed)
    return list(warm_ids) + rng.sample([s for s in pool_ids if s not in ws], B - len(warm_ids))


def validate_selection(chosen, warm_ids, B: int, pool_ids) -> dict:
    ps = set(pool_ids)
    return dict(n_unique_B=len(chosen) == B == len(set(chosen)), warm_prefix=list(chosen[:len(warm_ids)]) == list(warm_ids),
                in_pool=all(x in ps for x in chosen))


# ---------------------------------------------------------------- effects
def effects(nll: dict) -> dict:
    """nll: {(source, strategy): value} for source in {C4, WT}, strategy in {FR, SC}. Negative = first term better."""
    sc_c4 = nll[("C4", "SC")] - nll[("C4", "FR")]
    sc_wt = nll[("WT", "SC")] - nll[("WT", "FR")]
    return dict(delta_SC_C4=sc_c4, delta_SC_WT=sc_wt,
                delta_domain_FR=nll[("WT", "FR")] - nll[("C4", "FR")],
                delta_domain_SC=nll[("WT", "SC")] - nll[("C4", "SC")],
                interaction=sc_wt - sc_c4)


def summarize(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                min=float(v.min()), max=float(v.max()), n_negative=int((v < 0).sum()), n_positive=int((v > 0).sum()),
                values=[float(x) for x in v])


def label(summary: dict, ref_sd: float) -> str:
    """Prespecified (protocol.json) descriptive label for a grouped paired difference."""
    s = summary
    same_sign = s["n_negative"] == s["n"] or s["n_positive"] == s["n"]
    if same_sign and abs(s["mean"]) > ref_sd:
        return "CONSISTENT-NEGATIVE" if s["mean"] < 0 else "CONSISTENT-POSITIVE"
    return "NOT-DISTINGUISHABLE"
