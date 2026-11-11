"""Helpers, gates and the frozen decision rule of the model-matched EBSS generation pilot
(runs_ebss_generation_pilot/). Pure functions only (CPU-testable)."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import ebss_official_data as ED

MODEL = "Qwen/Qwen1.5-MoE-A2.7B"
SNAPSHOT = "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
OUT = "runs_ebss_generation_pilot"
CACHE = os.path.expanduser(os.environ.get("EBSS_PILOT_CACHE", "~/moe_cache_ebss_generation_pilot"))
N_PILOT, N_SMOKE = 16, 4
SMOKE_LEN = 32
PROGRESS_POSITIONS = (32, 64, 128, 256, 512)
TAU_REL_GAIN = 0.5                     # % (same margin as the official-artifact study)

DECISION_RULE = {
    "precedence": "RUN-BLOCKED -> QUALITY-GAIN (only if the optional B=16 GPTQ comparison completed validly) -> "
                  "balancing classes",
    "RUN-BLOCKED": "a generator correctness gate fails, the 16 x 512 pilot does not complete for BOTH methods, a "
                   "generated set is invalid (not 16 distinct full 512-token sequences of real ids), or resources "
                   "prevent the measurement",
    "MODEL-MATCHED-EBSS-QUALITY-GAIN": "B=16 GPTQ valid for all configs AND EBSS mean reconstruction is lower than "
                                       "Probability-Only by >= 0.5 % AND lower than every C4 Random B=16 seed AND "
                                       ">= 0.5 % lower than the C4 Random seed mean",
    "MODEL-MATCHED-EBSS-BALANCING-SUPPORTED": "EBSS set-level paper sigma (Eq. 3/4, all 24 MoE layers) AND mean "
                                              "per-layer load CV (all 24 layers) are both lower than Probability-Only",
    "MODEL-MATCHED-EBSS-BALANCING-NOT-SUPPORTED": "neither is lower (EBSS equal or worse on both)",
    "MODEL-MATCHED-EBSS-MIXED": "exactly one of the two balance metrics is lower",
    "notes": "set-level metrics over the 16 x 512 = 8,192 tokens of each set; lower-tail exposure (min / p5 / p10 / "
             "zero-exposure experts), per-layer results and layers 1/8/16/23 are reported separately and do not "
             "enter the label. With exactly k = 4 routed experts per token the per-layer mean count is fixed, so "
             "sigma_l and CV_l are proportional within a layer; the two means can still differ only through "
             "layer weighting (ddof differs). One seed, one start-token draw: no statistical generality.",
}


def decide(pilot_complete, gates_ok, d_sigma, d_cv, gptq):
    """d_* = EBSS - Probability-Only. gptq = None or dict(valid, ebss, po, c4=[3 seeds])."""
    if not (pilot_complete and gates_ok):
        return "RUN-BLOCKED"
    if gptq is not None and gptq.get("valid"):
        e, po, c4 = gptq["ebss"], gptq["po"], list(gptq["c4"])
        rel = lambda a, b: (a / b - 1) * 100
        if (rel(e, po) <= -TAU_REL_GAIN and all(e < c for c in c4)
                and rel(e, float(np.mean(c4))) <= -TAU_REL_GAIN):
            return "MODEL-MATCHED-EBSS-QUALITY-GAIN"
    a, b = d_sigma < 0, d_cv < 0
    if a and b:
        return "MODEL-MATCHED-EBSS-BALANCING-SUPPORTED"
    if not a and not b:
        return "MODEL-MATCHED-EBSS-BALANCING-NOT-SUPPORTED"
    return "MODEL-MATCHED-EBSS-MIXED"


SMOKE_CHECKS = ("incremental_equals_recount", "deterministic", "logprob_identity", "ranking_recomputed",
                "po_ignores_balance", "ebss_includes_balance", "length_exact", "token_ids_valid",
                "no_chat_template", "replay_logprob_close", "parents_intact")


def smoke_gate(checks):
    failed = [k for k in SMOKE_CHECKS if not checks.get(k)]
    if checks.get("dispatch_mismatches", 1) != 0:
        failed.append("dispatch_mismatches")
    return dict(passed=not failed, failed=failed)


def token_accounting(n_sentences, seqlen, w, vocab):
    """Denominators. emitted = final tokens kept; forward_rows = branch-token positions run through the model
    (1 start row per sentence, then w rows per step; the last step's forward only yields final routing);
    scored_candidates = all S||v expansions scored."""
    steps = seqlen - 1
    return dict(emitted_tokens=n_sentences * seqlen, generated_tokens=n_sentences * steps,
                forward_rows=n_sentences * (1 + steps * w),
                scored_candidates=n_sentences * (vocab + (steps - 1) * w * vocab),
                forward_calls=1 + steps)


def memory_estimate(rows, seqlen, n_layers, kv_heads, head_dim, dtype_bytes=2):
    kv = 2 * rows * seqlen * n_layers * kv_heads * head_dim * dtype_bytes
    return dict(kv_cache_gb=kv / 1e9)


def validate_generated(tokens, n, seqlen, vocab_limit):
    t = torch.as_tensor(tokens)
    hashes = {ED.window_hash(r.tolist()) for r in t} if t.dim() == 2 else set()
    checks = dict(shape_ok=tuple(t.shape) == (n, seqlen),
                  ids_in_vocab=bool(t.numel() and int(t.min()) >= 0 and int(t.max()) < vocab_limit),
                  distinct_sequences=len(hashes) == t.shape[0] if t.dim() == 2 else False,
                  padding_tokens=0)
    checks["valid"] = all(v for k, v in checks.items() if k != "padding_tokens")
    return checks


def distinct3(windows):
    return float(np.mean([ED.distinct_n(w, 3) for w in windows]))


def contamination(gen_windows, heldout_windows, n=ED.LEAK_NGRAM):
    idx = ED.build_ngram_index(heldout_windows, n)
    hits = [i for i, w in enumerate(gen_windows) if ED.shares_ngram(w, idx, n)
            and any(ED.exact_ngram_overlap(h, w, n) for h in heldout_windows)]
    return dict(ngram=n, windows_with_32gram_match=len(hits), windows=hits)


def write_jsonl(res, path, decode):
    with open(path + ".tmp", "w") as f:
        for i in range(res.tokens.shape[0]):
            ids = res.tokens[i].tolist()
            f.write(json.dumps(dict(index=i, method=res.method, first_token=ids[0], n_tokens=len(ids),
                                    padding_tokens=0, cum_logprob=float(res.cum_logp[i]),
                                    sigma=float(res.sigma[i]), final_score=float(res.final_score[i]),
                                    token_ids=ids, text=decode(ids)), ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
