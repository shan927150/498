"""Data: C4 candidate pool + held-out split, activation cache, WikiText-2 PPL.

Pool and holdout are drawn ONCE (by --data-seed) and cached; experiment seeds
only change *selection*, never the candidate pool. This keeps every strategy
and seed choosing from the identical pool (cleaner comparison) and lets the
expensive full-model forward be done a single time (cache_inputs.py).

Cache layout (cache_dir/):
  ids.pt            {"pool": LongTensor[P,T], "holdout": LongTensor[H,T]}
  L{l}.pt           bf16 tensor [P+H, T, d]: INPUT hidden states of layer l
                    rows 0..P-1 = pool, rows P..P+H-1 = holdout
  meta.json         sizes, seqlen, layers, measured forward cost per sequence
Global sequence id == row index in L{l}.pt.
"""
from __future__ import annotations
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset

from . import model_utils as MU


def get_splits(tokenizer, pool_size: int, holdout_size: int, seqlen: int,
               data_seed: int):
    """One shuffled pass over a C4 shard; first pool_size windows -> pool,
    next holdout_size -> holdout. One window per document => disjoint."""
    ds = load_dataset("allenai/c4", "en",
                      data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                      split="train")
    rng = random.Random(data_seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    need = pool_size + holdout_size
    out = []
    for idx in order:
        if len(out) >= need:
            break
        ids = tokenizer(ds[idx]["text"], return_tensors="pt").input_ids[0]
        if ids.shape[0] <= seqlen:
            continue
        st = rng.randint(0, ids.shape[0] - seqlen - 1)
        out.append(ids[st:st + seqlen])
    return out[:pool_size], out[pool_size:need]


def load_meta(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "meta.json")) as f:
        return json.load(f)


def load_layer_cache(cache_dir: str, layer: int) -> torch.Tensor:
    """Memory-mapped: rows are paged in from disk on access, so a 4.6 GB
    layer cache costs almost no RAM until used."""
    return torch.load(os.path.join(cache_dir, f"L{layer}.pt"), mmap=True)


def pool_and_holdout_ids(meta: dict):
    P, H = meta["pool_size"], meta["holdout_size"]
    return list(range(P)), list(range(P, P + H))


@torch.no_grad()
def wikitext2_ppl(model, tokenizer, seqlen: int = 2048,
                  max_windows: int | None = None) -> float:
    """Layer-wise WikiText-2 PPL (non-overlapping windows, GPTQ protocol).
    Uses whatever weights the CPU model currently holds (quantized or not)."""
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    n = enc.shape[1] // seqlen
    if max_windows:
        n = min(n, max_windows)
    windows = [enc[:, i * seqlen:(i + 1) * seqlen] for i in range(n)]
    hs = [MU.embed(model, w).cpu() for w in windows]
    for L in range(MU.num_layers(model)):
        with MU.layer_on_gpu(model, L):
            hs = [MU.run_layer(model, L, h.to(MU.GPU)).cpu() for h in hs]
    nll, cnt = 0.0, 0
    for w, h in zip(windows, hs):
        logits = model.lm_head(model.model.norm(h.to(MU.GPU))).float()
        lab = w.to(MU.GPU)
        nll += float(F.cross_entropy(logits[0, :-1], lab[0, 1:], reduction="sum"))
        cnt += seqlen - 1
    return math.exp(nll / cnt)
