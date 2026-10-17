"""Step 0 — build the layer-input activation cache (run ONCE, ~30-60 min).

Draws the C4 candidate pool + held-out set, runs the full model layer by
layer (one layer on the GPU at a time) and saves the INPUT hidden states of
each profiled layer to disk. E1/E2/E3 then only ever replay single layers on
these cached inputs, so they fit in a 20 GB MIG slice and never repeat the
expensive full-model forward.

Also records the measured full-forward cost per sequence in meta.json; E3
uses it to charge an honest routing-profile cost to selection strategies
(the "Total cost?" question).

Disk: (P+H) * T * 2048 * 2 bytes per layer = 4.6 GB/layer at the defaults.
"""
from __future__ import annotations
import argparse, json, os, time

import torch

from moe_calib import model_utils as MU
from moe_calib.data_utils import get_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--holdout-size", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    model, tok = MU.load_model()
    pool, holdout = get_splits(tok, args.pool_size, args.holdout_size,
                               args.seqlen, args.data_seed)
    assert len(pool) == args.pool_size and len(holdout) == args.holdout_size
    ids = torch.stack(pool + holdout)                       # [N, T]
    torch.save({"pool": ids[:len(pool)], "holdout": ids[len(pool):]},
               os.path.join(args.cache_dir, "ids.pt"))
    N, T = ids.shape
    B = args.batch
    print(f"[cache] {N} seqs x {T} tokens; layers {args.layers}")

    t0 = time.time()
    hs = torch.empty(N, T, MU.expert_dims(model)[0], dtype=MU.DTYPE)
    for i in range(0, N, B):
        hs[i:i + B] = MU.embed(model, ids[i:i + B]).cpu()
    last = max(args.layers)
    for L in range(last + 1):
        if L in args.layers:
            path = os.path.join(args.cache_dir, f"L{L}.pt")
            torch.save(hs, path)
            print(f"[cache] saved {path} ({hs.numel() * 2 / 1e9:.1f} GB)")
        if L == last:
            break
        tl = time.time()
        with MU.layer_on_gpu(model, L):
            for i in range(0, N, B):
                hs[i:i + B] = MU.run_layer(model, L, hs[i:i + B].to(MU.GPU)).cpu()
        print(f"[cache] layer {L:2d} forward {time.time() - tl:.1f}s")
    elapsed = time.time() - t0
    n_run = last  # layers actually executed
    per_layer_seq = elapsed / max(n_run, 1) / N
    meta = dict(pool_size=len(pool), holdout_size=len(holdout), seqlen=T,
                data_seed=args.data_seed, layers=args.layers,
                model=MU.MODEL_NAME, num_layers=MU.num_layers(model),
                cache_seconds=elapsed,
                # cost of ONE full-model forward of one sequence on this slice
                forward_s_per_seq=per_layer_seq * MU.num_layers(model))
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
