"""Real-model check of the router-profile top-k fix (pool sequences only).

For each layer and sequence, records three per-token expert selections from
the SAME forward pass:
  model : the topk(k) call inside Qwen2MoeSparseMoeBlock.forward (actual routing)
  new   : the fixed Profiler hook (topk(k) on softmax(dtype=float))
  old   : the previous hook rule, topk(k+1)[:k]
and compares per-expert counts against the tokens each expert module actually
received (Hessian accumulator). Output: runs_diag/routing_fix_check.json
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moe_calib import model_utils as MU                    # noqa: E402
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids  # noqa: E402
from moe_calib.profiler import Profiler                    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--n-seqs", type=int, default=32)
    a = ap.parse_args(argv)
    meta = load_meta(a.cache_dir)
    ids = split_ids(meta)["pool"][:a.n_seqs]
    model, _ = MU.load_model()
    E, k = MU.num_experts(model), MU.top_k(model)

    calls = []
    orig = torch.topk

    def rec(x, kk, *args, **kw):
        r = orig(x, kk, *args, **kw)
        if x.dim() == 2 and x.shape[-1] == E:
            calls.append((kk, r.indices.detach().clone()))
        return r

    out = dict(n_seqs=len(ids), top_k=k, layers={})
    torch.topk = rec
    try:
        for L in a.layers:
            cache = load_layer_cache(a.cache_dir, L)
            P = Profiler(model, L, K=1)
            sel = {"model": [], "new": [], "old": []}
            orig_run = MU.run_layer

            def run_layer(model_, L_, hs):
                calls.clear()
                y = orig_run(model_, L_, hs)
                # hook: topk(k) [new], topk(k+1) [margin / old rule]; then the model's topk(k)
                ks = [c[0] for c in calls]
                assert ks == [k, k + 1, k], ks
                sel["new"].append(calls[0][1])
                sel["old"].append(calls[1][1][:, :k])
                sel["model"].append(calls[2][1])
                return y

            MU.run_layer = run_layer
            try:
                P.run(cache, ids)
            finally:
                MU.run_layer = orig_run
            S = {n: torch.cat(v) for n, v in sel.items()}
            T = S["model"].shape[0]
            setdiff = lambda A, B: float((A.sort(-1).values != B.sort(-1).values).any(-1).float().mean())
            cnt = lambda A: torch.bincount(A.flatten().cpu(), minlength=E).numpy().astype(float)
            acc = np.array([P.acc.tokens((L, e, "gu")) for e in range(E)])
            hook_after = np.array([sum(P.seq_counts[s].get((L, e), 0) for s in ids) for e in range(E)], float)
            old_cnt = cnt(S["old"])
            r = dict(tokens=T,
                     frac_tokens_old_vs_model=setdiff(S["old"], S["model"]),
                     frac_tokens_new_vs_model=setdiff(S["new"], S["model"]),
                     before=dict(max_abs_count_diff=float(np.abs(old_cnt - acc).max()),
                                 mean_abs_count_diff=float(np.abs(old_cnt - acc).mean()),
                                 mean_abs_rel_diff=float(np.mean(np.abs(old_cnt - acc) / np.maximum(acc, 1)))),
                     after=dict(max_abs_count_diff=float(np.abs(hook_after - acc).max()),
                                mean_abs_count_diff=float(np.abs(hook_after - acc).mean()),
                                model_topk_counts_equal_acc=bool((cnt(S["model"]) == acc).all())))
            out["layers"][str(L)] = r
            print(f"[routing] L{L}: tokens={T} old!=model {r['frac_tokens_old_vs_model']:.4f} "
                  f"new!=model {r['frac_tokens_new_vs_model']:.4f} | before max/mean |dcount| "
                  f"{r['before']['max_abs_count_diff']:.0f}/{r['before']['mean_abs_count_diff']:.2f} "
                  f"after {r['after']['max_abs_count_diff']:.0f}/{r['after']['mean_abs_count_diff']:.2f}",
                  flush=True)
            P.free()
            del cache
    finally:
        torch.topk = orig
    os.makedirs("runs_diag", exist_ok=True)
    json.dump(out, open("runs_diag/routing_fix_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
