"""Exp4 — does an expert's benefit depend on HOW MANY routed tokens it gets,
or on how many DISTINCT sequences they come from?

Per-expert GPTQ (non-sequential) depends only on the expert's own routed
tokens, so "calibration regime" can matter only through the token count and
through their diversity (tokens inside one sequence are strongly correlated:
P0's effective rank stays far below the token count). P2c could not separate
count from regime because count ~ budget. Allocation changes per-expert
counts at a FIXED budget, so this matters for E3.

For each seed (pool ordering), layer and regime m (sequences), on frozen GPTQ:
  natural    all tokens of the first m sequences            (N tokens, m sources)
  spread     N tokens drawn uniformly from 256 sequences     (N tokens, many sources)
  conc       N tokens from the sequences richest in e        (N tokens, fewest sources)
  half_tok   a random N/2 of the natural tokens              (N/2 tokens, m sources)
  half_seq   all tokens of the first m/2 sequences           (~N/2 tokens, m/2 sources)
Score = mean tuning-split relative reconstruction error (E1's metric) and
the effective rank of each Hessian. Contrasts (paired over experts):
  spread vs natural vs conc      : diversity at fixed count
  half_tok vs half_seq           : sources at ~fixed count
  natural vs half_tok            : count at fixed sources

  MOE_LOAD_LAYERS=1,8,16,23 python diagnostics/exposure_diversity.py run
  python diagnostics/exposure_diversity.py report
Resumable per (seed, layer, regime). Tuning split only.
"""
from __future__ import annotations
import argparse, json, os, random, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import model_utils as MU                      # noqa: E402
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids   # noqa: E402
from moe_calib.profiler import Profiler                       # noqa: E402
from moe_calib.quant_eval import per_seq_rel_errors, quantize_from_H     # noqa: E402

CONDS = ("natural", "spread", "conc", "half_tok", "half_seq")
SALT = 9_100_003


def erank(H):
    lam = torch.linalg.eigvalsh(H.float()).clamp(min=0)
    s = lam.sum()
    if s <= 0:
        return 0.0
    p = lam[lam > 0] / s
    return float(torch.exp(-(p * p.log()).sum()))


def expert_rows(acts, L, e):
    """Concatenate an expert's routed activations over the collected sequences.
    Returns (Xgu, Xdown, seq_pos) on CPU; rows of gu and down are the same
    tokens in the same order (both hooks fire once per sequence call)."""
    gu, dn = acts.get((L, e, "gu"), []), acts.get((L, e, "down"), [])
    assert [s for s, _ in gu] == [s for s, _ in dn], "gu/down sequence order differs"
    if not gu:
        return None
    pos = np.concatenate([np.full(x.shape[0], i) for i, (_, x) in enumerate(gu)])
    sids = np.array([s for s, _ in gu])
    return (torch.cat([x for _, x in gu]), torch.cat([x for _, x in dn]), pos, sids)


def conditions(rows, rank, m, rng):
    """Row-index sets for one expert. rank: sequence id -> position in the pool order."""
    Xg, Xd, pos, sids = rows
    order_pos = np.array([rank[s] for s in sids])[pos]          # each row's sequence rank
    nat = np.flatnonzero(order_pos < m)
    N = len(nat)
    if N < 2:
        return None
    allr = np.arange(len(pos))
    per_seq = {}
    for i in allr:
        per_seq.setdefault(order_pos[i], []).append(i)
    richest = sorted(per_seq, key=lambda p: (-len(per_seq[p]), p))
    conc = np.concatenate([per_seq[p] for p in richest])[:N]
    return dict(natural=nat,
                spread=rng.choice(allr, size=min(N, len(allr)), replace=False),
                conc=conc,
                half_tok=rng.choice(nat, size=max(1, N // 2), replace=False),
                half_seq=np.flatnonzero(order_pos < m // 2))


def cmd_run(a):
    meta = load_meta(a.cache_dir)
    sp = split_ids(meta)
    raw = os.path.join(a.out, "exposure_diversity_raw.jsonl")
    os.makedirs(a.out, exist_ok=True)
    done = set()
    if os.path.exists(raw):
        done = {(r["seed"], r["layer"], r["m"]) for r in map(json.loads, open(raw))}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    for seed in a.seeds:
        order = random.Random(SALT + seed).sample(sp["pool"], a.pool_n)
        rank = {s: i for i, s in enumerate(order)}
        for L in a.layers:
            todo = [m for m in a.regimes if (seed, L, m) not in done]
            if not todo:
                continue
            cache = load_layer_cache(a.cache_dir, L)
            HO = Profiler(model, L, collect_hessian=False, collect_holdout=True) \
                .run(cache, sp["tune"]).holdout
            acts = Profiler(model, L, collect_hessian=False, collect_holdout=True,
                            holdout_tokens_per_seq=10 ** 9).run(cache, order).holdout
            for m in todo:
                rng = np.random.default_rng([seed, L, m])
                rec = dict(seed=seed, layer=L, m=m, percdamp=a.percdamp, experts={})
                for e in range(E):
                    rows = expert_rows(acts, L, e)
                    if rows is None:
                        continue
                    sets = conditions(rows, rank, m, rng)
                    if sets is None:
                        continue
                    Xg, Xd, pos, sids = rows
                    out = {}
                    for c, idx in sets.items():
                        if len(idx) == 0:
                            continue
                        it = torch.as_tensor(idx)
                        g = Xg[it].to(MU.GPU).float()
                        dd = Xd[it].to(MU.GPU).float()
                        Hg, Hd = g.T @ g, dd.T @ dd
                        Wq = quantize_from_H(model, L, e, Hg, Hd, a.bits, a.group_size, a.percdamp)
                        errs = per_seq_rel_errors(model, L, e, Wq, HO)
                        src = len(np.unique(pos[idx]))
                        out[c] = dict(err=float(np.mean(list(errs.values()))) if errs else None,
                                      tokens=int(len(idx)), sources=int(src),
                                      erank_gu=erank(Hg) if a.erank else None,
                                      erank_down=erank(Hd) if a.erank else None)
                        del g, dd, Hg, Hd
                    rec["experts"][str(e)] = out
                with open(raw, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                errs = {c: np.nanmean([v[c]["err"] for v in rec["experts"].values()
                                       if c in v and v[c]["err"] is not None]) for c in CONDS}
                print(f"[diversity] seed={seed} L={L} m={m} " +
                      " ".join(f"{c}={errs[c]:.5f}" for c in CONDS), flush=True)
            del cache, HO, acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def paired(recs, a_, b_):
    r, w = [], []
    for rec in recs:
        for v in rec["experts"].values():
            if a_ in v and b_ in v and v[a_]["err"] and v[b_]["err"]:
                r.append(v[a_]["err"] / v[b_]["err"])
    r = np.array(r)
    return (float(np.mean(r)), float(np.mean(r < 1)), len(r)) if len(r) else (np.nan, np.nan, 0)


def cmd_report(a):
    raw = os.path.join(a.out, "exposure_diversity_raw.jsonl")
    recs = [json.loads(l) for l in open(raw)]
    out, M = {}, ["# Exp4 — token count vs source diversity (frozen GPTQ, tuning split)", "",
                  "Ratio = error(condition A) / error(condition B), paired per expert; "
                  "'% A better' = share of experts where A has lower error.", ""]
    for m in sorted({r["m"] for r in recs}):
        R = [r for r in recs if r["m"] == m]
        mean = lambda c, k: float(np.nanmean([v[c][k] for r in R for v in r["experts"].values()
                                              if c in v and v[c][k] is not None]))
        M += [f"## regime m = {m} sequences", "",
              "| condition | tokens | sources | erank gate/up | erank down | mean err |",
              "|---|---|---|---|---|---|"]
        for c in CONDS:
            M.append(f"| {c} | {mean(c, 'tokens'):.0f} | {mean(c, 'sources'):.1f} | "
                     f"{mean(c, 'erank_gu'):.0f} | {mean(c, 'erank_down'):.0f} | {mean(c, 'err'):.5f} |")
        M += ["", "| contrast | ratio A/B | % A better | n |", "|---|---|---|---|"]
        out[m] = {}
        for A, B, what in (("spread", "natural", "diversity, same count (more sources)"),
                           ("conc", "natural", "diversity, same count (fewer sources)"),
                           ("half_tok", "half_seq", "sources at ~same count (m vs m/2)"),
                           ("natural", "half_tok", "count at same sources (N vs N/2)")):
            rt, fb, n = paired(R, A, B)
            out[m][f"{A}/{B}"] = dict(ratio=rt, frac_A_better=fb, n=n)
            M.append(f"| {A} / {B} — {what} | {rt:.4f} | {100 * fb:.1f}% | {n} |")
        M.append("")
    M += ["Reading: spread < 1 and conc > 1 (with half_tok/half_seq < 1) means diversity of "
          "sources matters beyond the token count, so exposure should be measured in sources or "
          "effective rank, not raw routed tokens. Ratios ~1 mean the count is what matters."]
    json.dump(out, open(os.path.join(a.out, "exposure_diversity.json"), "w"), indent=2)
    open(os.path.join(a.out, "exposure_diversity_report.md"), "w").write("\n".join(M) + "\n")
    print("\n".join(M))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--out", default="runs_diag")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--regimes", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--pool-n", type=int, default=256)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=1.0)
    ap.add_argument("--no-erank", dest="erank", action="store_false")
    a = ap.parse_args(argv)
    return dict(run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
