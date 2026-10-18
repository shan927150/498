"""Controlled FIXED-damping study (diagnostic only; production GPTQ untouched).

For percdamp in {0.01, 0.03, 0.05, 0.10}, act-order on, 4-bit, group 128, the
production solver (moe_calib.gptq.gptq_quantize) quantizes every expert of
layers 1/8/16/23 from STRICTLY NESTED calibration prefixes 16 < 32 < 64 <
128 < 256 of one pool ordering per characterization seed, and is scored on
the TUNING split only (the test split is never profiled). A fixed RTN
baseline (production zero-token fallback: H = I, act-order off) is scored
on the same tuning sequences.

Error = E1's tuning metric: mean over tuning sequences of
  sum_proj ||X_s (W - Wq)^T||^2 / sum_proj ||X_s W^T||^2
(first 128 routed tokens per sequence, as in E1), computed in one batched
pass per expert; checked against quant_eval.per_seq_rel_errors at start-up.

Pool ordering = diagnostics/low_data_diag.nested_order (the same orderings
as the earlier 0.01-only nested diagnostic).

  python diagnostics/damping_selection.py run      # resumable, per (seed, layer, prefix)
  python diagnostics/damping_selection.py report   # -> runs_diag/damping_selection.{json,csv}, _report.md
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from low_data_diag import (HKEY, OUT, PREFIXES, PROJ, gpu_weights, hessians, nested_order,  # noqa: E402
                           quant, read_jsonl, rtn, splits, tune_holdout)
from moe_calib import model_utils as MU                            # noqa: E402
from moe_calib.data_utils import load_layer_cache                  # noqa: E402
from moe_calib.quant_eval import per_seq_rel_errors                # noqa: E402

DAMPS = [0.01, 0.03, 0.05, 0.10]
RAW = f"{OUT}/damping_selection_raw.jsonl"
TRANS = list(zip(PREFIXES[:-1], PREFIXES[1:]))


# ------------------------------------------------------------ batched metric

def prep_tune(HO, L, e):
    """Stack an expert's tuning activations once: per proj (X fp16 on GPU,
    segment index into the expert's tuning-sequence list)."""
    sids = sorted({s for p in ("gu", "down") for s, _ in HO.get((L, e, p), [])})
    pos = {s: i for i, s in enumerate(sids)}
    out = dict(n=len(sids))
    for p in ("gu", "down"):
        lst = HO.get((L, e, p), [])
        if not lst:
            out[p] = None
            continue
        X = torch.cat([x for _, x in lst]).to(MU.GPU)
        seg = torch.cat([torch.full((x.shape[0],), pos[s], dtype=torch.long) for s, x in lst]).to(MU.GPU)
        out[p] = (X, seg)
    return out


@torch.no_grad()
def tune_err_fast(W, Q, T, den=None):
    if T["n"] == 0:
        return float("nan"), None
    num = torch.zeros(T["n"], device=MU.GPU, dtype=torch.float64)
    need_den = den is None
    if need_den:
        den = torch.zeros(T["n"], device=MU.GPU, dtype=torch.float64)
    for n in PROJ:
        if T[HKEY[n]] is None:
            continue
        X, seg = T[HKEY[n]]
        Xf = X.float()
        num.index_add_(0, seg, (Xf @ (W[n] - Q[n].to(MU.GPU)).T).pow(2).sum(1).double())
        if need_den:
            den.index_add_(0, seg, (Xf @ W[n].T).pow(2).sum(1).double())
    return float((num / (den + 1e-12)).mean()), den


# --------------------------------------------------------------------- run

def cmd_run(a):
    meta, sp = splits(a.cache_dir)
    os.makedirs(OUT, exist_ok=True)
    done = {(r["seed"], r["layer"], r["n"]) for r in read_jsonl(RAW)}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    checked = False
    for L in a.layers:
        todo = [s for s in a.seeds if any((s, L, n) not in done for n in PREFIXES)]
        if not todo:
            continue
        cache = load_layer_cache(a.cache_dir, L)
        HO = tune_holdout(model, L, cache, sp["tune"])
        Ws = {e: gpu_weights(model, L, e) for e in range(E)}
        Ts = {e: prep_tune(HO, L, e) for e in range(E)}
        dens, rtn_err = {}, []
        for e in range(E):
            m, dens[e] = tune_err_fast(Ws[e], rtn(Ws[e]), Ts[e])
            rtn_err.append(m)
        if not checked:                       # batched metric == E1's metric
            for e in range(3):
                q = rtn(Ws[e])
                ref = np.mean(list(per_seq_rel_errors(model, L, e, q, HO).values()))
                fast = tune_err_fast(Ws[e], q, Ts[e], dens[e])[0]
                assert abs(fast - ref) <= 1e-4 * ref, (e, fast, ref)
            checked = True
        for seed in todo:
            order = nested_order(sp["pool"], seed)
            run, tok, lo = None, None, 0
            for n in PREFIXES:
                Hc, tc = hessians(model, L, cache, order[lo:n], K=1)
                if run is None:
                    run, tok = Hc, tc
                else:
                    for e in range(E):
                        for p in ("gu", "down"):
                            run[e][p] += Hc[e][p]
                        tok[e] += tc[e]
                    del Hc
                lo = n
                if (seed, L, n) in done:
                    continue
                rec = dict(seed=seed, layer=L, n=n, tokens=[tok[e] for e in range(E)],
                           rtn_err=rtn_err, err={str(d): [] for d in DAMPS})
                for e in range(E):
                    for d in DAMPS:
                        q = quant(Ws[e], run[e], percdamp=d, actorder=True)
                        rec["err"][str(d)].append(tune_err_fast(Ws[e], q, Ts[e], dens[e])[0])
                with open(RAW, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[damp-sel] seed={seed} L={L} n={n} tok_med={np.median(rec['tokens']):.0f} "
                      f"RTN={np.mean(rtn_err):.5f} " +
                      " ".join(f"{d}:{np.mean(rec['err'][str(d)]):.5f}" for d in DAMPS), flush=True)
            del run
        del cache, HO, Ws, Ts
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ report

def budget_metrics(err, rtn_):
    err, rtn_ = np.asarray(err, float), np.asarray(rtn_, float)
    d = err - rtn_
    return dict(n=int(err.size), mean_err=float(err.mean()), median_err=float(np.median(err)),
                p90_err=float(np.percentile(err, 90)), mean_rtn=float(rtn_.mean()),
                mean_delta_vs_rtn=float(d.mean()), frac_better_than_rtn=float((d < 0).mean()),
                frac_worse_than_rtn=float((d > 0).mean()))


def trans_metrics(e0, e1):
    d = np.asarray(e0, float) - np.asarray(e1, float)
    return dict(n=int(d.size), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                frac_improved=float((d > 0).mean()), frac_worsened=float((d < 0).mean()))


def cmd_report(a):
    raw = read_jsonl(RAW)
    by = {(r["seed"], r["layer"], r["n"]): r for r in raw}
    seeds = sorted({r["seed"] for r in raw})
    layers = sorted({r["layer"] for r in raw})
    groups = {"all": (seeds, layers)}
    groups.update({f"seed={s}": ([s], layers) for s in seeds})
    groups.update({f"layer={L}": (seeds, [L]) for L in layers})
    groups.update({f"seed={s},layer={L}": ([s], [L]) for s in seeds for L in layers})

    def collect(ss, ls, n, d):
        rs = [by[(s, L, n)] for s in ss for L in ls if (s, L, n) in by]
        return sum((r["err"][str(d)] for r in rs), []), sum((r["rtn_err"] for r in rs), []), \
            sum((r["tokens"] for r in rs), [])

    out = dict(design=dict(percdamp=DAMPS, actorder=True, bits=4, group_size=128,
                           prefixes=PREFIXES, seeds=seeds, layers=layers,
                           split="tuning only", rtn="H=I, act-order off (production fallback)",
                           input_dims=dict(gate_up=2048, down=1408)),
               budget={}, transitions={}, summary={}, tokens={})
    rows = []
    for g, (ss, ls) in groups.items():
        out["budget"][g], out["transitions"][g], out["summary"][g] = {}, {}, {}
        for d in DAMPS:
            bm = {n: budget_metrics(*collect(ss, ls, n, d)[:2]) for n in PREFIXES}
            tm = {f"{n0}->{n1}": trans_metrics(collect(ss, ls, n0, d)[0], collect(ss, ls, n1, d)[0])
                  for n0, n1 in TRANS}
            out["budget"][g][str(d)] = bm
            out["transitions"][g][str(d)] = tm
            out["summary"][g][str(d)] = dict(
                mean_err_over_prefixes=float(np.mean([bm[n]["mean_err"] for n in PREFIXES])),
                mean_delta_vs_rtn_over_prefixes=float(np.mean([bm[n]["mean_delta_vs_rtn"] for n in PREFIXES])),
                frac_better_than_rtn_over_prefixes=float(np.mean([bm[n]["frac_better_than_rtn"] for n in PREFIXES])),
                **{f"delta_{k}": v["mean_diff"] for k, v in tm.items()})
            for n in PREFIXES:
                rows.append(dict(group=g, percdamp=d, kind="budget", prefix=n, **bm[n]))
            for k, v in tm.items():
                rows.append(dict(group=g, percdamp=d, kind="transition", prefix=k, **v))
    for n in PREFIXES:
        t = np.asarray(collect(seeds, layers, n, DAMPS[0])[2], float)
        out["tokens"][n] = dict(min=float(t.min()), p5=float(np.percentile(t, 5)), median=float(np.median(t)),
                                p95=float(np.percentile(t, 95)), max=float(t.max()))
    base = f"{OUT}/damping_selection"
    json.dump(out, open(base + ".json", "w"), indent=1)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("group", "percdamp", "kind", "prefix"), k))
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    M = ["# Fixed-damping selection (tuning split only)", "",
         f"act-order on, 4-bit, g128; seeds {seeds}; layers {layers}; nested prefixes {PREFIXES}; "
         "RTN = H=I, act-order off.", "",
         "## Summary (all seeds x layers)", "",
         "| percdamp | mean tuning err (avg over prefixes) | mean delta vs RTN | % experts better than RTN | "
         + " | ".join(f"{a}->{b} delta" for a, b in TRANS) + " |",
         "|---|---|---|---|" + "---|" * len(TRANS)]
    for d in DAMPS:
        s = out["summary"]["all"][str(d)]
        M.append(f"| {d} | {s['mean_err_over_prefixes']:.5f} | {s['mean_delta_vs_rtn_over_prefixes']:+.5f} | "
                 f"{100 * s['frac_better_than_rtn_over_prefixes']:.1f}% | " +
                 " | ".join(f"{s[f'delta_{a}->{b}']:+.5f}" for a, b in TRANS) + " |")
    M += ["", "## Per budget (all seeds x layers)", "",
          "| percdamp | prefix | GPTQ mean | median | p90 | RTN mean | delta vs RTN | % better | % worse |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in DAMPS:
        for n in PREFIXES:
            b = out["budget"]["all"][str(d)][n]
            M.append(f"| {d} | {n} | {b['mean_err']:.5f} | {b['median_err']:.5f} | {b['p90_err']:.5f} | "
                     f"{b['mean_rtn']:.5f} | {b['mean_delta_vs_rtn']:+.5f} | "
                     f"{100 * b['frac_better_than_rtn']:.1f}% | {100 * b['frac_worse_than_rtn']:.1f}% |")
    M += ["", "## Transitions (all): mean diff / median / % improved / % worsened", "",
          "| percdamp | " + " | ".join(f"{a}->{b}" for a, b in TRANS) + " |", "|---|" + "---|" * len(TRANS)]
    for d in DAMPS:
        M.append(f"| {d} | " + " | ".join(
            "{mean_diff:+.5f} / {median_diff:+.5f} / {imp:.0f}% / {wor:.0f}%".format(
                imp=100 * t["frac_improved"], wor=100 * t["frac_worsened"], **t)
            for t in out["transitions"]["all"][str(d)].values()) + " |")
    M += ["", "## Stability: mean tuning err averaged over prefixes, and best percdamp, per group", "",
          "| group | " + " | ".join(str(d) for d in DAMPS) + " | best | % better than RTN at best |",
          "|---|" + "---|" * (len(DAMPS) + 2)]
    for g in groups:
        s = out["summary"][g]
        best = min(DAMPS, key=lambda d: s[str(d)]["mean_err_over_prefixes"])
        M.append(f"| {g} | " + " | ".join(f"{s[str(d)]['mean_err_over_prefixes']:.5f}" for d in DAMPS) +
                 f" | {best} | {100 * s[str(best)]['frac_better_than_rtn_over_prefixes']:.1f}% |")
    M += ["", "## Per-group budget detail: delta vs RTN (% better) per prefix", ""]
    for g in groups:
        if g == "all" or "," in g:
            continue
        M += [f"### {g}", "", "| percdamp | " + " | ".join(str(n) for n in PREFIXES) + " |",
              "|---|" + "---|" * len(PREFIXES)]
        for d in DAMPS:
            M.append(f"| {d} | " + " | ".join(
                f"{b['mean_delta_vs_rtn']:+.5f} ({100 * b['frac_better_than_rtn']:.0f}%)"
                for b in out["budget"][g][str(d)].values()) + " |")
        M.append("")
    M += ["## Routed tokens per expert at each prefix (gate/up in-dim 2048, down in-dim 1408)", "",
          "| prefix | min | p5 | median | p95 | max |", "|---|---|---|---|---|---|"]
    for n, t in out["tokens"].items():
        M.append(f"| {n} | {t['min']:.0f} | {t['p5']:.0f} | {t['median']:.0f} | {t['p95']:.0f} | {t['max']:.0f} |")
    open(base + "_report.md", "w").write("\n".join(M) + "\n")
    print("\n".join(M))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args(argv)
    dict(run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
