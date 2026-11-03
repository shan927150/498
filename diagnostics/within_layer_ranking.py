"""Exp1 — is the layer-aware ranking gain ACTIONABLE by sequence selection?
CPU only, seconds; reads the frozen-damping E1 labels already in runs/.

Every selected sequence passes through every layer, so each layer gets the
same total routed tokens (B x seqlen x top-k) whatever is selected: sequence
selection cannot move exposure BETWEEN layers, only among experts WITHIN a
layer. Within one layer all experts share one benefit curve f_L(n); when f_L
decreases in n, f_L(n)/n orders experts exactly like coverage 1/n.

P2c scored rankings over all 240 experts pooled across layers. This script
repeats that capture test both pooled (P2c's view) and WITHIN each layer
(the only reallocation sequence selection can make), held out by
characterization seed (or, with one seed, by expert groups).

Utility rules (realized value = future_rel_delta / n, P2c's definition):
  coverage      1 / n
  global        max(f_G(n, budget), 0) / n     one flexible exposure model
  layer_aware   max(f_L(n, budget), 0) / n     a flexible model per layer
  oracle        realized value
Capture(q) = realized value of the top-q % by the rule / that of the oracle's
top-q % (q = 10, 25, 50; averaged). Within-layer: top-q % of each layer.

  python diagnostics/within_layer_ranking.py --outdir runs --percdamp 1.0
  -> runs/within_layer_ranking_b4_d1.{json,md}
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moe_calib.runid import damp_tag, load_e1_runs          # noqa: E402

QS = (0.10, 0.25, 0.50)


def load_labels(runs):
    """(seed, layer, expert, n0) -> count, mean rel_delta over replicates."""
    acc = {}
    for run in runs:
        for r in csv.DictReader(open(run["labels"])):
            k = (run["char_seed"], int(r["layer"]), int(r["expert"]), int(r["n0"]))
            a = acc.setdefault(k, {"n": float(r["tokens_n0"]), "y": []})
            a["y"].append(float(r["rel_delta"]))
    keys = sorted(acc)
    arr = lambda i: np.array([k[i] for k in keys], float)
    return dict(seed=arr(0), layer=arr(1), expert=arr(2), budget=arr(3),
                count=np.array([acc[k]["n"] for k in keys]),
                y=np.array([np.mean(acc[k]["y"]) for k in keys]))


def design(count, budget, budgets, mu, sd):
    lc = (np.log(count) - mu) / sd
    cols = [np.ones_like(lc), lc, lc ** 2]
    for b in budgets[1:]:
        D = (budget == b).astype(float)
        cols += [D, D * lc]
    return np.stack(cols, axis=1)


def fit(d, m, budgets, mu, sd):
    X = design(d["count"][m], d["budget"][m], budgets, mu, sd)
    beta, *_ = np.linalg.lstsq(X, d["y"][m], rcond=None)
    return beta


def capture(util, real, groups=None):
    """Mean over q of captured / oracle-captured realized value; with groups,
    the top-q % is taken inside each group and summed."""
    out = []
    for q in QS:
        got = best = 0.0
        for g in (np.unique(groups) if groups is not None else [None]):
            m = np.ones_like(real, bool) if g is None else groups == g
            k = max(1, int(round(q * m.sum())))
            u, r = util[m], real[m]
            got += r[np.argsort(-u, kind="stable")[:k]].sum()
            best += np.sort(r)[::-1][:k].sum()
        out.append(got / best if best > 0 else np.nan)
    return float(np.mean(out))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def folds_of(d, k=5):
    """Held-out unit: the characterization seed when there are >= 2 seeds;
    with a single seed, experts (expert id mod k: every layer's expert e in
    the same fold, so a held-out expert is never seen in training)."""
    if len(set(d["seed"].tolist())) >= 2:
        return d["seed"], "leave-one-seed-out"
    return d["expert"] % k, f"expert-grouped {k}-fold"


def run(d):
    layers, budgets = (sorted(set(d[k].tolist())) for k in ("layer", "budget"))
    group, _ = folds_of(d)
    rows = []
    for s in sorted(set(group.tolist())):
        tr = group != s
        mu, sd = np.log(d["count"][tr]).mean(), np.log(d["count"][tr]).std() + 1e-12
        bG = fit(d, tr, budgets, mu, sd)
        bL = {L: fit(d, tr & (d["layer"] == L), budgets, mu, sd) for L in layers}
        for b in budgets:
            m = (group == s) & (d["budget"] == b)
            n, real, lay = d["count"][m], d["y"][m] / d["count"][m], d["layer"][m]
            X = design(n, d["budget"][m], budgets, mu, sd)
            fG = X @ bG
            fL = np.array([X[i] @ bL[lay[i]] for i in range(len(n))])
            utils = dict(coverage=1.0 / n, global_=np.maximum(fG, 0) / n,
                         layer_aware=np.maximum(fL, 0) / n, oracle=real)
            for name, u in utils.items():
                rows.append(dict(fold=s, budget=b, rule=name.rstrip("_"),
                                 pooled=capture(u, real),
                                 within_layer=capture(u, real, lay),
                                 within_spearman=float(np.mean(
                                     [spearman(u[lay == L], real[lay == L]) for L in layers])),
                                 same_order_as_coverage=float(np.mean(
                                     [spearman(u[lay == L], 1.0 / n[lay == L]) for L in layers]))))
    return rows


def summarize(rows):
    rules = ["coverage", "global", "layer_aware", "oracle"]
    agg = {}
    for r in rules:
        rs = [x for x in rows if x["rule"] == r]
        agg[r] = {k: float(np.mean([x[k] for x in rs]))
                  for k in ("pooled", "within_layer", "within_spearman", "same_order_as_coverage")}
    cells = sorted({(x["fold"], x["budget"]) for x in rows})
    get = lambda r, c, k: next(x[k] for x in rows if x["rule"] == r and (x["fold"], x["budget"]) == c)
    wins = {k: sum(get("layer_aware", c, k) > get("coverage", c, k) + 1e-12 for c in cells)
            for k in ("pooled", "within_layer")}
    return agg, wins, len(cells)


def render(agg, wins, ncells, tag, validation=""):
    dp = agg["layer_aware"]["pooled"] - agg["coverage"]["pooled"]
    dw = agg["layer_aware"]["within_layer"] - agg["coverage"]["within_layer"]
    M = [f"# Exp1 — within-layer ranking check ({tag})", "", f"Validation: {validation}.", "",
         f"Layer-aware minus coverage capture: pooled across layers {100 * dp:+.2f} pp "
         f"(wins {wins['pooled']}/{ncells} cells), within layers {100 * dw:+.2f} pp "
         f"(wins {wins['within_layer']}/{ncells}).", "",
         "| rule | pooled capture | within-layer capture | within-layer Spearman vs realized | "
         "within-layer rank agreement with coverage |", "|---|---|---|---|---|"]
    for r, v in agg.items():
        M.append(f"| {r} | {v['pooled']:.4f} | {v['within_layer']:.4f} | "
                 f"{v['within_spearman']:.3f} | {v['same_order_as_coverage']:.3f} |")
    M += ["", "Reading: if the within-layer gain is ~0 and rank agreement with coverage is ~1, "
          "the P2c advantage came from cross-layer comparisons that sequence selection cannot "
          "act on; E3 can then only profit through cross-layer weighting inside the LP."]
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=1.0)
    a = ap.parse_args(argv)
    runs, _ = load_e1_runs(a.outdir, a.bits, None, a.percdamp)
    d = load_labels(runs)
    rows = run(d)
    agg, wins, ncells = summarize(rows)
    tag = f"b{a.bits}_{damp_tag(a.percdamp)}"
    base = os.path.join(a.outdir, f"within_layer_ranking_{tag}")
    json.dump(dict(summary=agg, wins=wins, cells=ncells, rows=rows), open(base + ".json", "w"),
              indent=2, default=float)
    md = render(agg, wins, ncells, tag, folds_of(d)[1])
    open(base + ".md", "w").write(md + "\n")
    print(md)
    return agg


if __name__ == "__main__":
    main()
