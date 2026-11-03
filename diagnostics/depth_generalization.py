"""Exp6 (analysis) — does the layer x regime exposure structure transfer to
layers that were never characterized?  CPU only.

Inputs: frozen-damping E1 labels for the characterized layers (runs/, all
seeds) and for NEW layers (e.g. 4/12/20, run into a separate outdir with a
separate cache). Each new layer is predicted WITHOUT using its labels:
  global        one flexible exposure model over all characterized layers
  nearest       the per-layer model of the nearest characterized layer
  interpolate   per-layer predictions of the two bracketing layers, linearly
                weighted by depth
  self_cv       reference: a per-layer model fitted on the new layer itself
                (2-fold over experts) — the best a layer-specific model can do
Per-layer model: y ~ 1 + lc + lc^2 + budget dummies + budget x lc
(y = future_rel_delta averaged over replicates, lc = standardized log count).
Also prints the depth profile: mean benefit by (layer, budget) for all layers.

  python diagnostics/depth_generalization.py --old-outdir runs --new-outdir runs_newlayers
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moe_calib.runid import load_e1_runs                    # noqa: E402


def load(outdir, bits, percdamp):
    runs, _ = load_e1_runs(outdir, bits, None, percdamp)
    acc = {}
    for run in runs:
        for r in csv.DictReader(open(run["labels"])):
            k = (run["char_seed"], int(r["layer"]), int(r["expert"]), int(r["n0"]))
            a = acc.setdefault(k, {"n": float(r["tokens_n0"]), "y": []})
            a["y"].append(float(r["rel_delta"]))
    keys = sorted(acc)
    col = lambda i: np.array([k[i] for k in keys], float)
    return dict(seed=col(0), layer=col(1), expert=col(2), budget=col(3),
                count=np.array([acc[k]["n"] for k in keys]),
                y=np.array([np.mean(acc[k]["y"]) for k in keys]))


class Model:
    def __init__(self, budgets, mu, sd):
        self.budgets, self.mu, self.sd = budgets, mu, sd

    def X(self, count, budget):
        lc = (np.log(count) - self.mu) / self.sd
        cols = [np.ones_like(lc), lc, lc ** 2]
        for b in self.budgets[1:]:
            D = (budget == b).astype(float)
            cols += [D, D * lc]
        return np.stack(cols, axis=1)

    def fit(self, count, budget, y):
        self.beta, *_ = np.linalg.lstsq(self.X(count, budget), y, rcond=None)
        return self

    def predict(self, count, budget):
        return self.X(count, budget) @ self.beta


def evaluate(old, new):
    budgets = sorted(set(old["budget"].tolist()))
    mu, sd = np.log(old["count"]).mean(), np.log(old["count"]).std() + 1e-12
    olds = sorted(set(old["layer"].tolist()))
    per = {L: Model(budgets, mu, sd).fit(old["count"][old["layer"] == L],
                                         old["budget"][old["layer"] == L],
                                         old["y"][old["layer"] == L]) for L in olds}
    glob_ = Model(budgets, mu, sd).fit(old["count"], old["budget"], old["y"])
    res = {}
    for L in sorted(set(new["layer"].tolist())):
        m = new["layer"] == L
        c, b, y = new["count"][m], new["budget"][m], new["y"][m]
        preds = {"global": glob_.predict(c, b)}
        near = min(olds, key=lambda o: (abs(o - L), o))
        preds["nearest"] = per[near].predict(c, b)
        lo = max([o for o in olds if o <= L], default=None)
        hi = min([o for o in olds if o >= L], default=None)
        if lo is None or hi is None or lo == hi:
            preds["interpolate"] = per[lo if lo is not None else hi].predict(c, b)
        else:
            w = (L - lo) / (hi - lo)
            preds["interpolate"] = (1 - w) * per[lo].predict(c, b) + w * per[hi].predict(c, b)
        ex = new["expert"][m]
        fold = (ex % 2).astype(int)
        sc = np.zeros_like(y)
        for f in (0, 1):
            tr, te = fold != f, fold == f
            sc[te] = Model(budgets, mu, sd).fit(c[tr], b[tr], y[tr]).predict(c[te], b[te])
        preds["self_cv"] = sc
        mse = {k: float(np.mean((y - p) ** 2)) for k, p in preds.items()}
        res[int(L)] = dict(n=int(m.sum()), mse=mse,
                           gain_vs_global={k: float(1 - v / mse["global"]) for k, v in mse.items()},
                           nearest_layer=int(near), bracket=[lo, hi])
    return res


def depth_profile(old, new):
    rows = {}
    for d, tag in ((old, "char"), (new, "new")):
        for L in sorted(set(d["layer"].tolist())):
            for b in sorted(set(d["budget"].tolist())):
                m = (d["layer"] == L) & (d["budget"] == b)
                if m.any():
                    rows.setdefault(int(L), {"src": tag})[int(b)] = float(d["y"][m].mean())
    return dict(sorted(rows.items()))


def render(res, prof):
    budgets = sorted({b for v in prof.values() for b in v if b != "src"})
    M = ["# Exp6 — layer x regime structure on uncharacterized layers", "",
         "Gain = reduction of MSE vs the global exposure model (new layers' labels unused, "
         "except self_cv).", "",
         "| new layer | n | nearest | interpolate (bracket) | self_cv (reference) |", "|---|---|---|---|---|"]
    for L, r in res.items():
        g = r["gain_vs_global"]
        M.append(f"| {L} | {r['n']} | {100 * g['nearest']:+.1f}% (L{r['nearest_layer']}) | "
                 f"{100 * g['interpolate']:+.1f}% ({r['bracket'][0]}–{r['bracket'][1]}) | "
                 f"{100 * g['self_cv']:+.1f}% |")
    M += ["", "## Depth profile: mean future_rel_delta by layer and budget", "",
          "| layer | source | " + " | ".join(f"n0={b}" for b in budgets) + " |",
          "|---|---|" + "---|" * len(budgets)]
    for L, v in prof.items():
        M.append(f"| {L} | {v['src']} | " + " | ".join(
            f"{v[b]:.4f}" if b in v else "–" for b in budgets) + " |")
    M += ["", "Reading: interpolation close to self_cv means a depth-smooth model can cover all "
          "24 layers from a few characterized ones; near zero or negative means every layer "
          "needs its own characterization."]
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-outdir", default="runs")
    ap.add_argument("--new-outdir", default="runs_newlayers")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=1.0)
    a = ap.parse_args(argv)
    old, new = load(a.old_outdir, a.bits, a.percdamp), load(a.new_outdir, a.bits, a.percdamp)
    res, prof = evaluate(old, new), depth_profile(old, new)
    json.dump(dict(results=res, depth_profile=prof),
              open(os.path.join(a.new_outdir, "depth_generalization.json"), "w"), indent=2)
    md = render(res, prof)
    open(os.path.join(a.new_outdir, "depth_generalization.md"), "w").write(md + "\n")
    print(md)
    return res


if __name__ == "__main__":
    main()
