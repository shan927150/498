"""Exp5 (analysis) — how much would a budget- or layer-adaptive GPTQ damping
gain over the single frozen value?  CPU only; reads the P0 raw file
(runs_diag/damping_threshold_raw.jsonl). Run the GPU part first to extend
the grid upward:  diagnostics/damping_threshold.py run --damps 3.0 5.0

Rules, each CHOSEN on one characterization seed and SCORED on the other
(mean tuning error, relative to the frozen value):
  per_prefix          one damping per calibration budget
  per_layer_prefix    one per (layer, budget)
  per_token_bin       one per bin of routed tokens / gate-up input dim (per expert)
  per_expert_oracle   best value for each expert (upper bound, not a rule)
Reference: the gain from doubling the calibration data at the frozen value.
This is an ablation on the side of the main line; the frozen config stays.

  python diagnostics/adaptive_damping.py [--frozen 1.0]
  -> runs_diag/adaptive_damping.{json,md}
"""
from __future__ import annotations
import argparse, json, os

import numpy as np

BINS = [0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]


def load(raw_path):
    by = {}
    for line in open(raw_path):
        r = json.loads(line)
        m = by.setdefault((r["seed"], r["layer"], r["n"]), dict(err={}))
        for f in ("tokens", "d_in"):
            if f in r:
                m[f] = r[f]
        for d, v in r["err"].items():
            m["err"][float(d)] = np.array([x["all"] for x in v])
    return {k: v for k, v in by.items() if "tokens" in v}


def analyse(by, frozen):
    seeds = sorted({k[0] for k in by})
    layers = sorted({k[1] for k in by})
    ns = sorted({k[2] for k in by})
    damps = sorted(set.intersection(*[set(v["err"]) for v in by.values()]))
    assert frozen in damps, f"frozen value {frozen} not in grid {damps}"
    d_gu = next(iter(by.values())).get("d_in", {}).get("gu", 2048)
    E = lambda s, L, n, d: by[(s, L, n)]["err"][d]
    rows = []
    for n in ns:
        acc = {k: [] for k in ("frozen", "per_prefix", "per_layer_prefix", "per_token_bin",
                               "per_expert_oracle", "doubled")}
        for s in seeds:
            others = [o for o in seeds if o != s]
            mean_o = lambda f: np.mean([f(o) for o in others])
            best_p = min(damps, key=lambda d: mean_o(lambda o: np.mean([E(o, L, n, d).mean() for L in layers])))
            # token-bin rule: pooled over prefixes and layers of the other seeds
            bin_best = {}
            for b in range(len(BINS) - 1):
                def bin_err(d, b=b):
                    v = []
                    for o in others:
                        for L in layers:
                            for nn in ns:
                                t = np.array(by[(o, L, nn)]["tokens"]) / d_gu
                                m = (t >= BINS[b]) & (t < BINS[b + 1])
                                if m.any():
                                    v.append(E(o, L, nn, d)[m])
                    return np.concatenate(v).mean() if v else np.inf
                bin_best[b] = min(damps, key=bin_err)
            for L in layers:
                acc["frozen"].append(E(s, L, n, frozen))
                acc["per_prefix"].append(E(s, L, n, best_p))
                bl = min(damps, key=lambda d: mean_o(lambda o: E(o, L, n, d).mean()))
                acc["per_layer_prefix"].append(E(s, L, n, bl))
                t = np.array(by[(s, L, n)]["tokens"]) / d_gu
                bi = np.clip(np.searchsorted(BINS, t, side="right") - 1, 0, len(BINS) - 2)
                acc["per_token_bin"].append(np.array([E(s, L, n, bin_best[b])[i]
                                                      for i, b in enumerate(bi)]))
                acc["per_expert_oracle"].append(np.min(np.stack([E(s, L, n, d) for d in damps]), 0))
                if (s, L, 2 * n) in by:
                    acc["doubled"].append(E(s, L, 2 * n, frozen))
        f = np.concatenate(acc["frozen"]).mean()
        gain = lambda k: (100 * (1 - np.concatenate(acc[k]).mean() / f)) if acc[k] else None
        rows.append(dict(n=n, frozen_err=float(f), **{k: gain(k) for k in acc if k != "frozen"},
                         best_per_prefix_all_seeds=min(damps, key=lambda d: np.mean(
                             [E(s, L, n, d).mean() for s in seeds for L in layers]))))
    return dict(damps=damps, seeds=seeds, layers=layers, frozen=frozen, rows=rows)


def render(res):
    M = [f"# Exp5 — adaptive damping vs frozen percdamp = {res['frozen']}", "",
         f"Grid {res['damps']}; seeds {res['seeds']}; layers {res['layers']}; each rule chosen "
         "on the other seed. Gain = reduction of mean tuning error vs the frozen value.", "",
         "| sequences | frozen err | per budget | per layer+budget | per token bin | per-expert "
         "oracle | doubling the data | best value (all seeds) |", "|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:+.2f}%"
    edge = []
    for r in res["rows"]:
        M.append(f"| {r['n']} | {r['frozen_err']:.5f} | {f(r['per_prefix'])} | {f(r['per_layer_prefix'])} | "
                 f"{f(r['per_token_bin'])} | {f(r['per_expert_oracle'])} | {f(r['doubled'])} | "
                 f"{r['best_per_prefix_all_seeds']} |")
        if r["best_per_prefix_all_seeds"] in (min(res["damps"]), max(res["damps"])):
            edge.append(r["n"])
    M += ["", ("Best value still at the grid edge for budgets " + ", ".join(map(str, edge))
               + ": extend the grid further.") if edge else "All per-budget optima are interior."]
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="runs_diag/damping_threshold_raw.jsonl")
    ap.add_argument("--frozen", type=float, default=1.0)
    ap.add_argument("--out", default="runs_diag")
    a = ap.parse_args(argv)
    res = analyse(load(a.raw), a.frozen)
    json.dump(res, open(os.path.join(a.out, "adaptive_damping.json"), "w"), indent=2, default=float)
    md = render(res)
    open(os.path.join(a.out, "adaptive_damping.md"), "w").write(md + "\n")
    print(md)
    return res


if __name__ == "__main__":
    main()
