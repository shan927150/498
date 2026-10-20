"""P1 gate — is the E1 target (future benefit of doubling calibration) sane
under the frozen GPTQ configuration?  Read-only; no model, no GPU.

Before any signal is validated (E2), additional calibration data must
generally HELP. Per starting budget n0 (n0 -> 2*n0), with each expert's
rel_delta first averaged over extension replicates:

  REQUIRED  median rel_delta > 0                         (for every n0)
  REQUIRED  fraction of experts with rel_delta < 0 <= --max-neg (for every n0)
  REQUIRED  per-characterization-seed median > 0         (every seed x n0)
  WARN      per-layer median > 0                         (every layer x n0)
  WARN      diminishing returns: median absolute delta non-increasing in n0

Exit code 0 = PASS (go to E2), 1 = FAIL (do not run E2: the target is not
yet cleanly defined). Writes <outdir>/e1_gate_b{bits}[_d{damp}].{json,md}.
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np

from moe_calib.runid import damp_tag, fp_percdamp, load_e1_runs


def load_rows(runs):
    """One row per (char_seed, layer, expert, n0): replicate-mean rel_delta / delta."""
    acc = {}
    for run in runs:
        for r in csv.DictReader(open(run["labels"])):
            k = (run["char_seed"], int(r["layer"]), int(r["expert"]), int(r["n0"]))
            a = acc.setdefault(k, {"rel": [], "abs": []})
            a["rel"].append(float(r["rel_delta"]))
            a["abs"].append(float(r["delta"]))
    return [dict(seed=k[0], layer=k[1], expert=k[2], n0=k[3],
                 rel=float(np.nanmean(v["rel"])), abs=float(np.nanmean(v["abs"])))
            for k, v in acc.items()]


def evaluate(rows, max_neg=0.25):
    n0s = sorted({r["n0"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    pick = lambda **kw: np.array([r["rel"] for r in rows
                                  if all(r[k] == v for k, v in kw.items())], float)
    per_n0, checks = {}, []
    for n0 in n0s:
        x = pick(n0=n0)
        a = np.array([r["abs"] for r in rows if r["n0"] == n0], float)
        per_n0[n0] = dict(n=int(x.size), median=float(np.median(x)), mean=float(x.mean()),
                          p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                          frac_negative=float(np.mean(x < 0)),
                          median_abs_delta=float(np.median(a)),
                          seed_median={s: float(np.median(pick(n0=n0, seed=s))) for s in seeds},
                          layer_median={L: float(np.median(pick(n0=n0, layer=L))) for L in layers})
        p = per_n0[n0]
        checks.append(dict(level="required", name=f"median rel_delta > 0 at n0={n0}",
                           ok=p["median"] > 0, value=p["median"]))
        checks.append(dict(level="required", name=f"frac negative <= {max_neg} at n0={n0}",
                           ok=p["frac_negative"] <= max_neg, value=p["frac_negative"]))
        bad_s = [s for s, v in p["seed_median"].items() if not v > 0]
        checks.append(dict(level="required", name=f"every seed median > 0 at n0={n0}",
                           ok=not bad_s, value=bad_s))
        bad_l = [L for L, v in p["layer_median"].items() if not v > 0]
        checks.append(dict(level="warn", name=f"every layer median > 0 at n0={n0}",
                           ok=not bad_l, value=bad_l))
    mad = [per_n0[n]["median_abs_delta"] for n in n0s]
    checks.append(dict(level="warn", name="diminishing returns (median |delta| non-increasing in n0)",
                       ok=all(b <= a for a, b in zip(mad, mad[1:])), value=mad))
    passed = all(c["ok"] for c in checks if c["level"] == "required")
    return dict(passed=passed, n0s=n0s, seeds=seeds, layers=layers,
                per_n0=per_n0, checks=checks)


def render(res, tag):
    M = [f"# E1 gate ({tag}): {'PASS' if res['passed'] else 'FAIL'}", "",
         "| n0 -> 2n0 | n | median rel_delta | mean | p10 | p90 | % negative | median abs delta |",
         "|---|---|---|---|---|---|---|---|"]
    for n0, p in res["per_n0"].items():
        M.append(f"| {n0} -> {2 * n0} | {p['n']} | {p['median']:+.4f} | {p['mean']:+.4f} | "
                 f"{p['p10']:+.4f} | {p['p90']:+.4f} | {100 * p['frac_negative']:.1f}% | "
                 f"{p['median_abs_delta']:.2e} |")
    M += ["", "| check | level | result | value |", "|---|---|---|---|"]
    for c in res["checks"]:
        M.append(f"| {c['name']} | {c['level']} | {'ok' if c['ok'] else 'FAIL'} | {c['value']} |")
    M += ["", "Per-seed / per-layer medians:", ""]
    for n0, p in res["per_n0"].items():
        M.append(f"- n0={n0}: seeds {', '.join(f'{k}:{v:+.3f}' for k, v in p['seed_median'].items())}; "
                 f"layers {', '.join(f'{k}:{v:+.3f}' for k, v in p['layer_median'].items())}")
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=None)
    ap.add_argument("--e1-dirs", nargs="+", default=None)
    ap.add_argument("--max-neg", type=float, default=0.25)
    ap.add_argument("--outdir", default="runs")
    a = ap.parse_args(argv)
    runs, common = load_e1_runs(a.outdir, a.bits, a.e1_dirs, a.percdamp)
    tag = f"b{a.bits}" + (f"_{damp_tag(a.percdamp)}" if a.percdamp is not None else "")
    res = evaluate(load_rows(runs), a.max_neg)
    res.update(percdamp=fp_percdamp(common), e1_runs=[r["dir"] for r in runs])
    base = os.path.join(a.outdir, f"e1_gate_{tag}")
    json.dump(res, open(base + ".json", "w"), indent=2, default=str)
    md = render(res, tag)
    open(base + ".md", "w").write(md + "\n")
    print(md)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
