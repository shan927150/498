"""E2 — Do cheap signals computed AT n0 predict the E1 labels, beyond
routed-token count?

Signals, all from the SAME base draw E1 used (read from runs/e1_draws.json,
so nothing depends on RNG replay):
  count    routed tokens n_e                         (exposure baseline)
  drift    prefix drift of normalized Hessian        (proposal's original)
  relse    between-block relative std. error of H_bar: the t-statistic /
           CI-width signal the instructor suggested
  margin   mean router top-k margin p_k - p_{k+1}    ("router uncertainty";
           LOW margin = uncertain routing -> oriented as -margin)
  entropy  mean routing entropy of tokens routed to e
Targets (E1): delta (effect size) and significant (BH q<0.1, majority over reps).
Metrics: AUROC, Spearman, count-binned Spearman, incremental R^2 over
log(count) — the last is the central "beyond exposure" test.
Template: Katharopoulos & Fleuret (ICML 2018) — predict whether non-uniform
treatment will pay off BEFORE spending compute on it; E2 is the go/no-go
gate for E3.
"""
from __future__ import annotations
import argparse, csv, json, os

import numpy as np

from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, load_layer_cache
from moe_calib.profiler import Profiler
from moe_calib.stats import auroc, spearman, binned_spearman, incremental_r2

SIGNALS = ("count", "drift", "relse", "margin", "entropy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--K", type=int, default=4, help="must match E1 --K")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()
    tag = f"b{args.bits}"
    meta = load_meta(args.cache_dir)
    draws = {int(k): v for k, v in
             json.load(open(os.path.join(args.outdir, "e1_draws.json"))).items()}

    lab: dict[tuple, dict] = {}
    for row in csv.DictReader(open(os.path.join(args.outdir, f"e1_labels_{tag}.csv"))):
        k = (int(row["layer"]), int(row["expert"]), int(row["n0"]))
        d = lab.setdefault(k, {"delta": [], "sig": []})
        d["delta"].append(float(row["rel_delta"]))
        d["sig"].append(int(row["significant"]))
    n0s = sorted({k[2] for k in lab})
    layers = sorted({k[0] for k in lab})

    model, _ = MU.load_model()
    E = MU.num_experts(model)
    blocks = list(range(args.K))
    sig_rows = []
    for L in layers:
        cache = load_layer_cache(args.cache_dir, L)
        for n0 in n0s:
            prof = Profiler(model, L, K=args.K).run(cache, draws[n0]["base"])
            for e in range(E):
                if (L, e, n0) not in lab:
                    continue
                key = (L, e, "gu")
                margin, ent = prof.router_signals(e)
                sig_rows.append(dict(
                    layer=L, expert=e, n0=n0, count=prof.acc.tokens(key),
                    drift=prof.acc.drift(key, blocks),
                    relse=prof.acc.rel_std_err(key, blocks),
                    margin=margin, entropy=ent,
                    delta=float(np.mean(lab[(L, e, n0)]["delta"])),
                    sig=int(np.mean(lab[(L, e, n0)]["sig"]) >= 0.5)))
            prof.free()
            print(f"[E2] L={L} n0={n0} done", flush=True)
        del cache

    report = {}
    for n0 in n0s:
        R = [r for r in sig_rows if r["n0"] == n0]
        delta = np.array([r["delta"] for r in R])
        sigl = np.array([r["sig"] for r in R])
        cnt = np.array([r["count"] for r in R], float)
        rep = {"n_experts": len(R), "n_significant": int(sigl.sum())}
        for s in SIGNALS:
            v = np.array([r[s] for r in R], float)
            v = -v if s == "margin" else v
            r0, r1, dr2 = incremental_r2(v, delta, cnt)
            rep[s] = dict(auroc=auroc(v, sigl), spearman=spearman(v, delta),
                          binned_spearman=binned_spearman(v, delta, cnt),
                          r2_count_only=r0, r2_with_signal=r1, incremental_r2=dr2)
        report[f"n0={n0}"] = rep
        print(f"\n[E2] n0={n0}: {rep['n_significant']}/{rep['n_experts']} significant")
        print(f"    {'signal':8s} {'AUROC':>6s} {'rho':>6s} {'bin_rho':>7s} {'dR2':>7s}")
        for s in SIGNALS:
            m = rep[s]
            print(f"    {s:8s} {m['auroc']:6.3f} {m['spearman']:6.3f} "
                  f"{m['binned_spearman']:7.3f} {m['incremental_r2']:7.4f}")

    with open(os.path.join(args.outdir, f"e2_report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(os.path.join(args.outdir, f"e2_signals_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
        w.writeheader()
        w.writerows(sig_rows)
    print(f"\n[E2] wrote {args.outdir}/e2_report_{tag}.json")


if __name__ == "__main__":
    main()
