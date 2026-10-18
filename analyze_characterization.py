"""Read-only report on finished E1/E2 outputs (no model, no GPU, no refits that
feed back into E2/E3).

A. E1 coverage per (char_seed, n0): attempted / labelled expert-replicates,
   exclusions by reason, routed-count distribution of labelled rows.
B. E2 predictive results per signal, pooled and per starting budget n0:
   leave-one-char-seed-out (LOSO) MSE, delta vs count-only, Spearman, count-
   binned Spearman, incremental R^2 over log(count).
     pooled_model_slice: the pooled LOSO model (what E3 uses), its held-out
                         errors restricted to rows with that n0;
     n0_specific:        LOSO refit using only rows with that n0 (does the
                         signal's value depend on calibration budget?).
C. Per-fold detail: held-out MSE vs count-only and coefficients per fold.
Rows are the same all-finite rows E2 fits on.

Output: <outdir>/characterization_report_b{bits}.json and .md
"""
from __future__ import annotations
import argparse, csv, json, os

import numpy as np

from moe_calib.runid import load_e1_runs
from moe_calib.stats import spearman, binned_spearman, incremental_r2
from moe_calib.utility import SIGNALS, ORIENT, fit_predictor, predict

E1_REASONS = ("zero_calib_tokens", "below_min_calib_tokens", "insufficient_paired_eval_seqs",
              "nonfinite_error", "nonpositive_base_error")
PCTS = (0, 5, 25, 50, 75, 95, 100)


def e1_coverage(runs):
    out = []
    for run in runs:
        rows = [json.loads(l) for l in open(os.path.join(run["dir"], "raw.jsonl"))]
        rows = [r for r in rows if not r.get("marker")]
        for n0 in run["fp"]["n0"]:
            rs = [r for r in rows if r["n0"] == n0]
            lab = [r for r in rs if "excluded" not in r]
            c = np.array([r["tokens_n0"] for r in lab], float)
            q = np.percentile(c, PCTS).tolist() if c.size else [float("nan")] * 7
            out.append(dict(char_seed=run["char_seed"], n0=n0, attempted=len(rs),
                            labelled=len(lab),
                            **{f"excl_{k}": sum(r.get("excluded") == k for r in rs)
                               for k in E1_REASONS},
                            **{f"count_p{p}": v for p, v in zip(PCTS, q)}))
    return out


def load_obs(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    d = {k: f(k) for k in ("characterization_seed", "starting_budget", "routed_count",
                           "future_rel_delta", "drift", "relse", "margin", "entropy", "layer")}
    ok = np.isfinite(d["future_rel_delta"]) & (d["routed_count"] > 0)
    for s in SIGNALS:
        if s != "count":
            ok &= np.isfinite(d[s])
    return {k: v[ok] for k, v in d.items()}


def _val(o, s):
    return None if s == "count" else o[s]


def loso(o, s):
    """Held-out predictions per row + per-fold (seed, mse, coef)."""
    pred = np.full(o["future_rel_delta"].size, np.nan)
    folds = []
    seeds = np.unique(o["characterization_seed"])
    for g in seeds:
        tr, te = o["characterization_seed"] != g, o["characterization_seed"] == g
        ent = fit_predictor(s, o["routed_count"][tr], None if s == "count" else o[s][tr],
                            o["future_rel_delta"][tr])
        pred[te] = predict(ent, o["routed_count"][te], None if s == "count" else o[s][te])
        folds.append(dict(heldout_seed=int(g),
                          mse=float(np.mean((pred[te] - o["future_rel_delta"][te]) ** 2)),
                          coef=ent["coef"]))
    return pred, folds


def metrics(o, mask, preds):
    y, cnt = o["future_rel_delta"][mask], o["routed_count"][mask]
    mse = {s: float(np.mean((preds[s][mask] - y) ** 2)) for s in SIGNALS}
    out = {}
    for s in SIGNALS:
        v = ORIENT[s] * (cnt if s == "count" else o[s][mask])
        r0, r1, dr2 = incremental_r2(v, y, cnt)
        out[s] = dict(loso_mse=mse[s], delta_vs_count=mse[s] - mse["count"],
                      rel_delta_vs_count=(mse[s] - mse["count"]) / mse["count"],
                      spearman=spearman(v, y), binned_spearman=binned_spearman(v, y, cnt),
                      incremental_r2=dr2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args(argv)
    tag = f"b{args.bits}"
    art = json.load(open(os.path.join(args.outdir, f"e2_model_{tag}.json")))
    runs, _ = load_e1_runs(args.outdir, args.bits, art["e1_runs"])
    o = load_obs(os.path.join(args.outdir, f"e2_signals_{tag}.csv"))
    n0s = sorted(set(o["starting_budget"].astype(int).tolist()))

    preds, folds = {}, {}
    for s in SIGNALS:
        preds[s], folds[s] = loso(o, s)
    allm = np.ones(o["future_rel_delta"].size, bool)
    report = dict(selected_signal=art["selected_signal"],
                  selection_criterion=art["selection_criterion"],
                  characterization_seeds=art["characterization_seeds"],
                  n_rows=int(allm.sum()),
                  e1_coverage=e1_coverage(runs),
                  pooled=metrics(o, allm, preds),
                  pooled_model_slice_by_n0={}, n0_specific_by_n0={},
                  folds={s: [dict(f, count_mse=folds["count"][i]["mse"])
                             for i, f in enumerate(folds[s])] for s in SIGNALS},
                  predictors={s: dict(coef=art["predictors"][s]["coef"],
                                      orientation=art["predictors"][s]["orientation"],
                                      count_range=art["predictors"][s]["count_range"])
                              for s in SIGNALS})
    for n0 in n0s:
        m = o["starting_budget"] == n0
        report["pooled_model_slice_by_n0"][str(n0)] = metrics(o, m, preds)
        sub = {k: v[m] for k, v in o.items()}
        sp = {s: loso(sub, s)[0] for s in SIGNALS}
        report["n0_specific_by_n0"][str(n0)] = metrics(sub, np.ones(m.sum(), bool), sp)
        report["n0_specific_by_n0"][str(n0)]["n_rows"] = int(m.sum())

    base = os.path.join(args.outdir, f"characterization_report_{tag}")
    json.dump(report, open(base + ".json", "w"), indent=2, default=float)

    L = [f"# Characterization report ({tag})", "",
         f"selected: **{report['selected_signal']}** ({report['selection_criterion']}); "
         f"seeds {report['characterization_seeds']}; {report['n_rows']} rows", "",
         "## E1 coverage (expert x replicate rows)", "",
         "| seed | n0 | attempted | labelled | " + " | ".join(E1_REASONS) +
         " | count min/p5/p25/med/p75/p95/max |", "|" + "---|" * (5 + len(E1_REASONS))]
    for r in report["e1_coverage"]:
        L.append(f"| {r['char_seed']} | {r['n0']} | {r['attempted']} | {r['labelled']} | " +
                 " | ".join(str(r[f'excl_{k}']) for k in E1_REASONS) + " | " +
                 "/".join(f"{r[f'count_p{p}']:.0f}" for p in PCTS) + " |")

    def table(title, m):
        L.extend(["", f"## {title}", "",
                  "| signal | LOSO MSE | delta vs count | rel. | Spearman | binned Spearman | incr. R2 |",
                  "|---|---|---|---|---|---|---|"])
        for s in SIGNALS:
            x = m[s]
            L.append(f"| {s} | {x['loso_mse']:.6g} | {x['delta_vs_count']:+.3g} | "
                     f"{100 * x['rel_delta_vs_count']:+.2f}% | {x['spearman']:.3f} | "
                     f"{x['binned_spearman']:.3f} | {x['incremental_r2']:.4f} |")
    table("Pooled", report["pooled"])
    for n0 in n0s:
        table(f"n0={n0}: pooled model, held-out rows with this n0",
              report["pooled_model_slice_by_n0"][str(n0)])
        table(f"n0={n0}: n0-specific LOSO refit "
              f"({report['n0_specific_by_n0'][str(n0)]['n_rows']} rows)",
              report["n0_specific_by_n0"][str(n0)])
    L.extend(["", "## Per-fold (held-out seed): MSE vs count-only, beta_signal", "",
              "| signal | " + " | ".join(f"seed {f['heldout_seed']}" for f in folds["count"]) + " |",
              "|---|" + "---|" * len(folds["count"])])
    for s in SIGNALS:
        cells = [f"{f['mse']:.4g} ({100 * (f['mse'] - f['count_mse']) / f['count_mse']:+.1f}%), "
                 f"b_sig={f['coef'].get('beta_signal', float('nan')):+.3g}, "
                 f"b_cnt={f['coef']['beta_count']:+.3g}" for f in report["folds"][s]]
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.extend(["", "## Final predictors (all seeds)", "", "| signal | orientation | coef | count support |",
              "|---|---|---|---|"])
    for s, p in report["predictors"].items():
        L.append(f"| {s} | {p['orientation']:+.0f} | " +
                 ", ".join(f"{k}={v:+.4g}" for k, v in p["coef"].items()) +
                 f" | [{p['count_range'][0]:.0f}, {p['count_range'][1]:.0f}] |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return report


if __name__ == "__main__":
    main()
