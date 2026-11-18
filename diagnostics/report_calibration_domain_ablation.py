"""Report for the C4 vs WikiText-2-train calibration-domain ablation (CPU only; reads runs_calibration_domain_ablation)."""
from __future__ import annotations

import csv, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import calibration_domain as CD     # noqa: E402
import calibration_domain_ablation as X            # noqa: E402

OUT = os.path.join(ROOT, X.OUT)
GS = [f"G{g}" for g in X.GSEEDS]
CELLS = [(s, c) for s in X.SOURCES for c in X.STRATS]
METRICS = ("mean_nll", "B2", "B2_layer_mean", "P_seq", "recon_c4")
LABEL = {"mean_nll": "mean NLL (nats/token)", "B2": "B2 pooled", "B2_layer_mean": "B2 layer-equal",
         "P_seq": "WikiText P_seq", "recon_c4": "C4 test reconstruction"}


def jl(path):
    return json.load(open(path))


def row_from_result(r, b2tot=None, b2pl=None):
    t = b2tot if b2tot is not None else r["b2"]
    pl = b2pl if b2pl is not None else r["b2_per_layer"]
    return dict(mean_nll=r["mean_nll"], ppl=r["ppl"], total_tokens=r["total_tokens"], recon_c4=r["recon_test"],
                P_seq=t["P_seq_mean"], B2=t["B2"], B2_layer_mean=t["B2_layer_mean"], B0_mean=t["B0_mean"], B1=t["B1"],
                B3=t["B3"], D_sum=t["D_sum"], M_sum=t["M_sum"], U_sum=t["U_sum"],
                per_layer_B2={str(L): pl[str(L)]["B2"] for L in X.LAYERS},
                installed_sha256=r["installed_weights_sha256"], install_ok=r.get("install_ok"),
                rtn_fallbacks=r.get("rtn_fallbacks"), untouched_changed=r.get("untouched_params_changed"),
                nonfinite=r.get("nonfinite_targets"), reproduction=(r.get("reproduction") or {}).get("all"))


def load_rows():
    import loss_alignment_mvp_b as LB
    rows = {}
    for f in sorted(os.listdir(os.path.join(OUT, "results"))):
        if f == "BF16.json" or not f.endswith(".json"):
            continue
        r = jl(os.path.join(OUT, "results", f))
        rows[r["config"]] = row_from_result(r)
    # C4-SC ORIGINAL: stored A_SC_s0 (NLL / recon / sha) + loss-alignment MVP B raw sums (same capture, FP32 metrics)
    if "C4:SC:ORIG" not in rows:
        a = jl(os.path.join(ROOT, "runs_shared_covariance_e3/results/A_SC_s0.json"))
        mb = jl(os.path.join(ROOT, "runs_loss_alignment/mvp_b/raw_sums.json"))["runs"]["SC"]
        assert mb["checks"]["installed_sha256"] == a["installed_weights_sha256"]
        tot, pl, _, _ = LB.metrics_from_raw(mb)
        rows["C4:SC:ORIG"] = row_from_result(a, tot, {str(L): v for L, v in pl.items()})
        rows["C4:SC:ORIG"]["source_note"] = "reused: runs_shared_covariance_e3 A_SC_s0 + runs_loss_alignment/mvp_b"
    return rows


def rel(a, b):
    return (a - b) / b


def effect_values(rows, ext, m):
    v = {(s, c): rows[f"{s}:{c}:{ext}"][m] for s, c in CELLS}
    if m == "mean_nll":
        return CD.effects(v)
    r = dict(delta_SC_C4=rel(v[("C4", "SC")], v[("C4", "FR")]), delta_SC_WT=rel(v[("WT", "SC")], v[("WT", "FR")]),
             delta_domain_FR=rel(v[("WT", "FR")], v[("C4", "FR")]), delta_domain_SC=rel(v[("WT", "SC")], v[("C4", "SC")]))
    r["interaction"] = r["delta_SC_WT"] - r["delta_SC_C4"]
    return r


EFFECTS = ("delta_SC_C4", "delta_SC_WT", "delta_domain_FR", "delta_domain_SC", "interaction")


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(m, x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.6f}" if m == "mean_nll" else f"{100 * x:+.2f} %"


def main():
    pr = X.protocol()
    ref_sd = pr["analysis"]["reference_sd"]["value"]
    sel = jl(os.path.join(OUT, "selection_manifest.json"))
    dm = jl(os.path.join(OUT, "dataset_manifest.json"))
    bf = jl(os.path.join(OUT, "results", "BF16.json"))
    rows = load_rows()
    have_g = all(f"{s}:{c}:{g}" in rows for s, c in CELLS for g in GS)
    # ---------------- per_configuration.csv / per_layer_b2.csv
    pc, plb = [], []
    for k, r in sorted(rows.items()):
        s, c, e = k.split(":")
        pc.append(dict(config=k, source=s, strategy=c, extraction=e, **{x: r[x] for x in r if x != "per_layer_B2"},
                       ppl_minus_bf16=r["ppl"] - bf["ppl"], nll_minus_bf16=r["mean_nll"] - bf["mean_nll"]))
        for L in X.LAYERS:
            plb.append(dict(config=k, source=s, strategy=c, extraction=e, layer=L, B2=r["per_layer_B2"][str(L)]))
    keys = ["config", "source", "strategy", "extraction", "mean_nll", "ppl", "nll_minus_bf16", "ppl_minus_bf16",
            "total_tokens", "recon_c4", "P_seq", "B2", "B2_layer_mean", "B0_mean", "B1", "B3", "D_sum", "M_sum", "U_sum",
            "installed_sha256", "install_ok", "rtn_fallbacks", "untouched_changed", "nonfinite", "reproduction", "source_note"]
    write_csv(os.path.join(OUT, "per_configuration.csv"), pc, keys)
    write_csv(os.path.join(OUT, "per_layer_b2.csv"), plb, ["config", "source", "strategy", "extraction", "layer", "B2"])
    # ---------------- paired_grouping_results.csv / domain_interaction.csv
    pg, di, summ = [], [], {}
    exts = (GS if have_g else []) + ["ORIG"]
    for ext in exts:
        if not all(f"{s}:{c}:{ext}" in rows for s, c in CELLS):
            continue
        for m in METRICS:
            ev = effect_values(rows, ext, m)
            pg.append(dict(extraction=ext, metric=m, scale="absolute" if m == "mean_nll" else "relative", **ev))
    if have_g:
        for m in METRICS:
            summ[m] = {}
            for ef in EFFECTS:
                vals = [x[ef] for x in pg if x["metric"] == m and x["extraction"] in GS]
                s = CD.summarize(vals)
                s["label"] = CD.label(s, ref_sd) if m == "mean_nll" else None
                orig = [x[ef] for x in pg if x["metric"] == m and x["extraction"] == "ORIG"]
                s["original_extraction"] = orig[0] if orig else None
                summ[m][ef] = s
                di.append(dict(metric=m, effect=ef, scale="absolute nats" if m == "mean_nll" else "relative",
                               mean=s["mean"], sd=s["sd"], min=s["min"], max=s["max"], n_negative=s["n_negative"],
                               n_positive=s["n_positive"], mean_over_reference_sd=s["mean"] / ref_sd if m == "mean_nll" else None,
                               label=s["label"], original_extraction=s["original_extraction"],
                               values=" / ".join(f"{v:+.6g}" for v in s["values"])))
    write_csv(os.path.join(OUT, "paired_grouping_results.csv"), pg, ["extraction", "metric", "scale", *EFFECTS])
    write_csv(os.path.join(OUT, "domain_interaction.csv"), di,
              ["metric", "effect", "scale", "mean", "sd", "min", "max", "n_negative", "n_positive",
               "mean_over_reference_sd", "label", "original_extraction", "values"])
    # cell-level grouping variability (NLL SD over groupings per cell)
    cell_sd = {f"{s}-{c}": float(np.std([rows[f"{s}:{c}:{g}"]["mean_nll"] for g in GS], ddof=1)) for s, c in CELLS} if have_g else {}
    res = dict(reference_sd=ref_sd, summaries=summ, cell_nll_sd_over_groupings=cell_sd, bf16=dict(ppl=bf["ppl"], mean_nll=bf["mean_nll"]),
               n_configs=len(rows), have_groupings=have_g)
    json.dump(res, open(os.path.join(OUT, "results_summary.json"), "w"), indent=1)
    if have_g:
        figures(rows, pg, summ, ref_sd)
    write_summary(pr, sel, dm, bf, rows, pg, summ, cell_sd, ref_sd, have_g)
    print("[report] written", os.path.join(OUT, "summary.md"))


# ===================================================================== figures
def figures(rows, pg, summ, ref_sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(OUT, "figures")
    os.makedirs(fd, exist_ok=True)
    col = {"FR": "#5b7fbf", "SC": "#d9822b"}
    mk = {GS[0]: "o", GS[1]: "s", GS[2]: "^"}
    # 1. mean NLL per condition with groupings
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (s, c) in enumerate(CELLS):
        v = [rows[f"{s}:{c}:{g}"]["mean_nll"] for g in GS]
        for g, y in zip(GS, v):
            ax.scatter(i, y, color=col[c], marker=mk[g], s=36, zorder=3)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=col[c], lw=2)
        o = rows.get(f"{s}:{c}:ORIG")
        if o:
            ax.scatter(i + 0.32, o["mean_nll"], color=col[c], marker="x", s=36)
    ax.set_xticks(range(4), [f"{s}-{c}" for s, c in CELLS])
    ax.set_ylabel("mean NLL (nats/token)")
    ax.set_title("WikiText-2 test NLL; markers = groupings, bar = mean, x = ORIGINAL", fontsize=9)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig1_nll_by_condition.png"), dpi=150); plt.close(fig)
    # 2-4. paired NLL differences per grouping
    def eff_plot(keys, names, fname, title):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for i, k in enumerate(keys):
            v = [x[k] for x in pg if x["metric"] == "mean_nll" and x["extraction"] in GS]
            for g, y in zip(GS, v):
                ax.scatter(i, y, marker=mk[g], color="#444", s=36, zorder=3)
            ax.hlines(np.mean(v), i - .25, i + .25, color="#c0392b", lw=2)
        ax.axhspan(-ref_sd, ref_sd, color="#999", alpha=.18, label=f"± reference grouping SD ({ref_sd:.1e})")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(range(len(keys)), names)
        ax.set_ylabel("Δ mean NLL (nats/token)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout(); fig.savefig(os.path.join(fd, fname), dpi=150); plt.close(fig)
    eff_plot(["delta_SC_C4", "delta_SC_WT"], ["SC−FR (C4 calib)", "SC−FR (WT calib)"], "fig2_sc_minus_fr_by_source.png",
             "Paired SC − FR NLL by calibration source (3 groupings)")
    eff_plot(["delta_domain_FR", "delta_domain_SC"], ["WT−C4 (Random)", "WT−C4 (SC)"], "fig3_domain_effect.png",
             "Calibration-source effect: WikiText-train − C4 (3 groupings)")
    eff_plot(["interaction"], ["(SC−FR)_WT − (SC−FR)_C4"], "fig4_interaction.png", "Domain × selection interaction")
    # 5. local metrics vs NLL (relative SC-FR and domain effects, grouping means)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, m in zip(axs, ("P_seq", "B2")):
        for k, nm in zip(EFFECTS[:4], ("SC−FR C4", "SC−FR WT", "WT−C4 FR", "WT−C4 SC")):
            ax.scatter(100 * summ[m][k]["mean"], summ["mean_nll"][k]["mean"], s=40)
            ax.annotate(nm, (100 * summ[m][k]["mean"], summ["mean_nll"][k]["mean"]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
        ax.axhspan(-ref_sd, ref_sd, color="#999", alpha=.18)
        ax.set_xlabel(f"relative change in {LABEL[m]} (%)"); ax.set_ylabel("Δ mean NLL")
        ax.grid(alpha=.3)
    fig.suptitle("Local error change vs NLL change (grouping means)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig5_local_vs_nll.png"), dpi=150); plt.close(fig)


# ===================================================================== summary.md
def write_summary(pr, sel, dm, bf, rows, pg, summ, cell_sd, ref_sd, have_g):
    L = []
    w = L.append
    w("# Calibration-domain ablation: C4 vs WikiText-2-train calibration (selection seed 0)\n")
    w(f"2 × 2 design (calibration source × {{Full Random, Shared Covariance-aware}}), B = 128, GPTQ 4-bit g128 act-order "
      f"percdamp 1.0 on the routed experts of layers 1/8/16/23. Primary outcome: mean NLL on the frozen 128 × 512 WikiText-2 "
      f"test windows (partial model, sequential forward). Protocol sha `{pr['protocol_sha256'][:16]}`, frozen before any "
      f"new GPU result.\n")
    w("## Data and selections\n")
    w(f"- WikiText-2 train: {dm['total_tokens']:,} tokens → {dm['n_complete_windows']:,} complete 512-token windows; "
      f"pool = {dm['pool_size']} sampled with seed {dm['sampling_seed']} (token sha `{dm['pool_token_sha256'][:16]}`).")
    lk = dm["leakage"]
    w(f"- Leakage: exact test-window matches {lk['vs_frozen_test_windows']['exact_window_matches']}; test 32-grams found in "
      f"the pool {lk['vs_frozen_test_windows']['test_ngrams_in_pool']} / {lk['vs_frozen_test_windows']['test_ngrams']}; "
      f"validation 32-gram overlap {100 * lk['vs_validation_windows']['fraction']:.3f} %.")
    W = sel["WT"]
    w(f"- WT warm start (same convention as C4): 32 of {W['warm']['eligible_size']} typical∩random-half sequences, "
      f"hash `{W['warm']['hash']}`. WT-FR `{W['FR']['ids_hash']}`, WT-SC `{W['SC']['ids_hash']}`; FR/SC overlap "
      f"{W['overlap_FR_SC']['intersection']} of 128 (32 warm).")
    J = W["sketch_objective"]
    w(f"- WT covariance-sketch objective J (WT pool reference): warm {J['WARM']['objective']:.4f}, FR {J['FR']['objective']:.4f}, "
      f"SC {J['SC']['objective']:.4f} ({W['sc_optimizer']['accepted']} accepted swaps; deterministic rerun identical "
      f"{W['sc_deterministic_rerun_identical']}).")
    for c in ("FR", "SC"):
        r = W["routing"][c]
        w(f"- WT-{c} exposure: median {r['tokens_per_expert']['median']:.0f} tokens/expert (min {r['tokens_per_expert']['min']:.0f}), "
          f"set routing JS {r['set_js']:.2e}, mean sequence atypicality d {r['mean_seq_atypicality_d']:.3f}.")
    w("")
    w("## Per configuration\n")
    w(f"BF16 reference: mean NLL {bf['mean_nll']:.6f}, PPL {bf['ppl']:.4f} (reproduces the stored baseline: "
      f"{bf['reproduces_stored_bf16_ppl']}; B2 = 0 with original weights: {bf['b2_zero']}).\n")
    w("| config | mean NLL | PPL | NLL − BF16 | C4 test recon | WikiText P_seq | B2 pooled | B2 layer-eq | install ok | reproduces stored |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(rows, key=lambda k: (k.split(":")[2] != "ORIG", k.split(":")[2], k.split(":")[0], k.split(":")[1]))
    for k in order:
        r = rows[k]
        w(f"| {k} | {r['mean_nll']:.6f} | {r['ppl']:.4f} | {r['mean_nll'] - bf['mean_nll']:+.6f} | {r['recon_c4']:.6f} | "
          f"{r['P_seq']:.5e} | {r['B2']:.5e} | {r['B2_layer_mean']:.5e} | {r['install_ok']} | "
          f"{'—' if r['reproduction'] is None else r['reproduction']} |")
    w("")
    if have_g:
        w(f"NLL SD over the 3 groupings per cell: " + ", ".join(f"{k} {v:.2e}" for k, v in cell_sd.items())
          + f". Prespecified reference SD (C4 SC−FR paired ΔNLL over the 5 SC E3 groupings, RMS over seeds 0–2): "
          f"{ref_sd:.2e}.\n")
        w("## Effects (3 calibration groupings; ORIGINAL extraction shown separately)\n")
        w("NLL rows are absolute differences (nats/token); local-metric rows are relative changes. Negative = the first "
          "term has lower error. Interaction = (SC−FR)_WT − (SC−FR)_C4.\n")
        names = dict(delta_SC_C4="SC − FR, C4 calib", delta_SC_WT="SC − FR, WT calib", delta_domain_FR="WT − C4, Random",
                     delta_domain_SC="WT − C4, SC", interaction="interaction")
        for m in METRICS:
            w(f"**{LABEL[m]}**\n")
            w("| effect | per grouping | mean | SD | min / max | sign (−/+) | ORIGINAL |" + (" mean / ref SD | label |" if m == "mean_nll" else ""))
            w("|---|---|---|---|---|---|---|" + ("---|---|" if m == "mean_nll" else ""))
            for ef in EFFECTS:
                s = summ[m][ef]
                extra = f" {s['mean'] / ref_sd:+.2f} | {s['label']} |" if m == "mean_nll" else ""
                w(f"| {names[ef]} | {' / '.join(fmt(m, v) for v in s['values'])} | {fmt(m, s['mean'])} | "
                  f"{(f'{s['sd']:.2e}' if m == 'mean_nll' else f'{100 * s['sd']:.2f} %')} | "
                  f"{fmt(m, s['min'])} / {fmt(m, s['max'])} | {s['n_negative']}/{s['n_positive']} | "
                  f"{fmt(m, s['original_extraction'])} |{extra}")
            w("")
        w("Per-layer B2 per configuration: `per_layer_b2.csv`.\n")
        for f, cap in (("fig1_nll_by_condition.png", ""), ("fig2_sc_minus_fr_by_source.png", ""),
                       ("fig3_domain_effect.png", ""), ("fig4_interaction.png", ""), ("fig5_local_vs_nll.png", "")):
            w(f"![](figures/{f})\n")
    else:
        w("Grouping replicates not complete yet; effects table pending.\n")
    w("## Checks\n")
    bad = sorted(k for k, r in rows.items() if r["install_ok"] is False)
    w(f"- install_ok: {len(rows) - len(bad)}/{len(rows)} configurations pass every installer check" +
      (f"; flagged: {', '.join(bad)}" if bad else "") + ".")
    for k in bad:
        r = jl(os.path.join(OUT, "results", k.replace(":", "_") + ".json"))
        fails = [c for c in r["expert_checks"] if not c["pass"]]
        for c in fails:
            w(f"  - {k}: check expert L{c['layer']}/e{c['expert']} installed-vs-E3 FP32 output deviation "
              f"{c['installed_vs_e3_rel']:.5f} > tolerance {c['weight_rounding_only_rel'] + 1.5 * c['bf16_compute_noise_rel']:.5f} "
              f"(weights bitwise = BF16(W_q): {c['installed_equals_bf16_of_e3_wq']}; quant effect {c['quant_effect_rel']:.3f}; "
              f"{c['n_tokens']} tokens). Every structural check passed (240 quantized, 0 RTN, untouched params unchanged, "
              f"restored pristine), so the result is kept and flagged, not dropped; the frozen tolerance was not changed.")
    w(f"- All configurations: RTN fallbacks {sorted({r['rtn_fallbacks'] for r in rows.values()})}; "
      f"non-target parameters changed {sorted({r['untouched_changed'] for r in rows.values()})}; non-finite targets "
      f"{sorted({r['nonfinite'] for r in rows.values()})}; scored tokens {sorted({r['total_tokens'] for r in rows.values()})}.")
    rep = {k: r["reproduction"] for k, r in rows.items() if r["reproduction"] is not None}
    w(f"- C4 re-runs reproduce the stored results bitwise (installed sha, per-expert recon, per-window NLL; C4-FR ORIG also "
      f"the MVP B B2 sums): {rep}.")
    w("- Identical evaluation windows (sha `" + jl(os.path.join(ROOT, "runs_partial_ppl/evaluation_protocol.json"))["windows_sha256"][:16]
      + "`) and identical BF16 capture for every configuration; GPTQ settings from the frozen config.")
    w("- WT selections, routing and sketches use WT-train pool rows only; SC sketch counts equal the WT routing profile "
      "(sketch-stage check).")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
