"""Figures + summary.md for the inference-time routing amplification MVP (reads the analysis JSONs)."""
from __future__ import annotations
import json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import inference_routing_amplification as D                    # noqa: E402

p, c, jload, jdump = D.p, D.c, D.jload, D.jdump
FR_C, FC_C, REF_C, INK, INK2, GRID = "#2a78d6", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"FR": FR_C, "FC": FC_C}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                         "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def save(fig, name):
    os.makedirs(p("figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(p("figures", f"{name}.png"), dpi=160)
    fig.savefig(p("figures", f"{name}.pdf"))


def fig1(plt, div):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for j, s in enumerate(("FR", "FC")):
        ax = axs[0, j]
        for i, g in enumerate(D.ORDERS):
            pl = div[f"{s}_O{g}"]["per_layer"]
            ax.plot(range(24), [100 * x["frac_set_changed"] for x in pl], color=COL[s], alpha=0.35 + 0.13 * i,
                    lw=1.5, marker="o", ms=3, label=f"O{g}")
        for tag, ls in (("ORIGINAL", "--"), ("FP64", ":")):
            pl = div[f"{s}_{tag}"]["per_layer"]
            ax.plot(range(24), [100 * x["frac_set_changed"] for x in pl], color=REF_C, ls=ls, lw=1.5, label=tag)
        ax.axhline(0, color=INK, lw=1)
        ax.text(23, 0, "BF16 = 0", ha="right", va="bottom", color=INK2, fontsize=8)
        ax.set_title(f"{s}: % tokens with changed top-4 set vs BF16", fontsize=9, color=INK)
        ax2 = axs[1, j]
        for i, g in enumerate(D.ORDERS):
            v = div[f"{s}_O{g}"]["vs_original_frac_set_changed_per_layer"]
            ax2.plot(range(24), [100 * x for x in v], color=COL[s], alpha=0.35 + 0.13 * i, lw=1.5, marker="o", ms=3,
                     label=f"O{g}")
        ax2.axhline(0, color=INK, lw=1)
        ax2.set_title(f"{s}: % tokens with changed set vs ORIGINAL FP32 (order-induced)", fontsize=9, color=INK)
        ax2.set_xlabel("MoE layer (shaded = quantized)")
        for a in (ax, ax2):
            for L in D.LAYERS_Q:
                a.axvline(L, color=GRID, lw=3, zorder=0)
    axs[0, 0].set_ylabel("% of tokens"); axs[1, 0].set_ylabel("% of tokens")
    axs[0, 1].legend(fontsize=7, ncol=2)
    save(fig, "fig1_routing_divergence_by_layer"); plt.close(fig)


def fig2(plt, po):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(po))
    for s in ("FR", "FC"):
        ax.plot(x, [r[f"{s}_natural"] for r in po], color=COL[s], marker="o", ms=6, label=f"{s} natural")
        ax.plot(x, [r[f"{s}_frozen"] for r in po], color=COL[s], ls="--", marker="o", ms=6, mfc="white", label=f"{s} frozen")
    ax.set_xticks(x, [str(r["order"]) for r in po])
    ax.set_xlabel("accumulation order"); ax.set_ylabel("WikiText-2 PPL")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("PPL under natural vs BF16-frozen top-4 routing", fontsize=9)
    save(fig, "fig2_natural_vs_frozen_ppl"); plt.close(fig)


def fig3(plt, po, var):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(po))
    ax.axhline(0, color=INK, lw=1)
    ax.plot(x, [r["delta_natural"] for r in po], color=INK, marker="o", ms=7, label=f"natural  SD {var['sd_delta_natural']:.5f}")
    ax.plot(x, [r["delta_frozen"] for r in po], color=FC_C, ls="--", marker="s", ms=7, mfc="white",
            label=f"frozen   SD {var['sd_delta_frozen']:.5f}")
    ax.set_xticks(x, [str(r["order"]) for r in po])
    ax.set_xlabel("accumulation order"); ax.set_ylabel("PPL(FC) - PPL(FR)")
    ax.set_title(f"Paired FC - FR difference;  R_SD = {var['R_SD']:.3f}", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "fig3_paired_delta_variability"); plt.close(fig)


def fig4(plt):
    ref = torch.load(c("bf16_reference_routing.pt"))
    rm = (ref["pk"][..., 3] - ref["p_next"]).flatten()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    from moe_calib import inference_routing_override as IRO
    for ax, s in zip(axs, ("FR", "FC")):
        ch_all = torch.zeros_like(rm, dtype=torch.bool)
        qm_ch, qm_st = [], []
        for g in D.ORDERS:
            d = torch.load(D.run_file(f"{s}_O{g}", "natural"))
            sc, _ = IRO.set_disagreement(d["idx"], ref["idx"])
            sc = sc.flatten()
            ch_all |= sc
            qm = (d["pk"][..., 3] - d["p_next"]).flatten()
            qm_ch.append(qm[sc]); qm_st.append(qm[~sc][::50])
        for vals, lab, col, ls in ((rm[~ch_all][::50], "BF16 margin, set stable in all 5 (1/50 sample)", REF_C, "-"),
                                   (rm[ch_all], "BF16 margin, set changed in any order", COL[s], "-"),
                                   (torch.cat(qm_st), "quantized margin, stable (1/50 sample)", REF_C, ":"),
                                   (torch.cat(qm_ch), "quantized margin, changed", COL[s], ":")):
            v = np.sort(np.clip(vals.double().numpy(), 1e-7, None))
            if v.size:
                ax.plot(v, np.arange(1, v.size + 1) / v.size, color=col, ls=ls, lw=1.8, label=f"{lab} (n={v.size})")
        ax.set_xscale("log"); ax.set_xlabel("p(4) - p(5)  (clipped at 1e-7)")
        ax.set_title(f"{s}: top-4/top-5 margin ECDF (5 orders pooled)", fontsize=9)
    axs[0].set_ylabel("cumulative fraction")
    axs[1].legend(fontsize=6.5)
    save(fig, "fig4_margins_vs_disagreement"); plt.close(fig)


def fig5(plt):
    z = np.load(c("concentration_curves.npz"))
    grid = z["grid"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k in z.files:
        if k == "grid":
            continue
        n, m = k.split("__")
        ax.plot(100 * grid, z[k], color=COL[n[:2]], ls="-" if m == "natural" else "--", lw=1, alpha=0.6)
    for s in ("FR", "FC"):
        for m, ls in (("natural", "-"), ("frozen", "--")):
            ax.plot([], [], color=COL[s], ls=ls, label=f"{s} {m}")
    ax.plot(100 * grid, grid, color=GRID, lw=1, label="uniform")
    ax.set_xscale("log"); ax.set_xlim(0.01, 100)
    ax.set_xlabel("% of scored tokens (largest |dNLL| first)"); ax.set_ylabel("share of total |dNLL|")
    ax.set_title("Token-level |NLL(order) - NLL(ORIGINAL)| concentration", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "fig5_token_nll_concentration"); plt.close(fig)


def fig6(plt, assoc):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ev in zip(axs, ("vs_ORIGINAL", "vs_BF16")):
        keys = ["no_local_no_prefix", "prefix_only", "local_any"]
        x = np.arange(len(keys))
        for j, s in enumerate(("FR", "FC")):
            vals = np.array([[assoc[f"{s}_O{g}"][ev]["prefix"][k]["mean"] or np.nan for k in keys] for g in D.ORDERS])
            for i in range(len(D.ORDERS)):
                ax.scatter(x + (j - 0.5) * 0.25, vals[i], color=COL[s], s=18, alpha=0.7, label=s if i == 0 else None)
        ns = [[assoc[f"{s}_O{g}"][ev]["prefix"][k]["n"] for s in ("FR", "FC") for g in D.ORDERS] for k in keys]
        ax.set_xticks(x, [f"{lab}\nn = {min(v)}-{max(v)}" for lab, v in
                          zip(["no flip at or\nbefore position", "flip only in\ncausal prefix", "flip at the\nposition"], ns)])
        ax.set_title(f"mean |dNLL| by routing-flip context ({ev.replace('_', ' ')})", fontsize=9)
    axs[0].set_ylabel("mean |NLL(order) - NLL(ORIGINAL)|")
    axs[1].legend(fontsize=7)
    fig.text(0.5, 0.005, "Descriptive association only (one dot per order); not causal attribution.", ha="center",
             fontsize=7, color=INK2)
    save(fig, "fig6_routing_vs_nll_sensitivity"); plt.close(fig)


def f(x, n=6):
    return "n/a" if x is None else f"{x:.{n}f}"


def report():
    plt = _style()
    dec = jload(p("decision.json"))
    div = jload(p("routing_divergence.json"))["per_config"]
    fig1(plt, div)
    if dec["verdict"] == "ROUTING-NO-FLIPS":
        return
    po = jload(p("paired_order_comparisons.json"))["per_order"]
    var, can = jload(p("variance_comparison.json")), jload(p("canonical_reference_comparison.json"))
    tokc, assoc = jload(p("token_nll_concentration.json")), jload(p("routing_nll_association.json"))["per_config"]
    corr, cost, val = jload(p("correctness_results.json")), jload(p("cost_results.json")), jload(p("routing_override_validation.json"))
    hreg = jload(p("historical_regression.json"))
    fig2(plt, po); fig3(plt, po, var); fig4(plt); fig5(plt); fig6(plt, assoc)
    L = []
    w = L.append
    w("# Inference-time routing amplification MVP - summary\n")
    w(f"Decision: **{dec['verdict']}** (conclusion {dec['conclusion']}); R_SD = {dec['R_SD']:.4f}; "
      f"protocol sha256 {dec['protocol_sha256'][:16]}...\n")
    w("## Parity\n")
    w("| test | result |\n|---|---|")
    w(f"| A record-only, BF16 | per-window bitwise {val['A_bf16']['record_vs_plain']['bitwise']}, layer hashes {val['A_bf16']['layer_hashes_equal']} |")
    w(f"| A record-only, {val['A_quant']['config']} | per-window bitwise {val['A_quant']['record_vs_plain']['bitwise']}, layer hashes {val['A_quant']['layer_hashes_equal']}, vs historical {val['A_quant']['plain_vs_historical']['bitwise']} |")
    b, cq = val["B_bf16_self_routing"], val["C_quant_self_routing"]
    w(f"| B BF16 self-routing | per-window {b['per_window']['bitwise']}, per-token {b['token_nll_bitwise']}, 24 layer hashes {b['layer_hashes_equal']}, final {b['final_hidden_equal']} |")
    w(f"| C quantized self-routing ({cq['config']}) | per-window {cq['per_window']['bitwise']}, per-token {cq['token_nll_bitwise']}, layer hashes {cq['layer_hashes_equal']}, selections {cq['selections_equal']}, router probs {cq['router_probs_equal']}, PPL {cq['ppl_equal']} |")
    w(f"\nHistorical regression: all natural evaluations reproduce historical per-window NLL bitwise: {hreg['all_bitwise']}.\n")
    w("## Routing divergence vs BF16 (natural inference)\n")
    w("| config | % tokens set changed (all layers) | % tokens any layer changed | first layer with change | L0/L1 identical | max-layer % |\n|---|---|---|---|---|---|")
    for n in D.PRIMARY + D.REFERENCES:
        d = div[n]
        mx = max(d["per_layer"], key=lambda x: x["frac_set_changed"])
        w(f"| {n} | {100 * d['frac_set_changed_all']:.3f} | {100 * d['frac_tokens_any_layer_changed']:.2f} | "
          f"{d['first_layer_with_any_change']} | {d['layers_0_1_identical']} | L{mx['layer']}: {100 * mx['frac_set_changed']:.3f} |")
    w("\n## PPL (natural vs BF16-frozen routing)\n")
    w("| order | FR nat | FR frozen | FC nat | FC frozen | Delta nat | Delta frozen | delta_g |\n|---|---|---|---|---|---|---|---|")
    for r in po:
        w(f"| {r['order']} | {f(r['FR_natural'])} | {f(r['FR_frozen'])} | {f(r['FC_natural'])} | {f(r['FC_frozen'])} | "
          f"{r['delta_natural']:+.6f} | {r['delta_frozen']:+.6f} | {r['delta_g']:+.6f} |")
    w(f"\nSD(Delta) natural {var['sd_delta_natural']:.6f} (historical {var['historical_sd_delta_natural']:.6f}), "
      f"frozen {var['sd_delta_frozen']:.6f}, **R_SD = {var['R_SD']:.4f}**. "
      f"SD FR {var['sd_FR_natural']:.6f} -> {var['sd_FR_frozen']:.6f}; SD FC {var['sd_FC_natural']:.6f} -> {var['sd_FC_frozen']:.6f}. "
      f"Mean Delta {var['mean_delta_natural']:+.6f} -> {var['mean_delta_frozen']:+.6f}; positive Delta "
      f"{var['n_positive_delta_natural']}/5 natural, {var['n_positive_delta_frozen']}/5 frozen.\n")
    w("## References (descriptive; not in the 5-order SD)\n")
    w("| reference | FR nat | FR frozen | FC nat | FC frozen | Delta nat | Delta frozen |\n|---|---|---|---|---|---|---|")
    for t in ("ORIGINAL", "FP64"):
        x = can[t]
        w(f"| {t} | {f(x['FR_natural'])} | {f(x['FR_frozen'])} | {f(x['FC_natural'])} | {f(x['FC_frozen'])} | "
          f"{x['delta_natural']:+.6f} | {x['delta_frozen']:+.6f} |")
    w("\n## Token-level NLL vs ORIGINAL (mean over the 10 primary configs)\n")
    for m in ("natural", "frozen"):
        a = tokc[f"mean_over_primary_{m}"]
        w(f"- {m}: mean |dNLL| {a['mean_abs']:.2e}, median {a['median_abs']:.2e}, p99 {a['p99']:.2e}, max {a['max']:.3f}; "
          f"top 1% tokens carry {100 * tokc[f'mean_top1pct_share_{m}']:.1f}% and top 10% carry "
          f"{100 * tokc[f'mean_top10pct_share_{m}']:.1f}% of total |dNLL|")
    w("\n## Correctness\n")
    w("| check | pass |\n|---|---|")
    for k, v in corr["checks"].items():
        w(f"| {k} | {v} |")
    w(f"\n## Cost\n\nnatural eval {cost['mean_natural_eval_s']:.1f} s, frozen {cost['mean_frozen_eval_s']:.1f} s, historical "
      f"plain {cost['mean_plain_eval_s']:.1f} s; peak GPU {cost['peak_gpu_gb']:.2f} GB; peak CPU {cost['peak_cpu_gb']:.1f} GB; "
      f"storage {cost['storage_bytes_runs'] / 1e9:.2f} GB; GPTQ solves 0.\n")
    open(p("summary.md"), "w").write("\n".join(L) + "\n")
    jdump(dict(decision=dec, variance=var, canonical=can, parity={k: v.get("passed") for k, v in val.items()},
               correctness_all_pass=corr["all_pass"], cost=cost,
               token_concentration={k: v for k, v in tokc.items() if k.startswith("mean")}), p("summary.json"))
    print("report written")
