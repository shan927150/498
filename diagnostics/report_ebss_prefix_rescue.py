"""Report for the C4-prefix diversity-rescue study: reconstruction / PPL tables, cost, frozen decisions, figures."""
from __future__ import annotations

import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import ebss_prefix as X          # noqa: E402

OUT = os.path.join(ROOT, "runs_ebss_prefix_rescue")
PILOT = os.path.join(ROOT, "runs_ebss_generation_pilot")
FID = os.path.join(ROOT, "runs_ebss_balance_fidelity")
BF16_PPL = 9.237090                               # frozen BF16 partial-model reference (pilot report)
LAYERS = [1, 8, 16, 23]
QSRC = dict(C4R_s0=os.path.join(PILOT, "results", "C4R_s0.json"), C4R_s1=os.path.join(PILOT, "results", "C4R_s1.json"),
            C4R_s2=os.path.join(PILOT, "results", "C4R_s2.json"), RS_PO=os.path.join(PILOT, "results", "PO.json"),
            RS_EBSS=os.path.join(FID, "results", "ALG1_RATE.json"), PX_PO=os.path.join(OUT, "results", "PX_PO.json"),
            PX_EBSS=os.path.join(OUT, "results", "PX_EBSS.json"))
LABEL = dict(C4R_s0="C4 Random s0 (= matched natural)", C4R_s1="C4 Random s1", C4R_s2="C4 Random s2",
             RS_PO="Random-start PO", RS_EBSS="Random-start ALG1_RATE", PX_PO="C4-prefix PO", PX_EBSS="C4-prefix ALG1_RATE")
# reference categorical palette (fixed slot order, validated upstream): entity -> slot, never by rank
COL = dict(RS_PO="#2a78d6", RS_EBSS="#eb6834", C4="#1baf7a", PX_PO="#eda100", PX_EBSS="#e87ba4")
INK, MUTED = "#222222", "#6b6b6b"


def p(*a):
    return os.path.join(OUT, *a)


def jl(path):
    return json.load(open(path))


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, path)


def col(name):
    return COL["C4"] if name.startswith("C4") else COL[name]


# ------------------------------------------------------------------ reconstruction / PPL
def per_expert(r):
    return {(e["layer"], e["expert"]): e for e in r["per_expert"]}


def recon_block(res):
    out = {}
    for n, r in res.items():
        pe = np.array([e["mean"] for e in r["per_expert"]])
        out[n] = dict(mean=r["recon"], median_expert=float(np.median(pe)), p90_expert=float(np.quantile(pe, .9)),
                      worst_expert=float(pe.max()),
                      per_layer={str(L): float(np.mean([e["mean"] for e in r["per_expert"] if e["layer"] == L])) for L in LAYERS},
                      valid=r["valid"], rtn_fallbacks=r["rtn_fallbacks"])
    return out


def pairwise(res, a, b):
    """a relative to b: per-expert relative difference (negative = a better)."""
    A, B = per_expert(res[a]), per_expert(res[b])
    keys = sorted(A)
    rel = np.array([A[k]["mean"] / B[k]["mean"] - 1 for k in keys]) * 100
    nat = np.array([B[k]["tokens"] for k in keys])
    dec = np.quantile(nat, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(dec, nat, side="right") - 1, 0, 9)
    return dict(delta_mean=res[a]["recon"] - res[b]["recon"], relative_pct=(res[a]["recon"] / res[b]["recon"] - 1) * 100,
                frac_experts_improved=float((rel < 0).mean()), median_expert_rel_pct=float(np.median(rel)),
                per_layer_rel_pct={str(L): float(np.mean([A[k]["mean"] for k in keys if k[0] == L])
                                                 / np.mean([B[k]["mean"] for k in keys if k[0] == L]) - 1) * 100 for L in LAYERS},
                by_natural_exposure_decile=[dict(decile=d, n=int((idx == d).sum()),
                                                 mean_rel_pct=float(rel[idx == d].mean()) if (idx == d).any() else None)
                                            for d in range(10)],
                per_expert_rel_pct=rel.tolist())


def main():
    have = {n: os.path.exists(f) for n, f in QSRC.items()}
    res = {n: jl(f) for n, f in QSRC.items() if have[n]}
    man = jl(p("logs", "capture_manifest.json")) if os.path.exists(p("logs", "capture_manifest.json")) else {}
    smoke = jl(p("smoke_results.json")) if os.path.exists(p("smoke_results.json")) else {}
    valid = bool(smoke.get("passed") and man.get("passed") and all(have.values())
                 and all(r["valid"] and r["rtn_fallbacks"] == 0 for r in res.values()))
    rb = recon_block(res)
    c4m = float(np.mean([res[f"C4R_s{s}"]["recon"] for s in range(3)])) if all(have[f"C4R_s{s}"] for s in range(3)) else None
    comps = {}
    pairs = [("PX_PO", "RS_PO"), ("PX_EBSS", "RS_EBSS"), ("PX_EBSS", "PX_PO")] + \
            [(m, f"C4R_s{s}") for m in ("PX_PO", "PX_EBSS", "RS_PO", "RS_EBSS") for s in range(3)]
    for a, b in pairs:
        if a in res and b in res:
            comps[f"{a}_vs_{b}"] = pairwise(res, a, b)
    deltas = {}
    if valid:
        deltas = dict(delta_PO=res["PX_PO"]["recon"] - res["RS_PO"]["recon"],
                      delta_EBSS=res["PX_EBSS"]["recon"] - res["RS_EBSS"]["recon"],
                      delta_balance_given_prefix=res["PX_EBSS"]["recon"] - res["PX_PO"]["recon"],
                      gap_recovery=dict(PO=X.gap_recovery(res["RS_PO"]["recon"], res["PX_PO"]["recon"], c4m),
                                        EBSS=X.gap_recovery(res["RS_EBSS"]["recon"], res["PX_EBSS"]["recon"], c4m)),
                      vs_c4_mean_pct={m: (res[m]["recon"] / c4m - 1) * 100 for m in ("RS_PO", "RS_EBSS", "PX_PO", "PX_EBSS")})
    jdump(dict(configs=rb, c4_mean=c4m, comparisons=comps, deltas=deltas, sources={n: os.path.relpath(f, ROOT) for n, f in QSRC.items()},
               note="negative deltas = improvement; RS_* vs PX_* is descriptive (initial content and generated-token "
                    "count differ); PX_PO vs PX_EBSS is the matched intervention"), p("reconstruction_results.json"))
    ppl = {}
    for n, r in res.items():
        ppl[n] = dict(ppl=r["ppl"], delta_vs_bf16=r["ppl"] - BF16_PPL,
                      delta_vs_c4r_s0=r["ppl"] - res["C4R_s0"]["ppl"] if "C4R_s0" in res else None,
                      per_window_nll=[a / b for a, b in zip(r["per_window_nll"], r["per_window_tokens"])])
    extra = {}
    if "PX_PO" in res and "PX_EBSS" in res:
        extra["PX_EBSS_minus_PX_PO"] = res["PX_EBSS"]["ppl"] - res["PX_PO"]["ppl"]
    extra["c4_seed_spread"] = [min(res[f"C4R_s{s}"]["ppl"] for s in range(3)), max(res[f"C4R_s{s}"]["ppl"] for s in range(3))] \
        if c4m is not None else None
    extra["noise_context"] = ("FP32 Hessian accumulation-order null: PPL SD ~0.0018 (update_hessian_order_null.md); "
                              "differences of this size are not interpreted")
    rec_rank = sorted(res, key=lambda n: res[n]["recon"]); ppl_rank = sorted(res, key=lambda n: res[n]["ppl"])
    from scipy.stats import spearmanr
    rho = float(spearmanr([res[n]["recon"] for n in res], [res[n]["ppl"] for n in res]).correlation) if len(res) > 2 else None
    jdump(dict(bf16=BF16_PPL, configs=ppl, **extra, recon_rank=rec_rank, ppl_rank=ppl_rank, spearman_recon_ppl=rho,
               windows="fixed 128 x 512 WikiText-2 windows"), p("ppl_results.json"))

    # ---------------- cost
    cost = {}
    for c in ("PX_PO", "PX_EBSS"):
        f = p("generation_logs", f"generation_cost_{c}.json")
        if os.path.exists(f):
            g = jl(f)
            cost[c] = dict(wall_s=g["wall_s"], gpu_seconds=g["gpu_seconds"], breakdown_s=g["breakdown_s"],
                           breakdown_frac=g["breakdown_frac"], accounting=g["accounting"],
                           s_per_generated_token=g["wall_s"] / g["accounting"]["emitted_continuation_tokens"],
                           s_per_final_token=g["wall_s"] / g["accounting"]["final_calibration_tokens"],
                           peak_gpu_gb=g["peak_gpu_gb"], peak_cpu_gb=g["peak_cpu_gb"], kv_cache_peak_gb=g["kv_cache_peak_gb"],
                           h2d_gb_per_s=g["h2d_gb_per_s"], bytes_host_to_device=g["bytes_host_to_device"])
    hp = jl(os.path.join(PILOT, "logs", "generation_cost_po.json"))
    hf = jl(os.path.join(FID, "logs", "generation_cost_ALG1_RATE.json"))
    cost["RS_PO"] = dict(wall_s=hp["wall_s"], generated_tokens=hp["accounting"]["generated_tokens"],
                         s_per_generated_token=hp["wall_s"] / hp["accounting"]["generated_tokens"],
                         peak_gpu_gb=hp["peak_gpu_gb"], peak_cpu_gb=hp["peak_cpu_gb"],
                         expert_transfer_frac=hp["timing"]["expert_transfer_s"] / hp["wall_s"])
    cost["RS_EBSS"] = dict(wall_s=hf["wall_excl_pool_capture_s"], wall_incl_audit_hook_s=hf["wall_s"],
                           generated_tokens=hf["accounting"]["generated_tokens"],
                           s_per_generated_token=hf["wall_excl_pool_capture_s"] / hf["accounting"]["generated_tokens"],
                           peak_gpu_gb=hf["peak_gpu_gb"], peak_cpu_gb=hf["peak_cpu_gb"],
                           expert_transfer_frac=hf["timing"]["expert_transfer_s"] / hf["wall_excl_pool_capture_s"],
                           note="excludes the read-only pool-capture audit hook of the fidelity study")
    cost["C4_NATURAL"] = dict(wall_s=0.0, note="no generation; data acquisition/tokenisation cost not measured in this "
                                               "repository and not invented")
    jdump(dict(conditions=cost, gpu="one 20 GB A100 MIG slice; GPU-seconds = wall",
               token_denominators="prefix runs: 16 x 448 generated continuation tokens; random-start: 16 x 511"),
          p("generation_cost.json"))
    qc = {}
    cap_share = man.get("capture_s", 0) * 16 / 80 if man else None
    for n, r in res.items():
        key = n if n in cost else ("C4_NATURAL" if n.startswith("C4") else n)
        tg = cost.get(key, {}).get("wall_s")
        qc[n] = dict(recon=r["recon"], ppl=r["ppl"], T_generation_s=tg,
                     T_activation_capture_s_derived=cap_share,
                     T_hessian_s=r["hessian_s"], T_gptq_s=r["gptq_s"], T_quantize_total_s=r["quantize_total_s"],
                     T_total_measured_s=(tg or 0.0) + r["quantize_total_s"] + (cap_share or 0.0),
                     T_data="not measured")
    jdump(dict(configs=qc, definition="T_total = T_generation + T_activation + T_quantize_total (Hessian + GPTQ + "
                                      "install/eval bookkeeping as measured by quant_config); T_activation is the B1 "
                                      "capture of this study's 80 rows scaled to 16 rows (derived); T_data not measured"),
          p("quality_cost_comparison.json"))

    # ---------------- decisions
    dv = jl(p("diversity_by_region.json")) if os.path.exists(p("diversity_by_region.json")) else None
    recon = {n: res[n]["recon"] for n in res}
    prim = X.primary_label(recon, valid)
    dec = dict(primary=prim, diversity=dv["diversity_classification"] if dv else None, valid_run=valid,
               deltas=deltas, ppl_note="PPL secondary; differences at the accumulation-order noise scale are not interpreted")
    jdump(dec, p("decision.json"))
    rd = jl(p("routing_diagnostics.json"))["compact"] if os.path.exists(p("routing_diagnostics.json")) else {}
    rep = jl(p("repetition_analysis.json")) if os.path.exists(p("repetition_analysis.json")) else {}
    summ = dict(primary=prim["label"], diversity=dec["diversity"]["label"] if dv else None,
                diversity_mixed=dec["diversity"]["mixed"] if dv else None,
                recon={n: rb[n]["mean"] for n in rb}, ppl={n: res[n]["ppl"] for n in res}, deltas=deltas, routing=rd,
                diversity_continuation={s: dv["sets"][s]["continuation"]["per_sequence_mean"] for s in dv["sets"]
                                        if "continuation" in dv["sets"][s]} if dv else None,
                diversity_full={s: dv["sets"][s]["full"]["per_sequence_mean"] for s in dv["sets"]} if dv else None,
                repetition={s: {k: rep["sets"][s][k] for k in ("n_with_loop", "loop_onset_median", "n_repeated_span_gt_64",
                                                               "n_single_token_loops_ge_16", "repeat_coverage_8gram_mean")}
                            for s in rep.get("sets", {})},
                cost={c: dict(wall_s=v.get("wall_s"), s_per_generated_token=v.get("s_per_generated_token")) for c, v in cost.items()})
    jdump(summ, p("summary.json"))
    write_md(summ, rb, c4m, deltas, cost, rd, dv)
    try:
        figures(res, rb, c4m, deltas, cost, rd, dv, rep, qc)
    except Exception as e:                                   # figures are optional; never block the report
        print("[report] figure error", repr(e))
    print("[report]", prim["label"], summ["diversity"])


def write_md(s, rb, c4m, deltas, cost, rd, dv):
    L = ["# C4-prefix diversity rescue (modified EBSS; B=16, P=64 + G=448)", "",
         f"**Primary: `{s['primary']}`**  |  **Diversity: `{s['diversity']}`**" + (" (mixed)" if s.get("diversity_mixed") else ""), ""]
    if dv:
        L += ["## Diversity by region (per-sequence means)", "",
              "| set | cont. distinct-2 | cont. repetition | full distinct-2 | full repetition | cont. NLL |", "|---|---|---|---|---|---|"]
        for k, v in dv["sets"].items():
            if "continuation" in v:
                c, f = v["continuation"]["per_sequence_mean"], v["full"]["per_sequence_mean"]
                L.append(f"| {k} | {c['distinct2']:.3f} | {c['repetition_rate']:.3f} | {f['distinct2']:.3f} | "
                         f"{f['repetition_rate']:.3f} | {v['continuation'].get('nll', {}).get('mean_per_token', float('nan')):.3f} |")
            else:
                f = v["full"]["per_sequence_mean"]
                L.append(f"| {k} | — | — | {f['distinct2']:.3f} | {f['repetition_rate']:.3f} | — |")
        L.append("")
    if rd:
        L += ["## Routing (CANONICAL-B1, 24 layers)", "", "| set | CV | min | p10 | routing L1 | atypical |", "|---|---|---|---|---|---|"]
        for k, v in rd.items():
            L.append(f"| {k} | {v['cv']:.4f} | {v['min']:.1f} | {v['p10']:.1f} | {v['routing_l1']:.3f} | {v['frac_atypical']:.2f} |")
        L.append("")
    L += ["## B=16 GPTQ (layers 1/8/16/23)", "", "| config | recon | vs C4 mean | PPL |", "|---|---|---|---|"]
    for n, v in rb.items():
        L.append(f"| {LABEL[n]} | {v['mean']:.7f} | {((v['mean'] / c4m - 1) * 100 if c4m else float('nan')):+.2f} % | {s['ppl'][n]:.6f} |")
    if deltas:
        L += ["", f"Δ_PO = {deltas['delta_PO']:+.3e}, Δ_EBSS = {deltas['delta_EBSS']:+.3e}, "
                  f"Δ_balance|prefix = {deltas['delta_balance_given_prefix']:+.3e}; gap recovery G_PO = "
                  f"{deltas['gap_recovery']['PO']:.3f}, G_EBSS = {deltas['gap_recovery']['EBSS']:.3f}"]
    L += ["", "See `../update_ebss_prefix_rescue.md` for the interpretation."]
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ figures
def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(p("figures", f"{name}.{ext}"), dpi=150)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)


def figures(res, rb, c4m, deltas, cost, rd, dv, rep, qc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)
    if dv:   # Fig 1: full vs continuation diversity
        sets = ["RS_PO", "PX_PO", "RS_EBSS", "PX_EBSS", "C4R_s0"]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, metric, title in ((axs[0], "distinct2", "distinct-2 (per-sequence mean)"),
                                  (axs[1], "repetition_rate", "repetition rate = 1 − distinct-1")):
            x = np.arange(len(sets))
            for j, reg in enumerate(("continuation", "full")):
                vals = [dv["sets"][s][reg]["per_sequence_mean"][metric] for s in sets]
                ax.bar(x + (j - .5) * .38, vals, .36, color=[col(s) for s in sets], alpha=1.0 if reg == "continuation" else .45,
                       edgecolor="white", linewidth=1)
            ax.set_xticks(x); ax.set_xticklabels([LABEL[s].replace(" (= matched natural)", "") for s in sets], rotation=25, ha="right")
            ax.set_title(title + "\nsolid = continuation [64:512], light = full [0:512]", fontsize=8, color=INK); _style(ax)
        _save(fig, "fig1_region_diversity"); plt.close(fig)
    if rep:  # Fig 2: repetition along position
        fig, ax = plt.subplots(figsize=(7, 3.2))
        xs = 64 + 32 * np.arange(14) + 16
        for s in ("RS_PO", "RS_EBSS", "PX_PO", "PX_EBSS", "C4R_s0"):
            ax.plot(xs, rep["coverage_by_position_32tok_bins"][s], color=col(s), lw=2, ls="--" if s.startswith("RS") else "-",
                    marker="o", ms=4, label=LABEL[s])
        ax.set_xlabel("token position (continuation region)", fontsize=8); ax.set_ylabel("fraction in repeated 8-grams", fontsize=8)
        ax.set_ylim(0, 1); ax.legend(fontsize=6, frameon=False); _style(ax)
        ax.set_title("Fig. 2  Repetition along the sequence", fontsize=9, color=INK)
        _save(fig, "fig2_repetition_position"); plt.close(fig)
    if rd:   # Fig 3 / 4
        sets = [s for s in ("C4R_s0", "C4R_s1", "C4R_s2", "RS_PO", "RS_EBSS", "PX_PO", "PX_EBSS") if s in rd]
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, m in zip(axs, ("cv", "min", "p10")):
            ax.bar(range(len(sets)), [rd[s][m] for s in sets], color=[col(s) for s in sets],
                   edgecolor="white", linewidth=1)
            ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets, rotation=35, ha="right"); ax.set_title(f"{m} (B1, 24 layers)", fontsize=8); _style(ax)
        fig.suptitle("Fig. 3  Expert exposure", fontsize=9); _save(fig, "fig3_expert_exposure"); plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for ax, m in zip(axs, ("routing_l1", "frac_atypical")):
            ax.bar(range(len(sets)), [rd[s][m] for s in sets], color=[col(s) for s in sets], edgecolor="white", linewidth=1)
            ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets, rotation=35, ha="right"); ax.set_title(m, fontsize=8); _style(ax)
        fig.suptitle("Fig. 4  Routing representativeness (vs frozen C4 reference)", fontsize=9)
        _save(fig, "fig4_routing"); plt.close(fig)
    if res:  # Fig 5
        sets = [s for s in QSRC if s in res]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
        axs[0].bar(range(len(sets)), [rb[s]["mean"] for s in sets], color=[col(s) for s in sets], edgecolor="white")
        lo = min(rb[s]["mean"] for s in sets)
        axs[0].set_ylim(lo * 0.95, max(rb[s]["mean"] for s in sets) * 1.01)
        axs[0].set_xticks(range(len(sets))); axs[0].set_xticklabels(sets, rotation=35, ha="right"); axs[0].set_title("mean held-out recon (↓)", fontsize=8)
        for j, s in enumerate(sets):
            axs[1].plot(LAYERS, [rb[s]["per_layer"][str(L)] for L in LAYERS], color=col(s), marker="o", ms=4, lw=2,
                        ls="--" if s.startswith("RS") else ("-" if s.startswith("PX") else ":"), label=s)
        axs[1].set_xticks(LAYERS); axs[1].set_title("per-layer recon", fontsize=8); axs[1].legend(fontsize=6, frameon=False)
        if "PX_EBSS" in res:
            data, names = [], []
            for a, b in (("PX_PO", "RS_PO"), ("PX_EBSS", "RS_EBSS"), ("PX_EBSS", "PX_PO"), ("PX_EBSS", "C4R_s0")):
                data.append(pairwise(res, a, b)["per_expert_rel_pct"]); names.append(f"{a}\nvs {b}")
            axs[2].boxplot(data, showfliers=False); axs[2].axhline(0, color=MUTED, lw=1)
            axs[2].set_xticklabels(names, fontsize=6); axs[2].set_title("per-expert relative difference (%)", fontsize=8)
        for ax in axs:
            _style(ax)
        fig.suptitle("Fig. 5  B=16 GPTQ reconstruction", fontsize=9); _save(fig, "fig5_reconstruction"); plt.close(fig)
    if deltas:  # Fig 6
        fig, ax = plt.subplots(figsize=(4.5, 3))
        g = deltas["gap_recovery"]
        ax.bar([0, 1], [g["PO"], g["EBSS"]], color=[COL["PX_PO"], COL["PX_EBSS"]], edgecolor="white")
        ax.axhline(0, color=MUTED, lw=1); ax.axhline(1, color=COL["C4"], lw=1, ls="--")
        ax.text(1.45, 1, "C4 mean", fontsize=7, color=MUTED, va="center")
        for i, v in enumerate((g["PO"], g["EBSS"])):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["PO", "ALG1_RATE"]); ax.set_ylabel("gap recovered G", fontsize=8)
        ax.set_ylim(min(-0.2, g["PO"] - .1, g["EBSS"] - .1), 1.15); _style(ax)
        ax.set_title("Fig. 6  Recovery of the random-start → C4 gap", fontsize=9); _save(fig, "fig6_gap_recovery"); plt.close(fig)
    if qc:  # Fig 7
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for n, v in qc.items():
            gen = v["T_generation_s"] or 0.0
            ax.scatter(gen, v["recon"], s=60, color=col(n), edgecolors="white", linewidths=2,
                       marker="s" if n.startswith("PX") else ("^" if n.startswith("RS") else "o"), zorder=3)
            if n.startswith("C4"):
                continue  # C4 seeds overlap; labelled once below
            off = {"PX_PO": (8, 6), "PX_EBSS": (8, -12), "RS_PO": (-40, -14), "RS_EBSS": (-60, 6)}.get(n, (5, 3))
            ax.annotate(f"{n} ({gen:.0f} s)", (gen, v["recon"]), fontsize=6, color=INK,
                        xytext=off, textcoords="offset points")
        c4 = [v for n, v in qc.items() if n.startswith("C4")]
        if c4:
            ax.annotate(f"C4 Random s0/s1/s2 (no generation)", (0.0,
                        min(v["recon"] for v in c4)), fontsize=6, color=INK, xytext=(8, 4), textcoords="offset points")
        ax.set_xlabel("measured generation GPU-seconds (capture + GPTQ are the same pipeline for every source;\n"
                      "their measured wall time varied 355–1366 s with machine load and is omitted)", fontsize=7)
        ax.set_ylabel("held-out reconstruction (↓)", fontsize=8); _style(ax)
        ax.set_title("Fig. 7  Quality vs cost", fontsize=9); _save(fig, "fig7_quality_cost"); plt.close(fig)


if __name__ == "__main__":
    main()
