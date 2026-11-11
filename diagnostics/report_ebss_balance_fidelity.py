"""Report for the EBSS balance-strength fidelity experiment: labels, summaries, figures (only for completed stages)."""
from __future__ import annotations

import json, os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs_ebss_balance_fidelity")
PILOT = os.path.join(ROOT, "runs_ebss_generation_pilot")
VARIANTS = ["EQ4_RATE", "EQ4_SHARE", "ALG1_SHARE", "ALG1_RATE"]
ALT = VARIANTS[1:]
COND = ["PROBABILITY_ONLY"] + VARIANTS
MARGIN = 0.5


def p(*a):
    return os.path.join(OUT, *a)


def jl(path):
    return json.load(open(path)) if os.path.exists(path) else None


def jdump(o, path):
    json.dump(o, open(path + ".tmp", "w"), indent=1, default=float); os.replace(path + ".tmp", path)


def rel(a, b):
    return (a / b - 1) * 100


def primary_label(regression_ok, any_material):
    if not regression_ok:
        return "BALANCE-NORMALIZATION-RUN-BLOCKED"
    return "BALANCE-NORMALIZATION-MATERIAL" if any_material else "BALANCE-NORMALIZATION-IMMATERIAL"


def quality_label(new, eq4, c4):
    """new {variant: recon}; eq4 = EQ4_RATE recon; c4 = 3 C4 Random seeds (frozen rule, 0.5 % margin)."""
    m = float(np.mean(c4))
    if any(rel(r, eq4) <= -MARGIN and all(r < c for c in c4) and rel(r, m) <= -MARGIN for r in new.values()):
        return "BALANCE-VARIANT-QUALITY-GAIN"
    if any(rel(r, eq4) <= -MARGIN or r < m for r in new.values()):
        return "BALANCE-VARIANT-QUALITY-MIXED"
    return "BALANCE-VARIANT-QUALITY-NO-GAIN"


def b16_material(full, gate):
    thr = []
    for v, g in (full or {}).get("gate_thresholds_at_B16", {}).items():
        hit = (abs(g["rel_cv"]) >= .10 or abs(g["rel_p10"]) >= .10 or abs(g["rel_min"]) >= .20 or
               abs(g["d_distinct2"]) >= .05 or abs(g["d_repetition"]) >= .05 or abs(g["rel_l1"]) >= .10 or
               abs(g["d_atypical"]) >= .25)
        thr.append((v, bool(hit)))
    return dict(thr)


def gptq_block():
    hist = {c: jl(os.path.join(PILOT, "results", f"{c}.json")) for c in ("C4R_s0", "C4R_s1", "C4R_s2", "PO", "EBSS")}
    new = {v: jl(p("results", f"{v}.json")) for v in ALT}
    new = {v: r for v, r in new.items() if r}
    if not new:
        return None
    rows = {"C4R_s0": hist["C4R_s0"], "C4R_s1": hist["C4R_s1"], "C4R_s2": hist["C4R_s2"],
            "PROBABILITY_ONLY": hist["PO"], "EQ4_RATE": hist["EBSS"], **new}
    c4 = [rows[f"C4R_s{s}"]["recon"] for s in range(3)]

    def pe(r):
        return {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert"]}

    def layer(r):
        out = {}
        for x in r["per_expert"]:
            out.setdefault(x["layer"], []).append(x["mean"])
        return {str(k): float(np.mean(v)) for k, v in out.items()}
    ref_dec = hist["C4R_s0"]
    recon = {}
    for c, r in rows.items():
        e = np.array([x["mean"] for x in r["per_expert"]])
        recon[c] = dict(recon=r["recon"], median_expert=float(np.median(e)), worst_expert=float(e.max()),
                        per_layer=layer(r), rtn_fallbacks=r["rtn_fallbacks"], valid=r["valid"],
                        by_natural_exposure_decile=r.get("by_natural_exposure_decile"))
    comps = {}
    for v in new:
        a = pe(rows[v])
        comps[v] = {}
        for ref in ["PROBABILITY_ONLY", "EQ4_RATE", "C4R_s0", "C4R_s1", "C4R_s2"]:
            b = pe(rows[ref])
            comps[v][ref] = dict(relative_pct=rel(rows[v]["recon"], rows[ref]["recon"]),
                                 frac_experts_improved=float(np.mean([a[k] < b[k] for k in a])),
                                 per_layer_relative_pct={L: rel(recon[v]["per_layer"][L], recon[ref]["per_layer"][L])
                                                         for L in recon[v]["per_layer"]})
        comps[v]["C4R_mean"] = dict(relative_pct=rel(rows[v]["recon"], float(np.mean(c4))))
    ppl = {c: dict(ppl=r["ppl"], per_window_nll=r.get("per_window_nll")) for c, r in rows.items()}
    label = quality_label({v: rows[v]["recon"] for v in new}, rows["EQ4_RATE"]["recon"], c4)
    rank_recon = sorted(rows, key=lambda c: rows[c]["recon"])
    rank_ppl = sorted(rows, key=lambda c: rows[c]["ppl"])
    return dict(recon=recon, comparisons=comps, ppl=ppl, label=label, c4_mean=float(np.mean(c4)),
                ranking_recon=rank_recon, ranking_ppl=rank_ppl, valid_all=all(r["valid"] for r in rows.values()))


def figures(cf, short, full, cost, gq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)
    col = dict(PROBABILITY_ONLY="#7f7f7f", EQ4_RATE="#1f77b4", EQ4_SHARE="#9ecae1", ALG1_SHARE="#ff7f0e",
               ALG1_RATE="#d62728", C4R="#2ca02c")
    pc = cf["short_EQ4_RATE"]["per_condition"]
    # 1 balance strength
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(VARIANTS, [pc[v]["median_balance_to_probability_spread"] for v in VARIANTS], color=[col[v] for v in VARIANTS])
    ax.axhline(1, ls="--", c="k", lw=.8)
    ax.set_ylabel("median balance spread / probability spread"); ax.set_yscale("log")
    ax.set_title("Fig. 1  Balance-term strength (EQ4_RATE short pools)")
    fig.tight_layout(); fig.savefig(p("figures", "fig1_balance_strength.png"), dpi=150); plt.close(fig)
    # 2 decision changes by prefix bin
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, ttl in ((axs[0], "D_vs_EQ4_RATE", "vs EQ4_RATE"), (axs[1], "D_vs_PO", "vs Probability-Only")):
        for v in COND[1:] if key == "D_vs_PO" else ALT:
            xs, ys = [], []
            for traj in ["short_EQ4_RATE"] + [k for k in cf if k.startswith("full_")]:
                for b, d in cf[traj]["per_condition"][v]["by_prefix_bin"].items():
                    if d["n"] and (traj == "short_EQ4_RATE" or int(b.split("-")[0]) > 128):
                        xs.append(b); ys.append(d[key])
            ax.plot(range(len(xs)), ys, "o-", c=col[v], label=v)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_title(f"Fig. 2  changed top-w sets {ttl}"); ax.set_xlabel("prefix length")
    axs[0].set_ylabel("fraction of sentence-steps"); axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(p("figures", "fig2_counterfactual_changes.png"), dpi=150); plt.close(fig)
    # 3 balance, 4 repetition, 5 routing (short; B16 if present)
    blocks = [("4x128", short["compact"], COND + ["C4R_s0_first128_first4"])]   # token-matched C4 (4 x 128)
    if full:
        blocks.append(("16x512", full["compact"], [c for c in ["PROBABILITY_ONLY", "EQ4_RATE"] + ALT + ["C4R_s0", "C4R_s1", "C4R_s2"]
                                                  if c in full["compact"]]))
    for fname, metrics, title in (("fig3_expert_balance", ("cv", "min", "p10"), "Fig. 3  Canonical expert balance (B1)"),
                                  ("fig4_repetition", ("distinct1", "distinct2", "distinct3", "longest_repeated_ngram_mean",
                                                       "model_ppl"), "Fig. 4  Diversity / repetition"),
                                  ("fig5_routing", ("routing_l1", "frac_atypical"), "Fig. 5  Routing representativeness")):
        fig, axs = plt.subplots(len(blocks), len(metrics), figsize=(3.2 * len(metrics), 3.2 * len(blocks)), squeeze=False)
        for bi, (lab, comp, names) in enumerate(blocks):
            for mi, m in enumerate(metrics):
                ax = axs[bi, mi]
                vals = [comp[n][m] for n in names]
                ax.bar(range(len(names)), vals, color=[col.get(n, col["C4R"]) for n in names])
                ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("PROBABILITY_ONLY", "PO")
                                                                      .replace("_first128_first4", "@4x128") for n in names],
                                                                     rotation=60, fontsize=6)
                ax.set_title(f"{m} ({lab})", fontsize=8)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(p("figures", fname + ".png"), dpi=150); plt.close(fig)
    # 6 reconstruction
    if gq:
        names = list(gq["recon"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar(range(len(names)), [gq["recon"][n]["recon"] for n in names],
               color=[col.get(n, col["C4R"]) for n in names])
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=7)
        ax.set_ylim(min(gq["recon"][n]["recon"] for n in names) * .95, max(gq["recon"][n]["recon"] for n in names) * 1.02)
        ax.set_ylabel("held-out expert reconstruction (lower better)"); ax.set_title("Fig. 6  B=16 GPTQ reconstruction")
        fig.tight_layout(); fig.savefig(p("figures", "fig6_reconstruction.png"), dpi=150); plt.close(fig)
    # 7 cost
    fig, ax = plt.subplots(figsize=(8, 3.6))
    parts = ["expert_weight_transfer", "model_forward_compute", "beam_pruning_selection", "expert_statistics_update",
             "beam_scoring", "pool_capture_audit", "other"]
    runs = [(f"{c}\n4x128", cost["short"][c]) for c in COND] + [(f"{v}\n16x512", c) for v, c in cost["full"].items()]
    bottom = np.zeros(len(runs))
    for part in parts:
        vals = np.array([r["breakdown_s"].get(part, 0.0) for _, r in runs])
        ax.bar(range(len(runs)), vals, bottom=bottom, label=part); bottom += vals
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([n for n, _ in runs], fontsize=6)
    ax.set_ylabel("seconds"); ax.legend(fontsize=6); ax.set_title("Fig. 7  Generation cost")
    fig.tight_layout(); fig.savefig(p("figures", "fig7_generation_cost.png"), dpi=150); plt.close(fig)


def main():
    dr, orc = jl(p("default_regression.json")), jl(p("normalization_oracle_results.json"))
    cf, short, gate = jl(p("counterfactual_ranking.json")), jl(p("short_generation_results.json")), jl(p("scientific_gate.json"))
    full = jl(p("full_generation_results.json"))
    scost = jl(p("logs", "short_generation_cost.json"))
    fcost = {v: jl(p("logs", f"generation_cost_{v}.json")) for v in ALT}
    fcost = {v: c for v, c in fcost.items() if c}
    regression_ok = bool(dr and dr.get("passed") and orc and orc["passed"] and cf and
                         all(cf[k].get("oracle_reproduces_production_decisions", True) for k in cf
                             if isinstance(cf[k], dict) and "per_condition" in cf[k]))
    m16 = b16_material(full, gate) if full else {}
    any_material = bool(gate and gate["short_stage_material"]) or any(m16.values())
    gq = gptq_block()
    label = primary_label(regression_ok, any_material)
    pc = cf["short_EQ4_RATE"]["per_condition"]
    dec = dict(decision=label, quality_label=gq["label"] if gq else "NOT-EVALUATED",
               regression_ok=regression_ok, short_stage_material=gate["short_stage_material"],
               qualifying_variants={v: gate["per_variant"][v]["qualifies"] for v in ALT},
               b16_material=m16, selected_for_B16=gate["selected_for_B16"], excluded=gate["excluded"],
               counterfactual_D={v: dict(vs_EQ4_RATE=pc[v]["D_vs_EQ4_RATE"], vs_PO=pc[v]["D_vs_PO"]) for v in COND},
               rule=json.load(open(p("protocol.json")))["decision"])
    if gq:
        dec["quality_numbers"] = {v: gq["comparisons"][v] for v in gq["comparisons"]}
    jdump(dec, p("decision.json"))
    # split outputs
    jdump(dict(short_4x128={c: short["full"][c]["text"] for c in short["full"]},
               b16=({c: full["full"][c]["text"] for c in full["full"]} if full else None)), p("generation_diversity.json"))
    jdump(dict(short_4x128={c: dict(b1=short["full"][c]["b1"], generation_time=short["full"][c].get("generation_time"),
                                    representativeness=short["full"][c]["representativeness"],
                                    gen_vs_b1=short["full"][c].get("generation_vs_b1_set_agreement"))
                            for c in short["full"]},
               b16=({c: dict(b1=full["full"][c]["b1"], generation_time=full["full"][c].get("generation_time"),
                             representativeness=full["full"][c]["representativeness"],
                             gen_vs_b1=full["full"][c].get("generation_vs_b1_set_agreement")) for c in full["full"]}
                    if full else None)), p("routing_diagnostics.json"))
    pilot_cost = jl(os.path.join(PILOT, "generation_cost.json"))
    eff = dict(short={c: scost["costs"][c] for c in COND}, full=fcost,
               historical_B16={k: pilot_cost[k] for k in ("PROBABILITY_ONLY", "EBSS")},
               note="new runs include a read-only pool-capture hook (pool_capture_audit); wall_excl_pool_capture_s "
                    "removes it; GPU-seconds = wall on one MIG slice")
    if gq and fcost:
        eb = gq["recon"]["EQ4_RATE"]["recon"]
        eff["per_unit"] = {v: dict(abs_improvement_vs_EQ4_RATE=eb - gq["recon"][v]["recon"],
                                   rel_improvement_pct=-rel(gq["recon"][v]["recon"], eb),
                                   incremental_gpu_s_vs_EQ4_RATE=fcost[v]["wall_excl_pool_capture_s"]
                                   - pilot_cost["EBSS"]["wall_s"]) for v in fcost if v in gq["recon"]}
    jdump(eff, p("generation_efficiency.json"))
    if gq:
        jdump(dict(metric="E3 held-out relative expert reconstruction (rows 2128-2255), mean over 240 experts",
                   B=16, configs=gq["recon"], comparisons=gq["comparisons"], c4r_mean=gq["c4_mean"],
                   ranking=gq["ranking_recon"], valid_all=gq["valid_all"]), p("reconstruction_results.json"))
        jdump(dict(configs=gq["ppl"], ranking=gq["ranking_ppl"]), p("ppl_results.json"))
    figures(cf, short, full, dict(short=scost["costs"], full=fcost), gq)
    summ = dict(decision=label, quality=dec["quality_label"], counterfactual=dec["counterfactual_D"],
                short=short["compact"], b16=(full["compact"] if full else None),
                recon=({c: v["recon"] for c, v in gq["recon"].items()} if gq else None),
                ppl=({c: v["ppl"] for c, v in gq["ppl"].items()} if gq else None))
    jdump(summ, p("summary.json"))
    write_md(dec, cf, short, full, gq, scost, fcost)
    print("[report]", label, dec["quality_label"])


def f(x, d=4):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_md(dec, cf, short, full, gq, scost, fcost):
    pc = cf["short_EQ4_RATE"]["per_condition"]
    L = [f"# EBSS balance-strength fidelity experiment\n",
         f"**Primary: {dec['decision']}**  |  **Quality: {dec['quality_label']}**\n",
         "## Counterfactual pruning on the EQ4_RATE 4 x 128 candidate pools\n",
         "| variant | strength | D vs EQ4_RATE | D vs PO | median balance/probability spread |", "|---|---|---|---|---|"]
    strength = dict(PROBABILITY_ONLY=0, EQ4_RATE=1, EQ4_SHARE=.25, ALG1_SHARE=1.936, ALG1_RATE=7.746)
    for v in COND:
        L.append(f"| {v} | {strength[v]} | {f(pc[v]['D_vs_EQ4_RATE'], 3)} | {f(pc[v]['D_vs_PO'], 3)} | "
                 f"{f(pc[v]['median_balance_to_probability_spread'], 3)} |")
    L += ["\n## Short generation 4 x 128 (canonical EQ4_RATE metrics, CANONICAL-B1 routing)\n",
          "| condition | CV | min | p10 | distinct-2 | repetition | longest rep. n-gram | routing L1 | atypical | PPL |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in short["compact"].items():
        L.append(f"| {c} | {f(s['cv'])} | {f(s['min'], 1)} | {f(s['p10'], 1)} | {f(s['distinct2'], 3)} | "
                 f"{f(s['repetition_rate'], 3)} | {f(s['longest_repeated_ngram_mean'], 1)} | {f(s['routing_l1'], 3)} | "
                 f"{f(s['frac_atypical'], 2)} | {f(s['model_ppl'], 3)} |")
    L.append(f"\nGate: qualifying {dec['qualifying_variants']}; B=16 run: {dec['selected_for_B16']}; excluded {dec['excluded']}\n")
    if full:
        L += ["## B=16 x 512\n", "| condition | CV | min | p10 | distinct-2 | repetition | routing L1 | atypical | PPL |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c, s in full["compact"].items():
            L.append(f"| {c} | {f(s['cv'])} | {f(s['min'], 1)} | {f(s['p10'], 1)} | {f(s['distinct2'], 3)} | "
                     f"{f(s['repetition_rate'], 3)} | {f(s['routing_l1'], 3)} | {f(s['frac_atypical'], 2)} | "
                     f"{f(s['model_ppl'], 3)} |")
    if gq:
        L += ["\n## B=16 GPTQ (layers 1/8/16/23)\n", "| config | recon | PPL |", "|---|---|---|"]
        for c in gq["recon"]:
            L.append(f"| {c} | {gq['recon'][c]['recon']:.7f} | {gq['ppl'][c]['ppl']:.6f} |")
        for v, cc in gq["comparisons"].items():
            L.append(f"- {v}: vs PO {cc['PROBABILITY_ONLY']['relative_pct']:+.2f} %, vs EQ4_RATE "
                     f"{cc['EQ4_RATE']['relative_pct']:+.2f} %, vs C4 mean {cc['C4R_mean']['relative_pct']:+.2f} %; "
                     f"experts improved vs EQ4_RATE {cc['EQ4_RATE']['frac_experts_improved']:.2f}")
    L.append("\nSee `../update_ebss_balance_fidelity.md` for the interpretation.")
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
