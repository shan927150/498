"""Report for the model-matched EBSS generation pilot: reconstruction/PPL aggregation, frozen decision, figures,
summary.json / summary.md."""
from __future__ import annotations

import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import ebss_pilot as P              # noqa: E402

LAYERS = [1, 8, 16, 23]
E = 60
QC = ["C4R_s0", "C4R_s1", "C4R_s2", "PO", "EBSS"]
COL = {"PROBABILITY_ONLY": "#4C78A8", "EBSS": "#B07AA1", "C4R": "#59A14F", "OFFICIAL": "#E15759"}


def p(*a):
    return os.path.join(ROOT, P.OUT, *a)


def J(path):
    return json.load(open(path))


def jdump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def gptq_block():
    if not all(os.path.exists(p("results", f"{c}.json")) for c in QC):
        return None
    res = {c: J(p("results", f"{c}.json")) for c in QC}
    prof = np.load(os.path.join(ROOT, "runs_low_budget", "routing_profile_v2.npz"))
    nat = prof["C"].sum(0)                                         # natural routed counts per (layer, expert)
    nat_p = np.concatenate([nat[li * E:(li + 1) * E] / nat[li * E:(li + 1) * E].sum() for li in range(4)])
    dec = np.minimum((np.argsort(np.argsort(nat_p)) * 10) // len(nat_p), 9)
    per = {c: np.array([e["mean"] for e in sorted(res[c]["per_expert"], key=lambda e: (e["layer"], e["expert"]))])
           for c in QC}
    c4m = np.mean([per[f"C4R_s{s}"] for s in range(3)], axis=0)
    rel = lambda a, b: (a / b - 1) * 100
    out = dict(metric="E3 held-out relative expert reconstruction (rows 2128-2255 of ~/moe_cache), mean over 240 "
                      "experts; lower is better", B=16, configs={})
    for c in QC:
        r = res[c]
        pl = {str(L): float(np.mean([e["mean"] for e in r["per_expert"] if e["layer"] == L])) for L in LAYERS}
        out["configs"][c] = dict(
            recon=r["recon"], per_layer=pl, rtn_fallbacks=r["rtn_fallbacks"], valid=r["valid"],
            per_expert_quantiles={k: float(np.quantile(per[c], q)) for k, q in
                                  (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1))},
            worst_expert=dict(zip(("layer", "expert"), divmod(int(np.argmax(per[c])), E))) | {
                "recon": float(per[c].max())},
            by_natural_exposure_decile=[float(per[c][dec == d].mean()) for d in range(10)],
            fallback_experts=[(e["layer"], e["expert"]) for e in r["per_expert"] if e["fallback"]],
            gptq_s=r["gptq_s"], hessian_s=r["hessian_s"], quantize_total_s=r["quantize_total_s"])
    out["c4r_mean_recon"] = float(np.mean([res[f"C4R_s{s}"]["recon"] for s in range(3)]))
    out["comparisons"] = {}
    for a_, b_ in (("EBSS", "PO"), ("EBSS", "C4R_mean"), ("PO", "C4R_mean")):
        A = per[a_]
        Bv = c4m if b_ == "C4R_mean" else per[b_]
        rb = out["c4r_mean_recon"] if b_ == "C4R_mean" else res[b_]["recon"]
        out["comparisons"][f"{a_}_vs_{b_}"] = dict(
            relative_pct=rel(res[a_]["recon"], rb), frac_experts_improved=float((A < Bv).mean()),
            per_layer_relative_pct={str(L): rel(out["configs"][a_]["per_layer"][str(L)],
                                                (np.mean([out["configs"][f"C4R_s{s}"]["per_layer"][str(L)] for s in range(3)])
                                                 if b_ == "C4R_mean" else out["configs"][b_]["per_layer"][str(L)]))
                                    for L in LAYERS},
            by_decile_relative_pct=[rel(A[dec == d].mean(), Bv[dec == d].mean()) for d in range(10)])
    out["valid_all"] = all(res[c]["valid"] for c in QC)
    ppl = dict(bf16=9.237090, configs={c: dict(ppl=res[c]["ppl"], delta_vs_bf16=res[c]["ppl"] - 9.237090,
                                               repeat_eval_abs_diff_ppl=res[c]["repeat_eval_abs_diff_ppl"],
                                               per_window_nll=res[c]["per_window_nll"]) for c in QC},
               note="secondary; four-layer partial-model PPL; earlier studies show grouping noise up to ~0.012")
    return out, ppl, per, dec


def paired_analysis():
    """Per-start comparison of the two sets (same start token) + balance-term vs probability-term spread."""
    import torch
    from moe_calib import ebss_generator as G
    po, eb = (torch.load(p("logs", f"generated_{m}.pt")) for m in ("po", "ebss"))
    cap = torch.load(p("logs", "capture_routing_nll.pt"))
    rows = torch.load(os.path.join(P.CACHE, "ids.pt"))["rows"]
    sel = cap["sel"]

    def sig(r):
        c = torch.stack([torch.bincount(sel[r, :, l].reshape(-1).long(), minlength=E) for l in range(24)])
        return float(G.balance_sigma(c, 512))
    sp = np.array([sig(r) for r in rows["po"]]); se = np.array([sig(r) for r in rows["ebss"]])
    same = (po["tokens"] == eb["tokens"]).all(1)
    first = [None if same[i] else int((po["tokens"][i] != eb["tokens"][i]).nonzero()[0]) for i in range(len(same))]
    spread = []
    for s in eb["step_log"]:
        if s["n_branches"] > 1:
            spread.append(dict(step=s["step"], sentence=s["sentence"],
                               balance_term_range=float(np.ptp(s["branch_sigma"]) / s["tau"]),
                               probability_term_range=float(np.ptp(s["branch_R"]) / (s["prefix_len"] + 1))))
    bal = np.array([x["balance_term_range"] for x in spread]); prb = np.array([x["probability_term_range"] for x in spread])
    return dict(identical_sentences=int(same.sum()), identical_indices=same.nonzero().flatten().tolist(),
                first_divergence_position=first,
                per_sentence_b1_sigma=dict(po=sp.tolist(), ebss=se.tolist(), diff=(se - sp).tolist(),
                                           n_ebss_lower=int((se - sp < 0).sum()), n_ebss_higher=int((se - sp > 0).sum()),
                                           mean_diff=float((se - sp).mean())),
                term_spread_across_live_branches=dict(
                    recorded_steps=sorted({x["step"] for x in spread}), median_balance_term_range=float(np.median(bal)),
                    median_probability_term_range=float(np.median(prb)),
                    ratio_median=float(np.median(prb) / np.median(bal)),
                    note="range over the w live branches of sigma/tau vs R/(l+1) at the recorded steps; the balance "
                         "term can only reorder branches whose probability terms differ by less than its range"))


def figures(rs, dv, cost, gq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(p("figures", name + ".png"), dpi=150)
        fig.savefig(p("figures", name + ".pdf"))
        plt.close(fig)
    S = rs["sets"]
    # 1 exposure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m in ("PROBABILITY_ONLY", "EBSS"):
        cv = [S[m]["b1"]["per_layer"][l]["cv"] for l in range(24)]
        ax[0].plot(range(24), cv, "o-", color=COL[m], label=m)
        sig = S[m]["b1"]["paper_sigma_per_layer"]
        ax[1].plot(range(24), sig, "o-", color=COL[m], label=m)
    c4cv = np.mean([[S[f"C4R_s{s}"]["b1"]["per_layer"][l]["cv"] for l in range(24)] for s in range(3)], 0)
    ax[0].plot(range(24), c4cv, "s--", color=COL["C4R"], label="C4 Random B16 (mean of 3)")
    ax[0].set(xlabel="MoE layer", ylabel="expert load CV", title="Per-layer load CV (B1 replay)")
    ax[1].set(xlabel="MoE layer", ylabel="sigma_l (Eq. 4)", title="Per-layer paper sigma")
    for m in ("PROBABILITY_ONLY", "EBSS"):
        h = np.sort(np.array(S[m]["b1"]["histogram"]), axis=1).mean(0)
        ax[2].plot(range(E), h, color=COL[m], label=m)
    h = np.mean([np.sort(np.array(S[f"C4R_s{s}"]["b1"]["histogram"]), axis=1).mean(0) for s in range(3)], 0)
    ax[2].plot(range(E), h, "--", color=COL["C4R"], label="C4 Random B16")
    ax[2].axhline(16 * 512 * 4 / E, color="gray", lw=.8)
    ax[2].set(xlabel="expert rank (sorted within layer)", ylabel="routed tokens (mean over 24 layers)",
              title="Sorted exposure profile")
    for a_ in ax:
        a_.legend(fontsize=7)
    save(fig, "fig1_exposure_balance")
    # 2 progression
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pos = list(P.PROGRESS_POSITIONS)
    for m in ("PROBABILITY_ONLY", "EBSS"):
        ax[0].plot(pos, [S[m]["progression_b1"][str(t)]["set_paper_sigma"] for t in pos], "o-", color=COL[m], label=m)
        ax[0].plot(pos, [S[m]["progression_b1"][str(t)]["per_sequence_sigma_mean"] for t in pos], "x:", color=COL[m],
                   label=m + " (per-sequence)")
        ax[1].plot(pos, [S[m]["progression_b1"][str(t)]["set_cv"] for t in pos], "o-", color=COL[m], label=m)
    for s in range(3):
        ax[0].plot(pos, [S[f"C4R_s{s}"]["progression_b1"][str(t)]["set_paper_sigma"] for t in pos], "--",
                   color=COL["C4R"], alpha=.6, label="C4 Random B16" if s == 0 else None)
        ax[1].plot(pos, [S[f"C4R_s{s}"]["progression_b1"][str(t)]["set_cv"] for t in pos], "--", color=COL["C4R"],
                   alpha=.6, label="C4 Random B16" if s == 0 else None)
    for a_ in ax:
        a_.set_xscale("log", base=2); a_.set_xlabel("tokens per sequence (prefix)"); a_.legend(fontsize=7)
    ax[0].set(ylabel="paper sigma (24 layers)", title="Balance vs generated length")
    ax[1].set(ylabel="mean per-layer CV", title="Set load CV vs length")
    save(fig, "fig2_balance_over_length")
    # 3 diversity
    D = dv["sets"]
    names = ["PROBABILITY_ONLY", "EBSS", "C4R_s0", "C4R_s1", "C4R_s2"]
    keys = [("distinct1_mean", "distinct-1"), ("distinct2_mean", "distinct-2"), ("distinct3_mean", "distinct-3"),
            ("repetition_rate_mean", "repetition rate")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(keys))
    for i, n in enumerate(names):
        ax[0].bar(x + i * .16, [D[n][k] for k, _ in keys], .16, label=n,
                  color=COL.get(n, COL["C4R"]), alpha=1 if n in COL else .4 + .2 * i / 4)
    off = dv["context_official_artifact_128_windows"]["text"]
    ax[0].scatter(x[:2] + .32, [off["distinct1_mean"], off["distinct2_mean"]], marker="*", s=120, color=COL["OFFICIAL"],
                  label="official EBSS_qwen (128 windows)", zorder=3)
    ax[0].set_xticks(x + .32); ax[0].set_xticklabels([l for _, l in keys]); ax[0].legend(fontsize=7)
    for i, n in enumerate(names):
        v = D[n]["model_nll"]["per_sequence"]
        ax[1].scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v, s=12,
                      color=COL.get(n, COL["C4R"]))
    ax[1].axhline(np.log(dv["context_official_artifact_128_windows"]["model_ppl"]), color=COL["OFFICIAL"], ls=":",
                  label="official artifact mean NLL")
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=20, fontsize=7)
    ax[1].set(ylabel="model NLL per token (B1)", title="Per-sequence model NLL"); ax[1].legend(fontsize=7)
    save(fig, "fig3_diversity_repetition")
    # 4 representativeness
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, n in enumerate(names):
        r = S[n]["representativeness"]
        ax[0].bar([j + i * .16 for j in range(4)], [r["routing_l1_per_layer"][str(L)] for L in LAYERS], .16, label=n,
                  color=COL.get(n, COL["C4R"]))
        ax[1].bar(i, r["frac_atypical"], color=COL.get(n, COL["C4R"]))
    oc = rs["context_official_artifact_B128_4layers"]["EBSS"]
    ax[0].axhline(oc["mean_over_layers"]["routing_l1"], color=COL["OFFICIAL"], ls=":", label="official artifact (mean)")
    ax[0].set_xticks([j + .32 for j in range(4)]); ax[0].set_xticklabels([f"L{L}" for L in LAYERS])
    ax[0].set(ylabel="routing L1 vs C4 pool", title="Routing mismatch"); ax[0].legend(fontsize=7)
    ax[1].axhline(oc["typicality"]["frac_atypical"], color=COL["OFFICIAL"], ls=":")
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=20, fontsize=7)
    ax[1].set(ylabel="fraction routing-atypical", title="Atypical sequences (d > d*)")
    save(fig, "fig4_routing_representativeness")
    # 5 cost
    fig, ax = plt.subplots(figsize=(8, 4))
    parts = list(cost["EBSS"]["breakdown_s"].keys())
    bottom = np.zeros(2)
    cmap = matplotlib.colormaps["tab10"]
    for i, k in enumerate(parts):
        v = np.array([cost[m]["breakdown_s"][k] for m in ("PROBABILITY_ONLY", "EBSS")])
        ax.bar([0, 1], v, bottom=bottom, label=k, color=cmap(i))
        bottom += v
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Probability-Only", "EBSS"])
    ax.set(ylabel="seconds (16 x 512, one MIG 2g.20gb slice)", title="Generation time breakdown"); ax.legend(fontsize=7)
    save(fig, "fig5_generation_cost")
    if gq is not None:
        g, ppl, per, dec = gq
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, c in enumerate(QC):
            ax[0].bar(i, g["configs"][c]["recon"], color=COL["EBSS"] if c == "EBSS" else COL["PROBABILITY_ONLY"]
                      if c == "PO" else COL["C4R"])
        ax[0].set_xticks(range(5)); ax[0].set_xticklabels(QC)
        lo = min(g["configs"][c]["recon"] for c in QC)
        ax[0].set_ylim(lo * .95, max(g["configs"][c]["recon"] for c in QC) * 1.02)
        ax[0].set(ylabel="mean expert reconstruction (B=16)", title="Held-out reconstruction")
        for c in ("PO", "EBSS"):
            ax[1].plot(range(10), g["comparisons"][f"{c}_vs_C4R_mean"]["by_decile_relative_pct"], "o-",
                       color=COL["EBSS" if c == "EBSS" else "PROBABILITY_ONLY"], label=f"{c} vs C4 Random mean")
        ax[1].axhline(0, color="gray", lw=.8)
        ax[1].set(xlabel="natural exposure decile (0 = rarest)", ylabel="relative error %",
                  title="Relative reconstruction by exposure decile"); ax[1].legend(fontsize=7)
        save(fig, "fig6_reconstruction")


def main():
    rs, dv, cost = J(p("routing_statistics.json")), J(p("diversity_statistics.json")), J(p("generation_cost.json"))
    smoke, feas, gate = J(p("smoke_test_results.json")), J(p("feasibility.json")), J(p("logs", "diagnostics_gate.json"))
    gq = gptq_block()
    gptq_in = None
    if gq is not None:
        g, ppl, _, _ = gq
        jdump(g, p("reconstruction_results.json"))
        jdump(ppl, p("ppl_results.json"))
        gptq_in = dict(valid=g["valid_all"], ebss=g["configs"]["EBSS"]["recon"], po=g["configs"]["PO"]["recon"],
                       c4=[g["configs"][f"C4R_s{s}"]["recon"] for s in range(3)])
    d = rs["ebss_minus_po"]
    pilot_complete = all(v["valid"] for v in gate["generated_valid"].values())
    label = P.decide(pilot_complete, gate["passed"], d["paper_sigma"], d["cv"], gptq_in)
    balance_only = P.decide(pilot_complete, gate["passed"], d["paper_sigma"], d["cv"], None)
    dec = dict(decision=label, balancing_class_ignoring_gptq=balance_only, rule=P.DECISION_RULE,
               inputs=dict(pilot_complete=pilot_complete, gates_ok=gate["passed"], d_paper_sigma=d["paper_sigma"],
                           d_cv=d["cv"], gptq=gptq_in),
               lower_tail=dict(d_min=d["min"], d_p10=d["p10"], d_n_zero=d["n_zero"]),
               layers_cv_improved_of_24=d["layers_cv_improved"],
               scope="16 x 512 pilot, one seed, one start-token draw; not full-budget EBSS effectiveness")
    jdump(dec, p("scientific_decision.json"))
    pa = paired_analysis()
    jdump(pa, p("paired_analysis.json"))
    figures(rs, dv, cost, gq)
    S = rs["sets"]
    summ = dict(decision=label, balancing_class_ignoring_gptq=balance_only,
                smoke=dict(passed=smoke["passed"], checks=smoke["checks"], control=smoke["control"],
                           replay={k: v["replay"] for k, v in smoke["methods"].items()}),
                feasibility=dict(status=feas["status"], per_step_s=feas["probe"]["per_step_s"]),
                balance={m: dict(paper_sigma=S[m]["b1"]["paper_sigma"], **S[m]["b1"]["mean_over_layers"],
                                 four_layers=S[m]["b1"]["layers_1_8_16_23"],
                                 representativeness=S[m]["representativeness"])
                         for m in S},
                ebss_minus_po=d, generation_vs_b1_routing=rs["generation_vs_b1_routing"],
                intervention=S["EBSS"]["intervention"], paired=pa,
                diversity={m: {k: dv["sets"][m][k] for k in ("distinct1_mean", "distinct2_mean", "distinct3_mean",
                                                             "repetition_rate_mean", "exact_duplicate_windows")}
                           | dict(model_ppl=dv["sets"][m]["model_nll"]["corpus_ppl"],
                                  longest_repeated_ngram_mean=dv["sets"][m]["longest_repeated_ngram"]["mean"],
                                  zlib_median=dv["sets"][m]["zlib_ratio"]["median"],
                                  near_dup_pairs_ge_0_5=dv["sets"][m]["near_duplicates"]["pairs_jaccard_ge_0.5"])
                           for m in dv["sets"]},
                contamination=dv["contamination"],
                cost={m: dict(wall_s=cost[m]["wall_s"], breakdown_frac=cost[m]["breakdown_frac"],
                              emitted_tokens_per_s=cost[m]["emitted_tokens_per_s"],
                              forward_rows_per_s=cost[m]["forward_rows_per_s"], peak_gpu_gb=cost[m]["peak_gpu_gb"],
                              kv_cache_peak_gb=cost[m]["kv_cache_peak_gb"], peak_cpu_gb=cost[m]["peak_cpu_gb"],
                              h2d_gb_per_s=cost[m]["h2d_gb_per_s"])
                      for m in ("PROBABILITY_ONLY", "EBSS")},
                gptq=None if gq is None else dict(recon={c: gq[0]["configs"][c]["recon"] for c in QC},
                                                  comparisons=gq[0]["comparisons"],
                                                  rtn_fallbacks={c: gq[0]["configs"][c]["rtn_fallbacks"] for c in QC},
                                                  ppl={c: gq[1]["configs"][c]["ppl"] for c in QC},
                                                  valid=gq[0]["valid_all"]))
    jdump(summ, p("summary.json"))
    write_md(summ, rs, dv, cost, gq)
    print(f"[report] decision {label} (balancing class {balance_only})")


def write_md(s, rs, dv, cost, gq):
    S = rs["sets"]
    po, eb = "PROBABILITY_ONLY", "EBSS"
    f = lambda x, n=4: f"{x:.{n}f}"
    L = [f"# Model-matched EBSS generation pilot (16 x 512, w = 4, tau = 1.3)\n",
         f"**Decision: {s['decision']}** (balancing class ignoring GPTQ: {s['balancing_class_ignoring_gptq']})\n",
         "## Main generation comparison (CANONICAL-B1 replay routing, all 24 MoE layers)\n",
         "| metric | Probability-only | EBSS | EBSS - PO | C4 Random B16 (mean of 3) |", "|---|---|---|---|---|"]
    c4 = lambda fn: np.mean([fn(f"C4R_s{i}") for i in range(3)])
    rows = [("paper sigma (Eq. 3/4)", lambda m: S[m]["b1"]["paper_sigma"], 5),
            ("load CV (mean of layers)", lambda m: S[m]["b1"]["mean_over_layers"]["cv"], 4),
            ("Gini", lambda m: S[m]["b1"]["mean_over_layers"]["gini"], 4),
            ("min expert exposure", lambda m: S[m]["b1"]["mean_over_layers"]["min"], 1),
            ("p10 expert exposure", lambda m: S[m]["b1"]["mean_over_layers"]["p10"], 1),
            ("zero-exposure experts / layer", lambda m: S[m]["b1"]["mean_over_layers"]["n_zero"], 2),
            ("routing L1 vs C4 (L1/8/16/23)", lambda m: S[m]["representativeness"]["routing_l1_mean"], 3),
            ("frac routing-atypical", lambda m: S[m]["representativeness"]["frac_atypical"], 2),
            ("model PPL on the set", lambda m: dv["sets"][m]["model_nll"]["corpus_ppl"], 3),
            ("distinct-1", lambda m: dv["sets"][m]["distinct1_mean"], 3),
            ("distinct-2", lambda m: dv["sets"][m]["distinct2_mean"], 3),
            ("repetition rate", lambda m: dv["sets"][m]["repetition_rate_mean"], 3)]
    for name, fn, n in rows:
        L.append(f"| {name} | {f(fn(po), n)} | {f(fn(eb), n)} | {fn(eb) - fn(po):+.{n}f} | {f(c4(fn), n)} |")
    L.append(f"| generation wall-s (= GPU-s) | {cost[po]['wall_s']:.0f} | {cost[eb]['wall_s']:.0f} | "
             f"{cost[eb]['wall_s'] - cost[po]['wall_s']:+.0f} | - |")
    L += ["", f"Intervention: {json.dumps(S[eb]['intervention'])}",
          f"Generation-time vs replay routing agreement: "
          f"{ {k: round(v['token_layer_set_agreement'], 4) for k, v in rs['generation_vs_b1_routing'].items()} }", ""]
    if gq is not None:
        g, ppl, _, _ = gq
        L += ["## B=16 GPTQ (layers 1/8/16/23, frozen GPTQ)\n", "| config | recon | RTN fallbacks | PPL |", "|---|---|---|---|"]
        for c in QC:
            L.append(f"| {c} | {g['configs'][c]['recon']:.7f} | {g['configs'][c]['rtn_fallbacks']} | "
                     f"{ppl['configs'][c]['ppl']:.6f} |")
        L.append("")
        for k, v in g["comparisons"].items():
            L.append(f"- {k}: {v['relative_pct']:+.2f} %, experts improved {v['frac_experts_improved']:.2f}")
    L.append("\nSee `update_ebss_generation_pilot.md` for the interpretation.")
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
