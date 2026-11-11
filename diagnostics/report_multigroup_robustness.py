"""Results, summary, decision (rule frozen in protocol.json) and figures for the multi-grouping robustness study."""
from __future__ import annotations
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import multigroup_robustness as MR      # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402
from moe_calib import multigroup_robustness as MG   # noqa: E402

P, G = MR.p, MR.GROUPS
COL = {"FR": "#2a78d6", "FC": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
HIST = ("ORIGINAL", "COMPACT", "SHUFFLED")


def load(*a):
    path = P(*a)
    return json.load(open(path)) if os.path.exists(path) else None


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0


def build():
    pr = MR.protocol()
    cm, act = load("cache_manifest.json"), load("activation_diagnostics.json")
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    b = base["ppl"]
    tau = 1e-4 * b
    res = {(c, g): load("results", f"{MR.rname(c, g)}.json") for c in MR.STRATS for g in G + ["ORIGINAL"]}
    missing = [MR.rname(c, g) for (c, g), r in res.items() if r is None]
    invalid = [MR.rname(c, g) for (c, g), r in res.items() if r is not None and not r["valid"]]
    ctrl_ok = all(res[(c, "ORIGINAL")] and res[(c, "ORIGINAL")]["determinism"]["original_reproduced_bitwise"]
                  for c in MR.STRATS)
    semantics_ok = bool(cm and cm.get("valid"))
    exec_order_ok = bool(cm and cm["semantics"]["execution_order_controls_passed"])
    repeat_ok = all(r["repeat_eval_abs_diff_ppl"] == 0 for r in res.values() if r)
    valid = semantics_ok and not invalid
    blocked = bool(missing or not ctrl_ok or not exec_order_ok or not repeat_ok)
    out = dict(protocol_sha256=pr["protocol_sha256"], decision_rule=pr["decision_rule"], n_configs_complete=10 -
               sum(1 for m in missing if not m.startswith("CTRL")), n_configs_expected=10, missing=missing, invalid=invalid,
               controls=dict(original_reproduced=ctrl_ok, execution_order=exec_order_ok, repeat_ppl_identical=repeat_ok,
                             cache_semantics_valid=semantics_ok))
    if missing or invalid:
        out["decision"] = MG.decide({}, G, b, valid=valid, blocked=True)
        return out, None
    ppl = {k: r["ppl"] for k, r in res.items()}
    s = MG.grouping_stats(ppl, G, MG.DELTA_ORIGINAL, tau)
    out["decision"] = MG.decide(ppl, G, b, valid=valid, blocked=blocked)
    tok = res[("FR", G[0])]["per_window_tokens"]
    assert all(r["per_window_tokens"] == tok for r in res.values())
    boot = {g: FD.paired_bootstrap(res[("FC", g)]["per_window_nll"], res[("FR", g)]["per_window_nll"], tok) for g in G}
    boot["ORIGINAL"] = FD.paired_bootstrap(res[("FC", "ORIGINAL")]["per_window_nll"], res[("FR", "ORIGINAL")]["per_window_nll"], tok)
    bsum = MG.bootstrap_summary({g: boot[g]["ci95"] for g in G}, {g: boot[g]["point"] for g in G})

    # ---------------- PPL results
    pplres = dict(bf16_ppl=b, tau=tau, evaluation=pr["evaluation"], configs={
        MR.rname(c, g): dict(strategy=c, grouping=g, ppl=r["ppl"], mean_nll=r["mean_nll"], total_tokens=r["total_tokens"],
                             ppl_minus_bf16=r["ppl"] - b, change_vs_original=r["ppl"] - ppl[(c, "ORIGINAL")],
                             installed_weights_sha256=r["installed_weights_sha256"],
                             repeat_eval_abs_diff_ppl=r["repeat_eval_abs_diff_ppl"],
                             per_window_nll=r["per_window_nll"], per_window_tokens=r["per_window_tokens"], valid=r["valid"])
        for (c, g), r in res.items()}, delta=s["delta"], delta_original=MG.DELTA_ORIGINAL,
        paired_window_bootstrap=boot, bootstrap_summary=bsum)
    MR.jdump(pplres, P("ppl_results.json"))

    # ---------------- reconstruction
    rec = {}
    for c in MR.STRATS:
        o = {(x["layer"], x["expert"]): x["mean"] for x in res[(c, "ORIGINAL")]["per_expert"]}
        for g in G:
            r = res[(c, g)]
            d = np.array([(x["mean"] - o[(x["layer"], x["expert"])]) / o[(x["layer"], x["expert"])] for x in r["per_expert"]])
            rec[MR.rname(c, g)] = dict(recon=r["recon"], recon_per_layer=r["recon_per_layer"],
                                       rel_change_vs_original=(r["recon"] - res[(c, "ORIGINAL")]["recon"]) / res[(c, "ORIGINAL")]["recon"],
                                       per_layer_rel_change_vs_original={L: (v - res[(c, "ORIGINAL")]["recon_per_layer"][L]) /
                                                                         res[(c, "ORIGINAL")]["recon_per_layer"][L]
                                                                         for L, v in r["recon_per_layer"].items()},
                                       per_expert_abs_rel_change=dict(median=float(np.median(np.abs(d))),
                                                                      p90=float(np.quantile(np.abs(d), .9)),
                                                                      max=float(np.abs(d).max()), frac_worse=float((d > 0).mean())),
                                       weight_change_vs_original=r["weight_change_vs_original"])
        rec[MR.rname(c, "ORIGINAL")] = dict(recon=res[(c, "ORIGINAL")]["recon"], recon_per_layer=res[(c, "ORIGINAL")]["recon_per_layer"])
    gap = {g: (res[("FC", g)]["recon"] - res[("FR", g)]["recon"]) / res[("FR", g)]["recon"] for g in G + ["ORIGINAL"]}
    rec_summary = dict(fc_minus_fr_recon_rel_gap=gap, fc_recon_worse_in=sum(gap[g] > 0 for g in G),
                       gap_range=max(gap[g] for g in G) - min(gap[g] for g in G),
                       max_abs_rel_change_vs_original={c: max(abs(rec[MR.rname(c, g)]["rel_change_vs_original"]) for g in G)
                                                       for c in MR.STRATS},
                       corr_recon_vs_ppl_within_strategy={c: MG.pearson([res[(c, g)]["recon"] for g in G],
                                                                        [ppl[(c, g)] for g in G]) for c in MR.STRATS},
                       corr_recon_gap_vs_delta=MG.pearson([gap[g] for g in G], [s["delta"][g] for g in G]),
                       note="5 points per correlation, one allocation seed: descriptive only")
    MR.jdump(dict(evaluator="E3 held-out relative expert reconstruction, ~/moe_cache test rows 2128-2255 (eval_cache_dir)",
                  configs=rec, summary=rec_summary), P("reconstruction_results.json"))

    # ---------------- historical comparison (different policies: context only)
    hist = json.load(open(os.path.join(ROOT, "runs_batch_sensitivity", "results.json")))
    htab = []
    for v in HIST:
        fr, fc = hist[f"FR|{v}"], hist[f"FC|{v}"]
        htab.append(dict(grouping=v, policy={"ORIGINAL": "cache_inputs.py, 176 x 8 consecutive original rows",
                                             "COMPACT": "345 rows (217 union + 128 old test), sorted, 43 x 8 + 1",
                                             "SHUFFLED": "345 rows, default_rng(20261110), 43 x 8 + 1"}[v],
                         sample="historical reference", fr_ppl=fr["ppl"], fc_ppl=fc["ppl"], delta=fc["ppl"] - fr["ppl"],
                         fr_recon=fr["recon"], fc_recon=fc["recon"]))
    for g in G:
        htab.append(dict(grouping=g, policy=f"217 calibration rows only, default_rng({g[1:]}), 27 x 8 + 1",
                         sample="primary randomized sample", fr_ppl=ppl[("FR", g)], fc_ppl=ppl[("FC", g)],
                         delta=s["delta"][g], fr_recon=res[("FR", g)]["recon"], fc_recon=res[("FC", g)]["recon"]))

    # ---------------- activation / routing summary
    actsum = None
    if act:
        actsum = {L: {g: {k: x[g][k] for k in ("relative_l2", "pct_bitwise_equal", "token_rel_l2_p50", "token_rel_l2_p99",
                                              "frac_tokens_top4_set_changed", "routing_distribution_l1")} for g in G}
                  for L, x in act["per_layer"].items()}
    wchg = {MR.rname(c, g): dict(frac_differ=sum(v["frac_differ"] * v["n"] for v in res[(c, g)]["weight_change_vs_original"].values()) /
                                 sum(v["n"] for v in res[(c, g)]["weight_change_vs_original"].values()),
                                 rel_l2=(sum(v["d2"] for v in res[(c, g)]["weight_change_vs_original"].values()) /
                                         sum(v["o2"] for v in res[(c, g)]["weight_change_vs_original"].values())) ** .5)
            for c in MR.STRATS for g in G if res[(c, g)]["weight_change_vs_original"]}
    runs = [res[k] for k in res]
    disk = dict(caches_gb=sum(dir_bytes(MR.gdir(g)) for g in G) / 1e9, weights_gb=dir_bytes(os.path.join(MR.MCACHE, "weights")) / 1e9)
    d0 = abs(MG.DELTA_ORIGINAL)
    out.update(
        bf16_ppl=b, tau=tau, stats=s, ppl={MR.rname(c, g): ppl[(c, g)] for (c, g) in ppl},
        decision_quantities=dict(abs_delta_original=d0, max_abs_interaction=s["max_abs_interaction"],
                                 max_abs_interaction_over_abs_delta_original=s["max_abs_interaction"] / d0,
                                 sd_delta=s["sd_delta"], sd_over_abs_delta_original=s["sd_over_abs_delta_original"],
                                 strategy_range_over_abs_delta_original={c: v / d0 for c, v in s["strategy_range"].items()},
                                 n_sign_reversals_beyond_tau=s["n_sign_reversals_beyond_tau"],
                                 floor=max(tau, 0.1 * d0)),
        exceeds_repeatability=dict(repeat_ppl_diff=0.0, all_nonzero_deltas_exceed_repeatability=all(abs(s["delta"][g]) > 0 for g in G),
                                   n_abs_delta_gt_tau=sum(abs(s["delta"][g]) > tau for g in G)),
        paired_window_bootstrap={g: dict(point=x["point"], ci95=x["ci95"]) for g, x in boot.items()},
        bootstrap_summary=bsum, reconstruction=rec_summary, historical_comparison=htab,
        activation_routing=actsum, weight_change_vs_original=wchg,
        determinism={c: res[(c, "ORIGINAL")]["determinism"] for c in MR.STRATS},
        execution_order_controls={g: cm["groupings"][g]["negative_control"]["passed"] for g in G},
        correctness={MR.rname(c, g): dict(experts=r["n_quantized_experts"], projections=r["n_projection_tensors"],
                                          rtn=r["rtn_fallbacks"], unchanged=r["projections_unchanged_from_bf16"],
                                          nonfinite=r["nonfinite_weights"], untouched_ok=r["untouched_params_unchanged"],
                                          expert_checks_pass=all(k["pass"] for k in r["expert_checks"]),
                                          repeat_eval=r["repeat_eval_abs_diff_ppl"], valid=r["valid"])
                     for (c, g), r in res.items()},
        compute=dict(quantize_s={MR.rname(c, g): res[(c, g)]["quantize_s"] for (c, g) in res},
                     mean_quantize_s=float(np.mean([r["quantize_s"] for r in runs])),
                     mean_eval_s=float(np.mean([r["eval_s"] for r in runs])),
                     cache_generation_s={g: cm["groupings"][g]["generation_s"] for g in G},
                     cache_negative_control_s={g: cm["groupings"][g]["negative_control"]["seconds"] for g in G},
                     diagnostics_s=act["seconds"] if act else None,
                     peak_gpu_gb=max(r["peak_gpu_gb"] for r in runs), peak_cpu_gb=max(r["peak_cpu_gb"] for r in runs),
                     disk=disk, cache_storage_bytes=cm["storage_bytes_total"]),
        environment=cm["model"]["environment"])
    return out, dict(res=res, ppl=ppl, s=s, gap=gap, act=act, boot=boot, hist=hist)


# ---------------------------------------------------------------- figures
def figures(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    d = P("figures")
    sv = lambda f, n: (f.savefig(os.path.join(d, n + ".pdf"), bbox_inches="tight"),
                       f.savefig(os.path.join(d, n + ".png"), dpi=200, bbox_inches="tight"), plt.close(f))
    ppl, s, res, gap = raw["ppl"], raw["s"], raw["res"], raw["gap"]
    lab = [g[-2:] for g in G]
    x = np.arange(len(G))

    # 1 PPL across groupings, paired lines
    f, ax = plt.subplots(figsize=(6, 3.4))
    for i, g in enumerate(G):
        ax.plot([i - .12, i + .12], [ppl[("FR", g)], ppl[("FC", g)]], color=MUTED, lw=1, zorder=1)
    for c, dx in (("FR", -.12), ("FC", .12)):
        ax.scatter(x + dx, [ppl[(c, g)] for g in G], s=36, color=COL[c], edgecolor="white", lw=1.5, zorder=3,
                   label=f"{MR.STRATS[c]}")
        ax.axhline(ppl[(c, "ORIGINAL")], color=COL[c], lw=1, ls=":", zorder=0)
    ax.set_xticks(x, [f"…{l}" for l in lab]); ax.set_xlabel("grouping seed 202611xx (5 prespecified groupings)")
    ax.set_ylabel("WikiText-2 PPL (4 layers GPTQ)")
    ax.set_title("FR vs FC PPL per calibration grouping (dotted: ORIGINAL grouping)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    sv(f, "fig1_ppl_by_grouping")

    # 2 Delta_g
    f, ax = plt.subplots(figsize=(6, 3.2))
    dl = [s["delta"][g] for g in G]
    ax.bar(x, dl, width=.5, color=["#eb6834" if v > 0 else "#2a78d6" for v in dl], zorder=2)
    ax.axhline(0, color=INK, lw=.8)
    ax.axhline(MG.DELTA_ORIGINAL, color=MUTED, lw=1.2, ls="--")
    ax.text(len(G) - .5, MG.DELTA_ORIGINAL, f"ORIGINAL +{MG.DELTA_ORIGINAL:.4f}", ha="right", va="bottom", color=MUTED, fontsize=8)
    for i, v in enumerate(dl):
        ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v > 0 else "top", fontsize=7.5, color=INK)
    ax.set_xticks(x, [f"…{l}" for l in lab]); ax.set_xlabel("grouping seed 202611xx")
    ax.set_ylabel("Δ_g = PPL(FC) − PPL(FR)")
    ax.set_title("Coverage penalty per grouping (>0: FC worse; <0: FC better)", fontsize=9, loc="left")
    sv(f, "fig2_delta_by_grouping")

    # 3 strategy-specific sensitivity
    f, ax = plt.subplots(figsize=(5, 3.4))
    for i, c in enumerate(("FR", "FC")):
        v = [ppl[(c, g)] for g in G]
        ax.vlines(i, min(v), max(v), color=COL[c], lw=6, alpha=.25)
        ax.scatter([i] * 5, v, s=30, color=COL[c], edgecolor="white", lw=1.2, zorder=3)
        hv = [raw["hist"][f"{c}|{h}"]["ppl"] for h in HIST]
        ax.scatter([i + .22] * 3, hv, s=26, facecolor="none", edgecolor=MUTED, lw=1, zorder=3,
                   label="historical (ORIGINAL/COMPACT/SHUFFLED)" if i == 0 else None)
        ax.text(i - .1, max(v), f"range {s['strategy_range'][c]:.4f}", ha="right", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks([0, 1], [MR.STRATS["FR"], MR.STRATS["FC"]]); ax.set_xlim(-.6, 1.6)
    ax.set_ylabel("PPL"); ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("PPL spread over the 5 new groupings (filled)", fontsize=9, loc="left")
    sv(f, "fig3_strategy_sensitivity")

    # 4 reconstruction vs PPL
    f, ax = plt.subplots(figsize=(5.6, 3.6))
    for c in ("FR", "FC"):
        ax.scatter([res[(c, g)]["recon"] for g in G], [ppl[(c, g)] for g in G], s=34, color=COL[c], edgecolor="white",
                   lw=1.2, label=f"{MR.STRATS[c]} (5 groupings)", zorder=3)
        ax.scatter([res[(c, "ORIGINAL")]["recon"]], [ppl[(c, "ORIGINAL")]], s=60, marker="D", facecolor="none",
                   edgecolor=COL[c], lw=1.4, label=f"{c} ORIGINAL", zorder=3)
    ax.set_xlabel("mean expert relative reconstruction error (held-out)"); ax.set_ylabel("PPL")
    ax.set_title("Reconstruction is nearly fixed per strategy while PPL moves", fontsize=9, loc="left")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0)); ax.legend(frameon=False, fontsize=7.5)
    sv(f, "fig4_recon_vs_ppl")

    # 5 activation / routing
    act = raw["act"]
    if act:
        Ls = [str(L) for L in MR.LAYERS]
        f, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for g in G:
            axs[0].plot(MR.LAYERS, [act["per_layer"][L][g]["relative_l2"] for L in Ls], color=MUTED, lw=1, marker="o", ms=4, alpha=.8)
            axs[1].plot(MR.LAYERS, [100 * act["per_layer"][L][g]["frac_tokens_top4_set_changed"] for L in Ls], color=MUTED,
                        lw=1, marker="o", ms=4, alpha=.8)
        hr = json.load(open(os.path.join(ROOT, "runs_batch_sensitivity", "routing_disagreement.json")))["per_layer"]
        ha = json.load(open(os.path.join(ROOT, "runs_batch_sensitivity", "activation_drift.json")))["per_layer"]
        for v, cc in (("COMPACT", "#1baf7a"), ("SHUFFLED", "#6250d6")):
            axs[0].plot(MR.LAYERS, [ha[L][v]["calibration_rows"]["relative_l2"] for L in Ls], color=cc, lw=1, ls="--", label=f"{v} (hist.)")
            axs[1].plot(MR.LAYERS, [100 * hr[L][v]["frac_any_expert_changed"] for L in Ls], color=cc, lw=1, ls="--")
        axs[0].plot([], [], color=MUTED, marker="o", ms=4, label="5 new groupings")
        axs[0].set_yscale("log"); axs[0].set_xlabel("layer input"); axs[0].set_ylabel("relative L2 vs ORIGINAL")
        axs[1].set_xlabel("layer"); axs[1].set_ylabel("% calibration tokens, top-4 set changed")
        axs[0].legend(frameon=False, fontsize=7.5)
        for a_ in axs:
            a_.set_xticks(MR.LAYERS)
        f.suptitle("Activation and routing differences vs ORIGINAL (217 calibration rows)", fontsize=9, x=.02, ha="left")
        f.tight_layout()
        sv(f, "fig5_activation_routing")


# ---------------------------------------------------------------- markdown
def markdown(o):
    s = o["stats"]
    L = [f"# Multi-grouping robustness: FR s0 vs FC s0 (four-layer GPTQ)", "",
         f"**Decision (rule frozen in protocol.json, sha256 `{o['protocol_sha256'][:16]}`): {o['decision']}**", "",
         f"BF16 PPL {o['bf16_ppl']:.6f}; tau = 1e-4 x BF16 = {o['tau']:.6f}; Delta_ORIGINAL = {MG.DELTA_ORIGINAL:+.6f}.", "",
         "| grouping | FR PPL | FC PPL | Delta_g | interaction | FC rank | window-bootstrap 95% CI (NLL diff) |",
         "|---|---|---|---|---|---|---|"]
    for g in G:
        ci = o["paired_window_bootstrap"][g]["ci95"]
        L.append(f"| {g} | {o['ppl'][f'FR_{g}']:.6f} | {o['ppl'][f'FC_{g}']:.6f} | {s['delta'][g]:+.6f} | "
                 f"{s['interaction'][g]:+.6f} | {'FC worse' if s['delta'][g] > 0 else 'FC better'} | [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
    L += ["", f"- mean Delta {s['mean_delta']:+.6f}, median {s['median_delta']:+.6f}, SD {s['sd_delta']:.6f}, "
              f"min {s['min_delta']:+.6f}, max {s['max_delta']:+.6f}, range {s['delta_range']:.6f}",
          f"- FC worse in {s['n_fc_worse']}/5, better in {s['n_fc_better']}/5; sign reversals vs ORIGINAL "
          f"{s['n_sign_reversals']}/5 ({s['n_sign_reversals_beyond_tau']} beyond tau)",
          f"- Range_FR {s['strategy_range']['FR']:.6f}, Range_FC {s['strategy_range']['FC']:.6f}; "
          f"max |I_g| {s['max_abs_interaction']:.6f}; SD/|Delta_O| {s['sd_over_abs_delta_original']:.2f}",
          f"- reconstruction FC-FR gap: " + ", ".join(f"{g} {100 * v:+.2f}%" for g, v in o["reconstruction"]["fc_minus_fr_recon_rel_gap"].items()),
          f"- controls: {o['controls']}", ""]
    L += ["## Historical comparison (different policies; not pooled)", "", "| grouping | sample | policy | FR | FC | Delta |",
          "|---|---|---|---|---|---|"]
    for h in o["historical_comparison"]:
        L.append(f"| {h['grouping']} | {h['sample']} | {h['policy']} | {h['fr_ppl']:.6f} | {h['fc_ppl']:.6f} | {h['delta']:+.6f} |")
    return "\n".join(L) + "\n"


def main():
    out, raw = build()
    MR.jdump(out, P("summary.json"))
    if raw is None:
        print(f"[report] incomplete: {out['missing']} invalid {out['invalid']} -> {out['decision']}"); return
    open(P("summary.md"), "w").write(markdown(out))
    figures(raw)
    print(f"[report] decision {out['decision']}; deltas " + " ".join(f"{v:+.5f}" for v in out["stats"]["delta"].values()))


if __name__ == "__main__":
    main()
