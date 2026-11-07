"""Final aggregate-constraint experiment: report on the LOCKED confirmation runs.

Reads runs_aggregate_constraint/locked_protocol.json (hash-verified) and the
15 confirmation evaluations in runs_aggregate_constraint/eval/. Fewer than 15
valid runs -> RUN-BLOCKED (no result is declared). Writes summary.md,
summary.json and figures/*.{pdf,png}. The decision rule is the one frozen in
the protocol.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import aggregate_constraint as DR                    # noqa: E402
import report_typicality as RT                       # noqa: E402
from moe_calib import aggregate_constraint as AG     # noqa: E402
from moe_calib import typicality as TY               # noqa: E402

STRATS, NAMES, WARM = DR.STRATS, DR.NAMES, DR.WARM
EFFECT = 0.005
# reference categorical order (dataviz palette.md, light mode), fixed per strategy
COLORS = {"FR": "#2a78d6", "TR": "#eb6834", "TC": "#1baf7a", "FC": "#eda100", "AG": "#e87ba4"}
MARKERS = {"FR": "o", "TR": "s", "TC": "^", "FC": "D", "AG": "P"}


def selection_metrics(chosen, C, row, ref, scores, layers, E, spec):
    r = dict(chosen=chosen, selected_counts=C[[row[s] for s in chosen]].sum(0).tolist())
    m = RT.set_metrics(r, C, row, ref, scores, layers, E)
    D = spec.D(np.asarray(r["selected_counts"]))
    m.update(aggregate_l1_by_layer={str(L): float(d) for L, d in zip(layers, D)},
             aggregate_l1_mean=float(D.mean()), satisfies_bounds=bool(spec.ok(np.asarray(r["selected_counts"]))))
    return m


def stat(d):
    v = np.array(list(d.values()))
    return dict(by_seed=d, mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                n_negative=int((v < 0).sum()))


def build(outdir, matched_dir, pool_dir):
    proto, h = DR.load_protocol(outdir)
    ev = os.path.join(outdir, "eval")
    runs, missing, invalid = {}, [], []
    for sd in proto["seeds"]:
        for k in STRATS:
            f = os.path.join(ev, f"{k}_s{sd}.json")
            ids = proto["selections"][str(sd)][k]["ids"]
            if not os.path.exists(f):
                missing.append(f"{k}_s{sd}")
                continue
            r = json.load(open(f))
            ok = (r["protocol_sha256"] == h and r["ids_hash"] == DR.ids_hash(ids) and r["chosen"] == ids
                  and r["n_experts_quantized_total"] == 240 and r["eval_split"] == "confirmation")
            (runs.__setitem__((k, sd), r) if ok else invalid.append(f"{k}_s{sd}"))
    res = dict(protocol_sha256=h, git_commit=proto["git_commit"], missing=missing, invalid=invalid,
               n_valid_runs=len(runs), groups={}, paired={}, runs={}, overlap={})
    if missing or invalid:
        res["decision"] = dict(label="RUN-BLOCKED", reason=f"missing {missing}, invalid {invalid}")
        return res, proto
    univ = {json.dumps(r["eval_universe"]) for r in runs.values()}
    if len(univ) != 1:
        res["decision"] = dict(label="RUN-BLOCKED", reason="evaluation universes differ")
        return res, proto
    res["eval_universe_size"] = len(json.loads(univ.pop()))
    # selection-side metrics from the original pool routing cache
    z = np.load(os.path.join(matched_dir, "routing_profile_v2.npz"))
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"].astype(float)
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    ref = TY.pool_reference(C, len(layers), E)
    scores = TY.sequence_scores(C, len(layers), E, ref)
    spec = AG.AggSpec(AG.pool_reference_flat(C, len(layers), E), proto["epsilon"]["eps"], 128,
                      AG.token_invariant(C, len(layers), E), len(layers), E)
    sel = json.load(open(os.path.join(outdir, "selections.json")))
    seeds = proto["seeds"]
    per = {}
    for (k, sd), r in runs.items():
        m = selection_metrics(r["chosen"], C, row, ref, scores, layers, E, spec)
        per[(k, sd)] = dict(recon_mean=r["recon_mean"], expert_median=r["recon_expert_median"],
                            expert_p90=r["recon_expert_p90"], expert_p95=r["recon_expert_p95"],
                            per_layer=r["recon_per_layer"], hessian_s=r["hessian_replay_s"],
                            gptq_s=r["gptq_s"], peak_gpu_mem_gb=r["peak_gpu_mem_gb"], **m)
        res["runs"][f"{k}_s{sd}"] = {x: y for x, y in per[(k, sd)].items()}
    keys = ("recon_mean", "expert_median", "expert_p90", "expert_p95", "d_mean", "d_post_warm",
            "js_set_vs_original_pool", "aggregate_l1_mean", "median_expert_tokens", "gini_expert_tokens",
            "n_low_exposure_experts", "median_effective_sources_per_expert", "hessian_s", "gptq_s")
    for k in STRATS:
        rs = [per[(k, sd)] for sd in seeds]
        v = np.array([r["recon_mean"] for r in rs])
        g = dict(name=NAMES[k], recon_by_seed={str(sd): per[(k, sd)]["recon_mean"] for sd in seeds},
                 recon_sd=float(v.std(ddof=1)))
        for f in keys:
            g[f] = float(np.mean([r[f] for r in rs]))
        g["per_layer"] = {str(L): float(np.mean([r["per_layer"][str(L)] for r in rs])) for L in layers}
        g["aggregate_l1_by_layer"] = {str(L): float(np.mean([r["aggregate_l1_by_layer"][str(L)] for r in rs]))
                                      for L in layers}
        g["satisfies_bounds_all_seeds"] = all(r["satisfies_bounds"] for r in rs)
        g["peak_gpu_mem_gb"] = float(max(r["peak_gpu_mem_gb"] or 0 for r in rs))
        res["groups"][k] = g
    Ev = lambda k, sd: per[(k, sd)]["recon_mean"]
    for b in ("TR", "FR", "TC", "FC"):
        rel = {str(sd): (Ev("AG", sd) - Ev(b, sd)) / Ev(b, sd) for sd in seeds}
        ab = {str(sd): Ev("AG", sd) - Ev(b, sd) for sd in seeds}
        pl = {str(L): stat({str(sd): (per[("AG", sd)]["per_layer"][str(L)] - per[(b, sd)]["per_layer"][str(L)])
                            / per[(b, sd)]["per_layer"][str(L)] for sd in seeds}) for L in layers}
        tail = {q: stat({str(sd): (per[("AG", sd)][q] - per[(b, sd)][q]) / per[(b, sd)][q] for sd in seeds})
                for q in ("expert_median", "expert_p90", "expert_p95")}
        res["paired"][f"AG_vs_{b}"] = dict(rel=stat(rel), abs=stat(ab), per_layer_rel=pl, tail_rel=tail)
    for b in ("FR", "TR", "TC", "FC"):
        res["overlap"][b] = {str(sd): len(set(runs[("AG", sd)]["chosen"][WARM:]) & set(runs[(b, sd)]["chosen"][WARM:]))
                             / (128 - WARM) for sd in seeds}
    # compute accounting (selection side from selections.json)
    trajs = {sd: sel["seeds"][str(sd)]["AG"]["trajectory"] for sd in seeds}
    ej = json.load(open(os.path.join(outdir, "epsilon.json")))
    feas = json.load(open(os.path.join(outdir, "feasibility.json")))
    fwd = 0.1784897712951011 * 2000
    res["cost"] = dict(
        routing_profile_full_forward_equiv_s=fwd, epsilon_construction_s=ej["construction_s"],
        ag_selection_s={str(sd): sel["seeds"][str(sd)]["AG"]["selection_s"] for sd in seeds},
        ag_lp_s={str(sd): float(sum(t["lp_s"] for t in trajs[sd])) for sd in seeds},
        ag_certificate_s={str(sd): float(sum(t["certificate_s"] for t in trajs[sd])) for sd in seeds},
        ag_repair_milp_s={str(sd): float(sum(t.get("milp_s", 0.0) for t in trajs[sd])) for sd in seeds},
        prepare_feasibility_milp_s={str(sd): feas["seeds"][str(sd)]["milp_s"] for sd in seeds},
        hessian_replay_s={k: res["groups"][k]["hessian_s"] for k in STRATS},
        gptq_s={k: res["groups"][k]["gptq_s"] for k in STRATS},
        peak_gpu_mem_gb=max(g["peak_gpu_mem_gb"] for g in res["groups"].values()),
        needs_routing_profile={"FR": False, "TR": True, "TC": True, "FC": True, "AG": True})
    res["controller"] = {str(sd): [dict(round=t["round"], repaired=t["repaired"], naive_feasible=t["naive_rounding_feasible"],
                                        n_swaps=t["n_rounding_repairs"], lp_objective=t["lp_objective"],
                                        naive_objective=t["naive_objective"], final_objective=t["final_objective"],
                                        objective_loss_rel=t["rounding_objective_loss_rel"],
                                        milp_status=t.get("milp_status"), milp_gap=t.get("milp_gap"))
                                   for t in trajs[sd]] for sd in seeds}
    # decision (frozen rule)
    P = res["paired"]
    tr, fc = P["AG_vs_TR"], P["AG_vs_FC"]
    G = res["groups"]
    worst_layer = max(v["mean"] for v in tr["per_layer_rel"].values())
    p95 = tr["tail_rel"]["expert_p95"]["mean"]
    bounds = G["AG"]["satisfies_bounds_all_seeds"]
    if (G["AG"]["recon_mean"] < G["TR"]["recon_mean"] and tr["rel"]["n_negative"] == 3 and tr["rel"]["mean"] <= -EFFECT
            and p95 <= EFFECT and worst_layer <= EFFECT and bounds):
        lab = "AGGREGATE-ALLOCATION-PASS"
    elif fc["rel"]["n_negative"] == 3 and G["AG"]["aggregate_l1_mean"] < G["FC"]["aggregate_l1_mean"]:
        lab = "AGGREGATE-ALLOCATION-PARTIAL"
    else:
        lab = "AGGREGATE-ALLOCATION-NO-GAIN"
    res["decision"] = dict(label=lab, delta_vs_TR=tr["rel"], worst_layer_rel_vs_TR=worst_layer,
                           p95_rel_vs_TR=p95, bounds_satisfied=bounds,
                           n_better_FC=fc["rel"]["n_negative"])
    return res, proto


# ---------------------------------------------------------------- figures

def figures(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a8a", "axes.labelcolor": "#333333", "xtick.color": "#555555",
                         "ytick.color": "#555555", "grid.color": "#e6e6e6"})
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    G, seeds = res["groups"], sorted(int(s) for s in res["groups"]["AG"]["recon_by_seed"])
    order = ["TR", "TC", "FR", "AG", "FC"]

    # 1. per-seed confirmation error, strategies on x, seeds as connected points
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, k in enumerate(order):
        ys = [G[k]["recon_by_seed"][str(s)] for s in seeds]
        ax.scatter([i] * len(ys), ys, s=36, color=COLORS[k], marker=MARKERS[k], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color="#333333", linewidth=2)
    for s in seeds:
        ax.plot(range(len(order)), [G[k]["recon_by_seed"][str(s)] for k in order], color="#bdbdbd",
                linewidth=1, zorder=1)
    ax.set_xticks(range(len(order)), [NAMES[k] for k in order], rotation=12)
    ax.set_ylabel("mean relative expert error\n(confirmation split, lower is better)")
    ax.set_title("Confirmation error by strategy (dots: seeds 0/1/2; bar: mean)", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"confirmation_recon.{ext}"), dpi=200)
    plt.close(fig)

    # 2. two routing proxies vs error (small multiples, shared error axis)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, key, lab in ((axes[0], "aggregate_l1_mean", "aggregate routing L1 to pool\n(mean over layers)"),
                         (axes[1], "d_post_warm", "mean per-sequence d(s)\nof the 96 post-warm picks")):
        for k in order:
            xs = [res["runs"][f"{k}_s{s}"][key] for s in seeds]
            ys = [res["runs"][f"{k}_s{s}"]["recon_mean"] for s in seeds]
            ax.scatter(xs, ys, s=40, color=COLORS[k], marker=MARKERS[k], label=NAMES[k], edgecolor="white",
                       linewidth=1.5, zorder=3)
        ax.set_xlabel(lab)
        ax.grid(True)
    axes[0].set_ylabel("mean relative expert error (confirmation)")
    axes[0].set_title("Aggregate shift", loc="left")
    axes[1].set_title("Per-sequence atypicality", loc="left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"shift_vs_error.{ext}"), dpi=200)
    plt.close(fig)

    # 3. per-layer paired relative change of AG vs each baseline
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    layers = list(res["paired"]["AG_vs_TR"]["per_layer_rel"])
    bases = ["TR", "TC", "FR", "FC"]
    w = 0.18
    for j, b in enumerate(bases):
        v = [100 * res["paired"][f"AG_vs_{b}"]["per_layer_rel"][L]["mean"] for L in layers]
        ax.bar(np.arange(len(layers)) + (j - 1.5) * w, v, width=w - 0.02, color=COLORS[b],
               label=f"vs {NAMES[b]}", zorder=3)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(range(len(layers)), [f"layer {L}" for L in layers])
    ax.set_ylabel("Aggregate Coverage-LP − baseline\n(% relative error, mean of 3 seeds)")
    ax.set_title("Per-layer paired change of Aggregate Coverage-LP (below 0 = better)", loc="left")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"per_layer_paired.{ext}"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- markdown

def pct(x):
    return f"{100 * x:+.2f} %"


def write_md(res, proto, path):
    L = []
    w = L.append
    w("# Aggregate-shift-constrained Coverage-LP — locked confirmation summary\n")
    w(f"Protocol `locked_protocol.json` sha256 `{res['protocol_sha256'][:16]}…`, commit `{res['git_commit'][:12]}`. "
      "Mean relative expert reconstruction error on the **locked confirmation split** (256 new, document-disjoint "
      "C4 windows); 240 routed experts, layers 1/8/16/23; partial-model non-sequential expert GPTQ (attention and "
      "shared experts BF16). Lower is better.\n")
    w(f"**Decision (rule frozen in the protocol): {res['decision']['label']}**\n")
    if res["decision"]["label"] == "RUN-BLOCKED":
        w(res["decision"]["reason"])
        open(path, "w").write("\n".join(L) + "\n")
        return
    G = res["groups"]
    seeds = sorted(G["AG"]["recon_by_seed"])
    w(f"Valid runs: {res['n_valid_runs']} / 15; evaluation universe {res['eval_universe_size']} experts (identical for all runs).\n")
    w("eps_L (median of 3,000 random completions of the fixed warm starts): " +
      ", ".join(f"L{l} {e:.5f}" for l, e in zip(proto["layers"], proto["epsilon"]["eps"])) + "\n")
    w("## Reconstruction (confirmation split)\n")
    w("| strategy | mean | SD | " + " | ".join(f"s{s}" for s in seeds) + " | expert median | p90 | p95 | " +
      " | ".join(f"L{l}" for l in proto["layers"]) + " |")
    w("|---" * (3 + len(seeds) + 3 + len(proto["layers"])) + "|")
    for k in ("TR", "TC", "FR", "AG", "FC"):
        g = G[k]
        nm = f"**{NAMES[k]}**" if k == "AG" else NAMES[k]
        w(f"| {nm} | {g['recon_mean']:.6f} | {g['recon_sd']:.6f} | " +
          " | ".join(f"{g['recon_by_seed'][s]:.6f}" for s in seeds) +
          f" | {g['expert_median']:.6f} | {g['expert_p90']:.6f} | {g['expert_p95']:.6f} | " +
          " | ".join(f"{v:.6f}" for v in g["per_layer"].values()) + " |")
    w("\n## Paired differences: Aggregate Coverage-LP − baseline\n")
    w("| baseline | Δ_s (rel.) s0 / s1 / s2 | mean | seeds AG better | per-layer mean (L1/L8/L16/L23) | p95 rel |")
    w("|---|---|---|---|---|---|")
    for b in ("TR", "TC", "FR", "FC"):
        p = res["paired"][f"AG_vs_{b}"]
        w(f"| {NAMES[b]} | " + " / ".join(pct(v) for v in p["rel"]["by_seed"].values()) +
          f" | {pct(p['rel']['mean'])} | {p['rel']['n_negative']}/3 | " +
          " / ".join(pct(v["mean"]) for v in p["per_layer_rel"].values()) +
          f" | {pct(p['tail_rel']['expert_p95']['mean'])} |")
    w("\n## Selection: routing shift, typicality, exposure (mean over seeds)\n")
    w("| strategy | aggregate L1 (L1/L8/L16/L23) | mean L1 | within all eps | set JS | mean d(s) | post-warm d | "
      "Gini | median tokens | median eff. sources |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k in ("TR", "TC", "FR", "AG", "FC"):
        g = G[k]
        w(f"| {NAMES[k]} | " + " / ".join(f"{v:.4f}" for v in g["aggregate_l1_by_layer"].values()) +
          f" | {g['aggregate_l1_mean']:.4f} | {'yes' if g['satisfies_bounds_all_seeds'] else 'no'} | "
          f"{g['js_set_vs_original_pool']:.5f} | {g['d_mean']:.4f} | {g['d_post_warm']:.4f} | "
          f"{g['gini_expert_tokens']:.4f} | {g['median_expert_tokens']:.0f} | {g['median_effective_sources_per_expert']:.1f} |")
    w("\nPost-warm overlap of AG with: " + "; ".join(
        f"{NAMES[b]} {np.mean(list(v.values())):.3f}" for b, v in res["overlap"].items()) + "\n")
    w("## Controller (coverage_lp_aggregate)\n")
    w("| seed | round | naive rounding feasible | repaired | swaps | LP obj | naive obj | final obj | loss vs naive | MILP status / gap |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for sd, ts in res["controller"].items():
        for t in ts:
            gap = "—" if t["milp_gap"] is None else f"{t['milp_status']} / {t['milp_gap']:.2e}"
            w(f"| {sd} | {t['round']} | {t['naive_feasible']} | {t['repaired']} | {t['n_swaps']} | "
              f"{t['lp_objective']:.2f} | {t['naive_objective']:.2f} | {t['final_objective']:.2f} | "
              f"{pct(t['objective_loss_rel'])} | {gap} |")
    c = res["cost"]
    w("\n## Compute\n")
    w(f"- Routing profile of the 2,000 candidates: ~{c['routing_profile_full_forward_equiv_s']:.0f} s full-model-forward "
      "equivalent (needed by TR, TC, FC, AG; not by FR).")
    w(f"- eps construction {c['epsilon_construction_s']:.1f} s; AG selection per seed " +
      ", ".join(f"{v:.0f} s" for v in c["ag_selection_s"].values()) + " (LP " +
      ", ".join(f"{v:.1f}" for v in c["ag_lp_s"].values()) + " s; integer certificates " +
      ", ".join(f"{v:.1f}" for v in c["ag_certificate_s"].values()) + " s; repair MILP " +
      ", ".join(f"{v:.0f}" for v in c["ag_repair_milp_s"].values()) + " s).")
    w("- Hessian replay / GPTQ per run (s): " + "; ".join(
        f"{NAMES[k]} {c['hessian_replay_s'][k]:.0f} / {c['gptq_s'][k]:.0f}" for k in STRATS) +
      f"; peak GPU {c['peak_gpu_mem_gb']:.1f} GB.")
    w("- Equal selected-sequence budget (B = 128) is not equal compute: every routing-based strategy pays "
      "the ~357 s profile (≈ 2,000 sequences of forward passes) plus selection time.\n")
    w("Figures: `figures/confirmation_recon.{pdf,png}`, `figures/shift_vs_error.{pdf,png}`, "
      "`figures/per_layer_paired.{pdf,png}`.")
    open(path, "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_aggregate_constraint")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--pool-dir", default="runs_typicality")
    a = ap.parse_args(argv)
    res, proto = build(a.outdir, a.matched_dir, a.pool_dir)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
    if res["decision"]["label"] != "RUN-BLOCKED":
        figures(res, a.outdir)
    write_md(res, proto, os.path.join(a.outdir, "summary.md"))
    print(open(os.path.join(a.outdir, "summary.md")).read())


if __name__ == "__main__":
    main()
