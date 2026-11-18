"""Report for the expert-wise calibration MVP: summary.json, summary.md and figures from the FINAL runs.

Interpretation text is NOT generated: if runs_expertwise_mvp/interpretation.md exists it is appended verbatim.
"""
from __future__ import annotations

import json, os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs_expertwise_mvp")
C4 = ("SR", "SC", "EC", "ER")
COLORS = dict(SR="#2a78d6", SC="#eb6834", EC="#1baf7a", ER="#eda100")   # reference categorical slots 1-4 (fixed order)
COMPS = [("EC", "SC", "constraint relaxation, given the covariance surrogate (PRIMARY)"),
         ("EC", "ER", "covariance surrogate, given expert-wise sets"),
         ("SC", "SR", "covariance surrogate, given a shared set"),
         ("ER", "SR", "constraint relaxation, given random selection")]


def jl(*a):
    return json.load(open(os.path.join(OUT, *a)))


def err(r, variant, c, e, split="test"):
    return r["results"][f"{variant}.{c}.{e}"][split]["mean"]


def rel_imp(r, x, y, e, variant="matched", split="test"):
    """(err_Y - err_X) / err_Y: positive = X better."""
    ey, ex = err(r, variant, y, e, split), err(r, variant, x, e, split)
    return (ey - ex) / ey


def verdict(per_seed, per_expert):
    cons = all(v > 0 for v in per_seed) and sum(v > 0 for v in per_expert) >= 6
    neg = all(v < 0 for v in per_seed) and sum(v < 0 for v in per_expert) >= 6
    m = float(np.mean(per_seed))
    if cons:
        return "CONSISTENT-IMPROVEMENT" + (" (material, >= 0.5 %)" if m >= 0.005 else " (below 0.5 % materiality)")
    if neg:
        return "CONSISTENT-DEGRADATION" + (" (material)" if m <= -0.005 else " (below 0.5 %)")
    return "INCONSISTENT / NO RELIABLE EFFECT"


def main():
    proto = jl("protocol.json")
    ex = proto["experts"]
    seeds = proto["settings"]["seeds"]
    R = {s: jl("results", f"final_seed{s}.json") for s in seeds}
    S = {s: jl("selections", f"seed{s}.json") for s in seeds}
    sk = jl("logs", "sketch.json")
    smoke = jl("results", "smoke_seed0.json") if os.path.exists(os.path.join(OUT, "results", "smoke_seed0.json")) else None
    summ = dict(experts=ex, seeds=seeds, expert_pool_tokens=proto["expert_pool_tokens"],
                expert_ranks=proto["expert_ranks"])

    # ---- 1. mean errors
    mean_err = {}
    for variant in ("matched", "unmatched"):
        for split in ("test", "confirm"):
            tab = {c: {str(s): float(np.mean([err(R[s], variant, c, e, split) for e in ex])) for s in seeds} for c in C4}
            for c in C4:
                tab[c]["mean"] = float(np.mean([tab[c][str(s)] for s in seeds]))
            mean_err[f"{variant}.{split}"] = tab
    summ["mean_error"] = mean_err

    # ---- 2/3/9. paired improvements + per-expert + verdicts
    comps = {}
    for variant in ("matched", "unmatched"):
        for split in ("test", "confirm"):
            for x, y, lab in COMPS:
                pe = {str(e): [rel_imp(R[s], x, y, e, variant, split) for s in seeds] for e in ex}
                ps = [float(np.mean([pe[str(e)][i] for e in ex])) for i in range(len(seeds))]
                pem = [float(np.mean(v)) for v in pe.values()]
                # ratio of means (expert-mean error) as a second summary
                rom = [(mean_err[f"{variant}.{split}"][y][str(s)] - mean_err[f"{variant}.{split}"][x][str(s)]) /
                       mean_err[f"{variant}.{split}"][y][str(s)] for s in seeds]
                comps[f"{variant}.{split}.{x}_vs_{y}"] = dict(
                    label=lab, per_seed=ps, mean=float(np.mean(ps)), sd=float(np.std(ps, ddof=1)),
                    per_expert={k: dict(seeds=v, mean=float(np.mean(v))) for k, v in pe.items()},
                    experts_improved=int(sum(v > 0 for v in pem)), expert_seed_wins=int(sum(
                        v > 0 for vv in pe.values() for v in vv)), ratio_of_means_per_seed=rom,
                    verdict=verdict(ps, pem))
    summ["comparisons"] = comps
    # interaction (relative): (EC vs SC) - (ER vs SR)
    summ["interaction_matched_test"] = [comps["matched.test.EC_vs_SC"]["per_seed"][i] -
                                        comps["matched.test.ER_vs_SR"]["per_seed"][i] for i in range(len(seeds))]

    # ---- 4. mismatch
    mis = {}
    for c in C4:
        mis[c] = {m: float(np.mean([R[s]["mismatch"][f"{c}.{e}"][m] for s in seeds for e in ex]))
                  for m in ("sketch_pre", "sketch_post", "full_pre", "full_post")}
    summ["mismatch_mean"] = mis
    summ["mismatch_per_expert"] = {f"{c}.{e}": {m: float(np.mean([R[s]["mismatch"][f"{c}.{e}"][m] for s in seeds]))
                                                for m in ("sketch_pre", "sketch_post", "full_pre", "full_post")}
                                   for c in C4 for e in ex}
    # surrogate vs reconstruction: within (seed, expert), condition deltas vs SR
    dx = {m: [] for m in ("sketch_post", "full_post")}
    dy = []
    for s in seeds:
        for e in ex:
            for c in ("SC", "EC", "ER"):
                for m in dx:
                    dx[m].append(R[s]["mismatch"][f"{c}.{e}"][m] - R[s]["mismatch"][f"SR.{e}"][m])
                dy.append(-rel_imp(R[s], c, "SR", e))
    summ["spearman_delta_mismatch_vs_delta_error"] = {
        m: dict(rho=float(spearmanr(dx[m], dy).correlation), pvalue=float(spearmanr(dx[m], dy).pvalue), n=len(dy))
        for m in dx}

    # ---- 5/6. tokens and diversity
    summ["quota"] = {str(s): R[s]["quota"] for s in seeds}
    summ["routed_tokens"] = {str(s): R[s]["routed_tokens"] for s in seeds}
    div = {}
    for c in C4:
        div[c] = {k: float(np.mean([R[s]["diversity"][f"{c}.{e}"][v][k2] for s in seeds for e in ex]))
                  for k, (v, k2) in dict(matched_distinct=("matched", "distinct"),
                                         matched_inv_simpson=("matched", "inv_simpson"),
                                         unmatched_distinct=("unmatched", "distinct"),
                                         unmatched_inv_simpson=("unmatched", "inv_simpson")).items()}
        div[c]["discarded_frac"] = float(np.mean([R[s]["diversity"][f"{c}.{e}"]["discarded_frac"]
                                                  for s in seeds for e in ex]))
    summ["diversity_mean"] = div

    # ---- 7. overlap
    summ["overlap"] = {str(s): S[s]["overlap"] for s in seeds}
    summ["union_replayed"] = {str(s): R[s]["union_replayed"] for s in seeds}

    # ---- 8. cost
    summ["cost"] = dict(sketch=sk["cost"], per_seed={str(s): dict(R[s]["cost"], selection_cpu_s=S[s]["selection_cpu_s"])
                                                    for s in seeds})
    # ---- checks
    summ["checks"] = {str(s): R[s]["checks"] for s in seeds}
    summ["sketch_checks"] = {k: v for k, v in sk.items() if k not in ("cost",)}
    if smoke:
        summ["smoke_equals_final_seed0"] = dict(
            errors=all(smoke["results"][k]["test"]["mean"] == R[0]["results"][k]["test"]["mean"]
                       for k in smoke["results"]),
            selection_and_quota=smoke["quota"] == R[0]["quota"])
    summ["all_checks_passed"] = bool(all(R[s]["checks"]["passed"] for s in seeds) and sk["passed"])
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    figures(R, ex, seeds, comps)
    write_md(summ, proto, R, S, ex, seeds)
    print("[report] written; all checks passed:", summ["all_checks_passed"])


# ===================================================================== figures
def figures(R, ex, seeds, comps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": ink})
    # --- Fig 1: per-expert paired improvement (small multiples, one comparison each)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    xs = np.arange(len(ex))
    for ax, (x, y, lab) in zip(axs, COMPS):
        cp = comps[f"matched.test.{x}_vs_{y}"]
        m = [100 * cp["per_expert"][str(e)]["mean"] for e in ex]
        ax.bar(xs, m, width=0.6, color=COLORS[x], zorder=2)
        for i, e in enumerate(ex):
            ax.scatter([i] * len(seeds), [100 * v for v in cp["per_expert"][str(e)]["seeds"]], s=14,
                       facecolor="white", edgecolor=ink, linewidth=0.8, zorder=3)
        ax.axhline(0, color=ink2, linewidth=0.8, zorder=1)
        ax.grid(axis="y", color=grid, linewidth=0.6, zorder=0)
        ax.set_xticks(xs, [str(e) for e in ex])
        ax.set_title(f"{x} vs {y}\n" + "\n".join(__import__("textwrap").wrap(lab, 34)), fontsize=9)
        ax.set_xlabel("expert (low → high pool frequency)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("held-out error reduction vs reference (%)\nbar = 3-seed mean, dots = seeds")
    fig.suptitle("Per-expert paired reconstruction improvement, token-matched, test split (positive = first better)",
                 color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "per_expert_improvement.png"), dpi=150)
    plt.close(fig)
    # --- Fig 2: mismatch vs reconstruction error (relative to SR within expert x seed)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, m, lab in ((axs[0], "sketch_post", "Δ sketch mismatch D_e (selection surrogate, after matching)"),
                       (axs[1], "full_post", "Δ full-dim relative Frobenius mismatch (after matching)")):
        for c in ("SC", "EC", "ER"):
            xv, yv = [], []
            for s in seeds:
                for e in ex:
                    xv.append(R[s]["mismatch"][f"{c}.{e}"][m] - R[s]["mismatch"][f"SR.{e}"][m])
                    yv.append(-100 * rel_imp(R[s], c, "SR", e))
            ax.scatter(xv, yv, s=40, color=COLORS[c], edgecolor="white", linewidth=1.5, label=c, zorder=3)
        ax.axhline(0, color=ink2, linewidth=0.8)
        ax.axvline(0, color=ink2, linewidth=0.8)
        ax.grid(color=grid, linewidth=0.6, zorder=0)
        ax.set_xlabel(lab + " vs SR")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("held-out error change vs SR (%)\n(negative = better than SR)")
    axs[1].legend(title="condition", frameon=False, loc="best")
    fig.suptitle("Covariance mismatch vs reconstruction error (each point: one expert × seed, relative to SR)",
                 color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "mismatch_vs_error.png"), dpi=150)
    plt.close(fig)


# ===================================================================== markdown
def pct(v, d=2):
    return f"{100 * v:+.{d}f} %"


def write_md(summ, proto, R, S, ex, seeds):
    L = []
    w = L.append
    w("# Expert-wise Activation Calibration MVP — summary\n")
    w("Qwen1.5-MoE-A2.7B, layer 16, 8 experts, candidate pool = 2,000 C4 sequences (rows 0-1999 of `~/moe_cache`), "
      "seq len 512, B = 128, seeds 0/1/2. GPTQ 4-bit / g128 / act-order / percdamp 1.0 (`quantize_expert` unchanged). "
      "Primary metric: E3 per-sequence relative reconstruction error (gate+up+down) on held-out test rows 2128-2255, "
      "mean over sequences, then mean over the 8 experts. Secondary: confirmation split (256 seqs). **Isolated expert "
      "reconstruction only — no full-model PPL, no inference-efficiency claim.**\n")
    w(f"All correctness checks passed: **{summ['all_checks_passed']}**"
      + (f"; smoke seed 0 reproduced bit-for-bit by final seed 0: **{summ['smoke_equals_final_seed0']['errors']}**"
         if "smoke_equals_final_seed0" in summ else "") + "\n")
    w("Experts (pool-frequency rank 0 = rarest of 60): " + ", ".join(
        f"{e} (rank {proto['expert_ranks'][str(e)]}, {proto['expert_pool_tokens'][str(e)]:,} pool tokens)" for e in ex)
      + "\n")
    # 1
    w("## 1. Mean reconstruction error (token-matched, primary)\n")
    for split in ("test", "confirm"):
        t = summ["mean_error"][f"matched.{split}"]
        w(f"**{split}** split\n")
        w("| condition | " + " | ".join(f"seed {s}" for s in seeds) + " | 3-seed mean |")
        w("|---|" + "---|" * (len(seeds) + 1))
        for c in C4:
            w(f"| {c} | " + " | ".join(f"{t[c][str(s)]:.6f}" for s in seeds) + f" | **{t[c]['mean']:.6f}** |")
        w("")
    # 2 + 9
    w("## 2. Paired relative improvements by seed (positive = first condition better)\n")
    w("r_e = (err_Y − err_X)/err_Y per expert, averaged over the 8 experts. Pre-registered rule: *consistent* = "
      "positive in all 3 seeds AND ≥ 6/8 experts improved on the seed average; *material* = also ≥ 0.5 %.\n")
    for variant, split in (("matched", "test"), ("matched", "confirm"), ("unmatched", "test")):
        w(f"**{variant} / {split}**" + (" — secondary, token quantity NOT controlled" if variant == "unmatched" else "")
          + "\n")
        w("| comparison | what it isolates | " + " | ".join(f"seed {s}" for s in seeds)
          + " | mean ± sd | experts improved | expert×seed wins | verdict |")
        w("|---|---|" + "---|" * (len(seeds) + 4))
        for x, y, lab in COMPS:
            cp = summ["comparisons"][f"{variant}.{split}.{x}_vs_{y}"]
            w(f"| {x} vs {y} | {lab} | " + " | ".join(pct(v) for v in cp["per_seed"])
              + f" | {pct(cp['mean'])} ± {100 * cp['sd']:.2f} | {cp['experts_improved']}/8 | "
              f"{cp['expert_seed_wins']}/{8 * len(seeds)} | {cp['verdict']} |")
        w("")
    w("Interaction (relaxation effect under the surrogate minus under random selection), matched/test, per seed: "
      + ", ".join(pct(v) for v in summ["interaction_matched_test"]) + "\n")
    # 3
    w("## 3. Per-expert differences (matched, test; 3-seed mean, seeds in brackets)\n")
    w("| expert (rank) | k_e mean | " + " | ".join(f"{x} vs {y}" for x, y, _ in COMPS) + " |")
    w("|---|---|" + "---|" * len(COMPS))
    for e in ex:
        kq = np.mean([summ["quota"][str(s)][str(e)] for s in seeds])
        cells = []
        for x, y, _ in COMPS:
            pe = summ["comparisons"][f"matched.test.{x}_vs_{y}"]["per_expert"][str(e)]
            cells.append(f"{pct(pe['mean'])} [{', '.join(f'{100 * v:+.1f}' for v in pe['seeds'])}]")
        w(f"| {e} ({proto['expert_ranks'][str(e)]}) | {kq:,.0f} | " + " | ".join(cells) + " |")
    w("\n![per-expert improvement](figures/per_expert_improvement.png)\n")
    # 4
    w("## 4. Covariance mismatch before / after token matching (mean over experts × seeds)\n")
    w("Sketch = the 64-d selection surrogate D_e; full = mean over gu/down of the full-dimensional relative Frobenius "
      "distance of the token-normalized Hessian to the full-pool Hessian (diagnostic only, never used for selection). "
      "Pre = all routed tokens of the selected sequences; post = the k_e matched tokens actually given to GPTQ.\n")
    w("| condition | sketch pre | sketch post | full pre | full post |")
    w("|---|---|---|---|---|")
    for c in C4:
        m = summ["mismatch_mean"][c]
        w(f"| {c} | {m['sketch_pre']:.4f} | {m['sketch_post']:.4f} | {m['full_pre']:.4f} | {m['full_post']:.4f} |")
    sp = summ["spearman_delta_mismatch_vs_delta_error"]
    w(f"\nSpearman(Δ mismatch vs SR, Δ error vs SR) over expert × seed × {{SC, EC, ER}} (n = {sp['sketch_post']['n']}): "
      f"sketch post ρ = {sp['sketch_post']['rho']:+.3f} (p = {sp['sketch_post']['pvalue']:.3g}); full post "
      f"ρ = {sp['full_post']['rho']:+.3f} (p = {sp['full_post']['pvalue']:.3g}).\n")
    w("![mismatch vs error](figures/mismatch_vs_error.png)\n")
    # 5
    w("## 5. Routed-token counts and matched quotas\n")
    w("| expert | " + " | ".join(f"seed {s}: SR / SC / EC / ER → k_e" for s in seeds) + " |")
    w("|---|" + "---|" * len(seeds))
    for e in ex:
        w(f"| {e} | " + " | ".join(" / ".join(f"{summ['routed_tokens'][str(s)][c][str(e)]:,}" for c in C4)
                                  + f" → **{summ['quota'][str(s)][str(e)]:,}**" for s in seeds) + " |")
    w("")
    # 6
    w("## 6. Source sequences and effective diversity (mean over experts × seeds)\n")
    w("Effective diversity = inverse Simpson 1/Σ p_s² of the per-sequence token shares.\n")
    w("| condition | distinct seqs (matched) | eff. diversity (matched) | distinct seqs (all tokens) | "
      "eff. diversity (all tokens) | discarded token fraction |")
    w("|---|---|---|---|---|---|")
    for c in C4:
        d = summ["diversity_mean"][c]
        w(f"| {c} | {d['matched_distinct']:.1f} | {d['matched_inv_simpson']:.1f} | {d['unmatched_distinct']:.1f} | "
          f"{d['unmatched_inv_simpson']:.1f} | {100 * d['discarded_frac']:.1f} % |")
    w("")
    # 7
    w("## 7. Shared vs expert-wise selection overlap\n")
    w("| seed | union SR | union SC | union EC | union ER | EC↔SC Jaccard | EC pairwise Jaccard | ER pairwise Jaccard |"
      " sequences replayed (all conditions) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        o = summ["overlap"][str(s)]
        w(f"| {s} | {o['SR']['union']} | {o['SC']['union']} | {o['EC']['union']} | {o['ER']['union']} | "
          f"{o['EC']['mean_jaccard_vs_SC']:.3f} | {o['EC']['mean_pairwise_jaccard']:.3f} | "
          f"{o['ER']['mean_pairwise_jaccard']:.3f} | {summ['union_replayed'][str(s)]} |")
    w("")
    # 8
    c = summ["cost"]["sketch"]
    w("## 8. Cost\n")
    w(f"* Sketch pass (2,000 pool sequences, 8 experts, gu+down sketches + full-dim reference): "
      f"**{c['sketch_replay_s']:.0f} s** vs plain layer replay {c['plain_replay_s']:.0f} s and routing-only Profiler "
      f"{c['routing_only_profiler_s']:.0f} s (the existing Profiler routing pass is slower than both because of its "
      f"per-expert Python bookkeeping) → extra profiling over a plain replay ≈ {c['extra_over_plain_s']:.0f} s "
      f"(+{100 * c['extra_over_plain_s'] / c['plain_replay_s']:.0f} %). Peak GPU {c['peak_gpu_gb']:.2f} GB, peak RSS "
      f"{c['peak_rss_gb']:.1f} GB.")
    w(f"* Sketch storage: {c['sketch_only_mb']:.0f} MB of per-(seq, expert, proj) sketches for 8 experts "
      f"(file incl. full-dim diagnostic references: {c['sketch_file_mb']:.0f} MB, outside the repo). Scales linearly in "
      f"experts: ≈ {c['sketch_only_mb'] / 8 * 60:.0f} MB per layer for all 60.")
    w("")
    w("| seed | selection CPU s | union replayed | capture s | s/seq | held-out capture s | GPTQ+eval s (64 quantizations) | "
      "peak GPU capture GB | peak GPU GPTQ GB |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        q = summ["cost"]["per_seed"][str(s)]
        w(f"| {s} | {q['selection_cpu_s']:.1f} | {summ['union_replayed'][str(s)]} | {q['capture_replay_s']:.0f} | "
          f"{q['capture_s_per_seq']:.3f} | {q['heldout_capture_s']:.0f} | {q['gptq_eval_s']:.0f} | "
          f"{q['peak_gpu_capture_gb']:.2f} | {q['peak_gpu_gptq_gb']:.2f} |")
    w("")
    w("Acquisition-cost note: SR/SC need 128 sequence forwards; EC/ER need the union of their per-expert sets "
      "(table 7) for just 8 experts — with all 60 experts the union approaches the whole pool.\n")
    # checks
    w("## Correctness checks\n")
    ck = summ["checks"][str(seeds[0])]
    w("Per seed: " + ", ".join(f"`{k}`" for k, v in ck.items() if isinstance(v, bool) and k != "passed")
      + " — all True in every seed: **" + str(all(summ["checks"][str(s)]["passed"] for s in seeds)) + "**. "
      f"Alignment checks per seed: {', '.join(str(summ['checks'][str(s)]['n_alignment_checks']) for s in seeds)}. "
      "Sketch pass: " + ", ".join(f"`{k}`={v}" for k, v in summ["sketch_checks"].items()
                                   if isinstance(v, bool)) + ".\n")
    ip = os.path.join(OUT, "interpretation.md")
    if os.path.exists(ip):
        w(open(ip).read())
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
