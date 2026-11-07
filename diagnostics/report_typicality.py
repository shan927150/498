"""Routing-typicality experiment: validation, full-pool reuse check, report (CPU).

    --check-full     the reused full-pool runs (runs_allocation_minimal) match the
                     frozen config, use no subset, and the same routing cache
    --validate F     one filtered-pool E3 output is complete and stays in its pool
    (default)        <outdir>/summary.{md,json}

Groups: pool in {full (F), typical_half (T), random_half (H)} x strategy in
{random (R), coverage_lp (C)} x seeds 0/1/2. Lower error is better.
  P_X = E(X,C) - E(X,R)   coverage penalty in pool X (> 0: coverage worse)
  I   = P_T - P_F         < 0: the penalty shrinks with typical candidates
  P_T - P_H               < 0: ... more than with a random half (pool-size control)
Relative changes are (A - B) / B with B the named reference.

Prespecified reading (fixed before any filtered-pool result):
  TYPICALITY-SUPPORTED  I < 0 and P_T - P_H < 0 on all 3 seeds, and coverage_lp
                        from T selects less atypical sequences than from F
  POOL-SIZE-EFFECT      I < 0 on all 3 seeds but P_T - P_H not < 0 on all 3
  NOT-SUPPORTED         I >= 0 on >= 2 seeds
  MIXED                 otherwise
Seeds are three paired replicates; no significance is claimed. Experts are
not independent replicates.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import report_minimal_allocation as RM               # noqa: E402
from moe_calib import typicality as TY               # noqa: E402
from moe_calib.frozen import load_frozen             # noqa: E402

POOLS = {"F": "full", "T": "typical_half", "H": "random_half"}
STRATS = {"R": "random", "C": "coverage_lp"}
SEEDS, B = (0, 1, 2), 128
LOW_SHARE = 0.25     # low-exposure expert: n_e < 25 % of an even share of its layer's tokens


def run_path(outdir, full_dir, pool, strat, seed):
    if pool == "full":
        return os.path.join(full_dir, f"e3_{strat}_B{B}_b4_d1_s{seed}.json")
    return os.path.join(outdir, f"e3_{strat}_B{B}_b4_d1_s{seed}_pool-{pool}.json")


def pool_file(outdir, pool):
    return os.path.join(outdir, f"pool_{pool}.json")


def validate(path, outdir, cfg):
    r = RM.validate(path, cfg)
    s = r["summary"]
    pool = path.rsplit("_pool-", 1)[1][:-5] if "_pool-" in path else "full"
    if pool == "full":
        if s.get("candidate_subset"):
            raise SystemExit(f"INVALID {path}: full-pool run carries a candidate subset")
        return r
    pf = json.load(open(pool_file(outdir, pool)))
    cs = s.get("candidate_subset") or {}
    if cs.get("fingerprint") != pf["fingerprint"] or cs.get("name") != pool:
        raise SystemExit(f"INVALID {path}: subset fingerprint/name differs from {pool_file(outdir, pool)}")
    if not set(r["chosen"]) <= set(pf["ids"]):
        raise SystemExit(f"INVALID {path}: selection outside pool {pool}")
    return r


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_full(full_dir, outdir, cfg):
    for st in STRATS.values():
        for sd in SEEDS:
            r = validate(run_path(outdir, full_dir, "full", st, sd), outdir, cfg)
            if r["summary"].get("frozen_config") != "configs/frozen_b4.json":
                raise SystemExit("reused full-pool run was not made with configs/frozen_b4.json")
    a, b = os.path.join(full_dir, "routing_profile_v2.npz"), os.path.join(outdir, "routing_profile_v2.npz")
    if sha(a) != sha(b):
        raise SystemExit("routing cache of the reused full-pool runs differs from the typicality one")
    print(f"full-pool runs in {full_dir} verified for reuse (6 runs, same frozen config and routing cache)")


# --------------------------------------------------------------- report

def gini(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def set_metrics(r, C, row, ref, scores, layers, E):
    ch = r["chosen"]
    idx = [row[s] for s in ch]
    post = [row[s] for s in ch[32:]]
    n = np.asarray(r["selected_counts"], float)
    out = dict(d_mean=float(scores["d"][idx].mean()), d_post_warm=float(scores["d"][post].mean()),
               entropy_mean=float(scores["entropy"][idx].mean()),
               js_set_vs_original_pool=TY.selected_set_js(C[idx], ref, len(layers), E))
    med, gin, low, src, eff, effm, l1 = [], [], [], [], [], [], []
    for li in range(len(layers)):
        x = n[li * E:(li + 1) * E]
        med.append(np.median(x))
        gin.append(gini(x))
        low.append(int((x < LOW_SHARE * x.sum() / E).sum()))
        X = C[idx][:, li * E:(li + 1) * E]
        src.append(np.median((X > 0).sum(0)))
        q = X / np.maximum(X.sum(0, keepdims=True), 1)          # sequence shares per expert
        effv = np.exp(-(np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0)).sum(0))
        eff.append(np.median(effv))
        effm.append(effv.mean())
        l1.append(float(np.abs(X.sum(0) / X.sum() - ref[li]).sum()))
    out.update(median_expert_tokens=float(np.mean(med)), gini_expert_tokens=float(np.mean(gin)),
               n_low_exposure_experts=int(sum(low)), median_sources_per_expert=float(np.mean(src)),
               median_effective_sources_per_expert=float(np.mean(eff)),
               mean_effective_sources_per_expert=float(np.mean(effm)),
               set_routing_l1_by_layer={str(L): v for L, v in zip(layers, l1)})
    return out


def build(outdir, full_dir, cfg):
    z = np.load(os.path.join(outdir, "routing_profile_v2.npz"))
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"].astype(float)
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    ref = TY.pool_reference(C, len(layers), E)                 # ORIGINAL pool, every condition
    scores = TY.sequence_scores(C, len(layers), E, ref)
    runs, missing = {}, []
    for pk, pool in POOLS.items():
        for sk, st in STRATS.items():
            for sd in SEEDS:
                p = run_path(outdir, full_dir, pool, st, sd)
                if os.path.exists(p):
                    runs[(pk, sk, sd)] = validate(p, outdir, cfg)
                else:
                    missing.append(f"{pool}/{st}/s{sd}")
    univ = {json.dumps(r["eval_universe"]) for r in runs.values()}
    if len(univ) != 1:
        raise SystemExit("runs do not share one evaluation universe")
    seeds = sorted({k[2] for k in runs})
    groups = [(p, s) for p in POOLS for s in STRATS if all((p, s, sd) in runs for sd in seeds)]
    res = dict(seeds=seeds, missing=missing, low_exposure_rule=f"n_e < {LOW_SHARE} x layer tokens / {E}",
               groups={}, primary={}, relative={}, overlap={}, mechanism={})
    E_ = lambda p, s, sd: runs[(p, s, sd)]["summary"]["recon_mean"]
    for p, s in groups:
        rs = [runs[(p, s, sd)] for sd in seeds]
        v = np.array([r["summary"]["recon_mean"] for r in rs])
        sm = lambda k: float(np.mean([r["summary"][k] for r in rs]))
        ms = [set_metrics(r, C, row, ref, scores, layers, E) for r in rs]
        g = dict(pool=POOLS[p], strategy=STRATS[s], recon_mean=float(v.mean()),
                 recon_sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                 recon_by_seed=dict(zip(map(str, seeds), v.tolist())),
                 expert_median=sm("recon_expert_median"), expert_p90=sm("recon_expert_p90"),
                 expert_p95=sm("recon_expert_p95"),
                 per_layer={str(L): float(np.mean([r["summary"]["recon_per_layer"][str(L)] for r in rs]))
                            for L in layers},
                 selected={k: (float(np.mean([m[k] for m in ms])) if not isinstance(ms[0][k], dict) else
                               {L: float(np.mean([m[k][L] for m in ms])) for L in ms[0][k]}) for k in ms[0]},
                 selected_by_seed={str(sd): m for sd, m in zip(seeds, ms)},
                 cost={k: float(np.mean([r["summary"]["cost"].get(k, 0.0) or 0.0 for r in rs]))
                       for k in ("selection_s", "hessian_replay_s", "gptq_s", "routing_full_forward_equiv_s")},
                 peak_gpu_mem_gb=float(max(r["summary"]["peak_gpu_mem_gb"] or 0 for r in rs)))
        res["groups"][f"{p}{s}"] = g
    have = lambda *gs: all(g in res["groups"] for g in gs)
    if have("FR", "FC", "TR", "TC", "HR", "HC"):
        pen = {p: {str(sd): E_(p, "C", sd) - E_(p, "R", sd) for sd in seeds} for p in POOLS}
        rel_pen = {p: {str(sd): pen[p][str(sd)] / E_(p, "R", sd) for sd in seeds} for p in POOLS}
        I = {sd: pen["T"][sd] - pen["F"][sd] for sd in pen["F"]}
        TH = {sd: pen["T"][sd] - pen["H"][sd] for sd in pen["F"]}
        HF = {sd: pen["H"][sd] - pen["F"][sd] for sd in pen["F"]}
        mean = lambda d: float(np.mean(list(d.values())))
        res["primary"] = dict(P=pen, P_rel=rel_pen, P_mean={p: mean(v) for p, v in pen.items()},
                              P_rel_mean={p: mean(v) for p, v in rel_pen.items()},
                              I_T_minus_F=I, I_mean=mean(I), T_minus_H=TH, T_minus_H_mean=mean(TH),
                              H_minus_F=HF, H_minus_F_mean=mean(HF))
        rel = lambda a, b: {str(sd): (E_(*a, sd) - E_(*b, sd)) / E_(*b, sd) for sd in seeds}
        for name, a, b in (("TR_vs_FR", ("T", "R"), ("F", "R")), ("TC_vs_FC", ("T", "C"), ("F", "C")),
                           ("TC_vs_TR", ("T", "C"), ("T", "R")), ("TC_vs_HC", ("T", "C"), ("H", "C")),
                           ("HC_vs_FC", ("H", "C"), ("F", "C")), ("HR_vs_FR", ("H", "R"), ("F", "R")),
                           ("TR_vs_HR", ("T", "R"), ("H", "R")), ("TC_vs_FR", ("T", "C"), ("F", "R"))):
            d = rel(a, b)
            res["relative"][name] = dict(by_seed=d, mean=mean(d), n_seeds_negative=sum(v < 0 for v in d.values()))
        for a, b in (("TC", "FC"), ("HC", "FC"), ("TC", "TR"), ("TC", "HC"), ("TR", "FR")):
            res["overlap"][f"{a}_vs_{b}"] = {
                str(sd): len(set(runs[(a[0], a[1], sd)]["chosen"]) & set(runs[(b[0], b[1], sd)]["chosen"])) / B
                for sd in seeds}
        G = res["groups"]
        bs = lambda g, k: [G[g]["selected_by_seed"][str(sd)][k] for sd in seeds]
        chk = {
            "1_typical_filter_lowers_selected_atypicality (TR<FR, TC<FC, d of 128 chosen)":
                [a < b for a, b in zip(bs("TR", "d_mean"), bs("FR", "d_mean"))]
                + [a < b for a, b in zip(bs("TC", "d_mean"), bs("FC", "d_mean"))],
            "2_coverage_lp_from_T_picks_more_typical_sequences (post-warm d, TC<FC)":
                [a < b for a, b in zip(bs("TC", "d_post_warm"), bs("FC", "d_post_warm"))],
            "3_set_JS_to_original_pool_decreases (TC<FC)":
                [a < b for a, b in zip(bs("TC", "js_set_vs_original_pool"), bs("FC", "js_set_vs_original_pool"))],
            "4_reconstruction_improves (TC<FC)": [E_("T", "C", sd) < E_("F", "C", sd) for sd in seeds],
            "5_coverage_penalty_shrinks (I<0)": [v < 0 for v in I.values()],
            "5b_more_than_random_half (P_T-P_H<0)": [v < 0 for v in TH.values()],
        }
        res["mechanism"] = {k: dict(by_seed=v, all=all(v)) for k, v in chk.items()}
        res["mechanism"]["6_consistent_across_seeds"] = all(x["all"] for x in res["mechanism"].values())
        nI, nTH = sum(v < 0 for v in I.values()), sum(v < 0 for v in TH.values())
        if nI == len(seeds) and nTH == len(seeds) and res["mechanism"][
                "2_coverage_lp_from_T_picks_more_typical_sequences (post-warm d, TC<FC)"]["all"]:
            lab = "TYPICALITY-SUPPORTED"
        elif nI == len(seeds):
            lab = "POOL-SIZE-EFFECT"
        elif len(seeds) - nI >= 2:
            lab = "NOT-SUPPORTED"
        else:
            lab = "MIXED"
        res["decision"] = dict(label=lab, n_seeds_I_negative=nI, n_seeds_T_minus_H_negative=nTH,
                               coverage_beats_random_in_T=all(v < 0 for v in pen["T"].values()))
    else:
        res["decision"] = dict(label="INCOMPLETE")
    return res


pct = lambda x: f"{100 * x:+.2f}%"
e6 = lambda x: f"{1e6 * x:+.1f}"


def render(res):
    G, M = res["groups"], []
    M += [f"# Routing-typicality experiment (B = {B}, seeds {res['seeds']})", "",
          f"Prespecified reading: **{res['decision']['label']}**.", ""]
    if res["missing"]:
        M += ["Missing: " + ", ".join(res["missing"]), ""]
    M += ["## Reconstruction (mean expert relative error, test split; lower is better)", "",
          "| pool | strategy | mean | sd | seed 0 / 1 / 2 | expert median | p90 | p95 |", "|---|---|---|---|---|---|---|---|"]
    for g in G.values():
        M.append(f"| {g['pool']} | {g['strategy']} | {g['recon_mean']:.6f} | "
                 + (f"{g['recon_sd']:.1e}" if g["recon_sd"] is not None else "–") + " | "
                 + " / ".join(f"{v:.6f}" for v in g["recon_by_seed"].values())
                 + f" | {g['expert_median']:.6f} | {g['expert_p90']:.6f} | {g['expert_p95']:.6f} |")
    if res["primary"]:
        P = res["primary"]
        M += ["", "## Coverage penalty and interaction (absolute error ×1e-6; negative = coverage better / penalty smaller)", "",
              "| quantity | seed 0 | seed 1 | seed 2 | mean |", "|---|---|---|---|---|"]
        for lab, d in (("P_F = E(F,C) − E(F,R)", P["P"]["F"]), ("P_T = E(T,C) − E(T,R)", P["P"]["T"]),
                       ("P_H = E(H,C) − E(H,R)", P["P"]["H"]), ("**I = P_T − P_F**", P["I_T_minus_F"]),
                       ("**P_T − P_H**", P["T_minus_H"]), ("P_H − P_F", P["H_minus_F"])):
            M.append(f"| {lab} | " + " | ".join(e6(v) for v in d.values()) + f" | {e6(np.mean(list(d.values())))} |")
        M.append("| relative penalty P_X / E(X,R): F / T / H (mean) | | | | "
                 + " / ".join(pct(P["P_rel_mean"][p]) for p in "FTH") + " |")
        M += ["", "## Relative error changes ((A − B) / B; negative = A better)", "",
              "| A vs B | seed 0 / 1 / 2 | mean | seeds A better |", "|---|---|---|---|"]
        names = {"F": "Full", "T": "Typical", "H": "RandHalf", "R": "Random", "C": "Cov-LP"}
        for k, v in res["relative"].items():
            a, b = k.split("_vs_")
            M.append(f"| {names[a[0]]} {names[a[1]]} vs {names[b[0]]} {names[b[1]]} | "
                     + " / ".join(pct(x) for x in v["by_seed"].values()) + f" | {pct(v['mean'])} | {v['n_seeds_negative']}/3 |")
    M += ["", "## Per-layer error (seed mean)", "", "| pool | strategy | " + " | ".join(
        f"L{L}" for L in next(iter(G.values()))["per_layer"]) + " |", "|---|---|" + "---|" * 4]
    for g in G.values():
        M.append(f"| {g['pool']} | {g['strategy']} | " + " | ".join(f"{v:.6f}" for v in g["per_layer"].values()) + " |")
    M += ["", "## Selected-set routing and exposure (seed mean; typicality always vs the ORIGINAL pool)", "",
          "| pool | strategy | d(s) of 128 chosen | d(s) after warm start | entropy | set JS vs pool | median expert tokens | Gini of expert tokens | low-exposure experts | median sources / expert | median effective sources / expert |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in G.values():
        s = g["selected"]
        M.append(f"| {g['pool']} | {g['strategy']} | {s['d_mean']:.4f} | {s['d_post_warm']:.4f} | {s['entropy_mean']:.4f} | "
                 f"{s['js_set_vs_original_pool']:.5f} | {s['median_expert_tokens']:.0f} | {s['gini_expert_tokens']:.3f} | "
                 f"{s['n_low_exposure_experts']:.1f} | {s['median_sources_per_expert']:.1f} | "
                 f"{s['median_effective_sources_per_expert']:.1f} |")
    M.append(f"\nLow exposure: {res['low_exposure_rule']}, summed over the 4 layers. Sources = selected sequences "
             "routing >= 1 token to the expert; effective sources = exp(entropy of the per-sequence shares of the "
             "expert's routed tokens).")
    if res["overlap"]:
        M += ["", "## Selected-set overlap (fraction of 128)", "", "| sets | seed 0 / 1 / 2 |", "|---|---|"]
        for k, v in res["overlap"].items():
            M.append(f"| {k.replace('_vs_', ' vs ')} | " + " / ".join(f"{x:.3f}" for x in v.values()) + " |")
    if res["mechanism"]:
        M += ["", "## Mechanism checks (per seed)", "", "| check | per seed | all |", "|---|---|---|"]
        for k, v in res["mechanism"].items():
            if isinstance(v, dict):
                M.append(f"| {k} | {' / '.join('yes' if x else 'no' for x in v['by_seed'])} | {'yes' if v['all'] else 'NO'} |")
        M.append(f"| 6_consistent_across_seeds | | {'yes' if res['mechanism']['6_consistent_across_seeds'] else 'NO'} |")
    M += ["", "## Cost (seed mean)", "", "| pool | strategy | selection s | Hessian s | GPTQ s | routing profile (full-forward equiv.) s | peak GPU GB |",
          "|---|---|---|---|---|---|---|"]
    for g in G.values():
        c = g["cost"]
        M.append(f"| {g['pool']} | {g['strategy']} | {c['selection_s']:.1f} | {c['hessian_replay_s']:.0f} | {c['gptq_s']:.0f} | "
                 f"{c['routing_full_forward_equiv_s']:.0f} | {g['peak_gpu_mem_gb']:.1f} |")
    M += ["", "Notes: full-pool runs are reused from `runs_allocation_minimal/` after validation. Random from a "
          "filtered pool also needs the full-pool routing profile (to define the pool) — not charged in its E3 cost "
          "field. Seed timings for parallel runs include contention. Equal-budget comparison only; no equal-compute claim."]
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_typicality")
    ap.add_argument("--full-dir", default="runs_allocation_minimal")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--validate", default=None)
    ap.add_argument("--check-full", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_frozen(a.frozen)
    if a.validate:
        validate(a.validate, a.outdir, cfg)
        print(f"valid {a.validate}")
        return 0
    if a.check_full:
        check_full(a.full_dir, a.outdir, cfg)
        return 0
    res = build(a.outdir, a.full_dir, cfg)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)
    md = render(res)
    open(os.path.join(a.outdir, "summary.md"), "w").write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
