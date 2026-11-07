"""Minimal allocation experiment (11/06): preflight, per-run validation and the
paired-seed report (CPU only; reads E3 JSONs, never re-runs anything).

    --preflight   check GPU / weights / cache / frozen split / artifact / disk / RAM
    --validate F  one E3 output is complete and frozen-config compliant (exit 1 if not)
    (default)     write <outdir>/summary.{md,json}

Statistics: three experiment seeds are three paired replicates; per-seed
paired differences are reported explicitly. Experts are NOT treated as
independent replicates; per-expert win fractions are descriptive only.
The decision rule (prespecified, before any final result was read):
  PASS     layer_aware < coverage_lp on all 3 seeds, mean relative gain >= 0.5 %,
           and no layer / expert-p95 regression > 0.5 % on the seed mean
  NO-GAIN  |mean relative gain| < 0.5 %, or layer_aware worse on >= 2 seeds
  MIXED    otherwise
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import alloc_models as AM          # noqa: E402
from moe_calib.frozen import load_frozen          # noqa: E402

STRATS = ["random", "coverage", "coverage_lp", "global_exposure", "layer_budget", "layer_aware"]
LEARNED = ["global_exposure", "layer_budget", "layer_aware"]
B, SEEDS, EFFECT = 128, (0, 1, 2), 0.005


def run_path(outdir, s, seed):
    return os.path.join(outdir, f"e3_{s}_B{B}_b4_d1_s{seed}.json")


# ----------------------------------------------------------- preflight

def preflight(a):
    cfg = load_frozen(a.frozen)
    ok, lines = True, []

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"[{'ok' if cond else 'MISSING'}] {name}{': ' + detail if detail else ''}")

    try:
        import torch
        cuda = torch.cuda.is_available()
        free, total = torch.cuda.mem_get_info() if cuda else (0, 0)
        chk("CUDA GPU", cuda and total > 15e9, f"{total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
    except Exception as e:                                      # pragma: no cover
        chk("CUDA GPU", False, repr(e))
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(cfg["model"], allow_patterns=["*.json", "*.safetensors"],
                              local_files_only=True)
        chk("model weights (local HF snapshot)", bool(glob.glob(os.path.join(p, "*.safetensors"))), p)
    except Exception as e:
        chk("model weights (local HF snapshot)", False, repr(e)[:200])
    meta_p = os.path.join(a.cache_dir, "meta.json")
    chk("activation cache meta", os.path.exists(meta_p), meta_p)
    if os.path.exists(meta_p):
        from moe_calib.data_utils import load_meta, split_ids
        meta = load_meta(a.cache_dir)
        chk("cache split == frozen split", split_ids(meta)["definition"] == cfg["split"])
        chk("cache model == frozen model", meta.get("model") == cfg["model"])
        for L in cfg["layers"]:
            chk(f"layer cache L{L}", os.path.exists(os.path.join(a.cache_dir, f"L{L}.pt")))
    rp = os.path.join(a.outdir, "routing_profile_v2.npz")
    lines.append(f"[info] routing_profile_v2.npz in outdir: {os.path.exists(rp)} "
                 "(built by the first run with the model_topk rule if absent)")
    try:
        art = json.load(open(a.alloc_models))
        for s in AM.STRATEGY_MODEL:
            AM.check_artifact(art, cfg, cfg["layers"], s)
        chk("allocation-model artifact", True, f"{a.alloc_models} fp={art['fingerprint']}")
    except Exception as e:
        chk("allocation-model artifact", False, repr(e)[:300])
    du = shutil.disk_usage(a.outdir if os.path.exists(a.outdir) else ".")
    chk("disk (>= 2 GB free)", du.free > 2e9, f"{du.free / 1e9:.0f} GB free")
    mem = float(os.environ.get("MEM_LIMIT", 0)) or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    chk("RAM (>= 16 GB; partial model load)", mem >= 16e9, f"{mem / 1e9:.0f} GB limit")
    print("\n".join(lines))
    print("PREFLIGHT OK" if ok else "RUN_BLOCKED")
    return 0 if ok else 1


# ----------------------------------------------------------- validation

def validate(path, cfg):
    r = json.load(open(path))
    s, ch = r["summary"], r["chosen"]
    P = cfg["split"]["pool"]
    errs = []
    want = dict(B=B, n_selected=B, n_unique_selected=B, bits=cfg["gptq"]["bits"],
                percdamp=float(cfg["gptq"]["percdamp"]), layers=list(cfg["layers"]),
                split=cfg["split"], eval_split="test",
                n_experts_quantized_total=60 * len(cfg["layers"]),
                routing_counts_match_hessian_tokens=True)
    errs += [f"{k}={s.get(k)!r} != {v!r}" for k, v in want.items() if s.get(k) != v]
    if len(set(ch)) != B or not all(P[0] <= x < P[1] for x in ch):
        errs.append("chosen is not B unique pool ids")
    if s["strategy"] in AM.LP_UTILITY_STRATEGIES:
        if len(r["trajectory"]) != 2 or r["chosen"][:32] != r["warm_start"]:
            errs.append("LP trajectory / warm start malformed")
        if s["strategy"] in AM.STRATEGY_MODEL and not s.get("alloc_models"):
            errs.append("no allocation-model fingerprint")
    if errs:
        raise SystemExit(f"INVALID {path}:\n  " + "\n  ".join(errs))
    return r


# --------------------------------------------------------------- report

def l1_counts(a, b, layers, E=60):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {}
    for li, L in enumerate(layers):
        x, y = a[li * E:(li + 1) * E], b[li * E:(li + 1) * E]
        out[str(L)] = float(np.abs(x / x.sum() - y / y.sum()).sum())
    return out


def load_all(outdir, cfg):
    runs, missing = {}, []
    for s in STRATS:
        for seed in SEEDS:
            p = run_path(outdir, s, seed)
            if os.path.exists(p):
                runs[(s, seed)] = validate(p, cfg)
            else:
                missing.append(f"{s} seed {seed}")
    univ = {r["summary"]["eval_universe_file"] for r in runs.values()}
    if len(univ) > 1:
        raise SystemExit(f"runs use different evaluation-universe files: {univ}")
    return runs, missing


def per_expert(r):
    return {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert"]}


def build(outdir, cfg):
    runs, missing = load_all(outdir, cfg)
    layers = list(cfg["layers"])
    seeds = sorted({k[1] for k in runs})
    full = [s for s in STRATS if all((s, sd) in runs for sd in seeds)]
    res = dict(outdir=outdir, B=B, seeds=seeds, missing=missing, strategies={}, paired={},
               overlap={}, routing={}, controller={}, cost={})
    m = lambda s, sd, k: runs[(s, sd)]["summary"][k]
    for s in full:
        v = np.array([m(s, sd, "recon_mean") for sd in seeds])
        e = dict(recon_mean_by_seed=dict(zip(map(str, seeds), v.tolist())), recon_mean=float(v.mean()),
                 recon_sd=float(v.std(ddof=1)) if len(v) > 1 else None)
        for k in ("recon_expert_median", "recon_expert_p90", "recon_expert_p95", "recon_p90",
                  "n_zero_exposure_final"):
            e[k] = float(np.mean([m(s, sd, k) for sd in seeds]))
        e["recon_per_layer"] = {str(L): float(np.mean([m(s, sd, "recon_per_layer")[str(L)] for sd in seeds]))
                                for L in layers}
        e["routing_shift_mean_l1"] = float(np.mean([m(s, sd, "routing_shift")["mean_l1"] for sd in seeds]))
        res["strategies"][s] = e
        c = [runs[(s, sd)]["summary"]["cost"] for sd in seeds]
        res["cost"][s] = {k: float(np.mean([x.get(k, 0.0) or 0.0 for x in c]))
                          for k in ("selection_s", "hessian_replay_s", "gptq_s",
                                    "routing_full_forward_equiv_s", "calib_full_forward_equiv_s",
                                    "equiv_extra_random_seqs")}
        res["cost"][s]["peak_gpu_mem_gb"] = float(np.max([m(s, sd, "peak_gpu_mem_gb") or 0 for sd in seeds]))
    # paired differences (negative = lower error = better than the reference)
    for ref in ("random", "coverage", "coverage_lp", "layer_budget"):
        if ref not in full:
            continue
        for s in full:
            if s == ref:
                continue
            d = {str(sd): m(s, sd, "recon_mean") - m(ref, sd, "recon_mean") for sd in seeds}
            rel = {str(sd): d[str(sd)] / m(ref, sd, "recon_mean") for sd in seeds}
            lay = {str(L): float(np.mean([(m(s, sd, "recon_per_layer")[str(L)] - m(ref, sd, "recon_per_layer")[str(L)])
                                          / m(ref, sd, "recon_per_layer")[str(L)] for sd in seeds])) for L in layers}
            tail = {k: float(np.mean([(m(s, sd, k) - m(ref, sd, k)) / m(ref, sd, k) for sd in seeds]))
                    for k in ("recon_expert_p90", "recon_expert_p95", "recon_expert_median")}
            win = {}
            for sd in seeds:
                a, b = per_expert(runs[(s, sd)]), per_expert(runs[(ref, sd)])
                win[str(sd)] = float(np.mean([a[k] < b[k] for k in b]))
            res["paired"][f"{s}_vs_{ref}"] = dict(
                abs_by_seed=d, rel_by_seed=rel, abs_mean=float(np.mean(list(d.values()))),
                rel_mean=float(np.mean(list(rel.values()))),
                n_seeds_better=int(sum(x < 0 for x in d.values())),
                rel_by_layer=lay, rel_tail=tail, expert_win_fraction_by_seed=win)
    # selection overlap / routing vs coverage_lp
    if "coverage_lp" in full:
        for s in full:
            if s == "coverage_lp":
                continue
            o, ow, l1 = {}, {}, {}
            for sd in seeds:
                a, b = runs[(s, sd)], runs[("coverage_lp", sd)]
                o[str(sd)] = len(set(a["chosen"]) & set(b["chosen"])) / B
                wa = set(b["warm_start"])
                ow[str(sd)] = len((set(a["chosen"]) & set(b["chosen"])) - wa) / (B - len(wa))
                l1[str(sd)] = l1_counts(a["selected_counts"], b["selected_counts"], layers)
            res["overlap"][s] = dict(overlap_with_coverage_lp=o, overlap_excl_warm=ow,
                                     identical_sets=[sd for sd, v in o.items() if v == 1.0])
            res["routing"][s] = dict(count_dist_l1_vs_coverage_lp={
                L: float(np.mean([l1[str(sd)][L] for sd in seeds])) for L in map(str, layers)})
    for s in full:
        if s not in AM.LP_UTILITY_STRATEGIES:
            continue
        rows = []
        for sd in seeds:
            for t in runs[(s, sd)]["trajectory"]:
                rows.append(dict(seed=sd, round=t["round"], m=t["regime_m"], bracket=t.get("regime_bracket"),
                                 lp=t["n_lp_positive_value_selected"], explore=t["n_exploration_sequences"],
                                 fill=t["n_zero_marginal_random_fill"], fallback=t["n_random_fallback_sequences"],
                                 fallback_frac={k: v for k, v in t["status_fractions"]["all"].items() if v},
                                 objective_by_layer={L: v["objective_contribution"] for L, v in t["per_layer"].items()},
                                 delta_hat_mean_by_layer={L: v["delta_hat_mean"] for L, v in t["per_layer"].items()}))
        n_lp = sum(r["lp"] for r in rows)
        n_all = sum(r["lp"] + r["explore"] + r["fill"] + r["fallback"] for r in rows)
        res["controller"][s] = dict(rounds=rows, lp_selected_fraction=n_lp / n_all if n_all else None)
    # mechanism vs random: does more exposure for an expert buy lower error?
    if "random" in full:
        for s in full:
            if s == "random":
                continue
            rows = []
            for sd in seeds:
                a, b = runs[(s, sd)], runs[("random", sd)]
                ta = {(x["layer"], x["expert"]): x["tokens"] for x in a["per_expert"]}
                tb = {(x["layer"], x["expert"]): x["tokens"] for x in b["per_expert"]}
                ea, eb = per_expert(a), per_expert(b)
                k = list(eb)
                lt = np.log([ta[x] / tb[x] for x in k])
                le = np.log([ea[x] / eb[x] for x in k])
                more = lt > 0
                rows.append(dict(frac_more_tokens=float(more.mean()),
                                 frac_worse_given_more_tokens=float((le[more] > 0).mean()) if more.any() else None,
                                 frac_worse_given_fewer_tokens=float((le[~more] > 0).mean()) if (~more).any() else None,
                                 corr_log_token_ratio_vs_log_err_ratio=float(np.corrcoef(lt, le)[0, 1]),
                                 js_vs_pool=m(s, sd, "routing_shift")["mean_js"]))
            res.setdefault("mechanism_vs_random", {})[s] = {
                kk: float(np.mean([r[kk] for r in rows if r[kk] is not None])) for kk in rows[0]}
        rp = os.path.join(outdir, "routing_profile_v2.npz")
        if os.path.exists(rp):
            # per-sequence atypicality: L1 between a sequence's own per-layer routing
            # distribution and the pool's, averaged over layers (CPU, routing cache only)
            z = np.load(rp)
            C, E = z["C"].astype(float), int(z["E"])
            aty = np.zeros(C.shape[0])
            ent = np.zeros(C.shape[0])
            for li in range(len(layers)):
                X = C[:, li * E:(li + 1) * E]
                q = X / X.sum(1, keepdims=True)
                pD = X.sum(0) / X.sum()
                aty += np.abs(q - pD).sum(1) / len(layers)
                ent += -(q * np.log(np.where(q > 0, q, 1))).sum(1) / len(layers)
            row = {int(s): i for i, s in enumerate(z["pool_ids"])}
            res["sequence_atypicality"] = dict(pool_mean_l1=float(aty.mean()), pool_mean_entropy=float(ent.mean()))
            for s in full:
                idx = [[row[x] for x in runs[(s, sd)]["chosen"][32:]] for sd in seeds]   # adaptive picks
                res["sequence_atypicality"][s] = dict(
                    l1_post_warm=float(np.mean([aty[i].mean() for i in idx])),
                    routing_entropy_post_warm=float(np.mean([ent[i].mean() for i in idx])))
        res["routing_js_vs_pool"] = {s: float(np.mean([m(s, sd, "routing_shift")["mean_js"] for sd in seeds]))
                                     for s in full}
    res["decision"] = decide(res)
    res["equal_compute"] = equal_compute(res)
    return res


def decide(res):
    p = res["paired"].get("layer_aware_vs_coverage_lp")
    if p is None or len(res["seeds"]) < 3:
        return dict(label="INCOMPLETE")
    worst_layer = max(p["rel_by_layer"].values())
    tail = p["rel_tail"]["recon_expert_p95"]
    if p["n_seeds_better"] == 3 and -p["rel_mean"] >= EFFECT and worst_layer <= EFFECT and tail <= EFFECT:
        lab = "ALLOCATION-PASS"
    elif abs(p["rel_mean"]) < EFFECT or p["n_seeds_better"] <= 1:
        lab = "ALLOCATION-NO-GAIN"
    else:
        lab = "ALLOCATION-MIXED"
    return dict(label=lab, rel_mean=p["rel_mean"], n_seeds_better=p["n_seeds_better"],
                worst_layer_rel=worst_layer, p95_rel=tail, effect_size=EFFECT)


def equal_compute(res):
    """Rough estimate ONLY (not a run): random error at B + overhead sequences,
    interpolated in log B on the existing random headroom curve, if present."""
    hr = os.path.join(ROOT, "runs_headroom_recon", "headroom_report.json")
    out = {s: dict(extra_random_seqs=c["equiv_extra_random_seqs"]) for s, c in res["cost"].items()}
    if not os.path.exists(hr):
        return out
    try:
        rows = json.load(open(hr))
        pts = sorted((int(r["B"]), float(r["recon_mean"])) for r in rows["rows"] if int(r["B"]) > 0)
    except Exception:
        return out
    xs, ys = np.log([p[0] for p in pts]), [p[1] for p in pts]
    for s, v in out.items():
        b = B + v["extra_random_seqs"]
        v["random_at_equal_compute_est"] = float(np.interp(np.log(b), xs, ys)) if b <= pts[-1][0] else None
        v["equal_compute_B"] = b
    return out


pct = lambda x: f"{100 * x:+.2f}%"


def render(res):
    S, P = res["strategies"], res["paired"]
    M = [f"# Minimal allocation experiment (B = {B}, seeds {res['seeds']})", "",
         f"Decision rule output: **{res['decision']['label']}** (prespecified rule in "
         "`diagnostics/report_minimal_allocation.py`; 0.5 % is a practical effect size, not a test).", ""]
    if res["missing"]:
        M += [f"Missing runs: {', '.join(res['missing'])}", ""]
    M += ["## Reconstruction (expert mean relative error, test split; lower is better)", "",
          "| strategy | mean | sd (seeds) | per seed | vs random | vs coverage | vs coverage_lp | expert p90 | expert p95 | zero-exposure experts |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s, e in S.items():
        cell = lambda ref: pct(P[f"{s}_vs_{ref}"]["rel_mean"]) if f"{s}_vs_{ref}" in P else "–"
        sd = f"{e['recon_sd']:.2e}" if e["recon_sd"] is not None else "–"
        M.append(f"| {s} | {e['recon_mean']:.6f} | {sd} | "
                 + ", ".join(f"{v:.6f}" for v in e["recon_mean_by_seed"].values())
                 + f" | {cell('random')} | {cell('coverage')} | {cell('coverage_lp')} | "
                   f"{e['recon_expert_p90']:.6f} | {e['recon_expert_p95']:.6f} | {e['n_zero_exposure_final']:.1f} |")
    M += ["", "## Paired per-seed differences (strategy − reference; negative = better)", "",
          "| comparison | Δ seed 0 / 1 / 2 (×1e-6) | rel. per seed | mean rel. | seeds better | per-layer rel. | p95 rel. | expert win fraction |",
          "|---|---|---|---|---|---|---|---|"]
    keys = [f"{s}_vs_coverage_lp" for s in LEARNED] + ["coverage_lp_vs_coverage", "layer_aware_vs_layer_budget"] \
        + [f"{s}_vs_random" for s in STRATS[1:]]
    for k in keys:
        if k not in P:
            continue
        p = P[k]
        M.append(f"| {k.replace('_vs_', ' vs ')} | " + " / ".join(f"{1e6 * v:+.2f}" for v in p["abs_by_seed"].values())
                 + " | " + " / ".join(pct(v) for v in p["rel_by_seed"].values())
                 + f" | {pct(p['rel_mean'])} | {p['n_seeds_better']}/{len(p['abs_by_seed'])} | "
                 + ", ".join(f"L{L} {pct(v)}" for L, v in p["rel_by_layer"].items())
                 + f" | {pct(p['rel_tail']['recon_expert_p95'])} | "
                 + " / ".join(f"{v:.2f}" for v in p["expert_win_fraction_by_seed"].values()) + " |")
    M += ["", "## Per-layer reconstruction (seed mean)", "",
          "| strategy | " + " | ".join(f"L{L}" for L in next(iter(S.values()))["recon_per_layer"]) + " |",
          "|---|" + "---|" * len(next(iter(S.values()))["recon_per_layer"])]
    for s, e in S.items():
        M.append(f"| {s} | " + " | ".join(f"{v:.6f}" for v in e["recon_per_layer"].values()) + " |")
    if res["overlap"]:
        M += ["", "## Selection overlap and routing vs coverage_lp", "",
              "| strategy | overlap (of 128) per seed | overlap excl. warm start | identical sets | per-layer L1 of routed-count distribution vs coverage_lp |",
              "|---|---|---|---|---|"]
        for s, o in res["overlap"].items():
            M.append(f"| {s} | " + " / ".join(f"{v:.3f}" for v in o["overlap_with_coverage_lp"].values())
                     + " | " + " / ".join(f"{v:.3f}" for v in o["overlap_excl_warm"].values())
                     + f" | {o['identical_sets'] or 'none'} | "
                     + ", ".join(f"L{L} {v:.3f}" for L, v in res["routing"][s]["count_dist_l1_vs_coverage_lp"].items()) + " |")
    if res["controller"]:
        M += ["", "## Controller diagnostics (per round; seeds pooled)", "",
              "| strategy | seed | round | m | bracket | LP / explore / fill / fallback | non-ok expert statuses | objective by layer | mean δ̂ by layer |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s, c in res["controller"].items():
            for r in c["rounds"]:
                br = r["bracket"]
                M.append(f"| {s} | {r['seed']} | {r['round']} | {r['m']} | "
                         + (f"{br['b_lo']:g}–{br['b_hi']:g} (w={br['w_hi']:.2f})" if br else "–")
                         + f" | {r['lp']} / {r['explore']} / {r['fill']} / {r['fallback']} | "
                         + (", ".join(f"{k} {100 * v:.1f}%" for k, v in r["fallback_frac"].items() if k != "ok") or "none")
                         + " | " + ", ".join(f"L{L} {v:.3f}" for L, v in r["objective_by_layer"].items())
                         + " | " + ", ".join(f"L{L} {v:.4f}" if v is not None else f"L{L} –"
                                             for L, v in r["delta_hat_mean_by_layer"].items()) + " |")
            M.append(f"| {s} | all | | | | LP-selected fraction of adaptive picks: "
                     f"{c['lp_selected_fraction']:.3f} | | | |")
    if res.get("mechanism_vs_random"):
        M += ["", "## Mechanism vs random (per expert, seed mean; descriptive)", "",
              "| strategy | routing JS vs pool | experts with more tokens than random | P(worse \\| more tokens) | P(worse \\| fewer tokens) | corr(log token ratio, log error ratio) |",
              "|---|---|---|---|---|---|"]
        M.append(f"| random | {res['routing_js_vs_pool']['random']:.4f} | – | – | – | – |")
        for s, v in res["mechanism_vs_random"].items():
            M.append(f"| {s} | {v['js_vs_pool']:.4f} | {v['frac_more_tokens']:.2f} | "
                     f"{v['frac_worse_given_more_tokens']:.2f} | {v['frac_worse_given_fewer_tokens']:.2f} | "
                     f"{v['corr_log_token_ratio_vs_log_err_ratio']:+.2f} |")
    if res.get("sequence_atypicality"):
        A = res["sequence_atypicality"]
        M += ["", "## Atypicality of the sequences picked after the warm start (routing cache; seed mean)", "",
              f"Pool mean: per-sequence L1 to pool routing {A['pool_mean_l1']:.3f}, "
              f"routing entropy {A['pool_mean_entropy']:.3f} nats.", "",
              "| strategy | mean L1(sequence routing, pool routing) | mean per-sequence routing entropy |", "|---|---|---|"]
        for s in res["strategies"]:
            if s in A:
                M.append(f"| {s} | {A[s]['l1_post_warm']:.3f} | {A[s]['routing_entropy_post_warm']:.3f} |")
    M += ["", "## Compute (seed mean)", "",
          "| strategy | selection s | Hessian replay s | GPTQ s | routing (full-forward equiv.) s | peak GPU GB | equiv. extra random seqs | est. random error at equal compute |",
          "|---|---|---|---|---|---|---|---|"]
    for s, c in res["cost"].items():
        ec = res["equal_compute"].get(s, {})
        est = ec.get("random_at_equal_compute_est")
        M.append(f"| {s} | {c['selection_s']:.1f} | {c['hessian_replay_s']:.0f} | {c['gptq_s']:.0f} | "
                 f"{c['routing_full_forward_equiv_s']:.0f} | {c['peak_gpu_mem_gb']:.1f} | {c['equiv_extra_random_seqs']:.0f} | "
                 + (f"{est:.6f} (B≈{ec['equal_compute_B']:.0f}, interpolated, NOT run)" if est else "–") + " |")
    M += ["", "Notes: three seeds = three paired replicates; no confidence intervals are claimed. "
          "Expert win fractions are descriptive (experts are not independent replicates). "
          "Equal-compute errors are interpolations of the existing random headroom curve, not runs."]
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_allocation_minimal")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--alloc-models", default="runs/allocation_models_b4_d1.json")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--validate", default=None)
    a = ap.parse_args(argv)
    if a.preflight:
        return preflight(a)
    cfg = load_frozen(a.frozen)
    if a.validate:
        validate(a.validate, cfg)
        print(f"valid {a.validate}")
        return 0
    res = build(a.outdir, cfg)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)
    md = render(res)
    open(os.path.join(a.outdir, "summary.md"), "w").write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
