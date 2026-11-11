"""Report for the official EBSS calibration-quality validation (CPU only).

Builds routing_diagnostics / reconstruction_results / ppl_results / comparisons / efficiency_results /
input_fingerprints / decision / summary (.json, .md) and six figures from the frozen artifacts and the
per-configuration results. Nothing here selects data or changes a threshold."""
from __future__ import annotations

import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import official_ebss_validation as DRV   # noqa: E402
import full_depth_pilot as FD            # noqa: E402
from moe_calib import ebss_official_data as ED   # noqa: E402
from moe_calib import typicality as TY            # noqa: E402

p = DRV.p
jdump = FD.jdump
LAYERS, E, K = DRV.LAYERS, 60, 4
C4 = [f"{c}_s{s}" for c in DRV.STRATS for s in DRV.SEEDS]
CONFIGS = DRV.CONFIGS
COL = {"FR": "#4C78A8", "TR": "#59A14F", "FC": "#E15759", "EBSS": "#B07AA1"}


def strat(c):
    return c.split("_")[0]


def load_results():
    return {c: json.load(open(p("results", f"{c}.json"))) for c in CONFIGS}


# ---------------------------------------------------------------- routing / calibration diagnostics
def routing_diagnostics(res):
    pol = json.load(open(p("execution_policy.json")))
    z = np.load(p("logs", "capture_routing_nll.npz"))
    C, nll, sec = z["C"].astype(float), z["nll"], z["sec"]
    prof = np.load(os.path.join(ROOT, "runs_low_budget", "routing_profile_v2.npz"))
    assert list(prof["layers"]) == LAYERS and int(prof["E"]) == E
    ref = TY.pool_reference(prof["C"], len(LAYERS), E)                      # [4, 60] C4-pool reference
    pool_scores = TY.sequence_scores(prof["C"], len(LAYERS), E, ref=ref)
    typ_ids = json.load(open(os.path.join(ROOT, "runs_typicality", "pool_typical_half.json")))["ids"]
    rebuilt = TY.typical_half(prof["pool_ids"], pool_scores["d"], 1000)
    d_star = float(np.nanmax(pool_scores["d"][np.array(typ_ids)]))
    sc = TY.sequence_scores(C, len(LAYERS), E, ref=ref)
    nc = pol["rows"]["n_c4"]
    c4_rows = pol["rows"]["c4_rows"]
    # extraction-policy drift for C4 rows: batch-1 routing vs the historical (batch-8) profile
    Ch = prof["C"][np.array(c4_rows)]
    drift_l1 = np.abs(C[:nc] / C[:nc].sum(1, keepdims=True) - Ch / Ch.sum(1, keepdims=True)).sum(1) / len(LAYERS)
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
    toks = torch.load(os.path.join(DRV.ECACHE, "ids.pt"))["tokens"]
    out = dict(definition=dict(
        exposure="routed calibration tokens per expert (top-4 slots of 128 x 512 tokens) at layers 1/8/16/23 from the "
                 "CANONICAL-B1 layer inputs (= E3 Profiler counts, cross-checked below)",
        uniform_mean=128 * 512 * K / E,
        routing_l1="|| aggregate routing distribution of the set - C4 pool reference ||_1 per layer",
        typicality=f"d(s) vs the C4 pool reference; atypical = d(s) > d* = {d_star:.6f} (largest d in the frozen "
                   "typical half)", typical_half_rebuilt_exactly=sorted(rebuilt) == sorted(typ_ids),
        model_nll="BF16 model mean NLL per target over each 512-token window (511 targets), CANONICAL-B1 forward",
        source_diversity="C4: one window per distinct C4 document (pool construction); EBSS: one window per "
                         "distinct official record"),
        policy_drift_c4=dict(rows=nc, mean_l1_per_layer_vs_historical_profile=float(drift_l1.mean()),
                             max=float(drift_l1.max()), frac_rows_identical=float((drift_l1 == 0).mean()),
                             note="C4 rows' batch-1 routing vs their historical batch-of-8 routing profile"),
        configs={})
    for c in CONFIGS:
        rows = DRV.config_rows(c)[0]
        Cc = C[rows]
        agg = Cc.sum(0)
        per_layer = {}
        for li, L in enumerate(LAYERS):
            cnt = agg[li * E:(li + 1) * E]
            per_layer[str(L)] = dict(ED.exposure_stats(cnt, uniform_mean=128 * 512 * K / E),
                                     routing_l1=ED.routing_l1(cnt, ref[li]), histogram=cnt.astype(int).tolist())
        # cross-check against E3 Profiler token counts of the quantization run
        prof_tokens = {(e["layer"], e["expert"]): e["tokens"] for e in res[c]["per_expert"]}
        ours = {(L, e): int(agg[li * E + e]) for li, L in enumerate(LAYERS) for e in range(E)}
        mismatch = sum(1 for k in prof_tokens if prof_tokens[k] != ours[k])
        W = [toks[r].tolist() for r in rows]
        d = sc["d"][rows]
        cjk = sum(any("一" <= ch <= "鿿" for ch in tok.decode(w)) for w in W)
        out["configs"][c] = dict(
            per_layer=per_layer,
            mean_over_layers={k: float(np.mean([per_layer[str(L)][k] for L in LAYERS]))
                              for k in ("min", "p5", "p10", "max", "sd", "cv", "gini", "routing_l1", "n_zero",
                                        "n_below_128", "n_below_0.25x_uniform")},
            profiler_token_count_mismatches=mismatch,
            typicality=dict(mean_d=float(np.mean(d)), median_d=float(np.median(d)),
                            frac_atypical=float(np.mean(d > d_star)), n_atypical=int(np.sum(d > d_star))),
            sequence_routing_entropy_mean=float(np.mean(sc["entropy"][rows])),
            model_nll=dict(mean=float(np.mean(nll[rows])), ppl=float(np.exp(np.mean(nll[rows]))),
                           median=float(np.median(nll[rows])), p10=float(np.quantile(nll[rows], .1)),
                           p90=float(np.quantile(nll[rows], .9))),
            text=ED.text_diagnostics(W), near_duplicates=ED.near_duplicate_stats(W),
            source=dict(n_parent_units=len(rows), unit="C4 document" if c != "EBSS" else "official EBSS record",
                        windows_with_cjk=cjk),
            per_row_extraction_s=float(sec[rows].sum()))
    return out, nll, sec, sc["d"], d_star


# ---------------------------------------------------------------- reconstruction
def per_expert_map(r):
    return {(e["layer"], e["expert"]): e["mean"] for e in r["per_expert"]}


def recon_tables(res):
    prof = np.load(os.path.join(ROOT, "runs_low_budget", "routing_profile_v2.npz"))
    pnat = prof["p"]                                                            # [240] natural frequency
    keys = [(L, e) for L in LAYERS for e in range(E)]
    pk = np.array([pnat[LAYERS.index(L) * E + e] for L, e in keys])
    dec = np.minimum((np.argsort(np.argsort(pk, kind="stable"), kind="stable") * 10) // len(keys), 9)
    out = {}
    for c, r in res.items():
        m = per_expert_map(r)
        v = np.array([m[k] for k in keys])
        out[c] = dict(recon=r["recon"], mean_check=float(v.mean()),
                      per_layer={str(L): float(np.mean([m[(L, e)] for e in range(E)])) for L in LAYERS},
                      median=float(np.median(v)), p90=float(np.quantile(v, .9)), worst=float(v.max()),
                      worst_expert=list(keys[int(v.argmax())]),
                      by_natural_exposure_decile=[float(v[dec == d].mean()) for d in range(10)],
                      rtn_fallbacks=r["rtn_fallbacks"], valid=r["valid"])
    for s in ("FR", "TR", "FC"):
        vals = [out[f"{s}_s{i}"]["recon"] for i in DRV.SEEDS]
        out[f"{s}_mean_over_seeds"] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)))
    return out, keys, dec


def comparisons(res, keys, dec, ppl_by):
    E_ = per_expert_map(res["EBSS"])
    e = np.array([E_[k] for k in keys])
    out = {}
    for c in C4:
        B_ = per_expert_map(res[c])
        b = np.array([B_[k] for k in keys])
        rel = (e / b - 1) * 100
        out[f"EBSS_vs_{c}"] = dict(
            baseline=c, baseline_recon=res[c]["recon"], ebss_recon=res["EBSS"]["recon"],
            relative_penalty_pct=ED.relative_penalty(res["EBSS"]["recon"], res[c]["recon"]),
            per_layer_relative_penalty_pct={str(L): ED.relative_penalty(
                np.mean([E_[(L, x)] for x in range(E)]), np.mean([B_[(L, x)] for x in range(E)])) for L in LAYERS},
            per_expert_relative_diff_pct=dict(median=float(np.median(rel)), p10=float(np.quantile(rel, .1)),
                                              p90=float(np.quantile(rel, .9)), min=float(rel.min()),
                                              max=float(rel.max())),
            frac_experts_improved=float(np.mean(e < b)),
            by_natural_exposure_decile_relative_pct=[ED.relative_penalty(e[dec == d].mean(), b[dec == d].mean())
                                                     for d in range(10)],
            ppl_ebss=ppl_by["EBSS"], ppl_baseline=ppl_by[c], delta_ppl=ppl_by["EBSS"] - ppl_by[c])
    summ = {}
    for s in ("FR", "TR", "FC"):
        rp = [out[f"EBSS_vs_{s}_s{i}"]["relative_penalty_pct"] for i in DRV.SEEDS]
        dp = [out[f"EBSS_vs_{s}_s{i}"]["delta_ppl"] for i in DRV.SEEDS]
        summ[s] = dict(mean_baseline_recon=float(np.mean([res[f"{s}_s{i}"]["recon"] for i in DRV.SEEDS])),
                       relative_penalty_pct=rp, mean_relative_penalty_pct=float(np.mean(rp)),
                       best_pct=float(min(rp)), worst_pct=float(max(rp)),
                       ebss_better_recon_all_seeds=bool(all(x < 0 for x in rp)),
                       delta_ppl=dp, ebss_lower_ppl_all_seeds=bool(all(x < 0 for x in dp)),
                       frac_experts_improved=[out[f"EBSS_vs_{s}_s{i}"]["frac_experts_improved"] for i in DRV.SEEDS])
    return out, summ


def bootstrap(a, b, tok, n=2000, seed=20261109):
    return FD.paired_bootstrap(a, b, tok, n_boot=n, seed=seed)


# ---------------------------------------------------------------- figures
def figures(diag, rec, comp, ppl, eff, bf16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(p("figures", f"{name}.{ext}"), dpi=160)
        plt.close(fig)

    # 1 exposure distribution
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, L in zip(axs, LAYERS):
        data, labels, cols = [], [], []
        for c in ["FR_s0", "TR_s0", "FC_s0", "EBSS"]:
            data.append(np.sort(diag["configs"][c]["per_layer"][str(L)]["histogram"])[::-1])
            labels.append(c); cols.append(COL[strat(c)])
        for d_, lab, col in zip(data, labels, cols):
            ax.plot(np.arange(1, E + 1), np.maximum(d_, 0.5), color=col, label=lab, lw=1.4)
        ax.set_yscale("log"); ax.set_title(f"layer {L}"); ax.set_xlabel("expert rank")
        ax.axhline(128 * 512 * K / E, color="0.6", ls=":", lw=1)
    axs[0].set_ylabel("routed calibration tokens (log)")
    axs[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 1. Expert exposure (sorted per layer; dotted = uniform mean; seed-0 C4 sets shown)")
    save(fig, "fig1_exposure")

    # 2 representativeness
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    mets = [("routing L1 to C4 pool (mean over layers)", lambda c: diag["configs"][c]["mean_over_layers"]["routing_l1"]),
            ("fraction atypical (d > d*)", lambda c: diag["configs"][c]["typicality"]["frac_atypical"]),
            ("BF16 model PPL on calibration windows (log)", lambda c: diag["configs"][c]["model_nll"]["ppl"]),
            ("distinct-2 (mean per window)", lambda c: diag["configs"][c]["text"]["distinct2_mean"])]
    for ax, (t, f) in zip(axs, mets):
        xs = np.arange(len(CONFIGS))
        ax.bar(xs, [f(c) for c in CONFIGS], color=[COL[strat(c)] for c in CONFIGS])
        ax.set_xticks(xs); ax.set_xticklabels(CONFIGS, rotation=60, fontsize=7); ax.set_title(t, fontsize=8)
        if "log" in t:
            ax.set_yscale("log")
    fig.suptitle("Fig 2. Calibration representativeness")
    save(fig, "fig2_representativeness")

    # 3 reconstruction
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    xs = np.arange(len(CONFIGS))
    axs[0].bar(xs, [rec[c]["recon"] for c in CONFIGS], color=[COL[strat(c)] for c in CONFIGS])
    lo = min(rec[c]["recon"] for c in CONFIGS)
    axs[0].set_ylim(lo * 0.95, max(rec[c]["recon"] for c in CONFIGS) * 1.02)
    axs[0].set_xticks(xs); axs[0].set_xticklabels(CONFIGS, rotation=60, fontsize=7)
    axs[0].set_title("mean held-out expert reconstruction error (lower = better)")
    for i, L in enumerate(LAYERS):
        for j, s in enumerate(("FR", "TR", "FC")):
            v = [comp[f"EBSS_vs_{s}_s{k}"]["per_layer_relative_penalty_pct"][str(L)] for k in DRV.SEEDS]
            x = i * 4 + j
            axs[1].scatter([x] * 3, v, color=COL[s], s=14, label=s if i == 0 else None)
    axs[1].axhline(0, color="0.4", lw=1)
    axs[1].set_xticks([i * 4 + 1 for i in range(4)]); axs[1].set_xticklabels([f"L{L}" for L in LAYERS])
    axs[1].set_ylabel("EBSS relative penalty, % (<0 = EBSS better)"); axs[1].legend(frameon=False)
    axs[1].set_title("per-layer EBSS vs baseline seeds")
    fig.suptitle("Fig 3. Held-out reconstruction")
    save(fig, "fig3_reconstruction")

    # 4 per-expert distribution EBSS vs FR/TR
    res = load_results()
    keys = [(L, e) for L in LAYERS for e in range(E)]
    Em = per_expert_map(res["EBSS"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    pos, data, cols, labs = [], [], [], []
    for j, c in enumerate([f"{s}_s{i}" for s in ("FR", "TR") for i in DRV.SEEDS]):
        Bm = per_expert_map(res[c])
        data.append([(Em[k] / Bm[k] - 1) * 100 for k in keys]); pos.append(j); cols.append(COL[strat(c)]); labs.append(c)
    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=True, flierprops=dict(ms=2))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(0, color="0.3", lw=1); ax.set_xticks(pos); ax.set_xticklabels(labs)
    ax.set_ylabel("per-expert (EBSS / baseline - 1) %")
    ax.set_title("Fig 4. EBSS vs Random by expert (240 experts; < 0 = EBSS better)")
    save(fig, "fig4_per_expert")

    # 5 PPL
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(xs, [ppl[c]["ppl"] for c in CONFIGS], color=[COL[strat(c)] for c in CONFIGS])
    ax.axhline(bf16, color="0.3", ls="--", lw=1, label=f"BF16 {bf16:.4f}")
    v = [ppl[c]["ppl"] for c in CONFIGS]
    ax.set_ylim(bf16 - 0.005, max(v) + 0.01)
    ax.set_xticks(xs); ax.set_xticklabels(CONFIGS, rotation=60, fontsize=7); ax.legend(frameon=False)
    ax.set_ylabel("WikiText-2 PPL (partial model)")
    ax.set_title("Fig 5. Model-level PPL, matched CANONICAL-B1 protocol")
    save(fig, "fig5_ppl")

    # 6 quality vs measured cost
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for c in CONFIGS:
        t = eff["configs"][c]["T_measured_s"]
        axs[0].scatter(t, rec[c]["recon"], color=COL[strat(c)], s=25)
        axs[1].scatter(t, ppl[c]["ppl"], color=COL[strat(c)], s=25)
        axs[0].annotate(c, (t, rec[c]["recon"]), fontsize=6)
        axs[1].annotate(c, (t, ppl[c]["ppl"]), fontsize=6)
    for ax, yl in zip(axs, ("mean reconstruction error", "PPL")):
        ax.set_xlabel("T_measured = tokenization + activation + Hessian + GPTQ (s)"); ax.set_ylabel(yl)
    fig.suptitle("Fig 6. Quality vs measured calibration cost (EBSS generation cost UNKNOWN and excluded)")
    save(fig, "fig6_quality_cost")


# ---------------------------------------------------------------- main
def main():
    proto = json.load(open(p("protocol.json")))
    res = load_results()
    reg = json.load(open(p("logs", "baseline_regression.json")))
    rep = json.load(open(p("logs", "capture_repeat.json")))
    repeat = json.load(open(p("results", f"{DRV.REPEAT_CONFIG}_repeat.json")))
    cm = json.load(open(p("logs", "capture_manifest.json")))
    man = json.load(open(p("ebss_selection_manifest.json")))
    comp_rec = json.load(open(p("compatibility.json")))
    audit = json.load(open(p("official_dataset_audit.json")))
    bf16 = reg["bf16"]["ppl"]

    diag, nll, sec, dvals, d_star = routing_diagnostics(res)
    jdump(diag, p("routing_diagnostics.json"))

    rec, keys, dec = recon_tables(res)
    hist = json.load(open(p("baseline_manifest.json")))["historical_context"]
    jdump(dict(metric="E3 held-out relative expert reconstruction, mean over 240 experts (lower = better); "
                      "~/moe_cache rows 2128-2255", policy="CANONICAL-B1 calibration for every configuration",
               exposure_deciles="deciles of the natural routing frequency p (C4 pool profile), decile 0 = rarest",
               configs=rec, historical_context_not_pooled={k: v["recon_partial_ppl"] for k, v in hist.items()}),
          p("reconstruction_results.json"))

    ppl = {c: dict(ppl=r["ppl"], delta_vs_bf16=r["ppl"] - bf16, rel_degradation_pct=(r["ppl"] / bf16 - 1) * 100,
                   total_nll=r["total_nll"], total_tokens=r["total_tokens"], per_window_nll=r["per_window_nll"],
                   per_window_tokens=r["per_window_tokens"], repeat_eval_abs_diff_ppl=r["repeat_eval_abs_diff_ppl"])
           for c, r in res.items()}
    ppl_by = {c: v["ppl"] for c, v in ppl.items()}
    seedstat = {s: dict(mean=float(np.mean([ppl_by[f"{s}_s{i}"] for i in DRV.SEEDS])),
                        sd=float(np.std([ppl_by[f"{s}_s{i}"] for i in DRV.SEEDS], ddof=1))) for s in DRV.STRATS}
    boot = {f"EBSS-{c}": bootstrap(res["EBSS"]["per_window_nll"], res[c]["per_window_nll"],
                                   res[c]["per_window_tokens"]) for c in C4}
    jdump(dict(protocol="runs_partial_ppl windows (128 x 512, 65,408 targets), EP.layerwise_nll batch 8, FP32 CE, "
                        "FP64 accumulation, natural routing",
               bf16=dict(ppl=bf16, historical=reg["bf16"]["historical"], exact=reg["bf16"]["exact_match_historical"]),
               configs=ppl, per_strategy_over_seeds=seedstat,
               paired_window_bootstrap_ebss_minus_baseline=boot,
               bootstrap_note="resamples evaluation windows for FIXED quantized models: benchmark-text variability "
                              "only, not calibration or execution-policy variability",
               repeat=dict(config=DRV.REPEAT_CONFIG, ppl=repeat["ppl"], equal=repeat["ppl"] == ppl_by[DRV.REPEAT_CONFIG],
                           weights_equal=repeat["installed_weights_sha256"] == res[DRV.REPEAT_CONFIG]["installed_weights_sha256"],
                           recon_equal=repeat["recon"] == res[DRV.REPEAT_CONFIG]["recon"]),
               historical_context_not_pooled={k: v["ppl_partial_ppl"] for k, v in hist.items()}),
          p("ppl_results.json"))

    comp, summ = comparisons(res, keys, dec, ppl_by)
    jdump(dict(definition="RelativePenalty = (Error_EBSS / Error_Baseline - 1) x 100 %; < 0 = EBSS better. One EBSS "
                          "set: no EBSS SD; experts are not independent replications", pairs=comp, by_strategy=summ),
          p("comparisons.json"))

    # efficiency
    tok_s = audit["tokenization_s"] * (man["records_scanned"] / audit["n_records"])
    eff = dict(definitions=dict(
        tokenization_s="EBSS: measured tokenization time of the 600 records x (records scanned / 600); C4: not "
                       "re-measured (pre-tokenized ids from the historical cache build) -> 0 in T_measured",
        activation_s="sum over the config's 128 rows of the measured per-row batch-1 compute time through layers "
                     "0..22 (+ layer 23 for the NLL diagnostic) + 128/N of the layer-transfer overhead",
        hessian_s="E3 Profiler time on the calibration rows (quantize_and_score t_h)",
        gptq_s="E3 quantization loop time (GPTQ solves + per-expert held-out scoring + write-back; the scoring "
               "cannot be separated without changing the frozen code)",
        T_measured="tokenization + activation + Hessian + GPTQ, each once",
        excluded=["EBSS sample generation (unknown; pre-generated artifact)", "C4 download / selection cost "
                  "(historical)", "model loading", "PPL evaluation (reported separately)"]),
        capture=dict(seconds_total=cm["seconds_total"], layer_transfer_s=cm["layer_transfer_s"], n_rows=cm["n_rows"],
                     peak_gpu_gb=cm["peak_gpu_gb"], peak_cpu_gb=cm["peak_cpu_gb"], cache_bytes=cm["bytes_on_disk"]),
        dataset=dict(download_s=json.load(open(p("source_provenance.json")))["download_s"],
                     tokenization_all_records_s=audit["tokenization_s"]),
        ebss_generation_cost=dict(measured=None, status="UNKNOWN: pre-generated artifact; the paper gives no time; "
                                  "the authors' rebuttal table R6 states 18 min for Qwen-MoE-14B on one A800 (for 128 "
                                  "x 512 sequences, settings not given) - external claim, not verified here"),
        configs={})
    for c in CONFIGS:
        rows = DRV.config_rows(c)[0]
        act = float(sec[rows].sum() + cm["layer_transfer_s"] * len(rows) / cm["n_rows"])
        t = ED.measured_time(tok_s if c == "EBSS" else 0.0, act, res[c]["hessian_s"], res[c]["gptq_s"])
        eff["configs"][c] = dict(t, ppl_eval_s=res[c]["eval_s"], quantize_total_s=res[c]["quantize_total_s"],
                                 peak_gpu_gb=res[c]["peak_gpu_gb"], peak_cpu_gb=res[c]["peak_cpu_gb"],
                                 calibration_tokens=128 * 512,
                                 weights_bytes=sum(os.path.getsize(os.path.join(res[c]["weights_dir"], f"L{L}.pt"))
                                                   for L in LAYERS))
    eff["storage"] = dict(cache_gb=cm["bytes_on_disk"] / 1e9,
                          weights_gb=sum(v["weights_bytes"] for v in eff["configs"].values()) / 1e9,
                          official_data_mb=os.path.getsize(DRV.raw_path()) / 1e6)
    jdump(eff, p("efficiency_results.json"))

    # input fingerprints
    jdump(dict(raw_dataset_sha256=man["source"]["raw_sha256"], ebss_windows_tensor_sha256=man["windows_tensor_sha256"],
               ebss_ordered_set_sha256=man["ordered_set_sha256"], cache_token_sha256=cm["token_sha256"],
               cache_files=cm["files"], c4_selection_ids_hash={c: v["ids_hash"] for c, v in
                                                     json.load(open(p("baseline_manifest.json")))["configs"].items()},
               ppl_windows_sha256=proto["ppl"]["windows_sha256"],
               reconstruction_cache="~/moe_cache (historical, unchanged), rows 2128-2255",
               installed_weights_sha256={c: r["installed_weights_sha256"] for c, r in res.items()},
               protocol_sha256=FD.sha256_file(p("protocol.json"))), p("input_fingerprints.json"))

    # decision
    correctness = dict(
        all_configs_valid=all(r["valid"] for r in res.values()),
        c4_rtn_fallbacks=sum(res[c]["rtn_fallbacks"] for c in C4), ebss_rtn_fallbacks=res["EBSS"]["rtn_fallbacks"],
        quant_pipeline_regression=reg["quant_pipeline"]["passed"], extraction_impl_regression=reg["extraction_impl"]["passed"],
        bf16_ppl_exact=reg["bf16"]["exact_match_historical"], capture_repeatability=rep["passed"],
        quant_repeat_bitwise=bool(repeat["installed_weights_sha256"] == res[DRV.REPEAT_CONFIG]["installed_weights_sha256"]
                                  and repeat["ppl"] == ppl_by[DRV.REPEAT_CONFIG] and repeat["recon"] == res[DRV.REPEAT_CONFIG]["recon"]),
        profiler_count_crosscheck_mismatches=sum(v["profiler_token_count_mismatches"] for v in diag["configs"].values()))
    correctness_ok = bool(correctness["all_configs_valid"] and correctness["c4_rtn_fallbacks"] == 0 and
                          correctness["quant_pipeline_regression"] and correctness["extraction_impl_regression"] and
                          correctness["bf16_ppl_exact"] and correctness["capture_repeatability"] and
                          correctness["quant_repeat_bitwise"])
    q = ED.quality_class(res["EBSS"]["recon"], [res[f"FR_s{i}"]["recon"] for i in DRV.SEEDS],
                         [res[f"TR_s{i}"]["recon"] for i in DRV.SEEDS], ppl_by["EBSS"],
                         [ppl_by[f"FR_s{i}"] for i in DRV.SEEDS], [ppl_by[f"TR_s{i}"] for i in DRV.SEEDS])
    prov = comp_rec["scientific_provenance"]["classification"]
    decision = ED.decide(man["status"] == "OK", correctness_ok, prov, q)
    jdump(dict(decision=decision, rule=proto["decision_rule"], data_ok=man["status"] == "OK", correctness=correctness,
               correctness_ok=correctness_ok, provenance=prov, quality=q,
               quality_class_ignoring_provenance=q["quality_class"],
               fc_negative_control=dict(ebss_vs_fc=summ["FC"])), p("decision.json"))

    summary = dict(decision=decision, provenance=prov, bf16_ppl=bf16,
                   configs={c: dict(recon=rec[c]["recon"], ppl=ppl_by[c], delta_ppl_vs_bf16=ppl_by[c] - bf16) for c in CONFIGS},
                   per_strategy=dict(recon={s: rec[f"{s}_mean_over_seeds"] for s in DRV.STRATS}, ppl=seedstat),
                   ebss=dict(recon=rec["EBSS"]["recon"], ppl=ppl_by["EBSS"], sd=None,
                             note="one fixed EBSS set: no SD"),
                   comparisons=summ, quality=q, correctness=correctness,
                   calibration=dict({c: dict(diag["configs"][c]["mean_over_layers"],
                                             frac_atypical=diag["configs"][c]["typicality"]["frac_atypical"],
                                             model_ppl=diag["configs"][c]["model_nll"]["ppl"],
                                             distinct1=diag["configs"][c]["text"]["distinct1_mean"],
                                             distinct2=diag["configs"][c]["text"]["distinct2_mean"],
                                             near_dup_windows_ge_0_8=diag["configs"][c]["near_duplicates"]["windows_in_a_pair_ge_0.8"],
                                             exact_dup=diag["configs"][c]["text"]["exact_duplicate_windows"])
                                     for c in CONFIGS}),
                   efficiency={c: eff["configs"][c]["T_measured_s"] for c in CONFIGS})
    jdump(summary, p("summary.json"))
    figures(diag, rec, comp, ppl, eff, bf16)
    write_md(summary, diag, rec, ppl, comp, summ, eff, correctness, reg, man, audit, decision, q, boot, seedstat)
    print(f"[report] decision {decision}; quality class ignoring provenance {q['quality_class']}")


def write_md(S, diag, rec, ppl, comp, summ, eff, corr, reg, man, audit, decision, q, boot, seedstat):
    f = lambda x, n=6: f"{x:.{n}f}"
    L = []
    L.append("# Official MoEQuant EBSS calibration-quality validation\n")
    L.append(f"**Decision: {decision}** (provenance {S['provenance']}; quality class ignoring provenance: "
             f"{q['quality_class']})\n")
    L.append("## Quality (CANONICAL-B1 extraction for every configuration)\n")
    L.append(f"BF16 PPL {f(S['bf16_ppl'])}.\n")
    L.append("| config | reconstruction | PPL | ΔPPL vs BF16 |\n|---|---|---|---|")
    for c in CONFIGS:
        L.append(f"| {c} | {rec[c]['recon']:.7f} | {f(ppl[c]['ppl'])} | {ppl[c]['delta_vs_bf16']:+.6f} |")
    L.append("\n## EBSS relative reconstruction penalty (%; < 0 = EBSS better) and ΔPPL (EBSS − baseline)\n")
    L.append("| baseline | s0 | s1 | s2 | mean | ΔPPL s0 / s1 / s2 | frac experts improved |\n|---|---|---|---|---|---|---|")
    for s in DRV.STRATS:
        v = summ[s]
        L.append(f"| {s} | " + " | ".join(f"{x:+.3f}" for x in v["relative_penalty_pct"]) +
                 f" | {v['mean_relative_penalty_pct']:+.3f} | " + " / ".join(f"{x:+.5f}" for x in v["delta_ppl"]) +
                 " | " + " / ".join(f"{x:.2f}" for x in v["frac_experts_improved"]) + " |")
    L.append("\n## Calibration diagnostics (means over layers 1/8/16/23)\n")
    L.append("| config | min exp. | p10 exp. | CV | Gini | routing L1 | atypical | model PPL | distinct-1 | distinct-2 | near-dup windows |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CONFIGS:
        d = S["calibration"][c]
        L.append(f"| {c} | {d['min']:.0f} | {d['p10']:.0f} | {d['cv']:.3f} | {d['gini']:.3f} | {d['routing_l1']:.3f} | "
                 f"{d['frac_atypical']:.2f} | {d['model_ppl']:.2f} | {d['distinct1']:.3f} | {d['distinct2']:.3f} | "
                 f"{d['near_dup_windows_ge_0_8']} |")
    L.append("\n## Correctness\n")
    for k, v in corr.items():
        L.append(f"- {k}: {v}")
    L.append("\n## Efficiency (seconds; EBSS generation cost unknown and excluded)\n")
    L.append("| config | tokenization | activation | Hessian | GPTQ | T_measured |\n|---|---|---|---|---|---|")
    for c in CONFIGS:
        e = eff["configs"][c]
        L.append(f"| {c} | {e['tokenization_s']:.1f} | {e['activation_s']:.1f} | {e['hessian_s']:.1f} | "
                 f"{e['gptq_s']:.1f} | {e['T_measured_s']:.1f} |")
    L.append("\nSee `update_ebss_official_validation.md` for the interpretation.")
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
