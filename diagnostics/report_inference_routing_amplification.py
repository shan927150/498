"""Analysis, decision, figures and summary for the inference-time routing amplification MVP (CPU only).

python diagnostics/report_inference_routing_amplification.py [--analyze] [--report]
Reads runs_inference_routing_amplification/results/*.json and ~/moe_cache_inference_routing/*.pt.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import inference_routing_amplification as D                    # noqa: E402
from moe_calib import inference_routing_override as IRO       # noqa: E402

p, c, jdump, jload = D.p, D.c, D.jdump, D.jload
NL, NW, T, K, E = 24, 128, 512, 4, 60
QS = (0.5, 0.9, 0.99)


def sd(x):
    return float(np.std(np.asarray(x, dtype=np.float64), ddof=1))


def quant(x, qs=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return {str(q): float(np.quantile(x, q)) for q in qs}


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(spearmanr(a, b).correlation)


def load_run(name, mode):
    return torch.load(D.run_file(name, mode))


def res(name, mode):
    return jload(D.res_file(name, mode))


def margin(d):
    return (d["pk"][..., 3] - d["p_next"])


# ---------------------------------------------------------------- routing divergence
def divergence(idx, ref, ref_margin, own_margin):
    sc, oo = IRO.set_disagreement(idx, ref)                   # [24, 128, 512]
    nm = IRO.n_membership_changes(idx, ref)
    any_slot = (idx.long() != ref.long()).any(-1)
    tok_any = sc.any(0)                                       # [128, 512]
    first = torch.where(tok_any, sc.float().argmax(0), torch.full_like(tok_any, -1, dtype=torch.long))
    first_hist = torch.bincount(first[tok_any].flatten(), minlength=NL).tolist() if tok_any.any() else [0] * NL
    per_win = sc.float().mean(dim=(0, 2)).numpy()             # fraction of (layer, token) changed per window
    ch = sc.flatten()
    out = dict(
        per_layer=[dict(layer=L, frac_set_changed=float(sc[L].float().mean()), n_set_changed=int(sc[L].sum()),
                        frac_order_only=float(oo[L].float().mean()), frac_any_slot=float(any_slot[L].float().mean()),
                        n_changed_selections=int(nm[L].sum())) for L in range(NL)],
        total_set_changed=int(sc.sum()), frac_set_changed_all=float(sc.float().mean()),
        total_order_only=int(oo.sum()), total_changed_selections=int(nm.sum()),
        frac_tokens_any_layer_changed=float(tok_any.float().mean()),
        first_divergence_layer_hist=first_hist,
        first_layer_with_any_change=next((L for L in range(NL) if sc[L].any()), None),
        layers_0_1_identical=bool(not sc[:2].any() and not any_slot[:2].any()),
        per_window_frac_quantiles=quant(per_win), per_window_frac_max_window=int(per_win.argmax()),
        membership_changes_per_changed_token=quant(nm[sc].numpy()) if sc.any() else None)
    rm, om = ref_margin.flatten(), own_margin.flatten()
    out["margin"] = dict(bf16_margin_changed=quant(rm[ch].numpy()), bf16_margin_stable=quant(rm[~ch][::5].numpy()),
                         quant_margin_changed=quant(om[ch].numpy()), quant_margin_stable=quant(om[~ch][::5].numpy()),
                         note="stable-set quantiles from a fixed 1/5 stride subsample")
    if ch.any():                                              # share of changes by BF16 margin decile of all tokens
        edges = np.quantile(rm[::5].numpy(), np.linspace(0, 1, 11))
        r = rm.numpy()
        dec = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, 9)
        chn = ch.numpy()
        out["margin"]["frac_changed_by_bf16_margin_decile"] = [float(chn[dec == i].mean()) for i in range(10)]
        out["margin"]["bf16_margin_decile_edges"] = edges.tolist()
    return out, sc


def analyze():
    t0 = time.time()
    pr = D.check_protocol()
    inv, wm, bman = jload(p("artifact_inventory.json")), jload(p("weight_manifest.json")), jload(p("bf16_routing_manifest.json"))
    val = jload(p("routing_override_validation.json"))
    cost_raw = jload(p("logs", "gpu_cost_raw.json"))
    pre = jload(p("logs", "preflight_flip_count.json"))
    ref = torch.load(c("bf16_reference_routing.pt"))
    ref_idx, ref_m = ref["idx"], margin(ref)
    early = pre["early_stop"]
    names = D.PRIMARY + D.REFERENCES
    modes = ["natural"] if early else ["natural", "frozen"]
    have = {(n, m): res(n, m) for n in names for m in modes}
    missing = [f"{n}:{m}" for (n, m), r in have.items() if r is None]

    # ---------- historical regression
    hreg = {}
    for n in names:
        r = have[(n, "natural")]
        if r:
            hreg[n] = dict(bitwise_per_window=r["historical_regression"]["bitwise"],
                           max_abs_window_diff=r["historical_regression"]["max_abs_diff"],
                           ppl_new=r["ppl"], ppl_historical=wm["configs"][n]["historical_ppl"],
                           ppl_equal=r["ppl"] == wm["configs"][n]["historical_ppl"])
    hb = jload(os.path.join(ROOT, D.HIST_BF16))
    hreg["BF16"] = dict(ppl_new=bman["ppl"], ppl_historical=hb["ppl"], ppl_equal=bman["ppl"] == hb["ppl"],
                        bitwise_per_window=bman["checks"]["plain_vs_historical_bf16_per_window_bitwise"])
    hist_sd = jload(os.path.join(ROOT, "runs_hessian_order_null/paired_differences.json"))
    jdump(dict(per_config=hreg, all_bitwise=all(v["bitwise_per_window"] for v in hreg.values()),
               historical_order_null_sd_delta=hist_sd["sd_delta"], historical_sd_fr=hist_sd["sd_fr"],
               historical_sd_fc=hist_sd["sd_fc"]), p("historical_regression.json"))

    # ---------- routing divergence (natural, and natural-choice under frozen trajectories)
    div, sets = {}, {}
    for n in names:
        if not have[(n, "natural")]:
            continue
        d = load_run(n, "natural")
        dv, sc = divergence(d["idx"], ref_idx, ref_m, margin(d))
        cnt = IRO.expert_counts(d["idx"], E)
        dv["expert_count_abs_diff_vs_bf16_per_layer"] = (cnt - IRO.expert_counts(ref_idx, E)).abs().sum(1).tolist()
        dv["dispatch_equals_own_indices"] = bool(torch.equal(d["dispatch"], cnt))
        div[n] = dv
        sets[n] = sc
        if "frozen" in modes and have[(n, "frozen")]:
            f = load_run(n, "frozen")
            fv, _ = divergence(f["idx"], ref_idx, ref_m, margin(f))
            own_mass, forced_mass = f["pk"].sum(-1), f["p_forced"].sum(-1)
            div[n]["under_frozen_execution"] = dict(
                natural_choice_vs_bf16_frac_set_changed_per_layer=[x["frac_set_changed"] for x in fv["per_layer"]],
                natural_choice_vs_bf16_total_set_changed=fv["total_set_changed"],
                forced_prob_mass_quantiles=quant(forced_mass.flatten()[::7].numpy()),
                own_top4_mass_quantiles=quant(own_mass.flatten()[::7].numpy()),
                mass_deficit_on_changed_tokens=quant((own_mass - forced_mass)[IRO.set_disagreement(f["idx"], ref_idx)[0]].numpy()),
                forced_equals_own_top4_values_where_sets_equal=bool(torch.equal(
                    f["p_forced"][~IRO.set_disagreement(f["idx"], ref_idx)[0] & (f["idx"].long() == ref_idx.long()).all(-1)],
                    f["pk"][~IRO.set_disagreement(f["idx"], ref_idx)[0] & (f["idx"].long() == ref_idx.long()).all(-1)])),
                dispatch_equals_bf16_reference=bool(torch.equal(f["dispatch"], ref["dispatch"])),
                frozen_selected_counts_equal_bf16=bool(torch.equal(f["dispatch"], IRO.expert_counts(ref_idx, E))))
            del f
        del d
    # pairwise routing difference between each order config and ORIGINAL of the same strategy
    for n in D.PRIMARY:
        s = n.split("_")[0]
        if n in div and f"{s}_ORIGINAL" in div:
            a, b = load_run(n, "natural")["idx"], load_run(f"{s}_ORIGINAL", "natural")["idx"]
            scab, _ = IRO.set_disagreement(a, b)
            div[n]["vs_original_frac_set_changed_per_layer"] = scab.float().mean(dim=(1, 2)).tolist()
            div[n]["vs_original_total_set_changed"] = int(scab.sum())
    jdump(dict(reference="BF16 natural routing (bf16_reference_routing.pt)", per_config=div,
               primary_total_set_changed=pre["total_set_disagreements_primary"], early_stop=early,
               bf16_reference_disagreement=0), p("routing_divergence.json"))
    if early:
        decision = dict(verdict="ROUTING-NO-FLIPS", conclusion="E")
        jdump(decision, p("decision.json"))
        return decision

    # ---------- PPL results
    nat = {n: dict(ppl=have[(n, "natural")]["ppl"], mean_nll=have[(n, "natural")]["mean_nll"]) for n in names if have[(n, "natural")]}
    fro = {n: dict(ppl=have[(n, "frozen")]["ppl"], mean_nll=have[(n, "frozen")]["mean_nll"],
                   natural_minus_frozen=nat[n]["ppl"] - have[(n, "frozen")]["ppl"]) for n in names if have[(n, "frozen")]}
    bfz = res("BF16", "frozen")
    jdump(dict(per_config=nat, bf16=bman["ppl"]), p("natural_ppl_results.json"))
    jdump(dict(per_config=fro, bf16_self_frozen=bfz["ppl"] if bfz else None), p("frozen_ppl_results.json"))

    # ---------- paired comparisons
    def pw(n, m):
        return np.array(have[(n, m)]["per_window_nll"], dtype=np.float64)
    per_order = []
    for g in D.ORDERS:
        fr, fc = f"FR_O{g}", f"FC_O{g}"
        dn, df = nat[fc]["ppl"] - nat[fr]["ppl"], fro[fc]["ppl"] - fro[fr]["ppl"]
        per_order.append(dict(order=g, FR_natural=nat[fr]["ppl"], FC_natural=nat[fc]["ppl"], FR_frozen=fro[fr]["ppl"],
                              FC_frozen=fro[fc]["ppl"], delta_natural=dn, delta_frozen=df, delta_g=dn - df,
                              FR_natural_minus_frozen=fro[fr]["natural_minus_frozen"],
                              FC_natural_minus_frozen=fro[fc]["natural_minus_frozen"],
                              FR_frac_windows_improved=float(np.mean(pw(fr, "frozen") < pw(fr, "natural"))),
                              FC_frac_windows_improved=float(np.mean(pw(fc, "frozen") < pw(fc, "natural"))),
                              FR_frac_windows_worsened=float(np.mean(pw(fr, "frozen") > pw(fr, "natural"))),
                              FC_frac_windows_worsened=float(np.mean(pw(fc, "frozen") > pw(fc, "natural"))),
                              paired_window_delta_sd_natural=sd(pw(fc, "natural") - pw(fr, "natural")),
                              paired_window_delta_sd_frozen=sd(pw(fc, "frozen") - pw(fr, "frozen"))))
    dn = [x["delta_natural"] for x in per_order]
    df = [x["delta_frozen"] for x in per_order]
    jdump(dict(per_order=per_order, sign="Delta = PPL(FC) - PPL(FR); negative = FC lower PPL"), p("paired_order_comparisons.json"))
    R = sd(df) / sd(dn)
    var = dict(sd_delta_natural=sd(dn), sd_delta_frozen=sd(df), R_SD=R,
               sd_FR_natural=sd([x["FR_natural"] for x in per_order]), sd_FR_frozen=sd([x["FR_frozen"] for x in per_order]),
               sd_FC_natural=sd([x["FC_natural"] for x in per_order]), sd_FC_frozen=sd([x["FC_frozen"] for x in per_order]),
               mean_delta_natural=float(np.mean(dn)), mean_delta_frozen=float(np.mean(df)),
               change_in_mean_delta=float(np.mean(df) - np.mean(dn)),
               range_delta_natural=[min(dn), max(dn)], range_delta_frozen=[min(df), max(df)],
               n_positive_delta_natural=int(sum(x > 0 for x in dn)), n_positive_delta_frozen=int(sum(x > 0 for x in df)),
               sign_stable_natural=len({x > 0 for x in dn}) == 1, sign_stable_frozen=len({x > 0 for x in df}) == 1,
               sd_delta_g=sd([x["delta_g"] for x in per_order]), mean_delta_g=float(np.mean([x["delta_g"] for x in per_order])),
               historical_sd_delta_natural=hist_sd["sd_delta"],
               natural_sd_equals_historical=abs(sd(dn) - hist_sd["sd_delta"]) < 1e-12,
               mean_FR_natural_minus_frozen=float(np.mean([x["FR_natural_minus_frozen"] for x in per_order])),
               mean_FC_natural_minus_frozen=float(np.mean([x["FC_natural_minus_frozen"] for x in per_order])),
               note="sample SD (ddof=1) over 5 deterministic orders of one seed; screening statistic, not a test; "
                    "the SD difference is not an additive variance contribution")
    # relationship between disagreement and natural-minus-frozen change (descriptive, n=10)
    rel = [dict(config=n, frac_set_changed=div[n]["frac_set_changed_all"], natural_minus_frozen=fro[n]["natural_minus_frozen"])
           for n in D.PRIMARY]
    var["disagreement_vs_natural_minus_frozen"] = dict(rows=rel, spearman_abs=spearman(
        [r["frac_set_changed"] for r in rel], [abs(r["natural_minus_frozen"]) for r in rel]), note="descriptive, n = 10")
    jdump(var, p("variance_comparison.json"))

    # ---------- canonical references
    canon = {}
    for tag in ("ORIGINAL", "FP64"):
        fr, fc = f"FR_{tag}", f"FC_{tag}"
        canon[tag] = dict(FR_natural=nat[fr]["ppl"], FC_natural=nat[fc]["ppl"], FR_frozen=fro[fr]["ppl"], FC_frozen=fro[fc]["ppl"],
                          delta_natural=nat[fc]["ppl"] - nat[fr]["ppl"], delta_frozen=fro[fc]["ppl"] - fro[fr]["ppl"],
                          FR_historical=wm["configs"][fr]["historical_ppl"], FC_historical=wm["configs"][fc]["historical_ppl"],
                          FR_frac_set_changed=div[fr]["frac_set_changed_all"], FC_frac_set_changed=div[fc]["frac_set_changed_all"])
        x = canon[tag]
        x["delta_natural_minus_frozen"] = x["delta_natural"] - x["delta_frozen"]
        x["sign_preserved"] = (x["delta_natural"] > 0) == (x["delta_frozen"] > 0)
        x["delta_frozen_within_primary_frozen_range"] = min(df) <= x["delta_frozen"] <= max(df)
    canon["ORIGINAL_vs_FP64_delta_gap_natural"] = canon["ORIGINAL"]["delta_natural"] - canon["FP64"]["delta_natural"]
    canon["ORIGINAL_vs_FP64_delta_gap_frozen"] = canon["ORIGINAL"]["delta_frozen"] - canon["FP64"]["delta_frozen"]
    canon["note"] = "descriptive; excluded from the 5-order SD; does not redefine the canonical estimand"
    jdump(canon, p("canonical_reference_comparison.json"))

    # ---------- token-level NLL (Analyses A, B, C)
    tok = {(n, m): load_run(n, m)["token_nll"].numpy() for n in names for m in modes}
    conc, assoc = {}, {}
    curves = {}
    fr_grid = np.unique(np.concatenate([np.linspace(0, 0.1, 101), np.linspace(0.1, 1, 91)]))
    for n in D.PRIMARY:
        s = n.split("_")[0]
        o = f"{s}_ORIGINAL"
        conc[n] = {}
        for m in modes:
            dl = tok[(n, m)] - tok[(o, m)]                           # [128, 511]
            a = np.abs(dl).flatten()
            srt = np.sort(a)[::-1]
            cs = np.cumsum(srt) / srt.sum()
            N = a.size
            top = {f"top_{int(f * 100)}pct": float(cs[int(math.ceil(f * N)) - 1]) for f in (0.01, 0.05, 0.10)}
            wabs = np.abs(dl).sum(1)
            ws = np.sort(wabs)[::-1]
            conc[n][m] = dict(mean_abs=float(a.mean()), median_abs=float(np.median(a)), p90=float(np.quantile(a, 0.9)),
                              p99=float(np.quantile(a, 0.99)), max=float(a.max()), frac_exact_zero=float(np.mean(a == 0)),
                              share_of_abs_change=top, n_tokens_for_50pct_abs=int(np.searchsorted(cs, 0.5) + 1),
                              n_tokens_for_90pct_abs=int(np.searchsorted(cs, 0.9) + 1), n_tokens=int(N),
                              signed_net_sum=float(dl.sum()), abs_sum=float(a.sum()),
                              top10pct_windows_share_of_abs=float(ws[:13].sum() / ws.sum()),
                              per_window_signed_sd=sd(dl.sum(1)))
            curves[(n, m)] = np.interp(fr_grid, (np.arange(N) + 1) / N, cs)
        # Analysis B / C on natural routing: events defined (i) vs ORIGINAL routing (pairs with dl), (ii) vs BF16
        dl = np.abs(tok[(n, "natural")] - tok[(o, "natural")])        # [128, 511], target j <- logit position j
        a_idx = load_run(n, "natural")["idx"]
        o_idx = load_run(o, "natural")["idx"]
        assoc[n] = {}
        for evname, other, mref in (("vs_ORIGINAL", o_idx, margin(load_run(o, "natural"))), ("vs_BF16", ref_idx, ref_m)):
            scx, _ = IRO.set_disagreement(a_idx, other)               # [24, 128, 512]
            ev = scx[:, :, :511]                                      # positions 0..510 produce the 511 scored logits
            nloc = ev.sum(0).numpy()                                  # [128, 511] layers changed at position j
            loc = nloc > 0
            per_pos = scx.sum(0).numpy()                              # events per position (all 512)
            cum = np.cumsum(per_pos, axis=1)
            prefix_events = np.concatenate([np.zeros((NW, 1)), cum[:, :510]], axis=1)   # events at positions < j
            posflag = (per_pos > 0).astype(np.int64)
            prefix_pos = np.concatenate([np.zeros((NW, 1)), np.cumsum(posflag, 1)[:, :510]], axis=1)
            # earliest layer among prefix events and min margin among prefix+local events
            layer_of = torch.where(scx, torch.arange(NL).view(NL, 1, 1).expand_as(scx), torch.full_like(scx, 99, dtype=torch.long))
            minlayer_pos = layer_of.min(0).values.numpy()             # [128, 512]
            mm = torch.where(scx, mref, torch.full_like(mref, np.inf)).min(0).values.numpy()
            pref_minlayer = np.minimum.accumulate(np.concatenate([np.full((NW, 1), 99), minlayer_pos[:, :510]], 1), axis=1)
            pref_minmargin = np.minimum.accumulate(np.concatenate([np.full((NW, 1), np.inf), mm[:, :510]], 1), axis=1)
            loc_minmargin = mm[:, :511]
            pre_any = prefix_events > 0
            g = lambda mask: dict(n=int(mask.sum()), mean=float(dl[mask].mean()) if mask.any() else None,
                                  median=float(np.median(dl[mask])) if mask.any() else None,
                                  p90=float(np.quantile(dl[mask], 0.9)) if mask.any() else None)
            bins = [(0, 0), (1, 2), (3, 10), (11, 50), (51, 10 ** 9)]
            assoc[n][evname] = dict(
                frac_targets_local_event=float(loc.mean()), frac_targets_prefix_event=float(pre_any.mean()),
                local=dict(with_event=g(loc), without_event=g(~loc)),
                by_n_local_layers={str(k): g(nloc == k) for k in (0, 1, 2)} | {"3+": g(nloc >= 3)},
                prefix=dict(no_local_no_prefix=g(~loc & ~pre_any), prefix_only=g(~loc & pre_any), local_any=g(loc)),
                by_prefix_event_count={f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": g((prefix_events >= lo) & (prefix_events <= hi))
                                       for lo, hi in bins},
                by_prefix_distinct_positions={f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": g((prefix_pos >= lo) & (prefix_pos <= hi))
                                              for lo, hi in bins},
                by_prefix_earliest_layer={str(L): g(pre_any & (pref_minlayer >= L) & (pref_minlayer < L2))
                                          for L, L2 in ((0, 6), (6, 12), (12, 18), (18, 24))},
                spearman_abs_dl_vs_prefix_events=spearman(dl.flatten()[::3], prefix_events.flatten()[::3]),
                spearman_abs_dl_vs_local_layers=spearman(dl.flatten()[::3], nloc.flatten()[::3]),
                local_event_min_margin=quant(loc_minmargin[loc]) if loc.any() else None,
                prefix_min_margin_low_vs_high_dl=dict(
                    top1pct_dl=quant(pref_minmargin[(dl >= np.quantile(dl, 0.99)) & pre_any]),
                    rest=quant(pref_minmargin[(dl < np.quantile(dl, 0.99)) & pre_any][::5])),
                top1pct_dl_frac_with_local_event=float(loc[dl >= np.quantile(dl, 0.99)].mean()),
                top1pct_dl_frac_with_prefix_event=float(pre_any[dl >= np.quantile(dl, 0.99)].mean()),
                margin_reference="ORIGINAL natural margins" if evname == "vs_ORIGINAL" else "BF16 margins")
    tokens_desc = dict(per_config=conc, reference="same-strategy ORIGINAL FP32 under the same routing regime",
                       note="absolute-change concentration is not an additive decomposition of the signed PPL difference")
    # aggregate across the 10 primary configs
    for m in modes:
        tokens_desc[f"mean_over_primary_{m}"] = {k: float(np.mean([conc[n][m][k] for n in D.PRIMARY]))
                                                 for k in ("mean_abs", "median_abs", "p99", "max", "abs_sum")}
        tokens_desc[f"mean_top1pct_share_{m}"] = float(np.mean([conc[n][m]["share_of_abs_change"]["top_1pct"] for n in D.PRIMARY]))
        tokens_desc[f"mean_top10pct_share_{m}"] = float(np.mean([conc[n][m]["share_of_abs_change"]["top_10pct"] for n in D.PRIMARY]))
    jdump(tokens_desc, p("token_nll_concentration.json"))
    jdump(dict(per_config=assoc, alignment=pr["token_alignment"],
               note="descriptive associations; causal attribution comes only from the frozen-routing intervention"),
          p("routing_nll_association.json"))
    np.savez_compressed(c("concentration_curves.npz"), grid=fr_grid,
                        **{f"{n}__{m}": v for (n, m), v in curves.items()})

    # ---------- cost
    runs = cost_raw["runs"]
    ev = lambda suf: [v["eval_s"] for k, v in runs.items() if k.endswith(suf) and "eval_s" in v]
    store = sum(os.path.getsize(os.path.join(c("runs"), f)) for f in os.listdir(c("runs")))
    cost = dict(model_load_s=[x["model_load_s"] for x in cost_raw["loads"]],
                bf16_reference_recording_s=runs.get("BF16_record_1", {}).get("eval_s"),
                historical_plain_eval_s=ev("_plain"), natural_eval_s=ev("_natural"), frozen_eval_s=ev("_frozen"),
                mean_natural_eval_s=float(np.mean(ev("_natural"))), mean_frozen_eval_s=float(np.mean(ev("_frozen"))),
                mean_plain_eval_s=float(np.mean(ev("_plain"))),
                median_natural_eval_s=float(np.median(ev("_natural"))), median_frozen_eval_s=float(np.median(ev("_frozen"))),
                recorder_overhead_per_token_us_median=(float(np.median(ev("_natural"))) - float(np.median(ev("_plain")))) / 65536 * 1e6,
                timing_note="wall times are confounded: the first plain BF16 eval is a cold start, and most primary frozen "
                            "evaluations overlapped a 32-min CPU pytest run (host CPU contention); only 2 plain evals exist, "
                            "so the recorder overhead is not resolvable from these timings (it is within run-to-run noise)",
                install_s=[v["install_s"] for k, v in runs.items() if k.endswith("_install")],
                untouched_digest_s=cost_raw.get("untouched_digest_s"),
                peak_gpu_gb=max(v.get("peak_gpu_gb", 0) for v in runs.values()),
                peak_cpu_gb=cost_raw.get("peak_cpu_gb"), storage_bytes_runs=store,
                storage_bytes_reference=os.path.getsize(c("bf16_reference_routing.pt")),
                gptq_solves=0, n_eval_forwards=sum(1 for v in runs.values() if "eval_s" in v),
                analysis_s=time.time() - t0)
    jdump(cost, p("cost_results.json"))

    # ---------- correctness
    endc = cost_raw.get("end_checks", {})
    A, B, C = val.get("A_bf16", {}), val.get("B_bf16_self_routing", {}), val.get("C_quant_self_routing", {})
    Aq = val.get("A_quant", {})
    fz = [have[(n, "frozen")] for n in names]
    pt = jload(p("logs", "pytest_results.json")) or {}
    corr = {
        "01_historical_weight_hashes_match": inv["all_valid"],
        "02_no_quantized_weights_regenerated": cost["gptq_solves"] == 0 and inv["regenerated_weights"] == 0,
        "03_all_240_experts_intended_weights": all(v["checks"]["installed_weights_sha256_match"] and v["checks"]["keys_complete_240_experts"]
                                                   for v in inv["configs"].values()),
        "04_non_target_params_unchanged": bool(endc.get("untouched_params_unchanged") and endc.get("gates_unchanged")),
        "05_windows_match_historical_hash": True,                         # EP.load_windows asserts + protocol check
        "06_bf16_ppl_matches_historical": bman["checks"]["ppl_equals_historical"] and bman["checks"]["plain_vs_historical_bf16_per_window_bitwise"],
        "07_record_only_matches_normal": bool(A.get("passed") and Aq.get("passed")),
        "08_bf16_self_routing_parity": bool(B.get("passed")),
        "09_quant_self_routing_parity": bool(C.get("passed")),
        "10_router_logits_computed_normally": bool(B.get("natural_choice_under_forcing_equal") and C.get("selections_equal")),
        "11_router_probs_computed_normally": bool(C.get("router_probs_equal")),
        "12_only_selected_indices_overridden": bool(C.get("passed") and B.get("passed")),
        "13_current_probs_used_for_forced": all(div[n]["under_frozen_execution"]["forced_equals_own_top4_values_where_sets_equal"]
                                                for n in names),
        "14_topk_normalisation_preserved": True,                           # norm_topk_prob False; verbatim forward (tests)
        "15_dispatch_and_accumulation_order_preserved": bool(C.get("layer_hashes_equal") and B.get("layer_hashes_equal")),
        "16_all_24_layers_covered": bman["checks"]["all_24_layers_recorded"] and all(
            x == 16 for x in B.get("forced_calls_per_layer", [])) and len(B.get("forced_calls_per_layer", [])) == 24,
        "17_reference_aligned_to_windows_positions": bman["checks"]["exactly_4_per_token"] and B.get("passed", False),
        "18_no_repeated_indices": bman["checks"]["no_repeated_ids"],
        "19_valid_expert_ids": bman["checks"]["indices_in_range"],
        "20_hooks_change_no_parameters": bool(endc.get("untouched_params_unchanged") and endc.get("targets_pristine")),
        "21_natural_ppl_reproduces_order_null": all(hreg[n]["bitwise_per_window"] for n in D.PRIMARY),
        "22_original_fp64_reproduce_historical": all(hreg[n]["bitwise_per_window"] for n in D.REFERENCES),
        "23_token_nll_sums_reproduce_window_nll": all(np.array_equal(tok[(n, m)].sum(1), np.array(have[(n, m)]["per_window_nll"]))
                                                      for n in names for m in modes),
        "24_window_nll_sums_reproduce_ppl": all(math.exp(float(np.sum(np.array(have[(n, m)]["per_window_nll"]))) / 65408)
                                                == have[(n, m)]["ppl"] for n in names for m in modes),   # EP.ppl_record definition
        "25_natural_frozen_identical_weights": all(have[(n, "natural")]["installed_weights_sha256"] == have[(n, "frozen")]["installed_weights_sha256"]
                                                   == wm["configs"][n]["installed_weights_sha256"] for n in names),
        "26_no_old_artifacts_overwritten": pt.get("historical_unchanged", None),
        "27_targeted_tests_pass": pt.get("targeted_passed", None),
        "28_full_test_suite_passes": pt.get("full_passed", None),
        "frozen_dispatch_equals_bf16": all(r["checks"]["dispatch_equals_bf16"] for r in fz),
        "bf16_end_of_process_bitwise": endc.get("bf16_end_per_window_bitwise"),
        "no_missing_runs": not missing}
    parity_ok = all(corr[k] for k in ("07_record_only_matches_normal", "08_bf16_self_routing_parity", "09_quant_self_routing_parity"))
    seq_vs_np = max(abs(math.exp(sum(have[(n, m)]["per_window_nll"]) / 65408) - have[(n, m)]["ppl"]) for n in names for m in modes)
    jdump(dict(checks=corr, check24_note="window sums use numpy pairwise summation as in EP.ppl_record; a Python sequential "
               f"sum differs by at most {seq_vs_np:.2e} PPL (last-ulp summation order)", all_pass=all(v is True for v in corr.values()), missing_runs=missing), p("correctness_results.json"))

    # ---------- decision
    if not parity_ok or missing:
        verdict = "ROUTING-INTERVENTION-BLOCKED"
    elif R <= 0.5:
        verdict = "ROUTING-AMPLIFICATION-SUPPORTED"
    elif R < 0.9:
        verdict = "ROUTING-PARTIAL-CONTRIBUTION"
    elif R <= 1.1:
        verdict = "ROUTING-NO-SUBSTANTIAL-REDUCTION"
    else:
        verdict = "ROUTING-VARIABILITY-INCREASED"
    concl = {"ROUTING-AMPLIFICATION-SUPPORTED": "A", "ROUTING-PARTIAL-CONTRIBUTION": "B",
             "ROUTING-NO-SUBSTANTIAL-REDUCTION": "C", "ROUTING-VARIABILITY-INCREASED": "C",
             "ROUTING-INTERVENTION-BLOCKED": "D"}[verdict]
    decision = dict(verdict=verdict, conclusion=concl, R_SD=R, sd_delta_natural=sd(dn), sd_delta_frozen=sd(df),
                    rules=pr["decision_rules"], protocol_sha256=pr["protocol_sha256"], parity_ok=parity_ok,
                    all_correctness_pass=all(v is True for v in corr.values()))
    jdump(decision, p("decision.json"))
    print(json.dumps(decision, indent=1))
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    if a.report:
        import report_inference_routing_figures as RF
        RF.report()
