"""Report for the FP32 Hessian accumulation-order null experiment: Hessian / FP64 control / weight / quantizer /
reconstruction / PPL / paired-difference analyses, the frozen decision, figures 1-6 (only for executed stages)."""
from __future__ import annotations
import json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import hessian_order_null as XH          # noqa: E402
from moe_calib import frozen_routing as FRT          # noqa: E402
from moe_calib import hessian_order_null as HN       # noqa: E402

p, STRATS, LAYERS, SEEDS = XH.p, XH.STRATS, XH.LAYERS, list(HN.ORDER_SEEDS)
jdump = XH.jdump
PROJ = ("gate_proj", "up_proj", "down_proj")
# earlier studies (context only; different perturbation mechanisms; never pooled)
CONTEXT = {"natural alternative routing (AA)": dict(FR=0.00442, FC=0.00429, Delta=0.00636),
           "frozen routing (AO)": dict(FR=0.00402, FC=0.00175, Delta=0.00561),
           "frozen routing + act-order (AOF)": dict(FR=0.00240, FC=0.00240, Delta=0.00166)}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def result(c, s, repeat=False):
    return load(p("results", f"{XH.cname(c, s, repeat)}.json"))


# ---------------------------------------------------------------- analyses
def weight_analysis(man, hpre, res):
    out = {}
    for c in STRATS:
        oo_wd = man["oo_references"][c]["weights_dir"]
        out[c] = {}
        Wo = {L: torch.load(os.path.join(oo_wd, f"L{L}.pt")) for L in LAYERS}
        for s in SEEDS:
            r = res[c][s]
            keys = hpre[c]["keys"][f"O{s}"]
            per_layer, per_expert, n_id, n = {}, [], 0, 0
            cross = dict(h_changed_w_changed=0, h_changed_w_same=0, h_same_w_changed=0, h_same_w_same=0)
            for L in LAYERS:
                W = torch.load(os.path.join(r["weights_dir"], f"L{L}.pt"))
                per_layer[str(L)] = FRT.weight_diff(W, Wo[L])
                for e in range(60):
                    ks = [f"{e}.{k}" for k in PROJ]
                    d = FRT.weight_diff({k: W[k] for k in ks}, {k: Wo[L][k] for k in ks})
                    hch = any(not keys[f"{L}.{e}.{pj}"]["bitwise_identical"] for pj in ("gu", "down"))
                    wch = d["n_differ"] > 0
                    cross[f"h_{'changed' if hch else 'same'}_w_{'changed' if wch else 'same'}"] += 1
                    per_expert.append(dict(layer=L, expert=e, rel_l2=d["rel_l2"], frac_differ=d["frac_differ"],
                                           n_tensors_identical=d["n_tensors_identical"], hessian_changed=hch,
                                           hessian_rel_fro_gu=keys[f"{L}.{e}.gu"]["rel_fro"],
                                           hessian_rel_fro_down=keys[f"{L}.{e}.down"]["rel_fro"]))
                    n_id += d["n_tensors_identical"]
                    n += 3
                del W
            tot = dict(n=sum(v["n"] for v in per_layer.values()), n_differ=sum(v["n_differ"] for v in per_layer.values()),
                       d2=sum(v["d2"] for v in per_layer.values()), o2=sum(v["o2"] for v in per_layer.values()))
            tot["frac_differ"] = tot["n_differ"] / tot["n"]
            tot["rel_l2"] = (tot["d2"] / tot["o2"]) ** .5
            out[c][str(s)] = dict(total=tot, per_layer=per_layer, n_projections=n, n_projections_identical=n_id,
                                  n_experts_changed=sum(x["frac_differ"] > 0 for x in per_expert),
                                  hessian_vs_weight_change=cross, per_expert=per_expert,
                                  installed_equal_OO=r["weights_equal_OO"])
            print(f"[report] weights {c} O{s}: {tot['frac_differ']:.4f} rel {tot['rel_l2']:.5f}", flush=True)
        del Wo
    return out


def quantizer_analysis(base, res):
    out = {}
    for c in STRATS:
        Qb = torch.load(base[c]["quantizer_file"])
        out[c] = {}
        for s in SEEDS:
            Q = torch.load(res[c][s]["quantizer_file"])
            rows = []
            for ek, ent in Qb["check_experts"].items():
                for k in PROJ:
                    a, b = ent[k], Q["check_experts"][ek][k]
                    sdiff = (b["scale"] - a["scale"]).abs() / a["scale"].abs()
                    grid_same = (b["scale"] == a["scale"]) & (b["zero"] == a["zero"])           # [groups, rows]
                    cd = b["codes"] != a["codes"]                                               # [rows, cols]
                    rows.append(dict(key=f"{ek}.{k}", frac_scale_differ=float((b["scale"] != a["scale"]).float().mean()),
                                     median_scale_rel_diff=float(sdiff.median()), max_scale_rel_diff=float(sdiff.max()),
                                     frac_zero_differ=float((b["zero"] != a["zero"]).float().mean()),
                                     frac_groups_grid_identical=float(grid_same.float().mean()),
                                     frac_codes_differ=float(cd.float().mean()), n_codes_differ=int(cd.sum()),
                                     max_code_change=int((b["codes"].int() - a["codes"].int()).abs().max())))
            ce = [float(Q["col_err2"][k].sum() / Qb["col_err2"][k].sum()) for k in Qb["col_err2"] if float(Qb["col_err2"][k].sum()) > 0]
            out[c][str(s)] = dict(per_projection=rows,
                                  frac_scale_differ=float(np.mean([r["frac_scale_differ"] for r in rows])),
                                  median_scale_rel_diff=float(np.median([r["median_scale_rel_diff"] for r in rows])),
                                  frac_zero_differ=float(np.mean([r["frac_zero_differ"] for r in rows])),
                                  frac_groups_grid_identical=float(np.mean([r["frac_groups_grid_identical"] for r in rows])),
                                  frac_codes_differ=float(np.mean([r["frac_codes_differ"] for r in rows])),
                                  n_codes_differ=int(sum(r["n_codes_differ"] for r in rows)),
                                  col_err2_ratio_median=float(np.median(ce)), col_err2_ratio_min=float(min(ce)),
                                  col_err2_ratio_max=float(max(ce)))
    return out


def ppl_analysis(man, res):
    out, oo = {}, {}
    for c in STRATS:
        ref = json.load(open(os.path.join(ROOT, man["oo_references"][c]["result"])))
        oo[c] = ref
        out[c] = {"OO": dict(ppl=ref["ppl"], mean_nll=ref["mean_nll"], reference=True)}
        w0 = np.array(ref["per_window_nll"])
        for s in SEEDS:
            r = res[c][s]
            w = np.array(r["per_window_nll"])
            out[c][str(s)] = dict(ppl=r["ppl"], mean_nll=r["mean_nll"], dev_from_OO=r["ppl"] - ref["ppl"],
                                  abs_dev_from_OO=abs(r["ppl"] - ref["ppl"]),
                                  per_window_nll_n_differ=int((w != w0).sum()),
                                  per_window_nll_max_abs_diff=float(np.abs(w - w0).max()),
                                  per_window_nll_mean_abs_diff=float(np.abs(w - w0).mean()),
                                  exceeds_tau=abs(r["ppl"] - ref["ppl"]) > HN.TAU_PPL,
                                  repeat_eval_identical=r["repeat_eval_per_window_identical"])
    return out, oo


def recon_analysis(man, res, oo):
    out = {}
    for c in STRATS:
        ref = oo[c]
        rp = {(d["layer"], d["expert"]): d["mean"] for d in ref["per_expert"]}
        out[c] = {"OO": dict(recon=ref["recon"], per_layer=ref["recon_per_layer"])}
        for s in SEEDS:
            r = res[c][s]
            d = np.array([x["mean"] - rp[(x["layer"], x["expert"])] for x in r["per_expert"]])
            rel = np.array([(x["mean"] - rp[(x["layer"], x["expert"])]) / rp[(x["layer"], x["expert"])] for x in r["per_expert"]])
            out[c][str(s)] = dict(recon=r["recon"], rel_change_vs_OO=(r["recon"] - ref["recon"]) / ref["recon"],
                                  per_layer=r["recon_per_layer"], n_experts_changed=int((d != 0).sum()),
                                  per_expert_max_abs_rel_change=float(np.abs(rel).max()),
                                  per_expert_median_abs_rel_change=float(np.median(np.abs(rel))))
    pen = {"OO": out["FC"]["OO"]["recon"] / out["FR"]["OO"]["recon"] - 1}
    for s in SEEDS:
        pen[str(s)] = out["FC"][str(s)]["recon"] / out["FR"][str(s)]["recon"] - 1
    out["FC_penalty_over_FR"] = pen
    return out


# ---------------------------------------------------------------- figures
def figures(o, executed_quant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figd = p("figures")
    col = {"FR": "#1f77b4", "FC": "#d62728"}
    x = np.arange(len(SEEDS))
    lab = [str(s)[-2:] for s in SEEDS]

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(figd, f"{name}.{ext}"), dpi=150)
        plt.close(fig)

    hd = o["hessian"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, c in enumerate(STRATS):
        keys = load(p("hessian_preflight", f"{c}.json"))["keys"]
        for j, s in enumerate(SEEDS):
            rf = [v["rel_fro"] for v in keys[f"O{s}"].values()]
            axs[0].scatter(np.full(len(rf), j + (i - .5) * .3) + np.random.default_rng(j).uniform(-.08, .08, len(rf)),
                           np.maximum(rf, 1e-12), s=3, alpha=.35, color=col[c], label=c if j == 0 else None)
        axs[1].plot(x, [hd["per_order"][c][f"O{s}"]["frac_elements_changed"] * 100 for s in SEEDS], marker="o", color=col[c], label=c)
        axs[2].plot(x, [np.median([v["median_ulp_changed"] for v in hd["diag_subset"][c][f"O{s}"].values()]) for s in SEEDS],
                    marker="o", color=col[c], label=f"{c} median ULP (changed elements)")
        axs[2].plot(x, [np.max([v["median_ulp_changed"] for v in hd["diag_subset"][c][f"O{s}"].values()]) for s in SEEDS],
                    marker="^", ls=":", color=col[c], label=f"{c} max over the 16 Hessians")
    axs[0].set_yscale("log"); axs[0].set_ylabel("relative Frobenius diff vs original order"); axs[0].set_title("480 Hessians per order")
    axs[1].set_ylabel("% Hessian elements differing bitwise"); axs[1].set_title("bitwise-difference fraction")
    axs[2].set_ylabel("FP32 ULP distance of changed elements"); axs[2].set_title("diagnostic subset (16 Hessians)")
    for a in axs:
        a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("order seed (2026112x)"); a.legend(fontsize=8)
    fig.suptitle("Fig 1 - Hessian sensitivity to FP32 accumulation order (original activations / routing)")
    save(fig, "fig1_hessian_order_sensitivity")

    fp = o.get("fp64")
    if fp:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        xs = np.arange(len(HN.FP64_SEEDS))
        for i, c in enumerate(STRATS):
            if c not in fp:
                continue
            sm = fp[c]["summary"]
            w = .2
            axs[0].bar(xs + (i - .5) * 2 * w - w / 2, [sm[f"O{s}"]["fp32_rel_fro_median"] for s in HN.FP64_SEEDS], w, color=col[c], label=f"{c} FP32")
            axs[0].bar(xs + (i - .5) * 2 * w + w / 2, [max(sm[f"O{s}"]["fp64_rel_fro_median"], 1e-20) for s in HN.FP64_SEEDS], w,
                       color=col[c], alpha=.4, hatch="//", label=f"{c} FP64")
            axs[1].bar(xs + (i - .5) * 2 * w - w / 2, [sm[f"O{s}"]["fp32_frac_elements_changed"] * 100 for s in HN.FP64_SEEDS], w, color=col[c], label=f"{c} FP32")
            axs[1].bar(xs + (i - .5) * 2 * w + w / 2, [sm[f"O{s}"]["fp64_cast_frac_elements_changed"] * 100 for s in HN.FP64_SEEDS], w,
                       color=col[c], alpha=.4, hatch="//", label=f"{c} FP64 -> FP32 cast")
        axs[0].set_yscale("log"); axs[0].set_ylabel("median relative Frobenius diff vs original order")
        axs[1].set_ylabel("% elements differing bitwise (FP32 values)")
        for a in axs:
            a.set_xticks(xs); a.set_xticklabels([str(s) for s in HN.FP64_SEEDS]); a.legend(fontsize=7)
        fig.suptitle("Fig 2 - FP32 vs FP64 accumulation of identical per-sequence contributions")
        save(fig, "fig2_fp32_vs_fp64")

    if not executed_quant:
        return
    wz, pp, pa, rc = o["weights"], o["ppl"], o["paired"], o["reconstruction"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for c in STRATS:
        axs[0].plot(x, [wz[c][str(s)]["total"]["frac_differ"] * 100 for s in SEEDS], marker="o", color=col[c], label=c)
        axs[1].plot(x, [wz[c][str(s)]["total"]["rel_l2"] for s in SEEDS], marker="o", color=col[c], label=c)
    axs[0].axhline(17.4, ls=":", color="grey", label="AOF (earlier, ~17-18 %)")
    axs[1].axhline(0.041, ls=":", color="grey", label="AOF (earlier, ~0.041)")
    axs[0].set_ylabel("% effective weight elements != OO (log)"); axs[1].set_ylabel("relative L2 vs OO, 720 tensors (log)")
    for a in axs:
        a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("order seed (2026112x)"); a.legend(fontsize=8); a.set_yscale("log")
    fig.suptitle("Fig 3 - GPTQ effective-weight sensitivity to accumulation order")
    save(fig, "fig3_weight_sensitivity")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, c in enumerate(STRATS):
        ax.plot(x + (i - .5) * .1, [pp[c][str(s)]["ppl"] for s in SEEDS], "o", color=col[c], label=f"{c} shuffled orders")
        ax.axhline(pp[c]["OO"]["ppl"], color=col[c], ls="--", lw=1, label=f"{c} OO (original order)")
        ax.axhspan(pp[c]["OO"]["ppl"] - HN.TAU_PPL, pp[c]["OO"]["ppl"] + HN.TAU_PPL, color=col[c], alpha=.08)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("order seed (2026112x)"); ax.set_ylabel("WikiText-2 PPL")
    ax.legend(fontsize=8, ncol=2); ax.set_title("Fig 4 - PPL across accumulation orders (band: OO +/- tau_PPL)")
    save(fig, "fig4_ppl")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, pa["delta"], "o-", color="k", label="Delta = FC - FR (shuffled orders)")
    ax.axhline(pa["delta_oo"], color="purple", ls="--", label=f"Delta_OO = {pa['delta_oo']:+.6f}")
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("order seed (2026112x)"); ax.set_ylabel("PPL difference")
    ax.legend(fontsize=8); ax.set_title(f"Fig 5 - paired strategy difference (SD {pa['sd_delta']:.5f})")
    save(fig, "fig5_paired_delta")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for c in STRATS:
        h = [o["hessian"]["per_order"][c][f"O{s}"]["rel_fro_median"] for s in SEEDS]
        w = [wz[c][str(s)]["total"]["rel_l2"] for s in SEEDS]
        axs[0].scatter(h, w, color=col[c], label=c)
        axs[1].scatter(w, [rc[c][str(s)]["rel_change_vs_OO"] * 100 for s in SEEDS], color=col[c], label=c)
        axs[2].scatter(w, [pp[c][str(s)]["dev_from_OO"] for s in SEEDS], color=col[c], label=c)
    axs[0].set_xlabel("median Hessian rel. Frobenius diff"); axs[0].set_ylabel("weight rel. L2 vs OO"); axs[0].set_xscale("log")
    axs[1].set_xlabel("weight rel. L2 vs OO"); axs[1].set_ylabel("reconstruction change vs OO (%)")
    axs[2].set_xlabel("weight rel. L2 vs OO"); axs[2].set_ylabel("PPL - PPL_OO")
    for v in (HN.TAU_PPL, -HN.TAU_PPL):
        axs[2].axhline(v, color="grey", ls=":")
    for a in axs:
        a.legend(fontsize=8); a.tick_params(axis="x", labelrotation=35, labelsize=8)
    for a in axs[1:]:
        a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("Fig 6 - Hessian perturbation vs weight, reconstruction and PPL change (10 configurations)")
    save(fig, "fig6_hessian_to_weight_ppl")


# ---------------------------------------------------------------- build
def build():
    man = load(p("input_manifest.json"))
    pr = load(p("protocol.json"))
    base = {c: load(p("baseline_regression", f"{c}.json")) for c in STRATS}
    jdump(dict(strategies={c: {k: v for k, v in (base[c] or {}).items() if k not in ("per_window_nll", "per_window_tokens", "per_expert",
                                                                                     "integrity_checks", "frozen_old_check", "replay_order", "blocks")}
                           for c in STRATS},
               passed=all((base[c] or {}).get("passed") for c in STRATS)), p("baseline_regression.json"))
    hd = XH.aggregate_hessian() or load(p("hessian_differences.json"))
    hpre = {c: load(p("hessian_preflight", f"{c}.json")) for c in STRATS}
    fp = {c: load(p("fp64_control", f"{c}.json")) for c in STRATS}
    fp = {c: v for c, v in fp.items() if v}
    if fp:
        jdump({c: dict(summary=v["summary"], design=v["design"], seconds=v["seconds"]) for c, v in fp.items()}, p("fp64_control.json"))
    blockers = []
    if not all((base[c] or {}).get("passed") for c in STRATS):
        blockers.append("baseline regression")
    if pr is None:
        blockers.append("protocol not frozen")
    if hd is None or not hd["checks_passed"]:
        blockers.append("Hessian preflight incomplete or invariance checks failed")
    o = dict(protocol_sha256=pr and pr["protocol_sha256"], hessian=hd, fp64={c: dict(summary=v["summary"]) for c, v in fp.items()})
    executed = False
    res = {c: {s: result(c, s) for s in SEEDS} for c in STRATS}
    rep = None
    if hd is not None and hd["checks_passed"] and not hd["early_stop"] and not blockers:
        missing = [XH.cname(c, s) for c in STRATS for s in SEEDS if res[c][s] is None]
        invalid = [XH.cname(c, s) for c in STRATS for s in SEEDS if res[c][s] is not None and not res[c][s]["valid"]]
        wrong_proto = [XH.cname(c, s) for c in STRATS for s in SEEDS if res[c][s] is not None
                       and res[c][s]["installed_weights_sha256"] is None]
        if missing:
            blockers.append(f"missing configurations: {missing}")
        if invalid or wrong_proto:
            blockers.append(f"invalid configurations: {invalid + wrong_proto}")
        rc = hd["repeat_config"]
        if rc:
            c0, s0 = rc
            r1, r2 = res[c0][s0], result(c0, s0, repeat=True)
            if r1 is None or r2 is None:
                blockers.append("repeatability control missing")
            else:
                rep = dict(config=XH.cname(c0, s0), weights_identical=r1["installed_weights_sha256"] == r2["installed_weights_sha256"],
                           weight_files_identical=all(r1["weight_files"][L]["sha256"] == r2["weight_files"][L]["sha256"] for L in r1["weight_files"]),
                           recon_identical=r1["recon"] == r2["recon"],
                           per_expert_identical=[d["mean"] for d in r1["per_expert"]] == [d["mean"] for d in r2["per_expert"]],
                           ppl_identical=r1["ppl"] == r2["ppl"], per_window_nll_identical=r1["per_window_nll"] == r2["per_window_nll"],
                           repeat_valid=r2["valid"])
                rep["passed"] = all(v for k, v in rep.items() if k != "config")
                if not rep["passed"]:
                    blockers.append("repeatability control differs")
        executed = not missing
    o["repeatability"] = rep
    if executed:
        o["weights"] = weight_analysis(man, hpre, res)
        o["quantizer"] = quantizer_analysis(base, res)
        o["ppl"], oo = ppl_analysis(man, res)
        o["reconstruction"] = recon_analysis(man, res, oo)
        o["paired"] = HN.paired([res["FR"][s]["ppl"] for s in SEEDS], [res["FC"][s]["ppl"] for s in SEEDS],
                                oo["FR"]["ppl"], oo["FC"]["ppl"])
        o["paired"]["delta_change_vs_OO"] = [d - o["paired"]["delta_oo"] for d in o["paired"]["delta"]]
        o["paired"]["n_delta_change_exceeding_tau"] = int(sum(abs(x) > HN.TAU_PPL for x in o["paired"]["delta_change_vs_OO"]))
        o["paired"]["sd_mean_nll"] = dict(FR=HN.sd([res["FR"][s]["mean_nll"] for s in SEEDS]),
                                          FC=HN.sd([res["FC"][s]["mean_nll"] for s in SEEDS]))
        devs = [o["ppl"][c][str(s)]["abs_dev_from_OO"] for c in STRATS for s in SEEDS]
        w_all_equal = all(res[c][s]["weights_equal_OO"] for c in STRATS for s in SEEDS)
        o["context_previous_sd"] = CONTEXT
        o["scale_comparison_vs_AOF"] = {k: o["paired"][f"sd_{k.lower()}"] / CONTEXT["frozen routing + act-order (AOF)"][k]
                                        for k in ("FR", "FC", "Delta")}
        o["affected"] = dict(FR=any(o["ppl"]["FR"][str(s)]["exceeds_tau"] for s in SEEDS),
                             FC=any(o["ppl"]["FC"][str(s)]["exceeds_tau"] for s in SEEDS),
                             delta_changes_materially=o["paired"]["n_delta_change_exceeding_tau"] > 0,
                             ranking_sign_reversals=o["paired"]["sign_reversals"])
    else:
        devs, w_all_equal = [], True
    any_diff = bool(hd and hd["any_hessian_differs"])
    decision = HN.decide(bool(blockers), any_diff, w_all_equal, devs)
    o.update(decision=decision, blockers=blockers, decision_rule=HN.DECISION_RULE, tau_ppl=HN.TAU_PPL, executed_quant=executed)
    o["compute"] = compute(base, hpre, fp, res if executed else None)
    for k, name in (("weights", "weight_differences.json"), ("quantizer", "quantizer_diagnostics.json"), ("ppl", "ppl_results.json"),
                    ("reconstruction", "reconstruction_results.json"), ("paired", "paired_differences.json")):
        if k in o:
            body = o[k] if k != "weights" else {c: {s: {kk: vv for kk, vv in v.items() if kk != "per_expert"} for s, v in o[k][c].items()} for c in STRATS}
            jdump(body, p(name))
    if executed:
        jdump({c: {s: v["per_expert"] for s, v in o["weights"][c].items()} for c in STRATS}, p("logs", "weight_per_expert.json"))
    return o


def compute(base, hpre, fp, res):
    out = dict(baseline_quantize_s={c: base[c]["quantize_s"] for c in STRATS if base[c]},
               hessian_replay_s={c: sum(v["replay_seconds"].values()) for c, v in hpre.items() if v},
               fp64_s={c: v["seconds"] for c, v in fp.items()},
               peak_gpu_gb=max([b["peak_gpu_gb"] for b in base.values() if b] + [v["peak_gpu_gb"] for v in hpre.values() if v]),
               peak_cpu_gb=max([b["peak_cpu_gb"] for b in base.values() if b] + [v["peak_cpu_gb"] for v in hpre.values() if v]))
    if res:
        rr = [r for c in STRATS for r in res[c].values()]
        out.update(n_completed_configurations=len(rr), quantize_s=[min(r["quantize_s"] for r in rr), max(r["quantize_s"] for r in rr)],
                   hessian_s_in_quant=[min(r["hessian_s"] for r in rr), max(r["hessian_s"] for r in rr)],
                   ppl_eval_s=[min(r["eval_s"] for r in rr), max(r["eval_s"] for r in rr)],
                   peak_gpu_gb=max(out["peak_gpu_gb"], max(r["peak_gpu_gb"] for r in rr)),
                   peak_cpu_gb=max(out["peak_cpu_gb"], max(r["peak_cpu_gb"] for r in rr)))
    du = 0
    for dp, _, fs in os.walk(XH.HCACHE):
        du += sum(os.path.getsize(os.path.join(dp, f)) for f in fs)
    out["disk_gb"] = du / 1e9
    return out


def markdown(o):
    L = [f"# FP32 Hessian accumulation-order null experiment - summary", "",
         f"Decision: **{o['decision']}**" + (f" (blockers: {o['blockers']})" if o["blockers"] else ""),
         f"Protocol sha256 {str(o['protocol_sha256'])[:16]}..., tau_PPL = {HN.TAU_PPL:.6f}", ""]
    hd = o["hessian"]
    if hd:
        L += ["## Hessians (480 per strategy and order; vs original order)", "",
              "| strategy | order | Hessians changed | elements changed | median rel Fro | max rel Fro | max abs |",
              "|---|---|---|---|---|---|---|"]
        for c in STRATS:
            for s in SEEDS:
                v = hd["per_order"][c][f"O{s}"]
                L.append(f"| {c} | {s} | {v['n_hessians_changed']}/480 | {v['frac_elements_changed'] * 100:.2f} % | "
                         f"{v['rel_fro_median']:.2e} | {v['rel_fro_max']:.2e} | {v['max_abs']:.3g} |")
        L.append("")
    if o["fp64"]:
        L += ["## FP64 control", "", "| strategy | order | FP32 median rel Fro | FP64 median rel Fro | FP64->FP32 cast elements changed |", "|---|---|---|---|---|"]
        for c, v in o["fp64"].items():
            for s in HN.FP64_SEEDS:
                sm = v["summary"][f"O{s}"]
                L.append(f"| {c} | {s} | {sm['fp32_rel_fro_median']:.2e} | {sm['fp64_rel_fro_median']:.2e} | {sm['fp64_cast_frac_elements_changed'] * 100:.4f} % |")
        L.append("")
    if o["executed_quant"]:
        L += ["## PPL", "", "| order | FR PPL | FC PPL | FC - FR |", "|---|---|---|---|",
              f"| original (OO) | {o['ppl']['FR']['OO']['ppl']:.6f} | {o['ppl']['FC']['OO']['ppl']:.6f} | {o['paired']['delta_oo']:+.6f} |"]
        for i, s in enumerate(SEEDS):
            L.append(f"| {s} | {o['ppl']['FR'][str(s)]['ppl']:.6f} | {o['ppl']['FC'][str(s)]['ppl']:.6f} | {o['paired']['delta'][i]:+.6f} |")
        pa = o["paired"]
        L += ["", f"SD FR {pa['sd_fr']:.5f}, SD FC {pa['sd_fc']:.5f}, SD Delta {pa['sd_delta']:.5f}; sign reversals {pa['sign_reversals']}/5", "",
              "## Weights vs OO", "", "| strategy | order | elements != OO | rel L2 | identical projections | experts changed |", "|---|---|---|---|---|---|"]
        for c in STRATS:
            for s in SEEDS:
                w = o["weights"][c][str(s)]
                L.append(f"| {c} | {s} | {w['total']['frac_differ'] * 100:.4f} % | {w['total']['rel_l2']:.5f} | {w['n_projections_identical']}/720 | {w['n_experts_changed']}/240 |")
    return "\n".join(L) + "\n"


def main():
    o = build()
    figures(o, o["executed_quant"])
    summ = {k: v for k, v in o.items() if k not in ("weights",)}
    summ["created"] = time.strftime("%F %T")
    jdump(summ, p("summary.json"))
    open(p("summary.md"), "w").write(markdown(o))
    print(f"[report] decision {o['decision']} blockers {o['blockers']}")


if __name__ == "__main__":
    main()
