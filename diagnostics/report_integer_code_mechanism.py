"""Analysis, decisions, figures and summary for runs_integer_code_mechanism."""
from __future__ import annotations
import glob, json, os

import numpy as np
import torch

import integer_code_mechanism as S
from moe_calib import code_spatial as CS

p, jload, jdump = S.p, S.jload, S.jdump
PROJ, DLAY, MLAY, GROUPS = S.PROJ, S.DLAY, S.MLAY, S.GROUPS
COND = {"B_with_grid_change": "cat_B", "C_same_level": "cat_C", "D_same_grid": "cat_D",
        "frac_bf16_changed_same_code": "frac_bf16_changed", "frac_code_changed_same_bf16": "frac_code_changed",
        "bf16_changed_grid_only_a_on_b": "frac_bf16_changed", "bf16_changed_grid_only_b_on_a": "frac_bf16_changed"}
UNCOND = ["frac_code_changed", "frac_bf16_changed", "frac_level_changed", "cat_A", "cat_B", "cat_C", "cat_D",
          "frac_scale_changed", "frac_zero_changed", "frac_grid_changed", "frac_group_changed", "code_changed_same_grid",
          "code_changed_changed_grid", "code_changed_same_group", "code_changed_changed_group"]


def wagg(ds):
    """Element-weighted aggregate of CS.decompose dicts (conditional shares weighted by their denominators)."""
    n = sum(d["n"] for d in ds)
    out = dict(n=n)
    for k in UNCOND:
        out[k] = sum(d[k] * d["n"] for d in ds) / n
    for k, den in COND.items():
        w = sum(d[den] * d["n"] for d in ds)
        out[k] = sum(d[k] * d[den] * d["n"] for d in ds) / w if w else float("nan")
    out["median_rel_scale_diff_changed_median"] = float(np.median([d["median_rel_scale_diff_changed"] for d in ds]))
    return out


# ---------------------------------------------------------------- decompositions
def dense_decomp():
    out = {}
    for va, rf in S.DENSE_PRIMARY + [S.DENSE_NULL] + S.DENSE_SECONDARY:
        for L in DLAY:
            ra, rb, wa, wb = S.dense_rec(va, L), S.dense_rec(rf, L), S.dense_w(va, L), S.dense_w(rf, L)
            for k in PROJ:
                d = CS.decompose(ra[k], rb[k], wa[k], wb[k])
                ap = group_pos(ra[k], rb[k])
                d.update(ap)
                out[f"{S.pkey(va, rf)}|L{L}|{k}"] = d
    return out


def group_pos(ra, rb):
    C = ra["group"].numel()
    pa, pb = torch.empty(C, dtype=torch.long), torch.empty(C, dtype=torch.long)
    pa[ra["perm"].long()] = torch.arange(C); pb[rb["perm"].long()] = torch.arange(C)
    return dict(frac_columns_position_changed=float((pa != pb).float().mean()),
                frac_columns_group_changed_cols=float((ra["group"] != rb["group"]).float().mean()))


def moe_decomp(em):
    out, W = {}, {}
    for g in GROUPS + ["ORIGINAL"]:
        W[g] = {L: S.moe_rec(g, L) for L in MLAY}
    for va, rf in S.MOE_PRIMARY + S.MOE_SECONDARY:
        for x in em["experts"]:
            L, e = x["layer"], x["expert"]
            for k in PROJ:
                ra, rb = W[va][L][e][k], W[rf][L][e][k]
                d = CS.decompose(ra, rb, CS.dequant(ra).to(torch.bfloat16), CS.dequant(rb).to(torch.bfloat16))
                d.update(group_pos(ra, rb))
                d.update(tokens=x["tokens_FR_ORIGINAL"], role=x["role"])
                out[f"{S.pkey(va, rf)}|L{L}|e{e}|{k}"] = d
    return out


def weight_metrics(pairs, loader, keys):
    import grouping_routing_amplification as GRA
    out = {}
    for va, rf in pairs:
        A, B = loader(va), loader(rf)
        out[S.pkey(va, rf)] = GRA.weight_diff({k: A[k] for k in keys}, {k: B[k] for k in keys})
    return out


# ---------------------------------------------------------------- spatial summaries
def spatial(pr):
    cells, nulls = {}, {}
    for f in sorted(glob.glob(p("logs", "dense_spatial", "*.json"))):
        d = json.load(open(f))
        key = S.pkey(*d["pair"])
        for ck, c in d["cells"].items():
            o = c["obs"]
            cells[f"{key}|{ck}"] = dict(role=d["role"], p=c["p"], flips=o["flips"], raw_E=c["raw_E"],
                                        excess_vs_segnull=c["excess_vs_segnull"], excess_vs_rownull=c["excess_vs_rownull"],
                                        runs=o.get("runs"), max_run=o.get("max_run"), row_dispersion=o.get("row_dispersion"),
                                        rows_with_flip=o.get("rows_with_flip"), interflip_hist=o.get("interflip_hist"),
                                        pairs_cross=o["pairs_cross"], edge_first=o["edge_first"], edge_last=o["edge_last"],
                                        windows={str(L): o[f"win_{L}"] for L in CS.WIN}, strata=c["strata"],
                                        group_change=c.get("group_change"), position_profile=o.get("position_profile"))
            if "null" in c:
                nulls[f"{key}|{ck}"] = dict(n_perm=c["n_perm"], seed=c["seed"], **c["null"])
    # classification (prespecified)
    prim = [k for k in cells if cells[k]["role"] == "primary" and k.split("|")[-1] in ("ref_exec", "var_exec")]
    def clustered(k):
        n = nulls.get(k, {}).get("E_1")
        return bool(n and n["obs"] > n["q995"] and n["ratio_to_null_mean"] >= 1.25)
    def boundary(k):
        hit = False
        for e in ("edge_first", "edge_last"):
            n = nulls.get(k, {}).get(e)
            if n and n["obs"] > n["q995"] and n["ratio_to_null_mean"] >= 1.25:
                hit = True
        return hit
    few = sum(cells[k]["flips"] < 200 for k in prim) / len(prim)
    fr = {o: np.mean([clustered(k) for k in prim if k.endswith(o)]) for o in ("ref_exec", "var_exec")}
    fproj = {pj: np.mean([clustered(k) for k in prim if f"|{pj}|" in k]) for pj in PROJ}
    fb = np.mean([boundary(k) for k in prim])
    if few > 0.5:
        cls = "INCONCLUSIVE"
    elif min(fr.values()) >= 0.75 and min(fproj.values()) >= 0.5:
        cls = "SPATIAL-CLUSTERING-SUPPORTED"
    elif max(fr.values()) <= 0.10 and fb < 0.5:
        cls = "SPATIAL-CLUSTERING-NOT-DETECTED"
    else:
        cls = "MIXED-SPATIAL-STRUCTURE"
    flags = {k: dict(clustered=clustered(k), boundary=boundary(k)) for k in prim}
    dec = dict(classification=cls, frac_cells_clustered_by_order=fr, frac_cells_clustered_by_projection=fproj,
               frac_cells_boundary=fb, frac_cells_lt200_flips=few, n_primary_cells=len(prim), cell_flags=flags,
               rule=pr["criteria"])
    return cells, nulls, dec


# ---------------------------------------------------------------- analysis
def analyze():
    pr = S.check_protocol()
    em = jload(p("moe_expert_manifest.json"))
    al = jload(p("dense_code_alignment_validation.json"))
    cells, nulls, dec = spatial(pr)
    jdump(dict(cells=cells, decision=dec,
               note="primary = 4 single-row perturbations vs ORIGINAL (1000 permutations); secondary = 6 grouping pairs "
                    "(200); null = G20261115 vs ORIGINAL (must be all-zero)"), p("dense_spatial_statistics.json"))
    jdump(dict(nulls=nulls, construction=pr["spatial"]["null"]), p("dense_permutation_nulls.json"))
    # MoE replay validation
    rep = {g: jload(p("logs", "moe_replay", f"{g}.json")) for g in ["ORIGINAL"] + GROUPS}
    moe_ok = all(r and r["all_parity"] for r in rep.values())
    jdump(dict(configs={g: dict(all_parity=r["all_parity"] if r else None, first_failure=r["first_failure"] if r else None,
                                n_tensors=3 * len(r["experts"]) if r else 0,
                                parity=r["experts"] if r else None) for g, r in rep.items()},
               all_parity=moe_ok, blocked=not moe_ok,
               requirement="replayed W_q.to(bf16) == historical saved BF16 weights bitwise for every sampled tensor"),
          p("moe_code_replay_validation.json"))
    # decompositions
    dd = dense_decomp()
    md = moe_decomp(em) if moe_ok else {}
    def dsel(pairs, pj=None, L=None):
        return [v for k, v in dd.items() if any(k.startswith(S.pkey(*pp) + "|") for pp in pairs)
                and (pj is None or k.endswith(pj)) and (L is None or f"|L{L}|" in k)]
    def msel(pairs, pj=None, L=None, filt=None):
        return [v for k, v in md.items() if any(k.startswith(S.pkey(*pp) + "|") for pp in pairs)
                and (pj is None or k.endswith(pj)) and (L is None or f"|L{L}|" in k) and (filt is None or filt(k, v))]
    dense_summary = dict(
        primary_vs_original=wagg(dsel(S.DENSE_PRIMARY)), null=wagg(dsel([S.DENSE_NULL])),
        grouping_pairs=wagg(dsel(S.DENSE_SECONDARY)), all_distinct_10=wagg(dsel(S.DENSE_PRIMARY + S.DENSE_SECONDARY)),
        by_projection={pj: wagg(dsel(S.DENSE_PRIMARY, pj)) for pj in PROJ},
        by_layer_projection={f"L{L}|{pj}": wagg(dsel(S.DENSE_PRIMARY, pj, L)) for L in DLAY for pj in PROJ},
        by_pair={S.pkey(*pp): wagg(dsel([pp])) for pp in S.DENSE_PRIMARY + S.DENSE_SECONDARY})
    out_c = dict(dense=dense_summary, per_tensor_dense=dd)
    if moe_ok:
        terc = np.quantile([x["tokens_FR_ORIGINAL"] for x in em["experts"]], [1 / 3, 2 / 3])
        def tok_bin(v):
            t = v["tokens"]
            return "low" if t <= terc[0] else ("mid" if t <= terc[1] else "high")
        per_expert = {}
        for x in em["experts"]:
            L, e = x["layer"], x["expert"]
            per_expert[f"L{L}|e{e}"] = dict(tokens=x["tokens_FR_ORIGINAL"], role=x["role"],
                                            **{pj: wagg(msel(S.MOE_PRIMARY, pj, filt=lambda k, v, L=L, e=e: f"|L{L}|e{e}|" in k))
                                               for pj in PROJ})
        rates = np.array([v[pj]["frac_code_changed"] for v in per_expert.values() for pj in PROJ])
        moe_summary = dict(
            primary_vs_original=wagg(msel(S.MOE_PRIMARY)), grouping_pairs=wagg(msel(S.MOE_SECONDARY)),
            by_projection={pj: wagg(msel(S.MOE_PRIMARY, pj)) for pj in PROJ},
            by_layer={str(L): wagg(msel(S.MOE_PRIMARY, L=L)) for L in MLAY},
            by_layer_projection={f"L{L}|{pj}": wagg(msel(S.MOE_PRIMARY, pj, L)) for L in MLAY for pj in PROJ},
            by_token_tertile={b: wagg(msel(S.MOE_PRIMARY, filt=lambda k, v, b=b: tok_bin(v) == b)) for b in ("low", "mid", "high")},
            token_tertile_edges=terc.tolist(),
            by_pair={S.pkey(*pp): wagg(msel([pp])) for pp in S.MOE_PRIMARY + S.MOE_SECONDARY},
            per_expert=per_expert,
            expert_heterogeneity=dict(code_rate_min=float(rates.min()), code_rate_median=float(np.median(rates)),
                                      code_rate_max=float(rates.max()), note="per expert x projection, primary pairs pooled"),
            coverage=em["coverage"])
        out_c["moe"] = moe_summary
        out_c["per_tensor_moe"] = md
        # weight metrics (identical Family D definition) on the sampled experts
        keys = [(x["layer"], x["expert"], k) for x in em["experts"] for k in PROJ]
        def mload(g):
            R = {L: S.moe_rec(g, L) for L in MLAY}
            return {(L, e, k): CS.dequant(R[L][e][k]).to(torch.bfloat16) for (L, e, k) in keys}
        out_c["moe_weight_metrics_sampled"] = weight_metrics(S.MOE_PRIMARY + S.MOE_SECONDARY, mload, keys)
    jdump(out_c, p("moe_code_difference_analysis.json"))
    jdump(dict(dense=dense_summary, moe=out_c.get("moe"),
               categories="A same code & same BF16; B same code & diff BF16; C diff code & same BF16; D diff code & diff BF16",
               note="counterfactual (a's codes on b's grid) shares are order-dependent; not an additive decomposition"),
          p("code_grid_decomposition.json"))
    # dense vs MoE comparison
    cmpd = None
    if moe_ok:
        D, M = dense_summary["primary_vs_original"], out_c["moe"]["primary_vs_original"]
        Rq = M["frac_code_changed"] / D["frac_code_changed"]
        Rqp = {pj: out_c["moe"]["by_projection"][pj]["frac_code_changed"] / dense_summary["by_projection"][pj]["frac_code_changed"]
               for pj in PROJ}
        if Rq <= 3 and all(v <= 3 for v in Rqp.values()):
            cc = "A-SIMILAR-CODE-DISAGREEMENT"
        elif Rq > 3 and all(v > 3 for v in Rqp.values()):
            cc = "B-MUCH-LARGER-MOE-CODE-DISAGREEMENT"
        else:
            cc = "C-PROJECTION-DEPENDENT"
        hist_d = jload(os.path.join(S.ROOT, S.DENSE_RUN, "dense_moe_comparison.json"))
        hs_d = jload(os.path.join(S.ROOT, S.DENSE_RUN, "hessian_sensitivity.json"))
        mw = out_c["moe_weight_metrics_sampled"]
        mwp = [mw[S.pkey(*pp)] for pp in S.MOE_PRIMARY]
        rows = {
            "model": ["Qwen2.5-1.5B (12 MLP tensors, all elements)", f"Qwen1.5-MoE-A2.7B ({em['coverage']}, 60 tensors)"],
            "comparisons": ["4 single-row perturbations vs ORIGINAL", "5 groupings vs ORIGINAL"],
            "bf16_changed": [D["frac_bf16_changed"], M["frac_bf16_changed"]],
            "integer_code_changed": [D["frac_code_changed"], M["frac_code_changed"]],
            "zero_relative_level_changed": [D["frac_level_changed"], M["frac_level_changed"]],
            "same_code_given_bf16_changed": [D["frac_bf16_changed_same_code"], M["frac_bf16_changed_same_code"]],
            "same_bf16_given_code_changed": [D["frac_code_changed_same_bf16"], M["frac_code_changed_same_bf16"]],
            "code_changed_on_identical_grid": [D["code_changed_same_grid"], M["code_changed_same_grid"]],
            "scale_changed_elements": [D["frac_scale_changed"], M["frac_scale_changed"]],
            "zero_changed_elements": [D["frac_zero_changed"], M["frac_zero_changed"]],
            "group_assignment_changed_elements": [D["frac_group_changed"], M["frac_group_changed"]],
            "actorder_position_changed_columns": [float(np.mean([v["frac_columns_position_changed"] for v in dsel(S.DENSE_PRIMARY)])),
                                                  float(np.mean([v["frac_columns_position_changed"] for v in msel(S.MOE_PRIMARY)]))],
            "bf16_diff_explained_by_grid_a_on_b": [D["bf16_changed_grid_only_a_on_b"], M["bf16_changed_grid_only_a_on_b"]],
            "rel_frobenius_sampled": [None, float(np.mean([v["rel_frobenius"] for v in mwp]))],
            "ge8ulp_share_sampled": [None, float(np.mean([v["ulp_share_of_changed"][">=8"] for v in mwp]))],
            "historical_whole_model_bf16_changed": [hist_d["rows"]["bf16_elements_changed_vs_original"][0],
                                                    hist_d["rows"]["bf16_elements_changed_vs_original"][1]],
            "historical_hessian_rel_fro_tn": [hs_d["vs_original"]["all"]["rel_fro_trace_normalized"]["median"],
                                              "AO 0.0039 / AA 0.0141 (16 diagnostic Hessians)"],
            "historical_sd_ppl_5_groupings": [hist_d["rows"]["within_seed_sd_ppl"][0], hist_d["rows"]["within_seed_sd_ppl"][1]],
        }
        cmpd = dict(rows=rows, columns=["dense", "MoE Family D (sampled experts)"], R_q=Rq, R_q_by_projection=Rqp,
                    R_bf16=M["frac_bf16_changed"] / D["frac_bf16_changed"], outcome=cc, rule=pr["criteria"]["code_comparison"])
        jdump(cmpd, p("dense_moe_code_comparison.json"))
    fx = jload(p("logs", "fixed_actorder.json"))
    figures(cells, nulls, dense_summary, out_c.get("moe"), dd, md, fx)
    write_summary(pr, al, cells, nulls, dec, dense_summary, out_c, cmpd, rep, moe_ok, fx, em)
    print(json.dumps(dict(spatial=dec["classification"], by_order=dec["frac_cells_clustered_by_order"],
                          by_proj=dec["frac_cells_clustered_by_projection"], boundary=dec["frac_cells_boundary"],
                          code=cmpd and cmpd["outcome"], R_q=cmpd and cmpd["R_q"]), indent=1, default=float))


# ---------------------------------------------------------------- figures
def figures(cells, nulls, ds, ms, dd, md, fx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = p("figures")
    # Fig 1: masks
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, (L, k) in enumerate(((9, "down_proj"), (19, "down_proj"), (9, "gate_proj"))):
        ra, rb = S.dense_rec("G20261111", L)[k], S.dense_rec("ORIGINAL", L)[k]
        F = (ra["q"] != rb["q"])
        Fe = CS.to_exec(F, rb["perm"])
        for j, (M, lab) in enumerate(((F, "original column order"), (Fe, "ORIGINAL act-order (execution)"))):
            ax = axs[j, i]
            ax.imshow(M[:256, :1024].float().numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
            for x in range(128, 1024, 128):
                ax.axvline(x - .5, color="tab:red", lw=.4, alpha=.6)
            ax.set_title(f"L{L} {k}: {lab}\n(rows 0-255, cols 0-1023; flip rate {float(F.float().mean()):.3f})", fontsize=8)
    fig.suptitle("Dense code-flip masks, G20261111 vs ORIGINAL (red: 128-column segments)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig1_dense_flip_masks.png"), dpi=140); plt.close(fig)
    # Fig 2: E1 obs vs null + lag excess
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    prim = [k for k in nulls if cells[k]["role"] == "primary"]
    cols = {"gate_proj": "tab:blue", "up_proj": "tab:orange", "down_proj": "tab:green"}
    for o, mk in (("original", "x"), ("ref_exec", "o"), ("var_exec", "s")):
        for k in prim:
            if not k.endswith(o):
                continue
            n = nulls[k]["E_1"]
            pj = k.split("|")[2]
            axs[0].errorbar(n["null_mean"], n["obs"], yerr=None, fmt=mk, color=cols[pj], ms=4, alpha=.7)
    lim = [0.5, max(max(nulls[k]["E_1"]["obs"] for k in prim), 2) * 1.05]
    axs[0].plot(lim, lim, "k--", lw=.8); axs[0].plot(lim, [1.25 * x for x in lim], ":", color="grey", lw=.8)
    axs[0].set_xlabel("null mean E1 (row x segment counts preserved)"); axs[0].set_ylabel("observed E1")
    axs[0].set_title("adjacent-flip enrichment; o ref exec, s var exec, x original (color = projection)", fontsize=8)
    lags = [str(d) for d in CS.LAGS]
    for pj in PROJ:
        for o, ls in (("ref_exec", "-"), ("original", ":")):
            ks = [k for k in prim if k.endswith(o) and f"|{pj}|" in k]
            y = [np.mean([cells[k]["excess_vs_segnull"][d] for k in ks]) for d in lags]
            axs[1].plot(CS.LAGS, y, ls, color=cols[pj], marker="o", ms=3, label=f"{pj} {o}")
    axs[1].axhline(1, color="k", lw=.6); axs[1].set_xscale("log", base=2)
    axs[1].set_xlabel("lag d (within 128-position segment)"); axs[1].set_ylabel("observed / segment-null lag-d pairs")
    axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig2_clustering_vs_null.png"), dpi=140); plt.close(fig)
    # Fig 3: runs / windows vs null
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for pj in PROJ:
        ks = [k for k in prim if k.endswith("ref_exec") and f"|{pj}|" in k]
        r = [[nulls[k][f"win_{L}"]["obs"] / max(nulls[k][f"win_{L}"]["null_mean"], 1e-9) for k in ks] for L in CS.WIN]
        axs[0].plot(CS.WIN, [np.median(x) for x in r], marker="o", color=cols[pj], label=pj)
        lo = [np.median([nulls[k][f"win_{L}"]["q005"] / max(nulls[k][f"win_{L}"]["null_mean"], 1e-9) for k in ks]) for L in CS.WIN]
        hi = [np.median([nulls[k][f"win_{L}"]["q995"] / max(nulls[k][f"win_{L}"]["null_mean"], 1e-9) for k in ks]) for L in CS.WIN]
        axs[0].fill_between(CS.WIN, lo, hi, color=cols[pj], alpha=.12)
    axs[0].set_yscale("log"); axs[0].axhline(1, color="k", lw=.6)
    axs[0].set_xlabel("run window length L"); axs[0].set_ylabel("observed / null mean (median over cells)")
    axs[0].set_title("all-flip windows (ref exec order); band = null 0.5-99.5 % (median)", fontsize=8); axs[0].legend(fontsize=7)
    labs = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"]
    for pj in PROJ:
        ks = [k for k in prim if k.endswith("ref_exec") and f"|{pj}|" in k and cells[k]["interflip_hist"]]
        h = np.sum([[cells[k]["interflip_hist"][b] for b in labs] for k in ks], 0)
        axs[1].plot(range(len(labs)), h / max(h.sum(), 1), marker="o", color=cols[pj], label=pj)
    axs[1].set_xticks(range(len(labs)), labs, fontsize=7); axs[1].set_xlabel("distance between consecutive flips in a row")
    axs[1].set_ylabel("share"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig3_flip_runs.png"), dpi=140); plt.close(fig)
    if ms is None:
        return
    # Fig 4: dense vs MoE bf16 vs codes
    fig, ax = plt.subplots(figsize=(8, 4))
    mets = [("frac_bf16_changed", "BF16 changed"), ("frac_code_changed", "integer code changed"),
            ("frac_level_changed", "level q-zero changed"), ("code_changed_same_grid", "code changed, same grid")]
    x = np.arange(len(mets))
    ax.bar(x - .2, [ds["primary_vs_original"][m] for m, _ in mets], .4, label="dense (4 vs ORIGINAL, all elements)", color="#2a6fbb")
    ax.bar(x + .2, [ms["primary_vs_original"][m] for m, _ in mets], .4, label="MoE (5 vs ORIGINAL, 20 sampled experts)", color="#d1603d")
    ax.set_xticks(x, [l for _, l in mets], fontsize=8); ax.set_ylabel("fraction of weight elements"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig4_dense_vs_moe_codes.png"), dpi=140); plt.close(fig)
    # Fig 5: categories
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, (lab, s) in enumerate((("dense", ds["primary_vs_original"]), ("MoE", ms["primary_vs_original"]))):
        b = 0
        for cat, col in (("cat_B", "#9ecae1"), ("cat_C", "#fdae6b"), ("cat_D", "#d62728")):
            axs[0].bar(i, s[cat], bottom=b, color=col, label=cat if i == 0 else None); b += s[cat]
    axs[0].set_xticks([0, 1], ["dense", "MoE"]); axs[0].set_ylabel("fraction of all elements (A = rest)")
    axs[0].legend(["B same code / diff BF16", "C diff code / same BF16", "D diff code / diff BF16"], fontsize=7)
    mets = [("frac_scale_changed", "scale"), ("frac_zero_changed", "zero"), ("frac_group_changed", "group"),
            ("bf16_changed_grid_only_a_on_b", "BF16 diff\nexplained by grid")]
    x = np.arange(len(mets))
    axs[1].bar(x - .2, [ds["primary_vs_original"][m] for m, _ in mets], .4, label="dense", color="#2a6fbb")
    axs[1].bar(x + .2, [ms["primary_vs_original"][m] for m, _ in mets], .4, label="MoE", color="#d1603d")
    axs[1].set_xticks(x, [l for _, l in mets], fontsize=8); axs[1].legend(fontsize=8); axs[1].set_title("grid diagnostics", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig5_code_grid_categories.png"), dpi=140); plt.close(fig)
    # Fig 6: projection level
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(DLAY))
    for i, pj in enumerate(PROJ):
        axs[0].bar(xs + (i - 1) * .27, [ds["by_layer_projection"][f"L{L}|{pj}"]["frac_code_changed"] for L in DLAY], .27, label=pj, color=cols[pj])
    axs[0].set_xticks(xs, [f"L{L}" for L in DLAY]); axs[0].set_title("dense: code-flip rate (4 vs ORIGINAL)", fontsize=9)
    axs[0].legend(fontsize=7)
    for pj in PROJ:
        pe = ms["per_expert"]
        axs[1].scatter([v["tokens"] for v in pe.values()], [v[pj]["frac_code_changed"] for v in pe.values()], color=cols[pj], s=18, label=pj)
    axs[1].set_xlabel("expert routed tokens (FR s0 ORIGINAL)"); axs[1].set_ylabel("code-flip rate (5 vs ORIGINAL)")
    axs[1].set_title("MoE sampled experts", fontsize=9); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig6_projection_sensitivity.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- summary
def fmt(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def write_summary(pr, al, cells, nulls, dec, ds, oc, cmpd, rep, moe_ok, fx, em):
    L_ = []
    w = L_.append
    w("# GPTQ integer-code sensitivity: spatial structure and dense/MoE comparison\n")
    w(f"**Spatial classification (dense): {dec['classification']}**. **Dense/MoE code comparison: "
      f"{cmpd['outcome'] if cmpd else 'BLOCKED'}**" + (f" (R_q = {cmpd['R_q']:.2f})" if cmpd else "") +
      f". Protocol sha256 {pr['protocol_sha256'][:16]}.\n")
    w("VERDICT_PARAGRAPH\n")
    w("## Validation\n")
    w(f"- Dense alignment: {len(al['per_tensor'])} tensors, all checks pass: {al['all_pass']} (valid permutation, group = position // 128, "
      "codes in [0, 15], scale*(q-zero) -> BF16 bitwise = saved weights); FP32 solver parity verified at production time.")
    w(f"- Dense null G20261115 vs ORIGINAL: code flips {ds['null']['frac_code_changed']}, BF16 changes {ds['null']['frac_bf16_changed']}.")
    w(f"- MoE replay (unchanged solver, read-only info capture, {em['coverage']}, 6 configs): bitwise = historical BF16 for all "
      f"{sum(3 * len(r['experts']) for r in rep.values() if r)} tensors: {moe_ok}; codes exact; FP32 dequant == solver output.\n")
    w("## Dense spatial structure (Experiment A)\n")
    w(f"Primary cells (4 pairs x 12 tensors x 2 execution orders = {dec['n_primary_cells']}): clustered fraction by order "
      + ", ".join(f"{k} {v:.2f}" for k, v in dec["frac_cells_clustered_by_order"].items()) + "; by projection "
      + ", ".join(f"{k} {v:.2f}" for k, v in dec["frac_cells_clustered_by_projection"].items())
      + f"; boundary-flagged {dec['frac_cells_boundary']:.2f}; cells with < 200 flips {dec['frac_cells_lt200_flips']:.2f}.\n")
    w("| pair | L.proj | order | flips | p | raw E1 | null E1 mean [99.5 %] | E1 / null | lag-1 pairs / seg-null | lag-8 / seg-null | lag-64 / seg-null | max run | edge first / null |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(nulls):
        if cells[k]["role"] != "primary" or k.split("|")[-1] == "original" or "G20261111" not in k:
            continue
        c, n = cells[k], nulls[k]
        pair, L, pj, o = k.split("|")
        w(f"| {pair} | {L}.{pj} | {o} | {c['flips']} | {c['p']:.4f} | {c['raw_E']['1']:.3f} | {n['E_1']['null_mean']:.3f} [{n['E_1']['q995']:.3f}] | "
          f"{n['E_1']['ratio_to_null_mean']:.3f} | {c['excess_vs_segnull']['1']:.3f} | {c['excess_vs_segnull']['8']:.3f} | "
          f"{c['excess_vs_segnull']['64']:.3f} | {c['max_run']} | {n['edge_first']['ratio_to_null_mean']:.2f} |")
    w("\nAggregated over the 4 primary pairs (median over pairs):\n")
    w("| L.proj | order | p | raw E1 | E1 / null mean | lag-1 excess vs seg-null | vs row-null | lag-128 cross pairs / null | edge first / null | edge last / null | group-changed cols flip rate / same |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for L in DLAY:
        for pj in PROJ:
            for o in ("original", "ref_exec", "var_exec"):
                ks = [k for k in nulls if cells[k]["role"] == "primary" and k.endswith(f"|L{L}|{pj}|{o}")]
                if not ks:
                    continue
                m = lambda f: float(np.median([f(k) for k in ks]))
                gc_ = [cells[k]["group_change"] for k in ks]
                gr = [g["flip_rate_group_changed_cols"] / g["flip_rate_group_same_cols"] for g in gc_
                      if g["flip_rate_group_changed_cols"] is not None and g["flip_rate_group_same_cols"]]
                w(f"| L{L}.{pj} | {o} | {m(lambda k: cells[k]['p']):.4f} | {m(lambda k: cells[k]['raw_E']['1']):.3f} | "
                  f"{m(lambda k: nulls[k]['E_1']['ratio_to_null_mean']):.3f} | {m(lambda k: cells[k]['excess_vs_segnull']['1']):.3f} | "
                  f"{m(lambda k: cells[k]['excess_vs_rownull']['1']):.3f} | {m(lambda k: nulls[k]['pairs_cross']['ratio_to_null_mean']):.3f} | "
                  f"{m(lambda k: nulls[k]['edge_first']['ratio_to_null_mean']):.2f} | {m(lambda k: nulls[k]['edge_last']['ratio_to_null_mean']):.2f} | "
                  f"{(np.median(gr) if gr else float('nan')):.2f} |")
    for o in ("original", "ref_exec", "var_exec"):
        ks = [k for k in cells if cells[k]["role"] == "primary" and k.endswith(o)]
        for d in ("1", "2", "8", "64"):
            v = [cells[k]["excess_vs_segnull"][d] for k in ks]
            w(f"- {o}: lag-{d} pairs / segment-null over {len(v)} primary cells: median {np.nanmedian(v):.4f}, range "
              f"{np.nanmin(v):.4f}-{np.nanmax(v):.4f}" + (f"; vs row-only null median {np.nanmedian([cells[k]['excess_vs_rownull'][d] for k in ks]):.2f}" if d == "1" else ""))
    w("")
    ms_ = jload(p("logs", "moe_spatial.json"))
    if ms_:
        w("**MoE secondary spatial (20 sampled experts x 3 projections, 5 groupings vs ORIGINAL, 200 permutations):**\n")
        w("| order | lag-1 pairs / seg-null (median, range) | vs row-only null | E1 / null mean | edge first / null |\n|---|---|---|---|---|")
        for o in ("original", "ref_exec", "var_exec"):
            cs = [c for cells_ in ms_.values() for k, c in cells_.items() if k.endswith(o)]
            ex = [c["excess_vs_segnull"]["1"] for c in cs]
            e1 = [c["null"]["E_1"]["ratio_to_null_mean"] for c in cs if "null" in c]
            ed = [c["null"]["edge_first"]["ratio_to_null_mean"] for c in cs if "null" in c]
            w(f"| {o} | {np.nanmedian(ex):.3f} ({np.nanmin(ex):.3f}-{np.nanmax(ex):.3f}) | {np.nanmedian([c['excess_vs_rownull']['1'] for c in cs]):.2f} | "
              f"{(np.nanmedian(e1) if e1 else float('nan')):.3f} | {(np.nanmedian(ed) if ed else float('nan')):.2f} |")
        n_cl = sum(1 for cells_ in ms_.values() for k, c in cells_.items() if "null" in c and c["null"]["E_1"]["obs"] > c["null"]["E_1"]["q995"]
                   and c["null"]["E_1"]["ratio_to_null_mean"] >= 1.25)
        n_ex = sum(1 for cells_ in ms_.values() for k, c in cells_.items() if "null" in c)
        w(f"\nMoE cells meeting the dense clustered-cell criterion: {n_cl}/{n_ex}.\n")
    st = [cells[k]["strata"] for k in cells if cells[k]["role"] == "primary" and k.endswith("ref_exec") and cells[k]["strata"]]
    if st:
        w("\nRow-flip-rate strata (ref exec, median over primary cells): lag-1 pairs / segment-null: " + ", ".join(
            f"{b} {np.median([s[b]['excess_1'] for s in st if b in s]):.3f}" for b in ("low", "mid", "high")) + ".\n")
    w("## Integer codes and quantization grid (Experiments A + C)\n")
    hdr = ["frac_bf16_changed", "frac_code_changed", "frac_level_changed", "cat_B", "cat_C", "cat_D", "frac_bf16_changed_same_code",
           "frac_code_changed_same_bf16", "code_changed_same_grid", "frac_scale_changed", "frac_zero_changed", "frac_group_changed",
           "bf16_changed_grid_only_a_on_b", "D_same_grid"]
    w("| set | " + " | ".join(hdr) + " |"); w("|---" * (len(hdr) + 1) + "|")
    sets = [("dense 4 vs ORIGINAL", ds["primary_vs_original"]), ("dense 6 grouping pairs", ds["grouping_pairs"]),
            ("dense null", ds["null"])] + [(f"dense {pj}", ds["by_projection"][pj]) for pj in PROJ]
    if oc.get("moe"):
        mo = oc["moe"]
        sets += [("MoE 5 vs ORIGINAL", mo["primary_vs_original"]), ("MoE 10 grouping pairs", mo["grouping_pairs"])] + \
                [(f"MoE {pj}", mo["by_projection"][pj]) for pj in PROJ] + \
                [(f"MoE tokens {b}", mo["by_token_tertile"][b]) for b in ("low", "mid", "high")] + \
                [(f"MoE L{L}", mo["by_layer"][str(L)]) for L in MLAY]
    for lab, s in sets:
        w(f"| {lab} | " + " | ".join(f"{s[h]:.4f}" for h in hdr) + " |")
    w("\nDense by layer x projection (4 vs ORIGINAL): " + "; ".join(
        f"L{L}.{pj} code {ds['by_layer_projection'][f'L{L}|{pj}']['frac_code_changed']:.4f} / BF16 {ds['by_layer_projection'][f'L{L}|{pj}']['frac_bf16_changed']:.4f}"
        for L in DLAY for pj in PROJ) + ".\n")
    if oc.get("moe"):
        h = oc["moe"]["expert_heterogeneity"]
        w(f"MoE per expert x projection code-flip rate: min {h['code_rate_min']:.4f}, median {h['code_rate_median']:.4f}, max {h['code_rate_max']:.4f} "
          f"(token tertile edges {oc['moe']['token_tertile_edges']}).\n")
    if cmpd:
        w("## Dense vs MoE\n")
        w("| metric | dense | MoE Family D (sampled) |\n|---|---|---|")
        for k, v in cmpd["rows"].items():
            w(f"| {k} | {fmt(v[0])} | {fmt(v[1])} |")
        w(f"\nR_q = {cmpd['R_q']:.2f} (by projection " + ", ".join(f"{k} {v:.2f}" for k, v in cmpd["R_q_by_projection"].items()) +
          f"); R_BF16 = {cmpd['R_bf16']:.2f}. Outcome {cmpd['outcome']}.\n")
    if fx:
        w("## Fixed-act-order replay (optional controlled check)\n")
        par = fx.get("parity", {})
        w(f"Parity (natural permutation supplied as override reproduces saved weights): {sum(par.values())}/{len(par)}.\n")
        for lab, block in (("dense", fx.get("dense", {})), ("MoE", fx.get("moe", {}))):
            if not block:
                continue
            for pj in PROJ:
                vs = [v for k, v in block.items() if k.endswith(pj)]
                if not vs:
                    continue
                nat, fix = wagg([v["natural"] for v in vs]), wagg([v["fixed"] for v in vs])
                w(f"- {lab} {pj}: code flips natural {nat['frac_code_changed']:.4f} -> fixed act-order {fix['frac_code_changed']:.4f}; "
                  f"BF16 {nat['frac_bf16_changed']:.4f} -> {fix['frac_bf16_changed']:.4f}; group changes {nat['frac_group_changed']:.4f} -> "
                  f"{fix['frac_group_changed']:.4f}; scale changed {nat['frac_scale_changed']:.4f} -> {fix['frac_scale_changed']:.4f}")
            if lab == "dense":
                e1 = [v["fixed_spatial"] for v in block.values() if v["fixed_spatial"]["null_E1"]]
                if e1:
                    w(f"- dense fixed-act-order spatial (ORIGINAL order, {len(e1)} cells, NaN = cells with < 2 flips excluded): median E1 / null "
                      f"{np.nanmedian([x['null_E1']['ratio_to_null_mean'] for x in e1]):.3f}; median lag-1 pairs / seg-null "
                      f"{np.nanmedian([x['excess_vs_segnull']['1'] for x in e1]):.4f}; median flip rate {np.median([x['p'] for x in e1]):.5f}")
        w("")
    w("ANSWERS_AND_LIMITATIONS\n")
    w("## Files\n\n- Code: `moe_calib/code_spatial.py`, `diagnostics/integer_code_mechanism.py`, `diagnostics/report_integer_code_mechanism.py`, "
      "`tests/test_code_spatial.py`.\n- Outputs: `runs_integer_code_mechanism/` (required JSONs, `figures/fig1`-`fig6`, `logs/`).\n"
      f"- MoE codes / Hessians of the sampled experts: `{S.CACHE}/moe_codes|moe_hessians/<config>/L<l>.pt`.\n")
    txt = "\n".join(L_) + "\n"
    it = jload(p("logs", "interpretation.json")) or {}
    txt = txt.replace("VERDICT_PARAGRAPH", it.get("verdict_paragraph", "_(interpretation pending)_"))
    txt = txt.replace("ANSWERS_AND_LIMITATIONS", it.get("answers_and_limitations", "_(interpretation pending)_"))
    open(p("summary.md"), "w").write(txt)
