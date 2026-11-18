"""GPTQ integer-code sensitivity: spatial structure of dense code flips (Experiment A) and MoE Family D integer codes
(Experiment C), plus an optional fixed-act-order replay.

Inputs (read only): dense codes / weights / Hessians in ~/moe_cache_dense_grouping_scope (runs_dense_grouping_scope);
MoE FR s0 historical weights in ~/moe_cache_multigroup_robustness/weights, grouping caches ~/moe_cache_multigroup_robustness/G*,
ORIGINAL cache ~/moe_cache (runs_multigroup_robustness). New MoE codes / Hessians -> ~/moe_cache_integer_code_mechanism.
MoE codes come from the UNCHANGED solver's existing read-only `info` capture; every replay must reproduce the historical
BF16 weights bitwise or the comparison is BLOCKED.

Stages: audit | protocol | dense_align | dense_spatial | moe_replay | moe_spatial | fixed_actorder | analyze
"""
from __future__ import annotations
import argparse, gc, hashlib, itertools, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import code_spatial as CS                                   # noqa: E402

OUT = "runs_integer_code_mechanism"
CACHE = os.path.expanduser(os.environ.get("ICM_CACHE", "~/moe_cache_integer_code_mechanism"))
DCACHE = os.path.expanduser("~/moe_cache_dense_grouping_scope")
MCACHE = os.path.expanduser("~/moe_cache_multigroup_robustness")
OCACHE = os.path.expanduser("~/moe_cache")
DENSE_RUN = "runs_dense_grouping_scope"
MG_RUN = "runs_multigroup_robustness"
GROUPS = [f"G{s}" for s in (20261111, 20261112, 20261113, 20261114, 20261115)]
DLAY = [1, 9, 19, 27]
MLAY = [1, 8, 16, 23]
PROJ = ("gate_proj", "up_proj", "down_proj")
GPTQ = dict(bits=4, group_size=128, percdamp=1.0)
N_PERM_PRIMARY, N_PERM_SECONDARY = 1000, 200
SEED_BASE = 20261201

DENSE_PRIMARY = [(g, "ORIGINAL") for g in GROUPS[:4]]                      # 4 single-row perturbations
DENSE_NULL = ("G20261115", "ORIGINAL")                                     # mandatory exact null
DENSE_SECONDARY = list(itertools.combinations(GROUPS[:4], 2))              # 6 distinct grouping pairs
MOE_PRIMARY = [(g, "ORIGINAL") for g in GROUPS]                            # 5 groupings vs ORIGINAL
MOE_SECONDARY = list(itertools.combinations(GROUPS, 2))                    # 10 grouping pairs
ORDERS = ("original", "ref_exec", "var_exec")


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def c(*a):
    return os.path.join(CACHE, *a)


def jdump(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(o, f, indent=1, default=float)
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def sha256_file(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def pkey(a, b):
    return f"{a}__{b}"


def dense_rec(cfg, L):
    return torch.load(os.path.join(DCACHE, "codes", cfg, f"L{L}.pt"))


def dense_w(cfg, L):
    return torch.load(os.path.join(DCACHE, "weights", cfg, f"L{L}.pt"))


def moe_wdir(cfg):
    return os.path.join(MCACHE, "weights", "CTRL_FR_ORIGINAL" if cfg == "ORIGINAL" else f"FR_{cfg}")


def code_hashes():
    fs = ["moe_calib/code_spatial.py", "moe_calib/dense_gptq.py", "moe_calib/gptq.py", "moe_calib/hessian.py",
          "moe_calib/profiler.py", "moe_calib/quant_eval.py", "diagnostics/integer_code_mechanism.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in fs}


# ---------------------------------------------------------------- audit
def stage_audit(a):
    inv = dict(dense={}, moe={})
    dinv = json.load(open(os.path.join(ROOT, DENSE_RUN, "artifact_inventory.json")))["configs"]
    for g in GROUPS + ["ORIGINAL"]:
        ok = {}
        for L in DLAY:
            f = dinv[g]["files"][str(L)]
            ok[str(L)] = dict(weights=sha256_file(os.path.join(DCACHE, "weights", g, f"L{L}.pt")) == f["weight_sha256"],
                              codes=sha256_file(os.path.join(DCACHE, "codes", g, f"L{L}.pt")) == f["codes_sha256"],
                              hessians=sha256_file(os.path.join(DCACHE, "hessians", g, f"L{L}.pt")) == f["hessian_sha256"])
        inv["dense"][g] = dict(files_match_inventory=ok, all_ok=all(all(v.values()) for v in ok.values()),
                               formats=dict(weights="{proj: bf16 [out, in]}", codes="{proj: {q uint8 [out,in] original col order, "
                                            "group int32 [in], perm int32 [in], scale/zero fp32 [n_groups, out]}}",
                                            hessians="{gu fp32 [1536,1536], down fp32 [8960,8960], tokens}"),
                               note="no FP32 dequantized weights saved; scale*(q-zero) was verified bitwise against the solver "
                                    "output at run time (runs_dense_grouping_scope results per_layer codes_reproduce_solver_bitwise)")
    # G20261115 == ORIGINAL bitwise
    eq = {}
    for L in DLAY:
        ra, rb, wa, wb = dense_rec("G20261115", L), dense_rec("ORIGINAL", L), dense_w("G20261115", L), dense_w("ORIGINAL", L)
        ha = torch.load(os.path.join(DCACHE, "hessians", "G20261115", f"L{L}.pt"))
        hb = torch.load(os.path.join(DCACHE, "hessians", "ORIGINAL", f"L{L}.pt"))
        eq[str(L)] = all(torch.equal(ra[k][x], rb[k][x]) for k in PROJ for x in ("q", "group", "perm", "scale", "zero")) and \
            all(torch.equal(wa[k], wb[k]) for k in PROJ) and torch.equal(ha["gu"], hb["gu"]) and torch.equal(ha["down"], hb["down"])
    inv["dense_null_G20261115_equals_ORIGINAL"] = eq
    # MoE
    import full_depth_pilot as FD
    from moe_calib import multigroup_robustness as MG
    _, sel = FD.load_selections()
    mpr = json.load(open(os.path.join(ROOT, MG_RUN, "protocol.json")))
    cman = json.load(open(os.path.join(ROOT, MG_RUN, "cache_manifest.json")))
    for g in GROUPS + ["ORIGINAL"]:
        r = json.load(open(os.path.join(ROOT, MG_RUN, "results", "CTRL_FR_ORIGINAL.json" if g == "ORIGINAL" else f"FR_{g}.json")))
        files = {str(L): sha256_file(os.path.join(moe_wdir(g), f"L{L}.pt")) == r["weight_files"][str(L)]["sha256"] for L in MLAY}
        if g == "ORIGINAL":
            cfiles = {str(L): sha256_file(os.path.join(OCACHE, f"L{L}.pt")) == mpr["references"]["original_cache_files"][str(L)] for L in MLAY}
        else:
            cfiles = {str(L): sha256_file(os.path.join(MCACHE, g, f"L{L}.pt")) == cman["groupings"][g]["files"][str(L)]["sha256"] for L in MLAY}
        inv["moe"][g] = dict(weights_dir=moe_wdir(g), weight_files_match_results=files, calibration_cache=r["calibration_cache"],
                             cache_files_match_manifest=cfiles, installed_weights_sha256=r["installed_weights_sha256"],
                             gptq=r["gptq"], valid=r["valid"], rtn_fallbacks=r["rtn_fallbacks"])
    inv["moe_saved"] = dict(bf16_weights=True, integer_codes=False, scales_zeros=False, act_order=False, hessians=False,
                            note="only BF16 installed weights were saved historically; codes/grids/perms/Hessians must be "
                                 "replayed from the preserved calibration caches with the unchanged solver")
    inv["moe_selection"] = dict(FR_ids_hash=sel["FR"]["ids_hash"], K=sel["FR"]["K"], n=len(sel["FR"]["chosen"]))
    jdump(inv, p("artifact_inventory.json"))
    # expert manifest: prespecified rule (diagnostic check experts + per-layer token-count strata from historical counts)
    r0 = json.load(open(os.path.join(ROOT, MG_RUN, "results", "CTRL_FR_ORIGINAL.json")))
    tok = {(e["layer"], e["expert"]): e["tokens"] for e in r0["per_expert"]}
    sel_e = []
    for L in MLAY:
        for e in (0, 30):
            sel_e.append(dict(layer=L, expert=e, role="diagnostic_check (EP.CHECK_EXPERTS / frozen_routing.DIAG_EXPERTS)",
                              tokens_FR_ORIGINAL=tok[(L, e)]))
        rest = sorted([(tok[(L, e)], e) for e in range(60) if e not in (0, 30)])
        for lab, (t, e) in (("stratum_min_tokens", rest[0]), ("stratum_median_tokens", rest[(len(rest) - 1) // 2]),
                            ("stratum_max_tokens", rest[-1])):
            sel_e.append(dict(layer=L, expert=e, role=lab, tokens_FR_ORIGINAL=t))
    alltok = np.array(list(tok.values()))
    jdump(dict(rule="per layer 1/8/16/23: experts 0 and 30 (historical diagnostic check experts, prespecified, not "
                    "outcome-based) + the min-, median- (lower) and max-token expert among the other 58 by historical FR s0 "
                    "ORIGINAL routed-token counts (runs_multigroup_robustness CTRL_FR_ORIGINAL per_expert.tokens)",
               experts=sel_e, n=len(sel_e), coverage=f"{len(sel_e)} / 240 routed experts",
               dims=dict(gate_proj=[1408, 2048], up_proj=[1408, 2048], down_proj=[2048, 1408]),
               all_expert_tokens=dict(min=float(alltok.min()), median=float(np.median(alltok)), max=float(alltok.max())),
               note="the 16 'diagnostic Hessians' historically = these 8 check experts x {gu, down}"), p("moe_expert_manifest.json"))
    print(json.dumps({k: (v["all_ok"] if "all_ok" in v else v) for k, v in inv["dense"].items()}),
          inv["dense_null_G20261115_equals_ORIGINAL"])
    print({g: (all(v["weight_files_match_results"].values()), all(v["cache_files_match_manifest"].values())) for g, v in inv["moe"].items()})


# ---------------------------------------------------------------- protocol
def stage_protocol(a):
    inv, em = jload(p("artifact_inventory.json")), jload(p("moe_expert_manifest.json"))
    assert inv and all(v["all_ok"] for v in inv["dense"].values()) and all(inv["dense_null_G20261115_equals_ORIGINAL"].values())
    if os.path.exists(p("logs", "moe_replay")) and os.listdir(p("logs", "moe_replay")):
        raise SystemExit("MoE replay results exist: protocol cannot be created now")
    body = dict(
        study="GPTQ integer-code sensitivity: dense spatial structure (A) and MoE Family D integer codes (C)",
        dense=dict(source=DENSE_RUN, layers=DLAY, primary_pairs=DENSE_PRIMARY, null_pair=DENSE_NULL,
                   secondary_pairs=DENSE_SECONDARY,
                   pair_convention="(variant, reference); G20261115 == ORIGINAL bitwise, so pairs with G20261115 other than "
                                   "the null are not separate comparisons (no double counting)",
                   distinct_comparisons="4 primary + 6 secondary = 10 distinct weight-set pairs"),
        moe=dict(strategy="FR s0", configs=GROUPS + ["ORIGINAL"], primary_pairs=MOE_PRIMARY, secondary_pairs=MOE_SECONDARY,
                 experts_manifest_sha256=obj_sha(em), experts=[(e["layer"], e["expert"]) for e in em["experts"]],
                 replay="Profiler(model, L, K=4).run(calibration cache, FR chosen ids [remapped to cache rows for groupings]) "
                        "-> quant_eval.quantize_from_acc(..., infos) -> unchanged gptq_quantize with read-only info capture "
                        "-> W_q.to(bf16) must equal the historical BF16 weights bitwise, else BLOCKED",
                 model_load="MOE_LOAD_LAYERS=1,8,16,23 (layer-wise replay on cached pristine inputs)"),
        alignment=dict(coordinates="original (output row, input column); q stored in original column order",
                       execution_position="pos[perm[p]] = p; group = pos // 128 (== lazy-batch block, blocksize 128)",
                       dequant="FP32 scale*(q-zero) with element grid S[r,j]=scale[group[j],r] (solver convention, verified), "
                               "then .to(bfloat16) (installation cast)",
                       separate=["code index", "scale / zero", "group assignment", "zero-relative level q-zero",
                                 "FP32 dequantized", "BF16 installed"]),
        spatial=dict(
            mask="F[r,j] = q_a[r,j] != q_b[r,j] in original coordinates",
            orders=dict(original="original column order, segments = contiguous 128-column blocks",
                        ref_exec="columns reordered by the REFERENCE's act-order, segments = its 128-position groups/blocks",
                        var_exec="columns reordered by the VARIANT's act-order"),
            statistics=["p", "E_d = P(F_{j+d}|F_j)/P(F_{j+d}) within segments, d in 1,2,4,...,64",
                        "cross-segment lag-128 pairs", "all-flip windows L=2,3,4,8 within segment", "runs, max run",
                        "inter-flip distance histogram (row, execution order)", "row dispersion vs binomial",
                        "row-flip-rate strata (tertiles of rows with >=2 flips): lag-1 pairs / analytic segment-null expectation",
                        "segment-edge flips (first/last 8 positions)", "flip rate in columns that changed group vs not"],
            null="per (row, segment) flip counts preserved; positions uniform within the segment; torch.Generator seeds "
                 f"{SEED_BASE} + index; {N_PERM_PRIMARY} permutations for each primary dense (pair, layer, projection, order), "
                 f"{N_PERM_SECONDARY} for secondary dense pairs and MoE primary pairs; analytic expectations k(k-1)(128-d)/(128*127) "
                 "(segment null) and row-only null reported alongside raw E_d",
            first_flip_alignment="not performed (optional; would need a matched null)"),
        criteria=dict(
            cell="primary dense cell = (pair, layer, projection, execution order in {ref_exec, var_exec})",
            clustered_cell="E_1 obs > null 99.5 % quantile AND E_1 obs / null mean >= 1.25",
            boundary_cell="segment-edge (first or last 8) flips > null 99.5 % quantile AND >= 1.25 x null mean",
            classification={"INCONCLUSIVE": "> 50 % of primary cells have < 200 flips",
                            "SPATIAL-CLUSTERING-SUPPORTED": ">= 75 % of primary cells clustered in BOTH execution orders and "
                                                            ">= 50 % in every projection type",
                            "SPATIAL-CLUSTERING-NOT-DETECTED": "<= 10 % of primary cells clustered in both execution orders and "
                                                              "< 50 % boundary cells",
                            "MIXED-SPATIAL-STRUCTURE": "otherwise (incl. projection- or layer-dependent clustering, or "
                                                       "boundary concentration without general clustering)"},
            code_comparison=dict(metric="R_q = MoE element-weighted code-flip rate / dense element-weighted code-flip rate, "
                                        "matched 'grouping vs ORIGINAL' comparisons (dense 4 non-null, MoE 5), overall and per projection",
                                 rule={"A-SIMILAR-CODE-DISAGREEMENT": "R_q <= 3 overall and in every projection",
                                       "B-MUCH-LARGER-MOE-CODE-DISAGREEMENT": "R_q > 3 overall and in every projection",
                                       "C-PROJECTION-DEPENDENT": "otherwise"},
                                 caveat="sampled 20/240 MoE experts; Hessian perturbation magnitudes and tokens per dimension differ"),
            causal="no error-feedback causality from spatial association; fixed-act-order replay is a controlled check only"),
        optional=dict(fixed_actorder="dense: variant Hessian with the ORIGINAL permutation for every tensor of the 4 primary "
                                     "pairs; MoE: the 20 experts x 5 groupings; parity: override with the natural permutation "
                                     "reproduces saved weights bitwise; 200-permutation null in the (shared) execution order",
                      feedback_test="not performed: no safe solver variant without substantial changes (design documented)"),
        code_sha256=code_hashes())
    body["protocol_sha256"] = obj_sha(body)
    body["created"] = time.strftime("%F %T")
    old = jload(p("protocol.json"))
    if old:
        assert old["protocol_sha256"] == body["protocol_sha256"], "protocol changed after freezing"
        print("protocol already frozen and consistent"); return
    jdump(body, p("protocol.json"))
    print(f"protocol frozen sha256 {body['protocol_sha256'][:16]}")


def check_protocol():
    pr = jload(p("protocol.json"))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    assert code_hashes() == pr["code_sha256"], "code changed since the protocol was frozen"
    return pr


# ---------------------------------------------------------------- dense alignment
def stage_dense_align(a):
    check_protocol()
    out = {}
    for g in GROUPS + ["ORIGINAL"]:
        for L in DLAY:
            rec, W = dense_rec(g, L), dense_w(g, L)
            for k in PROJ:
                out[f"{g}|L{L}|{k}"] = CS.validate_record(rec[k], W[k])
    res = jload(os.path.join(ROOT, DENSE_RUN, "results", "ORIGINAL.json"))
    fp32 = {g: all(all(jload(os.path.join(ROOT, DENSE_RUN, "results", f"{g}.json"))["per_layer"][str(L)]
                       ["codes_reproduce_solver_bitwise"].values()) for L in DLAY) for g in GROUPS + ["ORIGINAL"]}
    jdump(dict(per_tensor=out, all_pass=all(all(v.values()) for v in out.values()),
               fp32_solver_parity_at_run_time=fp32,
               note="FP32 parity: scale*(q-zero) == solver W_q bitwise was verified when the codes were produced (FP32 W_q not "
                    "saved); here: valid permutation, group == position//128, codes in [0,15], and bitwise BF16 reconstruction"),
          p("dense_code_alignment_validation.json"))
    print("dense alignment all pass:", all(all(v.values()) for v in out.values()), fp32)


# ---------------------------------------------------------------- spatial machinery
def masks_for(ra, rb, dev):
    F = (ra["q"] != rb["q"]).to(dev)
    return {"original": F, "ref_exec": CS.to_exec(F, rb["perm"]), "var_exec": CS.to_exec(F, ra["perm"])}


def strata(M):
    """Rows with >= 2 flips split into tertiles of row flip count: observed lag-1 pairs vs analytic segment null."""
    rc = M.sum(1)
    rows = (rc >= 2).nonzero().flatten()
    if rows.numel() < 30:
        return None
    qs = torch.quantile(rc[rows].double(), torch.tensor([1 / 3, 2 / 3], dtype=torch.float64, device=M.device))
    out = {}
    for lab, sel in (("low", rc[rows] <= qs[0]), ("mid", (rc[rows] > qs[0]) & (rc[rows] <= qs[1])), ("high", rc[rows] > qs[1])):
        sub = M[rows[sel]]
        if sub.shape[0] == 0:
            continue
        s, an = CS.stats(sub, full=False), CS.analytic_pairs(sub)
        out[lab] = dict(rows=int(sub.shape[0]), mean_row_flips=float(sub.sum(1).float().mean()),
                        pairs_1=s["pairs_1"], segnull_pairs_1=an["segnull_pairs_1"],
                        excess_1=s["pairs_1"] / an["segnull_pairs_1"] if an["segnull_pairs_1"] > 0 else float("nan"))
    return out


def analyze_cell(M, n_perm, seed, extra=None):
    s = CS.stats(M)
    an = CS.analytic_pairs(M)
    out = dict(obs={k: v for k, v in s.items()}, analytic=an,
               raw_E={str(d): CS.enrichment(s, d) for d in CS.LAGS},
               excess_vs_segnull={str(d): s[f"pairs_{d}"] / an[f"segnull_pairs_{d}"] if an[f"segnull_pairs_{d}"] > 0 else float("nan")
                                  for d in CS.LAGS},
               excess_vs_rownull={str(d): s[f"pairs_{d}"] / an[f"rownull_pairs_{d}"] if an[f"rownull_pairs_{d}"] > 0 else float("nan")
                                  for d in CS.LAGS},
               strata=strata(M), p=s["flips"] / s["n"])
    if n_perm and s["flips"] >= 2:
        nl = CS.permutation_null(M, n_perm, seed)
        o = {f"E_{d}": CS.enrichment(s, d) for d in CS.LAGS}
        o.update(pairs_cross=s["pairs_cross"], edge_first=s["edge_first"], edge_last=s["edge_last"], runs=s["runs"],
                 **{f"win_{L}": s[f"win_{L}"] for L in CS.WIN})
        out["null"] = {k: CS.summarize_null(o[k], nl[k]) for k in o}
        out["n_perm"], out["seed"] = n_perm, seed
    if extra:
        out.update(extra)
    return out


def group_change_info(ra, rb, F):
    gch = ra["group"] != rb["group"]
    C = gch.numel()
    pa, pb = torch.empty(C, dtype=torch.long), torch.empty(C, dtype=torch.long)
    pa[ra["perm"].long()] = torch.arange(C); pb[rb["perm"].long()] = torch.arange(C)
    Fc = F.cpu()
    return dict(frac_columns_group_changed=float(gch.float().mean()),
                frac_columns_position_changed=float((pa != pb).float().mean()),
                mean_abs_position_shift=float((pa - pb).abs().float().mean()),
                flip_rate_group_changed_cols=float(Fc[:, gch].float().mean()) if bool(gch.any()) else None,
                flip_rate_group_same_cols=float(Fc[:, ~gch].float().mean()) if bool((~gch).any()) else None)


def stage_dense_spatial(a):
    check_protocol()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(p("logs", "dense_spatial"), exist_ok=True)
    jobs = [(pr_, N_PERM_PRIMARY, "primary") for pr_ in DENSE_PRIMARY] + [(DENSE_NULL, 0, "null")] + \
           [(pr_, N_PERM_SECONDARY, "secondary") for pr_ in DENSE_SECONDARY]
    for (va, rf), n_perm, role in jobs:
        path = p("logs", "dense_spatial", f"{pkey(va, rf)}.json")
        if jload(path):
            continue
        t0, out = time.time(), dict(pair=[va, rf], role=role, cells={})
        for L in DLAY:
            ra, rb = dense_rec(va, L), dense_rec(rf, L)
            for k in PROJ:
                ms = masks_for(ra[k], rb[k], dev)
                gci = group_change_info(ra[k], rb[k], ms["original"])
                for o in ORDERS:
                    seed = SEED_BASE + int(hashlib.sha256(f"{va}|{rf}|{L}|{k}|{o}".encode()).hexdigest()[:6], 16)
                    out["cells"][f"L{L}|{k}|{o}"] = analyze_cell(ms[o], n_perm, seed, dict(group_change=gci))
                del ms
                torch.cuda.empty_cache()
            print(f"[dense_spatial] {va}__{rf} L{L} ({time.time() - t0:.0f} s)", flush=True)
        out["seconds"] = time.time() - t0
        jdump(out, path)


# ---------------------------------------------------------------- MoE replay (unchanged solver, read-only info capture)
def stage_moe_replay(a):
    pr = check_protocol()
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, MLAY))
    from moe_calib import model_utils as MU, dense_gptq as DG, multigroup_robustness as MG
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import quantize_from_acc
    from moe_calib.data_utils import load_layer_cache
    import full_depth_pilot as FD
    em = jload(p("moe_expert_manifest.json"))
    assert obj_sha(em) == pr["moe"]["experts_manifest_sha256"]
    _, sel = FD.load_selections()
    fr = sel["FR"]
    mpr = json.load(open(os.path.join(ROOT, MG_RUN, "protocol.json")))
    union = mpr["union_calibration_ids"]
    inv = jload(p("artifact_inventory.json"))
    by_layer = {}
    for e in em["experts"]:
        by_layer.setdefault(e["layer"], []).append(e["expert"])
    t0 = time.time()
    model, _ = MU.load_model()
    print(f"[moe] model (partial) loaded {time.time() - t0:.0f} s", flush=True)
    os.makedirs(p("logs", "moe_replay"), exist_ok=True)
    for g in ["ORIGINAL"] + GROUPS:
        path = p("logs", "moe_replay", f"{g}.json")
        if jload(path):
            continue
        assert all(inv["moe"][g]["weight_files_match_results"].values()) and all(inv["moe"][g]["cache_files_match_manifest"].values())
        if g == "ORIGINAL":
            cdir, chosen = OCACHE, fr["chosen"]
        else:
            cdir, chosen = os.path.join(MCACHE, g), MG.remap(fr["chosen"], union)
            assert [union[i] for i in chosen] == fr["chosen"]
        rec_out, t1 = {}, time.time()
        for L in MLAY:
            os.makedirs(c("moe_codes", g), exist_ok=True); os.makedirs(c("moe_hessians", g), exist_ok=True)
            cache = load_layer_cache(cdir, L)
            prof = Profiler(model, L, K=fr["K"]).run(cache, chosen, weights=[1.0] * len(chosen))
            hist = torch.load(os.path.join(moe_wdir(g), f"L{L}.pt"))
            codes, hes = {}, {}
            for e in by_layer[L]:
                infos = {}
                Wq = quantize_from_acc(model, prof.acc, L, e, None, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"],
                                       perms={}, infos=infos)
                codes[e], par = {}, {}
                for k in PROJ:
                    q, gid, exact = DG.integer_codes(Wq[k], infos[k], GPTQ["bits"], GPTQ["group_size"])
                    W16 = Wq[k].to(torch.bfloat16).cpu()
                    codes[e][k] = dict(q=q, group=gid, perm=infos[k]["perm"].int(), scale=infos[k]["scale"], zero=infos[k]["zero"])
                    par[k] = dict(bf16_equals_historical=bool(torch.equal(W16, hist[f"{e}.{k}"])), codes_exact=exact,
                                  dequant_equals_solver_fp32=bool(torch.equal(CS.dequant(codes[e][k]), Wq[k].cpu())))
                hes[e] = dict(gu=prof.acc.H_sum((L, e, "gu"), list(range(prof.K))).cpu(),
                              down=prof.acc.H_sum((L, e, "down"), list(range(prof.K))).cpu(),
                              tokens=prof.acc.tokens((L, e, "gu")))
                rec_out[f"L{L}|e{e}"] = dict(parity=par, tokens=hes[e]["tokens"])
            torch.save(codes, c("moe_codes", g, f"L{L}.pt"))
            torch.save(hes, c("moe_hessians", g, f"L{L}.pt"))
            prof.free()
            del prof, cache, codes, hes
            gc.collect(); torch.cuda.empty_cache()
        allp = all(all(all(v.values()) for v in r["parity"].values()) for r in rec_out.values())
        first_bad = next((k for k, r in rec_out.items() for kk, v in r["parity"].items() if not all(v.values())), None)
        jdump(dict(config=g, experts=rec_out, all_parity=allp, first_failure=first_bad, seconds=time.time() - t1,
                   codes_sha256={str(L): sha256_file(c("moe_codes", g, f"L{L}.pt")) for L in MLAY}), path)
        print(f"[moe] {g}: parity {allp} ({time.time() - t1:.0f} s){'' if allp else ' FIRST FAIL ' + str(first_bad)}", flush=True)


def moe_rec(cfg, L):
    return torch.load(c("moe_codes", cfg, f"L{L}.pt"))


def stage_moe_spatial(a):
    check_protocol()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = p("logs", "moe_spatial.json")
    out = jload(path) or {}
    em = jload(p("moe_expert_manifest.json"))
    for va, rf in MOE_PRIMARY:
        key = pkey(va, rf)
        if key in out:
            continue
        cells = {}
        A = {L: moe_rec(va, L) for L in MLAY}
        B = {L: moe_rec(rf, L) for L in MLAY}
        for ex in em["experts"]:
            L, e = ex["layer"], ex["expert"]
            for k in PROJ:
                ms = masks_for(A[L][e][k], B[L][e][k], dev)
                for o in ("ref_exec", "var_exec", "original"):
                    seed = SEED_BASE + int(hashlib.sha256(f"moe|{va}|{rf}|{L}|{e}|{k}|{o}".encode()).hexdigest()[:6], 16)
                    cells[f"L{L}|e{e}|{k}|{o}"] = analyze_cell(ms[o], N_PERM_SECONDARY if o != "original" else 0, seed)
        out[key] = cells
        jdump(out, path)
        print(f"[moe_spatial] {key}", flush=True)


# ---------------------------------------------------------------- optional fixed-act-order replay
def stage_fixed_actorder(a):
    check_protocol()
    from moe_calib.gptq import gptq_quantize
    from moe_calib import dense_gptq as DG
    from safetensors import safe_open
    import dense_grouping_scope as DS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = p("logs", "fixed_actorder.json")
    out = jload(path) or dict(dense={}, moe={}, parity={})
    # ---- dense
    sp = DS.snapshot_dir()
    with safe_open(os.path.join(sp, "model.safetensors"), framework="pt") as f:
        Wp = {L: {k: f.get_tensor(f"model.layers.{L}.mlp.{k}.weight") for k in PROJ} for L in DLAY}

    def solve(W, H, perm=None):
        info = {}
        Q = gptq_quantize(W.float().to(dev), H.to(dev), GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"],
                          actorder_override=perm, info=info)
        q, gid, ex = DG.integer_codes(Q, info, GPTQ["bits"], GPTQ["group_size"])
        return Q.cpu(), dict(q=q, group=gid, perm=info["perm"].int(), scale=info["scale"], zero=info["zero"]), ex
    dev_spatial = dev
    for L in DLAY:
        HO = torch.load(os.path.join(DCACHE, "hessians", "ORIGINAL", f"L{L}.pt"))
        RO, WO = dense_rec("ORIGINAL", L), dense_w("ORIGINAL", L)
        for va, rf in DENSE_PRIMARY:
            HV = torch.load(os.path.join(DCACHE, "hessians", va, f"L{L}.pt"))
            RV, WV = dense_rec(va, L), dense_w(va, L)
            for k in PROJ:
                key = f"{va}|L{L}|{k}"
                if key in out["dense"]:
                    continue
                hk = "down" if k == "down_proj" else "gu"
                if va == DENSE_PRIMARY[0][0]:              # parity: natural permutation supplied as override
                    Qn, rn, _ = solve(Wp[L][k], HO[hk], RO[k]["perm"])
                    out["parity"][f"dense|ORIGINAL|L{L}|{k}"] = bool(torch.equal(Qn.to(torch.bfloat16), WO[k]))
                    Qv, _, _ = solve(Wp[L][k], HV[hk], RV[k]["perm"])
                    out["parity"][f"dense|{va}|L{L}|{k}"] = bool(torch.equal(Qv.to(torch.bfloat16), WV[k]))
                Qf, rfx, ex = solve(Wp[L][k], HV[hk], RO[k]["perm"])
                Wf = Qf.to(torch.bfloat16)
                dn = CS.decompose(RV[k], RO[k], WV[k], WO[k])
                dfx = CS.decompose(rfx, RO[k], Wf, WO[k])
                M = (rfx["q"] != RO[k]["q"]).to(dev_spatial)
                seed = SEED_BASE + int(hashlib.sha256(f"fixed|{key}".encode()).hexdigest()[:6], 16)
                cell = analyze_cell(CS.to_exec(M, RO[k]["perm"]), N_PERM_SECONDARY, seed)
                out["dense"][key] = dict(natural=dn, fixed=dfx, codes_exact=ex,
                                         fixed_spatial=dict(p=cell["p"], E1=cell["raw_E"]["1"],
                                                            null_E1=cell.get("null", {}).get("E_1"),
                                                            excess_vs_segnull=cell["excess_vs_segnull"]))
            jdump(out, path)
        print(f"[fixed] dense L{L}", flush=True)
    # ---- MoE (selected experts, if replay passed)
    em = jload(p("moe_expert_manifest.json"))
    ok = all((jload(p("logs", "moe_replay", f"{g}.json")) or {}).get("all_parity") for g in ["ORIGINAL"] + GROUPS)
    out["moe_available"] = ok
    if ok:
        for L in MLAY:
            RO, HO = moe_rec("ORIGINAL", L), torch.load(c("moe_hessians", "ORIGINAL", f"L{L}.pt"))
            Wpm = moe_pristine(L, [x["expert"] for x in em["experts"] if x["layer"] == L])
            for va in GROUPS:
                RV = moe_rec(va, L)
                HV = torch.load(c("moe_hessians", va, f"L{L}.pt"))
                for x in em["experts"]:
                    if x["layer"] != L:
                        continue
                    e = x["expert"]
                    for k in PROJ:
                        key = f"{va}|L{L}|e{e}|{k}"
                        if key in out["moe"]:
                            continue
                        hk = "down" if k == "down_proj" else "gu"
                        if va == GROUPS[0]:
                            Qn, _, _ = solve(Wpm[e][k], HO[e][hk], RO[e][k]["perm"])
                            out["parity"][f"moe|ORIGINAL|L{L}|e{e}|{k}"] = bool(torch.equal(Qn, CS.dequant(RO[e][k])))
                        Qf, rfx, ex = solve(Wpm[e][k], HV[e][hk], RO[e][k]["perm"])
                        WVb, WOb = CS.dequant(RV[e][k]).to(torch.bfloat16), CS.dequant(RO[e][k]).to(torch.bfloat16)
                        out["moe"][key] = dict(natural=CS.decompose(RV[e][k], RO[e][k], WVb, WOb),
                                               fixed=CS.decompose(rfx, RO[e][k], Qf.to(torch.bfloat16), WOb), codes_exact=ex)
            jdump(out, path)
            print(f"[fixed] moe L{L}", flush=True)
    jdump(out, path)


def moe_pristine(L, experts):
    """Pristine BF16 expert weights straight from the checkpoint shards."""
    from safetensors import safe_open
    from moe_calib import model_utils as MU
    sp = MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B")
    wmap = json.load(open(os.path.join(sp, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for e in experts:
        out[e] = {}
        for k in PROJ:
            name = f"model.layers.{L}.mlp.experts.{e}.{k}.weight"
            with safe_open(os.path.join(sp, wmap[name]), framework="pt") as f:
                out[e][k] = f.get_tensor(name).to(torch.bfloat16)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "protocol", "dense_align", "dense_spatial", "moe_replay",
                                                        "moe_spatial", "fixed_actorder", "analyze"])
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "analyze":
        import report_integer_code_mechanism as R
        return R.analyze()
    dict(audit=stage_audit, protocol=stage_protocol, dense_align=stage_dense_align, dense_spatial=stage_dense_spatial,
         moe_replay=stage_moe_replay, moe_spatial=stage_moe_spatial, fixed_actorder=stage_fixed_actorder)[a.stage](a)


if __name__ == "__main__":
    main()
