"""Dense GPTQ numerical-perturbation dose-response (Qwen2.5-1.5B).

The calibration SET is fixed (FR s0's 128 sequences, frozen order). Per selected sequence there are two real layer-input
caches from the pristine BF16 model: the historical batch-size-8 extraction X^(8) (runs_dense_grouping_scope ORIGINAL
cache, cache_inputs.py consecutive batches of 8) and a batch-size-1 extraction X^(1). A dose-k configuration replaces the
historical cache of exactly k rows (subset S) by X^(1). A strong-path family uses a pilot-selected alternate deterministic
execution path instead of batch 1. Hessians: the historical DenseProfiler (BlockHessianAccumulator, K = 4 blocks, frozen
order); GPTQ under natural and fixed reference act-order (moe_calib.actorder_policy; production solver unchanged);
BF16 install; historical WikiText-2 windows.

Stages: audit | pilot_criteria | pilot | protocol | caches | gpu | refmatrix | func | analyze
"""
from __future__ import annotations
import argparse, gc, hashlib, itertools, json, math, os, shutil, stat, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import functional_diff as FD                                 # noqa: E402
from moe_calib import actorder_policy as AP                                 # noqa: E402

OUT = "runs_dense_dose_response"
CACHE = os.path.expanduser(os.environ.get("DDR_CACHE", "~/moe_cache_dense_dose_response"))
DCACHE = os.path.expanduser("~/moe_cache_dense_grouping_scope")
ACACHE = os.path.expanduser("~/moe_cache_actorder_functional_stability")
DENSE_RUN, AFS_RUN = "runs_dense_grouping_scope", "runs_actorder_functional_stability"
LAY = [1, 9, 19, 27]
PROJ = ("gate_proj", "up_proj", "down_proj")
GPTQ = dict(bits=4, group_size=128, percdamp=1.0)
GROUPS = [f"G{s}" for s in (20261111, 20261112, 20261113, 20261114, 20261115)]
HIST_SINGLETON = {"G20261111": 491, "G20261112": 1024, "G20261113": 1667, "G20261114": 1064}
DOSES_MAIN = [1, 4, 16, 64]
DOSES_OPT = [32, 96]
N_REP = 8
PERM_SEED_BASE = 20261301
POLICIES = ("natural", "fixed")
CHUNK = 4096


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


def tsave(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(o, path + ".tmp")
    os.replace(path + ".tmp", path)


def sha256_file(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()).hexdigest()


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def code_files():
    return ["moe_calib/functional_diff.py", "moe_calib/actorder_policy.py", "moe_calib/gptq.py", "moe_calib/dense_gptq.py",
            "moe_calib/hessian.py", "moe_calib/model_utils.py", "moe_calib/batch_sensitivity.py",
            "diagnostics/batch_sensitivity.py", "diagnostics/eval_partial_ppl.py", "diagnostics/dense_dose_response.py"]


def code_hashes():
    return {f: sha256_file(os.path.join(ROOT, f)) for f in code_files()}


def log_code_state(stage):
    path = p("logs", "code_state.json")
    st = jload(path) or {}
    st.setdefault(stage, []).append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), sha256=code_hashes()))
    jdump(st, path)


def check_protocol():
    pr = jload(p("protocol.json"))
    assert pr, "protocol not frozen"
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified after freeze"
    return pr


def cost_add(stage, key, seconds):
    path = p("logs", "cost_raw.json")
    cst = jload(path) or {}
    cst.setdefault(stage, {})[key] = seconds
    jdump(cst, path)


# ---------------------------------------------------------------- calibration
def calib():
    cal = jload(os.path.join(ROOT, DENSE_RUN, "calibration_manifest.json"))
    return cal["union_ids"], cal["fr_chosen_ordered"], cal["fr_union_rows"], cal


def replicate_perms():
    """Prespecified: replicate r -> default_rng(PERM_SEED_BASE + r).permutation(128) of the FR order; if a k=1 subset
    would duplicate an earlier replicate's, the seed is advanced by 1000 until unique (recorded)."""
    _, fr, _, _ = calib()
    out, firsts = [], set()
    for r in range(N_REP):
        s = PERM_SEED_BASE + r
        while True:
            perm = np.random.default_rng(s).permutation(len(fr))
            if int(perm[0]) not in firsts:
                break
            s += 1000
        firsts.add(int(perm[0]))
        out.append(dict(replicate=r, seed=int(s), order=[int(fr[i]) for i in perm]))
    return out


def configs(pr):
    """All dose configurations (name, family, k, replicate, subset of sequence ids)."""
    _, fr, _, _ = calib()
    reps = pr["perturbation"]["replicates"]
    out = [dict(name="b1_k000", family="b1", k=0, rep=None, S=[]),
           *[dict(name=f"hist_{g}", family="b1", k=1, rep=None, S=[s], historical=g) for g, s in HIST_SINGLETON.items()]]
    for k in DOSES_MAIN:
        for rp in reps:
            out.append(dict(name=f"b1_k{k:03d}_r{rp['replicate']}", family="b1", k=k, rep=rp["replicate"], S=rp["order"][:k]))
    out.append(dict(name="b1_k128", family="b1", k=128, rep=None, S=list(fr)))
    for k in DOSES_OPT:
        for rp in reps:
            out.append(dict(name=f"b1_k{k:03d}_r{rp['replicate']}", family="b1", k=k, rep=rp["replicate"], S=rp["order"][:k],
                            optional=True))
    sp = pr.get("strong_path", {})
    if sp.get("selected"):
        kk = sp["design"]["dose"]
        for rp in reps:
            out.append(dict(name=f"sp_k{kk:03d}_r{rp['replicate']}", family="sp", k=kk, rep=rp["replicate"], S=rp["order"][:kk]))
        out.append(dict(name="sp_k128", family="sp", k=128, rep=None, S=list(fr)))
    return out


# ---------------------------------------------------------------- audit
def stage_audit(a):
    log_code_state("audit")
    import dense_grouping_scope as DS
    union, fr, fr_rows, cal = calib()
    inv = jload(os.path.join(ROOT, DENSE_RUN, "artifact_inventory.json"))["configs"]
    cm = jload(os.path.join(ROOT, DENSE_RUN, "cache_manifest.json"))
    gman = jload(os.path.join(ROOT, DENSE_RUN, "grouping_manifest.json"))["groupings"]
    out = dict(calibration=dict(n_union=len(union), n_fr=len(fr), fr_order_first10=fr[:10], fr_union_rows_first10=fr_rows[:10],
                                fr_ids_hash=cal["fr_ids_hash"], K_blocks=cal["K_blocks"], block_rule=cal["block_rule"],
                                fr_token_sha256=cal["fr_token_sha256"]))
    # caches
    out["caches"] = {}
    for g in GROUPS + ["ORIGINAL"]:
        out["caches"][g] = {str(L): sha256_file(os.path.join(DCACHE, "cache", g, f"L{L}.pt")) == cm[g]["files"][str(L)]["sha256"]
                            for L in LAY}
    out["cache_format"] = dict(shape=cm["ORIGINAL"]["files"]["1"]["shape"], content="decoder-layer-L INPUT hidden states of the 217 "
                               "union rows in sorted union order (bf16); built by diagnostics/batch_sensitivity.forward_plan",
                               original_plan="moe_calib.batch_sensitivity.original_batches(union, 8, 2256): consecutive "
                                             "original-id batches of 8 incl. non-union batch-mates", n_original_batches=len(gman["ORIGINAL"]["batches"]))
    # grouping caches differ from ORIGINAL only in the singleton row
    diff = {}
    for g in GROUPS:
        sing = gman[g]["batches"][-1]
        for L in LAY:
            A = torch.load(os.path.join(DCACHE, "cache", g, f"L{L}.pt"), mmap=True)
            O = torch.load(os.path.join(DCACHE, "cache", "ORIGINAL", f"L{L}.pt"), mmap=True)
            rows = [i for i in range(A.shape[0]) if not torch.equal(A[i], O[i])]
            diff[f"{g}|L{L}"] = dict(rows_differ=[union[i] for i in rows], singleton=sing)
    out["grouping_cache_vs_original"] = diff
    out["singletons_match"] = all(set(v["rows_differ"]) <= set(v["singleton"]) for v in diff.values())
    # Hessians / weights / codes
    art = {}
    for g in GROUPS + ["ORIGINAL"]:
        for L in LAY:
            f = inv[g]["files"][str(L)]
            art[f"{g}|L{L}"] = dict(weights=sha256_file(os.path.join(DCACHE, "weights", g, f"L{L}.pt")) == f["weight_sha256"],
                                    codes=sha256_file(os.path.join(DCACHE, "codes", g, f"L{L}.pt")) == f["codes_sha256"],
                                    hessians=sha256_file(os.path.join(DCACHE, "hessians", g, f"L{L}.pt")) == f["hessian_sha256"])
    out["historical_artifacts"] = art
    # reference act-order permutations
    rp = {}
    for L in LAY:
        H, R = torch.load(os.path.join(DCACHE, "hessians", "ORIGINAL", f"L{L}.pt")), torch.load(os.path.join(DCACHE, "codes", "ORIGINAL", f"L{L}.pt"))
        for k in PROJ:
            hk = "down" if k == "down_proj" else "gu"
            rp[f"L{L}|{k}"] = bool(torch.equal(AP.natural_perm_cpu(H[hk].to(dev())).int(), R[k]["perm"]))  # solver device (GPU argsort)
    out["reference_perm_reproduced"] = rp
    import eval_partial_ppl as EP
    win, prot = EP.load_windows()
    out["windows"] = dict(sha256=prot["windows_sha256"], shape=list(win.shape))
    fam = jload(os.path.join(ROOT, AFS_RUN, "functional_activation_manifest.json"))["dense"]
    out["functional_capture"] = dict(source=AFS_RUN, nll_equals_historical_bf16_bitwise=fam["nll_equals_historical_bf16_bitwise"],
                                     files={L: v["sha256"] for L, v in fam["files"].items()},
                                     files_ok={L: sha256_file(v["path"]) == v["sha256"] for L, v in fam["files"].items()},
                                     definition="R_func numerator ||f_cfg - f_ORIGINAL||, denominator ||f_ORIGINAL - f_BF16||, "
                                                "FP32 SiLU-GLU on captured BF16 MLP inputs (runs_actorder_functional_stability)")
    hist = {g: jload(os.path.join(ROOT, DENSE_RUN, "results", f"{g}.json"))["ppl"] for g in GROUPS + ["ORIGINAL"]}
    out["historical_ppl"] = hist
    out["bf16_ppl"] = jload(os.path.join(ROOT, DENSE_RUN, "bf16_baseline.json"))["ppl"]
    out["all_ok"] = all(all(v.values()) for v in out["caches"].values()) and out["singletons_match"] and \
        all(all(v.values()) for v in art.values()) and all(rp.values()) and all(out["functional_capture"]["files_ok"].values())
    jdump(out, p("logs", "audit.json"))
    print("audit ok", out["all_ok"], "singletons", out["singletons_match"], flush=True)


# ---------------------------------------------------------------- strong-path pilot
PILOT_CRITERIA = dict(
    candidates={"b1": "each row alone (batch 1) [main family; reference strength]",
                "b2": "each historical batch of 8 split into consecutive pairs",
                "b4": "each historical batch of 8 split into consecutive halves",
                "b16": "consecutive historical batches merged pairwise (batch 16)",
                "sdpa8": "historical batches of 8, attention via torch SDPA instead of eager (same math)",
                "bf16full8": "historical batches of 8 with torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False"},
    pilot_rows="FR rows inside the first 8 historical batches that contain an FR row (with their batch-mates)",
    strength="median over target layers 1/9/19/27 of the pooled relative L2 difference ||X_alt - X_8|| / ||X_8|| of the "
             "FR rows' decoder-layer inputs",
    eligibility=["deterministic: a repeat of the candidate is bitwise identical",
                 "per-row well-defined: a different batch composition of the same size / path gives bitwise identical FR rows",
                 "changes >= 90 % of pilot FR rows at layer 27"],
    selection="the eligible candidate (other than b1) with the largest strength, adopted only if strength >= 2 x the b1 "
              "strength; otherwise no strong path (failure condition, Phase 3.3)",
    design_if_selected="dose 64, the same 8 nested replicate subsets as the main family, natural + fixed act-order, plus "
                       "the k = 128 full-path endpoint")


def stage_pilot_criteria(a):
    path = p("logs", "pilot_criteria.json")
    if os.path.exists(path):
        print("criteria exist"); return
    d = dict(criteria=PILOT_CRITERIA, sha256=obj_sha(PILOT_CRITERIA), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(d, path)
    print("pilot criteria", d["sha256"][:16])


def set_path(model, path):
    """Switch the alternate execution path (restored by set_path(model, None))."""
    if path == "sdpa8":
        model.config._attn_implementation = "sdpa"
    elif path == "bf16full8":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    else:
        model.config._attn_implementation = "eager"
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def path_batches(path, base, fr_set, alt=False, fillers=None):
    """Batch plan for an execution path over historical batches `base` (lists of 8 ids). alt=True: a different batch
    composition of the same sizes (non-FR batch-mates replaced by unused non-calibration filler rows) to test per-row
    well-definedness (batch-mate invariance)."""
    if alt:
        it = iter(fillers)
        base = [[r if r in fr_set else next(it) for r in b] for b in base]
    if path == "b1":
        return [[r] for b in base for r in b if r in fr_set]
    if path in ("b8", "sdpa8", "bf16full8"):
        return base
    if path in ("b2", "b4"):
        s = 2 if path == "b2" else 4
        return [b[i:i + s] for b in base for i in range(0, len(b), s)]
    if path == "b16":
        return [base[i] + (base[i + 1] if i + 1 < len(base) else []) for i in range(0, len(base), 2)]
    raise ValueError(path)


def capture_plan(model, alltok, batches, rows):
    import batch_sensitivity as DB
    keep = {}

    def on_layer(L, X):
        if L in LAY:
            keep[L] = X.clone()
    DB.forward_plan(model, alltok, batches, rows, on_layer)
    return keep


def stage_pilot(a):
    log_code_state("pilot")
    crit = jload(p("logs", "pilot_criteria.json"))
    assert crit and crit["sha256"] == obj_sha(PILOT_CRITERIA), "pilot criteria missing or changed"
    if jload(p("strong_path_pilot.json")):
        print("pilot done"); return
    import dense_grouping_scope as DS, batch_sensitivity as DB
    from moe_calib import batch_sensitivity as BS
    union, fr, _, _ = calib()
    fr_set = set(fr)
    alltok = DB.all_tokens()
    base_all = BS.original_batches(union, 8, int(alltok.shape[0]))
    base = [b for b in base_all if any(r in fr_set for r in b)][:8]
    rows = [r for b in base for r in b if r in fr_set]
    used = set(r for b in base_all for r in b)
    fillers = [r for r in range(int(alltok.shape[0])) if r not in used]
    model, _ = DS.load_model()
    t0 = time.time()
    set_path(model, None)
    X8 = capture_plan(model, alltok, base, rows)
    O = {L: torch.load(os.path.join(DCACHE, "cache", "ORIGINAL", f"L{L}.pt"), mmap=True) for L in LAY}
    uidx = [union.index(r) for r in rows]
    b8_equals_cache = all(torch.equal(X8[L], O[L][uidx]) for L in LAY)
    res = dict(rows=rows, n_rows=len(rows), b8_equals_historical_cache=b8_equals_cache, candidates={})
    for path in ("b1", "b2", "b4", "b16", "sdpa8", "bf16full8"):
        set_path(model, path)
        try:
            bt = path_batches(path, base, fr_set)
            X = capture_plan(model, alltok, bt, rows)
            Xr = capture_plan(model, alltok, bt, rows)
            Xa = capture_plan(model, alltok, path_batches(path, base, fr_set, alt=True, fillers=fillers), rows)
        finally:
            set_path(model, None)
        per = {}
        for L in LAY:
            d = (X[L].float() - X8[L].float())
            per[str(L)] = dict(rel_l2=float(d.norm() / X8[L].float().norm()),
                               frac_rows_changed=float(np.mean([not torch.equal(X[L][i], X8[L][i]) for i in range(len(rows))])),
                               frac_elements_changed=float((X[L] != X8[L]).float().mean()),
                               median_row_rel=float(torch.tensor([float((X[L][i].float() - X8[L][i].float()).norm() / X8[L][i].float().norm())
                                                                  for i in range(len(rows))]).median()))
        res["candidates"][path] = dict(
            per_layer=per, strength=float(np.median([per[str(L)]["rel_l2"] for L in LAY])),
            deterministic=all(torch.equal(X[L], Xr[L]) for L in LAY),
            per_row_well_defined=all(torch.equal(X[L], Xa[L]) for L in LAY),
            changes_90pct_L27=per["27"]["frac_rows_changed"] >= 0.9)
        print(path, res["candidates"][path]["strength"], res["candidates"][path]["deterministic"],
              res["candidates"][path]["per_row_well_defined"], per["27"]["frac_rows_changed"], flush=True)
    s1 = res["candidates"]["b1"]["strength"]
    elig = {k: v for k, v in res["candidates"].items() if k != "b1" and v["deterministic"] and v["per_row_well_defined"]
            and v["changes_90pct_L27"]}
    best = max(elig, key=lambda k: elig[k]["strength"]) if elig else None
    sel = best if best and elig[best]["strength"] >= 2 * s1 else None
    res.update(b1_strength=s1, eligible=sorted(elig), best_eligible=best,
               best_over_b1=elig[best]["strength"] / s1 if best else None, selected=sel,
               rule=PILOT_CRITERIA["selection"], criteria_sha256=crit["sha256"], seconds=time.time() - t0)
    jdump(res, p("strong_path_pilot.json"))
    print("selected", sel, "best", best, res["best_over_b1"], flush=True)


# ---------------------------------------------------------------- protocol
def stage_protocol(a):
    au = jload(p("logs", "audit.json"))
    assert au and au["all_ok"]
    pil = jload(p("strong_path_pilot.json"))
    assert pil, "pilot must precede the protocol (the pilot never evaluates PPL)"
    if os.path.exists(p("protocol.json")):
        raise SystemExit("protocol already frozen")
    union, fr, fr_rows, cal = calib()
    reps = replicate_perms()
    sel = pil["selected"]
    pr = dict(
        title="Dense GPTQ numerical-perturbation dose-response and functional stability",
        model=dict(id="Qwen/Qwen2.5-1.5B", revision="8faed761d45a263340a0528343f099c05c9a4323", layers=LAY, projections=list(PROJ)),
        quantization=dict(**GPTQ, act_order_policies=list(POLICIES), solver="moe_calib.gptq.gptq_quantize (default path unchanged)",
                          natural="argsort(diag H_cfg, descending) inside the solver",
                          fixed="actorder_override = natural permutation of the historical ORIGINAL Hessian (per projection); "
                                "Hessian, scales, zero-points, error feedback live",
                          install="fp32 W_q -> bf16 (set_expert_weights cast); every solve starts from pristine BF16 weights"),
        calibration=dict(manifest_sha256=obj_sha(cal), fr_order=fr, fr_union_rows=fr_rows, union_ids=union, K_blocks=cal["K_blocks"],
                         rule="identical 128 sequences, identical order and K=4 blocks in every configuration"),
        perturbation=dict(
            X8="historical ORIGINAL layer-input cache (~/moe_cache_dense_grouping_scope/cache/ORIGINAL), batch 8",
            X1="batch-size-1 extraction of each of the 128 FR rows (forward_plan, one row per batch, pristine BF16), built once",
            hybrid="cache = X8 with the rows in S replaced by X1 (no interpolation, no synthetic noise)",
            doses_main=[0] + DOSES_MAIN + [128], doses_optional=DOSES_OPT, n_replicates=N_REP,
            replicates=reps, replicate_rule="replicate r: default_rng(20261301 + r).permutation of the FR order (seed advanced "
                                            "by 1000 if its first row duplicates an earlier replicate's); S_k,r = first k rows (nested)",
            endpoint="k = 128: one configuration (no replicates; a rerun is a correctness check only)",
            historical_singletons=HIST_SINGLETON,
            validation=["X1 of each historical singleton row == the historical grouping cache row (bitwise)",
                        "hybrid for each historical singleton == historical grouping cache (all 217 rows, bitwise)",
                        "physical mixed execution (S rows batch 1, remaining historical batches with non-calibration fillers "
                        "replacing S) == hybrid (all 217 rows) for replicate 0 at k = 4 and k = 16",
                        "historical singleton configs reproduce historical Hessians, permutations, codes, weights, PPL bitwise"]),
        strong_path=dict(pilot="strong_path_pilot.json (activation differences only; no PPL)", pilot_criteria_sha256=pil["criteria_sha256"],
                         selected=sel, best_eligible=pil["best_eligible"], best_over_b1=pil["best_over_b1"],
                         design=dict(dose=64, replicates="same 8 nested subsets", endpoint=128) if sel else None,
                         failure_rule="if none selected: complete the main family, report its range, no extrapolation to MoE claimed"),
        metrics=dict(
            hessian="D_H = ||H_cfg - H_0||_F / ||H_0||_F per layer & projection (gu, down) + historical trace-normalized "
                    "(moe_calib.frozen_routing.hessian_metrics rel_fro_trace_normalized); median / mean / per layer; diag "
                    "change, act-order position and group changes; calibration activation perturbation norms; principal "
                    "scalar D_H = sqrt(sum_l,p ||dH||^2 / sum_l,p ||H_0||^2) (pooled)",
            R_func="sqrt(sum_l ||f_cfg,l - f_0,l||^2) / sqrt(sum_l ||f_0,l - f_BF16,l||^2): full FP32 SiLU-GLU MLP outputs on the "
                   "runs_actorder_functional_stability held-out BF16 MLP inputs; f_0 = historical natural ORIGINAL (= k=0 both "
                   "policies); fixed denominator (PRIMARY functional metric)",
            secondary_functional=["MLP RMSE", "relative to BF16 output norm", "per layer", "per projection ||dW X|| / ||W_BF16 X||",
                                  "per-token p50/p95/p99", "integer-code disagreement", "rel Frobenius"],
            ppl="PPL, log PPL, per-window NLL on runs_partial_ppl/windows.pt; repeat-eval parity on k=0, historical singletons, k=128",
            variability="s_k = SD_r(PPL_k,r), s_k^log (ddof=1) per dose with >1 distinct subset; R_PPL,k = s_k / Q_0 with "
                        "Q_0 = PPL_0 - PPL_BF16 (PRIMARY); R_PPL,k^mean = s_k / (mean_k - PPL_BF16); log companion "
                        "SD(log PPL) / (log PPL_0 - log PPL_BF16)",
            moe_reference="historical MoE FR s0: R_func 0.82 (block incl. shared expert, pooled, vs ORIGINAL quant error), "
                          "R_PPL recomputed with both denominators"),
        statistics=dict(
            unit="replicate trajectory (8); nested doses within a replicate are correlated",
            bootstrap="2000 resamples of whole replicate trajectories (seed 20261315) for mean / RMS R_func, s_k and ratios; "
                      "also chi-square 95 % CI for s_k",
            power_law="OLS of log R_func on log D_H over all configs with D_H > 0 per policy; slope b with trajectory-bootstrap CI; "
                      "supported if R^2 >= 0.8",
            ppl_models="Y = R_PPL vs X = dose RMS R_func: M1 aX, M2 a sqrt(X), M3 aX/(b+X); fitted by maximum likelihood on "
                       "replicate-level PPL (per-dose free means, sd_k = Q_0 * Y(X_k)); AIC compared; exploratory",
            mitigation="R_mitigation,k = RMS fixed / RMS natural R_func; R_SD,k = SD fixed / SD natural (k with replicates only); "
                       "paired per-replicate differences in R_func, |dlogPPL from ORIGINAL| and PPL"),
        interpretation=dict(
            monotonic="Spearman(D_H, R_func) >= 0.8 over configs AND dose means non-decreasing in k (<= 1 violation)",
            dynamic_range="insufficient for an MoE comparison if max dose RMS R_func (natural) < 0.41 (= 0.5 x 0.82)",
            solver_saturation="claimed only if the log-log slope between the two largest D_H doses < 0.25 while D_H increases >= 2x",
            evaluation_saturation="claimed only if range is sufficient AND M3 beats M1 and M2 by AIC >= 2",
            mitigation_effective="R_mitigation,k CI upper < 1 at that dose",
            ppl_benefit="pooled paired mean of (dlogPPL_nat^2 - dlogPPL_fix^2) over replicates x doses with trajectory-bootstrap "
                        "CI excluding 0"),
        reference_matrix=dict(design="Q_ij = GPTQ(H_j, pi_i), i, j in G20261111..15 historical Hessians; pi_i = natural "
                                     "permutation of H_i; diagonal = historical natural; G20261115 = ORIGINAL",
                              metrics=["mean / SD PPL per reference over j", "R_func vs natural ORIGINAL and vs Q_i,5",
                                       "penalty", "act-order disagreement between references"]),
        parity=["k=0 natural == historical ORIGINAL weights bitwise", "k=0 fixed == k=0 natural bitwise",
                "historical singleton parity (Hessians, perms, codes, weights, PPL)", "repeat eval bitwise",
                "only 12 target tensors change; restore to pristine"],
        cache_dir=CACHE, code_sha256_at_freeze=code_hashes())
    pr["protocol_sha256"] = obj_sha(pr)
    pr["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(pr, p("protocol.json"))
    shutil.copy(p("protocol.json"), p("logs", "protocol_readonly_copy.json"))
    os.chmod(p("logs", "protocol_readonly_copy.json"), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    jdump(dict(replicates=reps, configs=configs(pr)), p("perturbation_manifest.json"))
    jdump(dict(union_ids=union, fr_order=fr, fr_union_rows=fr_rows, K_blocks=cal["K_blocks"],
               source_manifest=os.path.join(DENSE_RUN, "calibration_manifest.json"), source_sha256=obj_sha(cal)),
          p("calibration_manifest.json"))
    print("protocol", pr["protocol_sha256"][:16], "strong path", sel, flush=True)


# ---------------------------------------------------------------- alternate caches + validation
def alt_cache_path(fam, L):
    return c("alt", fam, f"L{L}.pt")


def stage_caches(a):
    pr = check_protocol(); log_code_state("caches")
    import dense_grouping_scope as DS, batch_sensitivity as DB
    from moe_calib import batch_sensitivity as BS
    union, fr, _, _ = calib()
    fr_set = set(fr)
    alltok = DB.all_tokens()
    val = jload(p("activation_cache_validation.json")) or {}
    model, _ = DS.load_model()
    fams = ["b1"] + (["sp"] if pr["strong_path"]["selected"] else [])
    base_all = BS.original_batches(union, 8, int(alltok.shape[0]))
    base_fr = [b for b in base_all if any(r in fr_set for r in b)]
    for fam in fams:
        if all(os.path.exists(alt_cache_path(fam, L)) for L in LAY):
            continue
        t0 = time.time()
        path = "b1" if fam == "b1" else pr["strong_path"]["selected"]
        set_path(model, path)
        try:
            bt = path_batches(path, base_fr, fr_set)
            X = capture_plan(model, alltok, bt, fr)
        finally:
            set_path(model, None)
        for L in LAY:
            tsave(X[L], alt_cache_path(fam, L))
        cost_add("caches", fam, time.time() - t0)
        O = {L: torch.load(os.path.join(DCACHE, "cache", "ORIGINAL", f"L{L}.pt"), mmap=True) for L in LAY}
        uidx = [union.index(r) for r in fr]
        val[fam] = dict(path=path, n_batches=len(bt), files={str(L): dict(sha256=sha256_file(alt_cache_path(fam, L)),
                                                                          shape=list(X[L].shape)) for L in LAY},
                        per_layer={str(L): dict(rel_l2=float((X[L].float() - O[L][uidx].float()).norm() / O[L][uidx].float().norm()),
                                                frac_rows_changed=float(np.mean([not torch.equal(X[L][i], O[L][uidx[i]]) for i in range(len(fr))])),
                                                row_rel=[float((X[L][i].float() - O[L][uidx[i]].float()).norm() / O[L][uidx[i]].float().norm())
                                                         for i in range(len(fr))])
                                   for L in LAY}, seconds=time.time() - t0)
        jdump(val, p("activation_cache_validation.json"))
        print(f"[cache] {fam}: {val[fam]['per_layer']['27']['rel_l2']:.3e} rows changed L27 {val[fam]['per_layer']['27']['frac_rows_changed']}", flush=True)
    # historical singleton validation
    if "historical_singletons" not in val:
        X1 = {L: torch.load(alt_cache_path("b1", L)) for L in LAY}
        hs = {}
        for g, s in HIST_SINGLETON.items():
            ok_row, ok_full = True, True
            for L in LAY:
                G = torch.load(os.path.join(DCACHE, "cache", g, f"L{L}.pt"), mmap=True)
                Hy = hybrid(L, [s], "b1", X1)
                ok_row &= torch.equal(X1[L][fr.index(s)], G[union.index(s)])
                ok_full &= torch.equal(Hy, G)
            hs[g] = dict(row=s, x1_row_equals_historical=bool(ok_row), hybrid_equals_historical_cache=bool(ok_full))
        val["historical_singletons"] = hs
        jdump(val, p("activation_cache_validation.json"))
        print("[hist]", hs, flush=True)
    # physical mixed execution
    reps = pr["perturbation"]["replicates"]
    for k in (4, 16):
        key = f"physical_b1_r0_k{k}"
        if key in val:
            continue
        S = reps[0]["order"][:k]
        used = set(r for b in base_all for r in b)
        fillers = [r for r in range(int(alltok.shape[0])) if r not in used]
        plan, fi = [], 0
        for b in base_all:
            nb = []
            for r in b:
                if r in S:
                    nb.append(fillers[fi]); fi += 1
                else:
                    nb.append(r)
            plan.append(nb)
        plan += [[s] for s in S]
        set_path(model, "b1")
        Xp = capture_plan(model, alltok, plan, union)
        X1 = {L: torch.load(alt_cache_path("b1", L)) for L in LAY}
        eq = {str(L): bool(torch.equal(Xp[L], hybrid(L, S, "b1", X1))) for L in LAY}
        val[key] = dict(S=S, n_fillers=fi, filler_ids=fillers[:fi], all_217_rows_bitwise=eq, ok=all(eq.values()))
        jdump(val, p("activation_cache_validation.json"))
        print("[physical]", key, eq, flush=True)
    if "sp" in fams and "physical_sp_r0_k16" not in val:
        val["physical_sp_r0_k16"] = dict(note="strong path is an execution-path switch applied to S rows' own historical batches; "
                                              "per-row well-definedness verified in the pilot (batch-mate invariance)")
        jdump(val, p("activation_cache_validation.json"))
    val["all_ok"] = all(v.get("x1_row_equals_historical") and v.get("hybrid_equals_historical_cache")
                        for v in val["historical_singletons"].values()) and \
        all(val[f"physical_b1_r0_k{k}"]["ok"] for k in (4, 16))
    jdump(val, p("activation_cache_validation.json"))
    print("cache validation ok", val["all_ok"], flush=True)


_ORIG = {}


def hybrid(L, S, fam, Xalt=None):
    union, fr, _, _ = calib()
    if L not in _ORIG:
        _ORIG[L] = torch.load(os.path.join(DCACHE, "cache", "ORIGINAL", f"L{L}.pt"))
    X = _ORIG[L].clone()
    if S:
        A = Xalt[L] if Xalt is not None else torch.load(alt_cache_path(fam, L), mmap=True)
        ui = torch.tensor([union.index(s) for s in S])
        fi = torch.tensor([fr.index(s) for s in S])
        X[ui] = A[fi]
    return X


# ---------------------------------------------------------------- GPU: Hessians, solves, PPL
def cfg_dir(name):
    return c("cfg", name)


def stage_gpu(a):
    pr = check_protocol(); log_code_state("gpu")
    val = jload(p("activation_cache_validation.json"))
    assert val and val["all_ok"], "activation cache validation failed -> BLOCKED"
    from moe_calib import dense_gptq as DG, frozen_routing as FRT
    import dense_grouping_scope as DS, eval_partial_ppl as EP
    union, fr, fr_rows, cal = calib()
    D = dev()
    cfgs = [x for x in configs(pr) if a.only is None or x["name"].startswith(tuple(a.only.split(",")))]
    if not a.optional:
        cfgs = [x for x in cfgs if not x.get("optional")]
    win, prot = EP.load_windows()
    model, _ = DS.load_model()
    vc = DS.verify_checkpoint(model)
    assert vc["all_equal"], vc
    snap = DG.snapshot(model, LAY)
    tgt0, unt0 = DG.target_digests(model, LAY), DG.untouched_digests(model, LAY)
    W0 = {L: {k: w.detach().cpu().clone() for k, w in DG.get_weights(model, L).items()} for L in LAY}
    H0 = {L: torch.load(os.path.join(DCACHE, "hessians", "ORIGINAL", f"L{L}.pt")) for L in LAY}
    ref_perm = {(L, k): AP.natural_perm_cpu(H0[L]["down" if k == "down_proj" else "gu"].to(D)) for L in LAY for k in PROJ}
    corr = jload(p("actorder_correctness.json")) or {}
    Xalt = {}
    os.makedirs(p("results"), exist_ok=True)
    for cf in cfgs:
        name = cf["name"]
        rp = p("results", f"{name}.json")
        if (jload(rp) or {}).get("complete"):
            continue
        t_all = time.time()
        fam = cf["family"]
        if cf["S"] and fam not in Xalt:
            Xalt = {fam: {L: torch.load(alt_cache_path(fam, L)) for L in LAY}}
        rec = dict(config=cf, hessian={}, policies={})
        th = tq = 0.0
        for L in LAY:
            t0 = time.time()
            X = hybrid(L, cf["S"], fam, Xalt.get(fam))
            prof = DG.DenseProfiler(model, L, cal["K_blocks"]).run(X, fr_rows)
            H = {"gu": prof.H("gu"), "down": prof.H("down")}
            ntok = dict(gu=prof.tokens("gu"), down=prof.tokens("down"))
            th += time.time() - t0
            tsave(dict(gu=H["gu"].cpu(), down=H["down"].cpu(), tokens=ntok), os.path.join(cfg_dir(name), f"H_L{L}.pt"))
            for hk in ("gu", "down"):
                m = FRT.hessian_metrics(H[hk].cpu(), H0[L][hk])
                d0, d1 = AP.solver_diag(H0[L][hk]).double(), AP.solver_diag(H[hk]).double()
                pn = AP.natural_perm_cpu(H[hk])
                po = ref_perm[(L, "down_proj" if hk == "down" else "gate_proj")]
                pos_a, pos_b = torch.empty_like(pn), torch.empty_like(po)
                pos_a[pn] = torch.arange(len(pn)); pos_b[po] = torch.arange(len(po))
                m.update(sq_diff=float((H[hk].cpu().double() - H0[L][hk].double()).pow(2).sum()),
                         sq_ref=float(H0[L][hk].double().pow(2).sum()),
                         diag_rel_change_median=float(((d1 - d0) / d0).abs().median()),
                         frac_actorder_pos_changed=float((pn != po).float().mean()),
                         frac_group_changed=float(((pos_a // 128) != (pos_b // 128)).float().mean()), tokens=ntok[hk])
                rec["hessian"][f"L{L}|{hk}"] = m
            # calibration activation perturbation at this layer input
            if cf["S"]:
                Oh = hybrid(L, [], fam)
                rec["hessian"][f"L{L}|act"] = dict(rel_l2_fr_rows=float((X[fr_rows].float() - Oh[fr_rows].float()).norm() / Oh[fr_rows].float().norm()))
            # solves
            tq0 = time.time()
            for pol in POLICIES:
                wl, cl, meta = {}, {}, {}
                for k in PROJ:
                    hk = "down" if k == "down_proj" else "gu"
                    info = {}
                    Q = AP.solve(W0[L][k], H[hk], pol, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"],
                                 ref_perm=ref_perm[(L, k)], info=info)
                    q, gid, ex = DG.integer_codes(Q, info, GPTQ["bits"], GPTQ["group_size"])
                    wl[k] = Q.to(torch.bfloat16).cpu()
                    cl[k] = dict(q=q, group=gid, perm=info["perm"].int(), scale=info["scale"], zero=info["zero"])
                    meta[k] = dict(codes_exact=ex, finite=bool(torch.isfinite(Q).all()), weight_sha256=tensor_sha(wl[k]))
                tsave(dict(weights=wl, codes=cl, meta=meta), os.path.join(cfg_dir(name), f"Q_{pol}_L{L}.pt"))
                rec["policies"].setdefault(pol, dict(meta={}))["meta"][str(L)] = meta
            tq += time.time() - tq0
            del H, prof, X
            torch.cuda.empty_cache()
        # PPL per policy
        repeat = name in ("b1_k000", "b1_k128", "sp_k128") or name.startswith("hist_")
        for pol in POLICIES:
            assert DG.target_digests(model, LAY) == tgt0
            Wq = {L: torch.load(os.path.join(cfg_dir(name), f"Q_{pol}_L{L}.pt"))["weights"] for L in LAY}
            for L in LAY:
                DG.set_weights(model, L, Wq[L])
            inst_ok = all(torch.equal(DG.get_weights(model, L)[k].detach().cpu(), Wq[L][k]) for L in LAY for k in PROJ)
            dig = DG.installed_digest(model, LAY)
            unt_ok = DG.untouched_digests(model, LAY) == unt0
            n1, k1, s1 = DS.evaluate(model, win)
            r = EP.ppl_record(n1, k1, s1)
            if repeat:
                n2, _, _ = DS.evaluate(model, win)
                r["repeat_per_window_bitwise"] = bool(np.array_equal(n1, n2))
            r.update(installed_weights_sha256=dig, installed_equals_solver=inst_ok, untouched_unchanged=unt_ok,
                     installed_unchanged_during_eval=DG.installed_digest(model, LAY) == dig)
            DG.restore(model, snap)
            eq, tot = DG.matches(model, snap)
            r["restored"] = eq == tot and DG.target_digests(model, LAY) == tgt0
            rec["policies"][pol].update(r)
        # parity checks
        nat = {L: torch.load(os.path.join(cfg_dir(name), f"Q_natural_L{L}.pt")) for L in LAY}
        if name == "b1_k000":
            fx = {L: torch.load(os.path.join(cfg_dir(name), f"Q_fixed_L{L}.pt")) for L in LAY}
            corr["k0_natural_equals_historical_ORIGINAL"] = all(torch.equal(nat[L]["weights"][k], torch.load(os.path.join(DCACHE, "weights", "ORIGINAL", f"L{L}.pt"))[k])
                                                                for L in LAY for k in PROJ)
            corr["k0_fixed_equals_natural"] = all(torch.equal(nat[L]["weights"][k], fx[L]["weights"][k]) for L in LAY for k in PROJ)
            corr["k0_hessians_equal_historical"] = all(torch.equal(torch.load(os.path.join(cfg_dir(name), f"H_L{L}.pt"))[hk], H0[L][hk])
                                                       for L in LAY for hk in ("gu", "down"))
            h = jload(os.path.join(ROOT, DENSE_RUN, "results", "ORIGINAL.json"))
            corr["k0_ppl_per_window_equals_historical"] = bool(np.array_equal(np.array(rec["policies"]["natural"]["per_window_nll"]),
                                                                              np.array(h["per_window_nll"])))
        if name.startswith("hist_"):
            g = cf["historical"]
            HG = {L: torch.load(os.path.join(DCACHE, "hessians", g, f"L{L}.pt")) for L in LAY}
            CG = {L: torch.load(os.path.join(DCACHE, "codes", g, f"L{L}.pt")) for L in LAY}
            WG = {L: torch.load(os.path.join(DCACHE, "weights", g, f"L{L}.pt")) for L in LAY}
            h = jload(os.path.join(ROOT, DENSE_RUN, "results", f"{g}.json"))
            corr[f"hist|{g}"] = dict(
                hessians=all(torch.equal(torch.load(os.path.join(cfg_dir(name), f"H_L{L}.pt"))[hk], HG[L][hk]) for L in LAY for hk in ("gu", "down")),
                perms=all(torch.equal(nat[L]["codes"][k]["perm"], CG[L][k]["perm"]) for L in LAY for k in PROJ),
                codes=all(torch.equal(nat[L]["codes"][k]["q"], CG[L][k]["q"]) for L in LAY for k in PROJ),
                weights=all(torch.equal(nat[L]["weights"][k], WG[L][k]) for L in LAY for k in PROJ),
                ppl=bool(np.array_equal(np.array(rec["policies"]["natural"]["per_window_nll"]), np.array(h["per_window_nll"]))))
        rec.update(hessian_s=th, gptq_s=tq, total_s=time.time() - t_all, complete=True,
                   valid=all(rec["policies"][pp]["installed_equals_solver"] and rec["policies"][pp]["untouched_unchanged"]
                             and rec["policies"][pp]["restored"] and rec["policies"][pp]["installed_unchanged_during_eval"]
                             and rec["policies"][pp].get("repeat_per_window_bitwise", True) for pp in POLICIES))
        jdump(rec, rp)
        jdump(corr, p("actorder_correctness.json"))
        cost_add("gpu", name, rec["total_s"])
        print(f"[{name}] PPL nat {rec['policies']['natural']['ppl']:.6f} fix {rec['policies']['fixed']['ppl']:.6f} "
              f"D_H(L19 down) {rec['hessian']['L19|down']['rel_fro_raw']:.2e} valid {rec['valid']} {rec['total_s']:.0f}s", flush=True)
        if name == "b1_k000" and not all(corr[x] for x in ("k0_natural_equals_historical_ORIGINAL", "k0_fixed_equals_natural",
                                                           "k0_hessians_equal_historical", "k0_ppl_per_window_equals_historical")):
            raise SystemExit("k=0 parity failed -> BLOCKED")
        if name.startswith("hist_") and not all(corr[f"hist|{cf['historical']}"].values()):
            raise SystemExit(f"historical singleton parity failed for {name} -> BLOCKED")
        if not rec["valid"]:
            raise SystemExit(f"{name}: integrity failed -> BLOCKED")
        gc.collect()
    n3, _, _ = DS.evaluate(model, win)
    bf = jload(os.path.join(ROOT, DENSE_RUN, "bf16_baseline.json"))
    corr["end_" + time.strftime("%H%M%S")] = dict(bf16_reeval_bitwise=bool(np.array_equal(n3, np.array(bf["per_window_nll"]))),
                                                  untouched_unchanged=DG.untouched_digests(model, LAY) == unt0,
                                                  targets_pristine=DG.target_digests(model, LAY) == tgt0)
    jdump(corr, p("actorder_correctness.json"))


# ---------------------------------------------------------------- 5 x 5 reference-order matrix
def stage_refmatrix(a):
    pr = check_protocol(); log_code_state("refmatrix")
    from moe_calib import dense_gptq as DG
    import dense_grouping_scope as DS, eval_partial_ppl as EP
    D = dev()
    path = p("logs", "refmatrix_raw.json")
    out = jload(path) or {}
    win, _ = EP.load_windows()
    model, _ = DS.load_model()
    snap = DG.snapshot(model, LAY)
    tgt0, unt0 = DG.target_digests(model, LAY), DG.untouched_digests(model, LAY)
    W0 = {L: {k: w.detach().cpu().clone() for k, w in DG.get_weights(model, L).items()} for L in LAY}
    perms = {i: {L: torch.load(os.path.join(DCACHE, "codes", i, f"L{L}.pt")) for L in LAY} for i in GROUPS}
    perms = {i: {(L, k): perms[i][L][k]["perm"].long() for L in LAY for k in PROJ} for i in GROUPS}
    prev_fixed = os.path.join(ACACHE, "ao", "fixed")
    for j in GROUPS:
        Hj = {L: torch.load(os.path.join(DCACHE, "hessians", j, f"L{L}.pt")) for L in LAY}
        for i in GROUPS:
            key = f"{i}|{j}"
            if key in out:
                continue
            t0 = time.time()
            wq, codes = {}, {}
            for L in LAY:
                wq[L], codes[L] = {}, {}
                for k in PROJ:
                    hk = "down" if k == "down_proj" else "gu"
                    info = {}
                    Q = AP.solve(W0[L][k], Hj[L][hk].to(D), "fixed", GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"],
                                 ref_perm=perms[i][(L, k)], info=info)
                    q, gid, _ = DG.integer_codes(Q, info, GPTQ["bits"], GPTQ["group_size"])
                    wq[L][k] = Q.to(torch.bfloat16).cpu()
                    codes[L][k] = dict(q=q, group=gid, perm=info["perm"].int(), scale=info["scale"], zero=info["zero"])
            tsave(dict(weights=wq, codes=codes), c("refmatrix", f"{i}__{j}.pt"))
            chk = {}
            if i == j:
                chk["diag_equals_historical_natural"] = all(torch.equal(wq[L][k], torch.load(os.path.join(DCACHE, "weights", j, f"L{L}.pt"))[k])
                                                            for L in LAY for k in PROJ)
            if i == "G20261115":
                chk["equals_previous_fixed_run"] = all(torch.equal(wq[L][k], torch.load(os.path.join(prev_fixed, j, f"L{L}.pt"))["weights"][k])
                                                       for L in LAY for k in PROJ)
            for L in LAY:
                DG.set_weights(model, L, wq[L])
            n1, k1, s1 = DS.evaluate(model, win)
            r = EP.ppl_record(n1, k1, s1)
            DG.restore(model, snap)
            r.update(checks=chk, untouched_unchanged=DG.untouched_digests(model, LAY) == unt0,
                     restored=DG.target_digests(model, LAY) == tgt0, seconds=time.time() - t0)
            out[key] = {kk: v for kk, v in r.items()}
            jdump(out, path)
            print(f"[ref] pi={i} H={j} PPL {r['ppl']:.6f} {chk}", flush=True)
            if chk and not all(chk.values()):
                raise SystemExit("reference-matrix parity failed -> BLOCKED")


# ---------------------------------------------------------------- functional
@torch.no_grad()
def func_cache():
    """f_0 (natural ORIGINAL) and f_BF16 MLP outputs per layer + projection-level BF16 output norms (computed once)."""
    path = c("func_ref.pt")
    if os.path.exists(path):
        return torch.load(path)
    FD_ = FD
    D = dev()
    import actorder_functional_stability as AFS
    W0 = AFS.dense_pristine()
    out = {}
    for L in LAY:
        act = torch.load(os.path.join(ACACHE, "act", "dense", f"L{L}.pt"), mmap=True)
        Wo = {k: v.to(D).float() for k, v in torch.load(os.path.join(DCACHE, "weights", "ORIGINAL", f"L{L}.pt")).items()}
        Wb = {k: v.to(D).float() for k, v in W0[L].items()}
        f0, fb, dens = [], [], {k: 0.0 for k in PROJ}
        for i in range(0, act["X"].shape[0], CHUNK):
            x = act["X"][i:i + CHUNK].to(D).float(); xd = act["Xd"][i:i + CHUNK].to(D).float()
            f0.append(FD_.mlp_fp32(x, Wo["gate_proj"], Wo["up_proj"], Wo["down_proj"]).cpu())
            fb.append(FD_.mlp_fp32(x, Wb["gate_proj"], Wb["up_proj"], Wb["down_proj"]).cpu())
            for k in PROJ:
                xin = xd if k == "down_proj" else x
                dens[k] += float((xin @ Wb[k].T).double().pow(2).sum())
        out[L] = dict(f0=torch.cat(f0), fb=torch.cat(fb), proj_den=dens)
    tsave(out, path)
    return out


@torch.no_grad()
def functional_one(wq, ref, Wo, wref=None):
    """wq {L: {proj: bf16}} vs natural ORIGINAL: MLP sums, per-token norms, projection-level ||dW X||."""
    fp32 = FD
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    D = dev()
    res = {}
    for L in LAY:
        act = torch.load(os.path.join(ACACHE, "act", "dense", f"L{L}.pt"), mmap=True)
        W = {k: wq[L][k].to(D).float() for k in PROJ}
        dW = {k: W[k] - Wo[L][k].to(D).float() for k in PROJ}
        Wr = {k: wref[L][k].to(D).float() for k in PROJ} if wref is not None else None
        num = numb = numr = 0.0
        pnum = {k: 0.0 for k in PROJ}
        tok = []
        for i in range(0, act["X"].shape[0], CHUNK):
            x = act["X"][i:i + CHUNK].to(D).float(); xd = act["Xd"][i:i + CHUNK].to(D).float()
            f = fp32.mlp_fp32(x, W["gate_proj"], W["up_proj"], W["down_proj"])
            d = f - ref[L]["f0"][i:i + CHUNK].to(D)
            num += float(d.double().pow(2).sum())
            numb += float((f - ref[L]["fb"][i:i + CHUNK].to(D)).double().pow(2).sum())
            if Wr is not None:
                numr += float((f - fp32.mlp_fp32(x, Wr["gate_proj"], Wr["up_proj"], Wr["down_proj"])).double().pow(2).sum())
            tok.append(d.norm(dim=1).cpu())
            for k in PROJ:
                xin = xd if k == "down_proj" else x
                pnum[k] += float((xin @ dW[k].T).double().pow(2).sum())
        t = torch.cat(tok).double()
        q = torch.quantile(t, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
        res[str(L)] = dict(sq_diff_vs_orig=num, sq_diff_vs_bf16=numb, sq_diff_vs_wref=numr if Wr is not None else None, tok_p50=float(q[0]), tok_p95=float(q[1]), tok_p99=float(q[2]),
                           tok_mean=float(t.mean()), n_el=int(act["X"].shape[0]) * int(ref[L]["f0"].shape[1]),
                           proj_sq_diff=pnum)
    return res


def code_rates(cA, cB):
    n = ch = lv = 0
    per = {}
    from moe_calib import code_spatial as CS
    for L in LAY:
        for k in PROJ:
            a_, b_ = cA[L][k], cB[L][k]
            Sa, Za = CS.element_grid(a_); Sb, Zb = CS.element_grid(b_)
            cd = a_["q"] != b_["q"]
            l_ = ((a_["q"].float() - Za) != (b_["q"].float() - Zb))
            per[f"L{L}|{k}"] = dict(code=float(cd.float().mean()), level=float(l_.float().mean()),
                                    group=float((a_["group"] != b_["group"]).float().mean()))
            n += cd.numel(); ch += int(cd.sum()); lv += int(l_.sum())
    return dict(code=ch / n, level=lv / n, per=per)


def stage_func(a):
    check_protocol(); log_code_state("func")
    ref = func_cache()
    Wo = {L: torch.load(os.path.join(DCACHE, "weights", "ORIGINAL", f"L{L}.pt")) for L in LAY}
    Co = {L: torch.load(os.path.join(DCACHE, "codes", "ORIGINAL", f"L{L}.pt")) for L in LAY}
    import actorder_functional_stability as AFS
    W0 = AFS.dense_pristine()
    path = p("logs", "functional_raw.json")
    out = jload(path) or {}
    if "_ref" not in out:
        out["_ref"] = {str(L): dict(sq_quant=float((ref[L]["f0"].double() - ref[L]["fb"].double()).pow(2).sum()),
                                    sq_bf16=float(ref[L]["fb"].double().pow(2).sum()), proj_den=ref[L]["proj_den"]) for L in LAY}
        # consistency with runs_actorder_functional_stability: historical G20261111 natural
        r = functional_one({L: torch.load(os.path.join(DCACHE, "weights", "G20261111", f"L{L}.pt")) for L in LAY}, ref, Wo)
        num = sum(r[str(L)]["sq_diff_vs_orig"] for L in LAY)
        den = sum(out["_ref"][str(L)]["sq_bf16"] for L in LAY)
        prev = jload(os.path.join(ROOT, AFS_RUN, "dense_functional_difference.json"))["pooled_over_layers"]["G20261111__ORIGINAL"]["mlp"]
        out["_consistency_G20261111"] = dict(E_mlp_here=math.sqrt(num / den), E_mlp_previous=prev,
                                             rel_diff=abs(math.sqrt(num / den) - prev) / prev)
        jdump(out, path)
        print("consistency", out["_consistency_G20261111"], flush=True)
    names = sorted(f[:-5] for f in os.listdir(p("results")) if f.endswith(".json"))
    jobs = [(n, pol) for n in names for pol in POLICIES] + \
           [(f"ref:{k}", None) for k in sorted((jload(p("logs", "refmatrix_raw.json")) or {}).keys())]
    for name, pol in jobs:
        key = f"{name}|{pol}" if pol else name
        if key in out:
            continue
        t0 = time.time()
        if pol:
            d = {L: torch.load(os.path.join(cfg_dir(name), f"Q_{pol}_L{L}.pt")) for L in LAY}
            wq, cq = {L: d[L]["weights"] for L in LAY}, {L: d[L]["codes"] for L in LAY}
        else:
            i, j = name[4:].split("|")
            d = torch.load(c("refmatrix", f"{i}__{j}.pt"))
            wq, cq = d["weights"], d["codes"]
        wref = torch.load(c("refmatrix", f"{i}__G20261115.pt"))["weights"] if not pol else None
        r = functional_one(wq, ref, Wo, wref)
        cr = code_rates(cq, Co)
        rf = {f"L{L}|{k}": float((wq[L][k].float() - Wo[L][k].float()).norm() / Wo[L][k].float().norm()) for L in LAY for k in PROJ}
        bfc = {f"L{L}|{k}": float((wq[L][k] != Wo[L][k]).float().mean()) for L in LAY for k in PROJ}
        out[key] = dict(per_layer=r, codes=cr, rel_frob=rf, bf16_changed=bfc, seconds=time.time() - t0)
        jdump(out, path)
        print(f"[func] {key} R_func {math.sqrt(sum(r[str(L)]['sq_diff_vs_orig'] for L in LAY) / sum(out['_ref'][str(L)]['sq_quant'] for L in LAY)):.4f} "
              f"code {cr['code']:.4f} {time.time() - t0:.0f}s", flush=True)
        del d, wq, cq
        gc.collect()


def stage_analyze(a):
    import report_dense_dose_response as R
    R.main()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--optional", action="store_true")
    a = ap.parse_args(argv)
    dict(audit=stage_audit, pilot_criteria=stage_pilot_criteria, pilot=stage_pilot, protocol=stage_protocol,
         caches=stage_caches, gpu=stage_gpu, refmatrix=stage_refmatrix, func=stage_func, analyze=stage_analyze)[a.stage](a)


if __name__ == "__main__":
    main()
