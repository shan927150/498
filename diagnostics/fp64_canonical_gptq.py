"""FP64-canonical GPTQ (FR s0 / FC s0, layers 1/8/16/23, ORIGINAL activations / routing R_O / act-order P_O).

Canonical Hessian = the ORIGINAL FP32 per-sequence contributions accumulated per original block in FP64, blocks summed
in FP64, cast once to FP32. The unchanged order-null quantization pathway (`hessian_order_null.quant_pass`, act-order
override P_O, original replay order) is run with the Profiler's accumulator replaced by one that serves the stored
canonical Hessian (`moe_calib/fp64_canonical.serve_hessians`). The solver is unchanged.

Stages (--stage):
  protocol                        audit + frozen protocol (before any canonical Hessian / PPL)
  hessians --strategy S           reconstruct canonical Hessians with the historical Fp64Accumulator for orders
                                  O0 / 20261121 / 20261122: FP32 path == stored preflight Hessians (480/480), FP64 and
                                  FP32-cast bitwise order-invariant, FP64 == historical control (exact rel-Frobenius
                                  reproduction), Hessian-level diagnostics vs ORIGINAL FP32
  regression --strategy S         loaded-accumulator pathway serving the ORIGINAL FP32 Hessians -> weights == OO bitwise
  quant --strategy S [--repeat]   canonical GPTQ + PPL (fresh process)
  timing                          FP32 vs FP64 block accumulation, one layer, one 15-expert chunk
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import hessian_order_null as HON        # noqa: E402  (validated order-null pathway)
from moe_calib import fp64_canonical as FC64           # noqa: E402
from moe_calib import hessian_order_null as HN         # noqa: E402

XA, DB, EP, FRD, XR, MRD = HON.XA, HON.DB, HON.EP, HON.FRD, HON.XR, HON.MRD
AA, FRT = HON.AA, HON.FRT
OUT = "runs_fp64_canonical_reproducibility"
STORE = os.path.expanduser("~/moe_cache_fp64_canonical")
LAYERS, STRATS = HON.LAYERS, HON.STRATS
ORDERS = (None,) + tuple(HN.FP64_SEEDS)          # O0, 20261121, 20261122 (the historical control's orders)
CHUNK = 15
jdump, sha256_file = HON.jdump, HON.sha256_file
CODE = ["moe_calib/fp64_canonical.py", "diagnostics/fp64_canonical_gptq.py", "moe_calib/gptq.py", "moe_calib/profiler.py",
        "moe_calib/hessian.py", "moe_calib/hessian_order_null.py", "diagnostics/hessian_order_null.py",
        "moe_calib/quant_eval.py", "e3_allocation.py", "moe_calib/partial_model_quant.py", "diagnostics/eval_partial_ppl.py"]
HIST = ["runs_hessian_order_null/input_manifest.json", "runs_hessian_order_null/protocol.json",
        "runs_hessian_order_null/ppl_results.json", "runs_hessian_order_null/paired_differences.json",
        "runs_hessian_order_null/weight_differences.json", "runs_hessian_order_null/fp64_control/FR.json",
        "runs_hessian_order_null/fp64_control/FC.json", "runs_hessian_order_null/baseline_regression/FR.json",
        "runs_hessian_order_null/baseline_regression/FC.json", "runs_actorder_ablation/ppl_sensitivity.json",
        "runs_actorder_ablation/weight_sensitivity.json", "runs_cross_seed_grouping/paired_differences.json",
        "runs_cross_seed_grouping/variance_analysis.json", "runs_cross_seed_grouping/ppl_results.json",
        "runs_cross_seed_grouping/reconstruction_results.json", "runs_multigroup_robustness/ppl_results.json",
        "runs_partial_ppl/windows.pt", "runs_partial_ppl/evaluation_protocol.json"]


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def jload(*a):
    return json.load(open(p(*a)))


def hon(*a):
    return os.path.join(ROOT, HON.OUT, *a)


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def env():
    def run(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as ex:                     # noqa: BLE001
            return f"unavailable ({ex})"
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, host=platform.node(),
                git_head=run(["git", "-C", ROOT, "rev-parse", "HEAD"]))


# ===================================================================== protocol
def stage_protocol(a):
    os.makedirs(p("logs"), exist_ok=True)
    if os.path.exists(p("protocol.json")):
        print("[protocol] frozen; not rewritten"); return
    man = json.load(open(hon("input_manifest.json")))
    cfg, sel, _, _ = MRD.selections()
    inv = dict(created=time.strftime("%F %T"), environment=env(),
               selections={c: dict(ids_hash=sel[c]["ids_hash"], expected=HON.SEL_PREFIX[c],
                                   match=sel[c]["ids_hash"] == HON.SEL_PREFIX[c],
                                   selection_sha256=HN.selection_sha(sel[c]["chosen"]),
                                   matches_order_null_manifest=HN.selection_sha(sel[c]["chosen"]) ==
                                   man["selections"][c]["selection_sha256"], K=sel[c]["K"]) for c in STRATS},
               original_cache=dict(dir=man["original_cache_dir"], files={
                   L: dict(sha256=sha256_file(os.path.join(man["original_cache_dir"], f"L{L}.pt")), expected=h)
                   for L, h in man["original_cache_files"].items()}),
               routing_original=dict(man["routing_original"], sha_now=sha256_file(man["routing_original"]["file"])),
               original_actorder=dict(man["original_actorder"], sha_now=sha256_file(man["original_actorder"]["file"])),
               order_null_hessian_files={f"{c}_{HON.oname(s)}": dict(path=HON.hess_file(c, s),
                                                                     sha256=sha256_file(HON.hess_file(c, s)))
                                         for c in STRATS for s in ORDERS},
               fp64_control=dict(files={c: hon("fp64_control", f"{c}.json") for c in STRATS},
                                 stored_fp64_hessians=False,
                                 note="the control stored comparisons only (no FP64 matrices) -> canonical Hessians are "
                                      "reconstructed with the same Fp64Accumulator and orders, and must reproduce the "
                                      "control's recorded FP32-vs-FP64 rel-Frobenius values exactly"),
               oo_references={c: dict(man["oo_references"][c],
                                      weight_files_sha_now={L: sha256_file(os.path.join(man["oo_references"][c]["weights_dir"], f"L{L}.pt"))
                                                            for L in man["oo_references"][c]["weight_files"]})
                              for c in STRATS},
               order_null_results={f"{c}_O{s}": os.path.exists(hon("results", f"{c}_O{s}.json"))
                                   for c in STRATS for s in HN.ORDER_SEEDS},
               windows=dict(sha256=man["windows_sha256"], batch=man["eval_batch"], bf16_ppl=man["bf16_ppl"]),
               historical_sha256={f: sha256_file(os.path.join(ROOT, f)) for f in HIST},
               calibration_test_disjoint=dict(cfg_split=cfg["split"], max_calibration_row={c: int(max(sel[c]["chosen"]))
                                                                                         for c in STRATS}),
               extraction_policy="ORIGINAL CANONICAL activation cache (~/moe_cache rows from the C4 pool; built once, "
                                 "before and independently of the FR/FC selections, used by both strategies); no "
                                 "regrouped cache is used")
    inv["calibration_test_disjoint"]["ok"] = all(v < cfg["split"]["test"][0] for v in
                                                 inv["calibration_test_disjoint"]["max_calibration_row"].values())
    jdump(inv, p("artifact_inventory.json"))
    proto = dict(
        study="FP64 canonical GPTQ + numerical reproducibility synthesis",
        model="Qwen/Qwen1.5-MoE-A2.7B (partial-model GPTQ: routed experts of layers 1/8/16/23; everything else BF16)",
        strategies={c: dict(ids_hash=sel[c]["ids_hash"], selection_sha256=inv["selections"][c]["selection_sha256"])
                    for c in STRATS},
        gptq=cfg["gptq"], layers=LAYERS, activation_extraction="ORIGINAL (frozen order-null input manifest)",
        routing="ORIGINAL R_O forced, asserted equal to natural top-k for every token",
        act_order="ORIGINAL P_O as explicit override (runs_actorder_ablation/original_actorder.pt); natural argsort of "
                  "the canonical Hessian recorded and compared, never used",
        hessian_accumulation=dict(
            contributions="outer_s = Xf.T @ Xf, FP32 GEMM per (sequence, expert, projection), unchanged",
            blocks="original positional blocks (i*4)//128 of the ORIGINAL replay order",
            canonical="per-block FP64 accumulation of the FP32 contributions, blocks summed in FP64 "
                      "(H64[K,d,d].sum(0)), cast ONCE to FP32; no normalization; damping unchanged (solver-internal)",
            reconstruction="HN.Fp64Accumulator (the historical control's class) replayed for orders O0/20261121/20261122",
            gptq_input="stored canonical FP32 Hessian served by LoadedHessianAccumulator inside the unchanged E3 path"),
        order_invariance=dict(orders=["O0", "O20261121", "O20261122"],
                              require="FP64 sum and FP32 cast bitwise identical across the three orders for all 480 "
                                      "Hessians per strategy; FP32 path == stored preflight Hessians; identical token "
                                      "counts per block; finite and symmetric",
                              claim_scope="tested orders only; not a universal invariance claim"),
        evaluation=dict(windows_sha256=man["windows_sha256"], batch=man["eval_batch"],
                        metric="PPL = exp(sum NLL / scored targets), EP.layerwise_nll, BF16, natural inference routing",
                        repeat="two evaluations per configuration, per-window NLL must be identical"),
        regression=dict(pathway="loaded-accumulator pathway serving ORIGINAL FP32 Hessians must reproduce the frozen OO "
                                "weights bitwise (FR)",
                        expected=dict(FR=HON.OO_PPL["FR"], FC=HON.OO_PPL["FC"]),
                        order_null_expected={c: {f"O{s}": json.load(open(hon("results", f"{c}_O{s}.json")))["ppl"]
                                                 for s in HN.ORDER_SEEDS} for c in STRATS}),
        determinism="FR canonical quantized twice in fresh processes: weights, reconstruction, PPL and per-window NLL "
                    "bitwise equal",
        estimands=dict(
            canonical="Delta_FP64 = PPL(FC, S, G0) - PPL(FR, S, G0), one deterministic configuration G0 (no interval)",
            order_null="five FP32 accumulation orders (reference distribution of the same pathway; not pooled with "
                       "other mechanisms)",
            execution_marginal="E_{S,G}[PPL(FC,S,G) - PPL(FR,S,G)] over 3 allocation seeds x 5 groupings (existing "
                               "cross-seed grouping study, ORIGINAL excluded); also the paired mean-NLL version",
            window_conditional="paired per-window bootstrap for fixed weights (existing results), conditional on G"),
        decision=dict(
            canonical=dict(VALIDATED="all mandatory checks pass, canonical Hessians order-invariant under the tested "
                                     "orders, both configurations quantize, FR repeat bitwise reproducible",
                           NOT_ORDER_INVARIANT="canonical Hessians or weights differ across tested orders / repeats",
                           BLOCKED="historical artifacts fail validation or a mandatory check fails",
                           note="independent of which strategy wins"),
            execution=dict(MATERIAL="matched execution variants (existing studies) change the sign of FC-FR, or the "
                                    "FC-FR range across matched variants >= 2 x |the reference FC-FR|, while repeated "
                                    "evaluation of identical weights is deterministic",
                           LIMITED="otherwise",
                           note="decided from the existing studies only; the FP64 result is not used")),
        reporting_rules=["exact stored values, no rounded table values", "no pooling across mechanisms",
                         "no additive variance decomposition", "FP64 point is one deterministic reference, not an "
                         "estimate of the grouping mean", "no significance claims from 5 permutations or 3 seeds"],
        code_sha256={f: sha256_file(os.path.join(ROOT, f)) for f in CODE},
        created=time.strftime("%F %T"))
    jdump(proto, p("protocol.json"))
    bad = [c for c in STRATS if not inv["selections"][c]["match"]] + \
          [L for L, v in inv["original_cache"]["files"].items() if v["sha256"] != v["expected"]]
    print("[protocol] frozen; inventory problems:", bad or "none")


# ===================================================================== canonical Hessians
@torch.no_grad()
def stage_hessians(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    c = a.strategy
    outp = p("logs", f"hessians_{c}.json")
    if os.path.exists(outp):
        print(f"[hessians] {c} done"); return
    os.makedirs(os.path.join(STORE, "hessians"), exist_ok=True)
    man = json.load(open(hon("input_manifest.json")))
    chosen, K = man["selections"][c]["replay_order"], man["selections"][c]["K"]
    hp = {HON.oname(s): torch.load(HON.hess_file(c, s)) for s in ORDERS}
    ctl = json.load(open(hon("fp64_control", f"{c}.json")))["keys"]
    tables = FRD.load_tables("O")
    P = XA.load_original_perms(c)
    model, _ = DB.load_model()
    E = MU.num_experts(model)
    d_gu, d_down = MU.expert_dims(model)
    dims = dict(gu=d_gu, down=d_down)
    canon, orig32, rec = {}, {}, {}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    replay_s = 0.0
    for L in LAYERS:
        cache = load_layer_cache(HON.OCACHE, L)
        for c0 in range(0, E, CHUNK):
            keys = [(L, e, pj) for e in range(c0, min(c0 + CHUNK, E)) for pj in ("gu", "down")]
            base = {}
            for s in ORDERS:
                order, blocks, _ = HN.order_spec(chosen, s, K)
                prof = HN.OrderedForcedProfiler(model, L, tables[L], {x: x for x in chosen}, True, chosen, order,
                                                blocks, K=K, collect_hessian=False)
                prof.acc = HN.Fp64Accumulator(keys, dims, K, MU.GPU)
                t1 = time.time()
                prof.run(cache, chosen, weights=[1.0] * len(chosen))
                torch.cuda.synchronize(); replay_s += time.time() - t1
                assert prof.n_ordered_diff == 0
                o = HON.oname(s)
                for key in keys:
                    sk = f"{key[0]}.{key[1]}.{key[2]}"
                    H32, H64 = prof.acc.sums(key)
                    cast = H64.to(torch.float32)
                    r = rec.setdefault(sk, {})
                    r[f"fp32_equals_preflight_{o}"] = FRT.tensor_sha(H32) == hp[o]["sha"][sk]
                    r[f"block_tokens_equal_{o}"] = prof.acc.raw_n[key].tolist() == hp["O0"]["block_tokens"][sk]
                    r[f"sha64_{o}"], r[f"sha_cast_{o}"] = FRT.tensor_sha(H64), FRT.tensor_sha(cast)
                    if s is None:
                        r["fp32_vs_fp64_rel_fro_O0"] = float((H32.double() - H64).norm() / H64.norm())
                        r["reproduces_control_fp64"] = r["fp32_vs_fp64_rel_fro_O0"] == ctl[sk]["fp32_vs_fp64_rel_fro_O0"]
                        canon[sk], orig32[sk] = cast.cpu().clone(), H32.cpu().clone()
                        r["finite"] = bool(torch.isfinite(cast).all())
                        r["symmetric"] = bool(torch.equal(cast, cast.T))
                        r["fp32_original_symmetric"] = bool(torch.equal(H32, H32.T))
                        hd = HN.hdiff(cast, H32)
                        r["vs_original_fp32"] = {k: hd[k] for k in ("n_diff", "frac_diff", "rel_fro", "max_abs")}
                        r["vs_original_fp32"]["trace_rel_diff"] = float(
                            (torch.trace(cast.double()) - torch.trace(H32.double())) / torch.trace(H32.double()))
                        pk = "gate_proj" if key[2] == "gu" else "down_proj"
                        pref = P[f"{key[0]}.{key[1]}.{pk}"]
                        r["natural_vs_PO"] = AA.perm_compare(HN.natural_perm(cast).cpu(), pref)
                        r["original_natural_vs_PO"] = AA.perm_compare(HN.natural_perm(H32).cpu(), pref)
                        if key in [tuple(x) for x in HON.DIAG_KEYS]:
                            r["factorization_vs_original"] = HN.factorization_sensitivity(cast, H32, pref, 1.0)
                        base[sk] = (r[f"sha64_{o}"], r[f"sha_cast_{o}"])
                    else:
                        r[f"fp64_equal_O0_{o}"] = r[f"sha64_{o}"] == base[sk][0]
                        r[f"cast_equal_O0_{o}"] = r[f"sha_cast_{o}"] == base[sk][1]
                prof.acc = None
                del prof
                gc.collect(); torch.cuda.empty_cache()
            print(f"[hessians] {c} L{L} experts {c0}-{min(c0 + CHUNK, E) - 1} ({time.time() - t0:.0f} s)", flush=True)
        del cache
    path = os.path.join(STORE, "hessians", f"{c}.pt")
    torch.save(dict(canonical=canon, original_fp32=orig32,
                    sha_canonical={k: rec[k]["sha_cast_O0"] for k in canon},
                    sha_original_fp32={k: hp["O0"]["sha"][k] for k in canon}), path)
    vals = list(rec.values())
    summ = dict(
        n_hessians=len(vals),
        fp32_path_equals_preflight_all=all(v[f"fp32_equals_preflight_{HON.oname(s)}"] for v in vals for s in ORDERS),
        block_tokens_equal_all=all(v[f"block_tokens_equal_{HON.oname(s)}"] for v in vals for s in ORDERS),
        fp64_order_invariant_all=all(v[f"fp64_equal_O0_O{s}"] for v in vals for s in HN.FP64_SEEDS),
        cast_order_invariant_all=all(v[f"cast_equal_O0_O{s}"] for v in vals for s in HN.FP64_SEEDS),
        reproduces_control_fp64_all=all(v["reproduces_control_fp64"] for v in vals),
        finite_all=all(v["finite"] for v in vals), symmetric_all=all(v["symmetric"] for v in vals),
        n_bitwise_equal_original_fp32=sum(v["vs_original_fp32"]["n_diff"] == 0 for v in vals),
        frac_elements_differ_from_original=float(np.sum([v["vs_original_fp32"]["n_diff"] for v in vals]) /
                                                 sum(canon[k].numel() for k in canon)),
        rel_fro_vs_original_median=float(np.median([v["vs_original_fp32"]["rel_fro"] for v in vals])),
        rel_fro_vs_original_max=float(np.max([v["vs_original_fp32"]["rel_fro"] for v in vals])),
        trace_rel_diff_max_abs=float(np.max(np.abs([v["vs_original_fp32"]["trace_rel_diff"] for v in vals]))),
        natural_perm_equal_PO=sum(v["natural_vs_PO"]["identical"] for v in vals),
        original_natural_perm_equal_PO=sum(v["original_natural_vs_PO"]["identical"] for v in vals),
        natural_frac_columns_changed_group_mean=float(np.mean([v["natural_vs_PO"]["frac_columns_changed_group"] for v in vals])),
        natural_frac_columns_changed_group_max=float(np.max([v["natural_vs_PO"]["frac_columns_changed_group"] for v in vals])),
        n_hessians_with_group_change_without_override=sum(v["natural_vs_PO"]["frac_columns_changed_group"] > 0 for v in vals),
        factorization_diag=[dict(key=k, **v["factorization_vs_original"]) for k, v in rec.items()
                            if "factorization_vs_original" in v],
        replay_s_total=replay_s, wall_s=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_rss_gb=rss_gb(), store=path, store_sha256=sha256_file(path), store_bytes=os.path.getsize(path))
    jdump(dict(strategy=c, summary=summ, keys=rec), outp)
    print(f"[hessians] {c}: " + json.dumps({k: v for k, v in summ.items() if k != "factorization_diag"}), flush=True)


# ===================================================================== quantization (canonical / regression)
def run_quant(c, source, name, ppl=True):
    store = torch.load(os.path.join(STORE, "hessians", f"{c}.pt"))
    H = store["canonical" if source == "canonical" else "original_fp32"]
    ref_sha = store["sha_canonical" if source == "canonical" else "sha_original_fp32"]
    assert {k: FRT.tensor_sha(v) for k, v in H.items()} == ref_sha, "stored Hessians corrupted"
    del store
    HON.HCACHE = STORE                                   # weights / quantizer files go to this study's store only
    man = json.load(open(hon("input_manifest.json")))
    t0 = time.time()
    with FC64.serve_hessians(H):
        model, rec, q = HON.quant_pass(c, None, ref_sha)
    tq = time.time() - t0
    evals = HON.evaluate(model, man["windows_sha256"]) if ppl else []
    wd, files, qf, qsha = HON.save_outputs(model, rec, name)
    oo = json.load(open(HON.XR.oo_result_path(c)))
    out = HON.slim(rec) | dict(
        strategy=c, name=name, hessian_source=source, weights_dir=wd, weight_files=files, quantizer_file=qf,
        quantizer_file_sha256=qsha, perm_equal_original=f"{sum(rec['perm_ok'])}/{len(rec['perm_ok'])}",
        hessians_served_equal_canonical=f"{sum(rec['hess_sha_equal'])}/{len(rec['hess_sha_equal'])}",
        natural_actorder=HON.perm_summary(rec["natural_vs_PO"]), weights_equal_OO=rec["installed_weights_sha256"] ==
        oo["installed_weights_sha256"], recon_equal_OO=rec["recon"] == oo["recon"], quant_wall_s=tq)
    if evals:
        out.update(evals[0])
        out["repeat_eval_per_window_identical"] = evals[0]["per_window_nll"] == evals[1]["per_window_nll"]
        out["repeat_eval_abs_diff_ppl"] = abs(evals[0]["ppl"] - evals[1]["ppl"])
        out["eval_s_both"] = [e["eval_s"] for e in evals]
    out["valid"] = bool(HON.base_ok(rec) and (not evals or out["repeat_eval_per_window_identical"]))
    return out


def stage_regression(a):
    c = a.strategy
    path = p("logs", f"regression_{c}.json")
    if os.path.exists(path):
        print(f"[regression] {c} done"); return
    out = run_quant(c, "original_fp32", f"{c}_FP32_LOADED_REGRESSION", ppl=False)
    out["passed"] = bool(out["valid"] and out["weights_equal_OO"] and out["recon_equal_OO"])
    jdump(out, path)
    print(f"[regression] {c}: weights==OO {out['weights_equal_OO']} recon==OO {out['recon_equal_OO']} valid {out['valid']}")


def stage_quant(a):
    c = a.strategy
    name = f"{c}_s0_FP64_CANONICAL" + ("_repeat" if a.repeat else "")
    path = p("results", f"{name}.json")
    if os.path.exists(path):
        print(f"[quant] {name} done"); return
    hs = jload("logs", f"hessians_{c}.json")["summary"]
    assert hs["fp64_order_invariant_all"] and hs["cast_order_invariant_all"] and hs["fp32_path_equals_preflight_all"], \
        "canonical Hessians not validated: CANONICAL-BLOCKED / NOT-ORDER-INVARIANT; no GPTQ"
    out = run_quant(c, "canonical", name)
    os.makedirs(p("results"), exist_ok=True)
    jdump(out, path)
    print(f"[quant] {name}: PPL {out['ppl']:.9f} (OO {HON.OO_PPL[c]:.9f}) recon {out['recon']:.9f} weights==OO "
          f"{out['weights_equal_OO']} perms {out['perm_equal_original']} valid {out['valid']} ({out['quant_wall_s']:.0f} s)")
    if not out["valid"]:
        raise SystemExit(f"{name} failed correctness checks")


# ===================================================================== accumulation timing
class _Acc:
    def __init__(self, keys, dims, K, device, dtype):
        self.keys, self.dtype = set(keys), dtype
        self.H = {k: torch.zeros(K, dims[k[2]], dims[k[2]], dtype=dtype, device=device) for k in keys}

    @torch.no_grad()
    def add(self, key, X, block, weight=1.0):
        if key not in self.keys or X.numel() == 0:
            return
        Xf = X.to(torch.float32)
        outer = Xf.T @ Xf
        self.H[key][block] += outer if self.dtype == torch.float32 else outer.to(torch.float64)


@torch.no_grad()
def stage_timing(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    if os.path.exists(p("logs", "timing_raw.json")):
        print("[timing] done"); return
    man = json.load(open(hon("input_manifest.json")))
    c = "FR"
    chosen, K = man["selections"][c]["replay_order"], man["selections"][c]["K"]
    tables = FRD.load_tables("O")
    model, _ = DB.load_model()
    d_gu, d_down = MU.expert_dims(model)
    L = 8
    cache = load_layer_cache(HON.OCACHE, L)
    keys = [(L, e, pj) for e in range(CHUNK) for pj in ("gu", "down")]
    order, blocks, _ = HN.order_spec(chosen, None, K)
    res = {"none": [], "fp32": [], "fp64": []}

    def one(kind):
        prof = HN.OrderedForcedProfiler(model, L, tables[L], {x: x for x in chosen}, True, chosen, order, blocks, K=K,
                                        collect_hessian=False)
        if kind != "none":
            prof.acc = _Acc(keys, dict(gu=d_gu, down=d_down), K, MU.GPU, torch.float32 if kind == "fp32" else torch.float64)
        else:
            prof.acc = _Acc([], dict(gu=d_gu, down=d_down), K, MU.GPU, torch.float32)
        torch.cuda.synchronize(); t = time.time()
        prof.run(cache, chosen, weights=[1.0] * len(chosen))
        torch.cuda.synchronize(); dt = time.time() - t
        prof.acc = None; del prof; gc.collect(); torch.cuda.empty_cache()
        return dt
    one("fp64")
    for rep in range(3):
        for k in (["none", "fp32", "fp64"][rep:] + ["none", "fp32", "fp64"][:rep]):
            res[k].append(one(k))
    jdump(dict(layer=L, experts=f"0-{CHUNK - 1}", keys=len(keys), seconds=res,
               median={k: float(np.median(v)) for k, v in res.items()},
               note="one layer, 15 experts x (gu, down), 128 sequences, original order; 'none' = same replay with hooks "
                    "but no accumulation for these keys; FP32 = original arithmetic, FP64 = canonical"),
          p("logs", "timing_raw.json"))
    print("[timing]", {k: float(np.median(v)) for k, v in res.items()})


def stage_tests(a):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + (["tests"] if a.full else ["tests/test_fp64_canonical.py"])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    open(p("logs", "pytest_full.log" if a.full else "pytest_targeted.log"), "w").write(r.stdout + r.stderr)
    print((r.stdout.strip().splitlines() or [r.stderr[-1500:]])[-1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--strategy", choices=STRATS)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    for d in ("logs", "results", "figures"):
        os.makedirs(p(d), exist_ok=True)
    dict(protocol=stage_protocol, hessians=stage_hessians, regression=stage_regression, quant=stage_quant,
         timing=stage_timing, tests=stage_tests)[a.stage](a)


if __name__ == "__main__":
    main()
