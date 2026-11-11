"""GPTQ act-order ablation after frozen-routing calibration (FR s0 / FC s0, four layers, five groupings).

Stages (--stage):
  preflight                         inputs, references, resources
  audit                             independent installation audit of the FC AO G20261114 tolerance exceedance
                                    (+ passing / historical experts, negative controls)
  capture_order --strategy S        OO re-run through the modified solver (default path): weights / recon must equal
                                    the frozen OO; every projection re-quantized with its own natural permutation
                                    (bitwise); the applied permutations are saved
  merge_order                       original_actorder.pt + manifest
  solver_regression                 FR AO G20261111 re-run: default path == stored AO weights, natural-permutation
                                    override bitwise, Hessians == stored; on success freezes protocol.json
                                    (integrity protocol + decision rule + code hashes) BEFORE any AOF run
  aof --strategy S --grouping G     one AOF configuration (fresh process)
  report
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from contextlib import contextmanager

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB          # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import frozen_routing as FRD            # noqa: E402
import frozen_routing_robustness as XR  # noqa: E402
import multigroup_robustness as MRD     # noqa: E402
from moe_calib import actorder_ablation as AA          # noqa: E402
from moe_calib import frozen_routing as FRT            # noqa: E402
from moe_calib import frozen_routing_robustness as RB  # noqa: E402

OUT = "runs_actorder_ablation"
ACACHE = os.path.expanduser(os.environ.get("ACTORDER_CACHE", "~/moe_cache_actorder_ablation"))
LAYERS, STRATS, OCACHE, G = FRD.LAYERS, list(RB.STRATS), FRD.OCACHE, list(RB.GROUPINGS)
CHECK = [tuple(x) for x in EP.CHECK_EXPERTS]
jdump, sha256_file = FRD.jdump, FRD.sha256_file
CODE_FILES = ["moe_calib/actorder_ablation.py", "moe_calib/gptq.py", "moe_calib/quant_eval.py",
              "diagnostics/actorder_ablation.py", "diagnostics/report_actorder_ablation.py"]


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def aof_name(c, g):
    return f"{c}_AOF_{g}"


def aof_result_path(c, g):
    return p("results", f"{aof_name(c, g)}.json")


def oo_weights_dir(c):
    return json.load(open(XR.oo_result_path(c)))["weights_dir"]


def oo_hessian_path(c):
    return os.path.join(FRD.FCACHE, "hessians", f"{c}_OO.pt")


def code_hashes():
    return {f: sha256_file(os.path.join(ROOT, f)) for f in CODE_FILES if os.path.exists(os.path.join(ROOT, f))}


# ---------------------------------------------------------------- preflight
def stage_preflight(a):
    cfg, sel, checks, union = MRD.selections()
    MRD.protocol(); FRD.protocol(); XR.protocol()
    for c in STRATS:
        assert sel[c]["ids_hash"] == RB.SELECTION_HASHES[c]
    cman = json.load(open(os.path.join(ROOT, MRD.OUT, "cache_manifest.json")))
    for g in G:
        MRD.verify_cache(g)
        assert cman["groupings"][g]["sequence_ids"] == union
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    assert abs(base["ppl"] - 9.237090) < 1e-6 and prot["scored_targets_total"] == 65408 and tuple(win.shape) == (128, 512)
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    refs = {}
    for c in STRATS:
        r = json.load(open(XR.oo_result_path(c)))
        refs[f"{c}_OO"] = dict(path=os.path.relpath(XR.oo_result_path(c), ROOT), result_sha256=sha256_file(XR.oo_result_path(c)),
                               ppl=r["ppl"], installed_weights_sha256=r["installed_weights_sha256"], weights_dir=r["weights_dir"],
                               hessian_file=oo_hessian_path(c), hessian_file_sha256=sha256_file(oo_hessian_path(c)))
        for g in G:
            path = XR.ao_result_path(c, g)
            r = json.load(open(path))
            refs[f"{c}_AO_{g}"] = dict(path=os.path.relpath(path, ROOT), result_sha256=sha256_file(path), ppl=r["ppl"],
                                       valid_under_frozen_check=r["valid"], installed_weights_sha256=r["installed_weights_sha256"],
                                       weights_dir=r["weights_dir"], hessian_file=r["hessian_file"],
                                       hessian_file_sha256=r["hessian_file_sha256"],
                                       weight_files={L: v["sha256"] for L, v in r["weight_files"].items()})
            assert sha256_file(r["hessian_file"]) == r["hessian_file_sha256"]
    rob = json.load(open(os.path.join(ROOT, XR.OUT, "summary.json")))
    assert rob["decision"] == "RUN-BLOCKED"                          # the earlier frozen decision, unchanged
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    disk = shutil.disk_usage(os.path.expanduser("~")).free
    pre = dict(selections=checks, environment=DB.env_record(), host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6,
               gpu_free_gb=free / 1e9, gpu_total_gb=tot / 1e9, disk_free_tb=disk / 1e12, created=time.strftime("%F %T"),
               feasible=bool(free > 12e9 and disk > 60e9))
    jdump(pre, p("logs", "preflight.json"))
    jdump(dict(selections={c: dict(ids_hash=sel[c]["ids_hash"], K=sel[c]["K"], chosen=sel[c]["chosen"]) for c in STRATS},
               union_calibration_ids=union, groupings={g: cman["groupings"][g]["files"] for g in G},
               grouping_manifest_sha256={g: json.load(open(os.path.join(ROOT, MRD.OUT, "grouping_manifests", f"{g}.json")))["manifest_sha256"] for g in G},
               routing_original_sha256=json.load(open(FRD.p("routing_original.json")))["file_sha256"],
               windows_sha256=prot["windows_sha256"], gptq=cfg["gptq"], references=refs,
               earlier_decision=dict(study=XR.OUT, decision=rob["decision"],
                                     blocked_by=list(rob["blocked_by_failed_ao_checks"]))), p("input_manifest.json"))
    print(f"preflight ok: GPU free {free / 1e9:.1f} GB, disk {disk / 1e12:.0f} TB; earlier decision {rob['decision']} kept")
    if not pre["feasible"]:
        raise SystemExit("preflight: insufficient resources")


# ---------------------------------------------------------------- installation audit
def audit_configs():
    """Every configuration whose full Hessians for the 8 check experts and installed weights are stored."""
    out = []
    for c in STRATS:
        for k in ("OO", "AO", "OA", "AA"):
            r = json.load(open(FRD.p("results", f"{c}_{k}.json")))
            out.append(dict(name=f"{c}_{k}_G20261111", hessian=r["hessian_file"], weights_dir=r["weights_dir"],
                            checks=r["expert_checks"], result=FRD.p("results", f"{c}_{k}.json")))
        for g in RB.NEW_AO:
            r = json.load(open(XR.ao_result_path(c, g)))
            out.append(dict(name=f"{c}_AO_{g}", hessian=r["hessian_file"], weights_dir=r["weights_dir"],
                            checks=r["expert_checks"], result=XR.ao_result_path(c, g)))
            r = json.load(open(XR.aa_result_path(c, g)))
            out.append(dict(name=f"{c}_AA_{g}", hessian=XR.aa_hessian_path(c, g), weights_dir=r["weights_dir"],
                            checks=r["expert_checks"], result=XR.aa_result_path(c, g)))
    return out


def stage_audit(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.gptq import quantize_expert
    from moe_calib.profiler import Profiler
    from moe_calib.data_utils import load_layer_cache
    if (load_json(p("installation_audit.json")) or {}).get("decision"):
        print(f"[audit] done: {load_json(p('installation_audit.json'))['decision']}"); return
    cfg, sel, _, union = MRD.selections()
    test_ids = list(range(*cfg["split"]["test"]))
    gq = cfg["gptq"]
    model, _ = DB.load_model()
    dev = MU.GPU
    W0 = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()} for L, e in CHECK}
    X = {}
    for L in LAYERS:
        cache = load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]]
        for e in (0, 30):
            X[(L, e)] = PQ.expert_inputs(model, L, e, cache)
    xsha = {f"{L}.{e}": FRT.tensor_sha(X[(L, e)]) for L, e in CHECK}

    def audit_expert(L, e, Wq32, installed):
        mod = MU.expert_module(model, L, e)
        MU.set_expert_weights(model, L, e, installed)
        try:
            return AA.precision_decomposition(mod, Wq32, W0[(L, e)], X[(L, e)], dev)
        finally:
            MU.set_expert_weights(model, L, e, W0[(L, e)])

    cases, t0 = [], time.time()
    for cf in audit_configs():
        H = torch.load(cf["hessian"])
        stored = {(ck["layer"], ck["expert"]): ck for ck in cf["checks"]}
        for L in LAYERS:
            Wf = torch.load(os.path.join(cf["weights_dir"], f"L{L}.pt"))
            for e in (0, 30):
                Wq32 = {k: v.cpu() for k, v in quantize_expert({k: v.float() for k, v in W0[(L, e)].items()},
                                                               H["full"][f"{L}.{e}.gu"], H["full"][f"{L}.{e}.down"],
                                                               gq["bits"], gq["group_size"], gq["percdamp"]).items()}
                inst = {k: Wf[f"{e}.{k}"] for k in AA.PROJ}
                dec = audit_expert(L, e, Wq32, inst)
                s = stored[(L, e)]
                oc = dec["expert"]["old_check"]
                dec.update(config=cf["name"], layer=L, expert_id=e,
                           solver_recomputed_cast_equals_stored_file=all(torch.equal(Wq32[k].to(torch.bfloat16), inst[k]) for k in AA.PROJ),
                           stored_old_check=dict(installed_vs_e3_rel=s["installed_vs_e3_rel"], tolerance=s["tolerance"],
                                                 weight_rounding_only_rel=s["weight_rounding_only_rel"],
                                                 bf16_compute_noise_rel=s["bf16_compute_noise_rel"], passed=s["pass"]),
                           old_check_reproduced_exactly=bool(oc["installed_vs_e3_rel"] == s["installed_vs_e3_rel"]
                                                             and oc["weight_rounding_only_rel"] == s["weight_rounding_only_rel"]
                                                             and oc["bf16_compute_noise_rel"] == s["bf16_compute_noise_rel"]),
                           old_check_max_abs_diff=max(abs(oc[k] - s[k]) for k in ("installed_vs_e3_rel", "weight_rounding_only_rel",
                                                                                 "bf16_compute_noise_rel")),
                           integrity_pass=AA.integrity_pass(dec))
                cases.append(dec)
            del Wf
        print(f"[audit] {cf['name']}: reproduced {sum(c['old_check_reproduced_exactly'] for c in cases[-8:])}/8, "
              f"integrity {sum(c['integrity_pass'] for c in cases[-8:])}/8 ({time.time() - t0:.0f} s)", flush=True)
    # historical highest-ratio expert: partial-PPL FC s2, layer 1 expert 0 (0.954 x tolerance; weights not stored ->
    # validated by reproducing the stored check floats)
    man = json.load(open(os.path.join(ROOT, EP.OUT, "selected_set_manifest.json")))
    ent = man["configs"]["FC_s2"]
    st = {(ck["layer"], ck["expert"]): ck for ck in json.load(open(os.path.join(ROOT, EP.OUT, "ppl_FC_s2.json")))["expert_checks"]}[(1, 0)]
    pf = Profiler(model, 1, K=ent["K"]).run(load_layer_cache(OCACHE, 1), ent["chosen"], weights=[1.0] * len(ent["chosen"]))
    Wq32 = {k: v.cpu() for k, v in quantize_expert({k: v.float() for k, v in W0[(1, 0)].items()}, pf.acc.H_sum((1, 0, "gu"), list(range(ent["K"]))),
                                                   pf.acc.H_sum((1, 0, "down"), list(range(ent["K"]))), gq["bits"], gq["group_size"], gq["percdamp"]).items()}
    pf.free()
    dec = audit_expert(1, 0, Wq32, {k: Wq32[k].to(torch.bfloat16) for k in AA.PROJ})
    oc = dec["expert"]["old_check"]
    dec.update(config="partial_ppl_FC_s2 (historical max ratio 0.954)", layer=1, expert_id=0,
               solver_recomputed_cast_equals_stored_file="weights not stored",
               stored_old_check=dict(installed_vs_e3_rel=st["installed_vs_e3_rel"], weight_rounding_only_rel=st["weight_rounding_only_rel"],
                                     bf16_compute_noise_rel=st["bf16_compute_noise_rel"]),
               old_check_reproduced_exactly=bool(oc["installed_vs_e3_rel"] == st["installed_vs_e3_rel"]
                                                 and oc["bf16_compute_noise_rel"] == st["bf16_compute_noise_rel"]),
               old_check_max_abs_diff=max(abs(oc[k] - st[k]) for k in ("installed_vs_e3_rel", "weight_rounding_only_rel", "bf16_compute_noise_rel")),
               integrity_pass=AA.integrity_pass(dec))
    hist = dec
    # negative controls on the failing expert (FC AO G20261114, L1 e0): the new checks must detect real errors
    fail = next(c for c in cases if c["config"] == "FC_AO_G20261114" and c["layer"] == 1 and c["expert_id"] == 0)
    cf = next(c for c in audit_configs() if c["name"] == "FC_AO_G20261114")
    H = torch.load(cf["hessian"])
    Wf = torch.load(os.path.join(cf["weights_dir"], "L1.pt"))
    Wq32 = {k: v.cpu() for k, v in quantize_expert({k: v.float() for k, v in W0[(1, 0)].items()}, H["full"]["1.0.gu"],
                                                   H["full"]["1.0.down"], gq["bits"], gq["group_size"], gq["percdamp"]).items()}
    neg = {}
    wrong = {k: Wf[f"30.{k}"] for k in AA.PROJ}                                  # expert 30's weights in expert 0
    d = audit_expert(1, 0, Wq32, wrong)
    neg["wrong_expert_installed"] = dict(detected=not AA.integrity_pass(d), I1=all(d["installed_bitwise_equal_cast"].values()),
                                         I2=d["expert"]["module_bitwise_equal_independent_reference"],
                                         old_check_ratio=d["expert"]["old_check"]["ratio"])
    one = {k: Wf[f"0.{k}"].clone() for k in AA.PROJ}
    one["down_proj"].view(torch.int16)[0, 0] += 1                                # one BF16 ulp in one element
    d = audit_expert(1, 0, Wq32, one)
    neg["one_element_perturbed"] = dict(detected=not AA.integrity_pass(d), I1=all(d["installed_bitwise_equal_cast"].values()),
                                        I2=d["expert"]["module_bitwise_equal_independent_reference"],
                                        old_check_ratio=d["expert"]["old_check"]["ratio"])
    # truncation instead of round-to-nearest (an incorrect cast) must violate I1 and usually I3
    trunc = {k: (Wq32[k].contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16) for k in AA.PROJ}
    d = audit_expert(1, 0, Wq32, trunc)
    neg["truncating_cast"] = dict(detected=not AA.integrity_pass(d), I1=all(d["installed_bitwise_equal_cast"].values()),
                                  I3=all(x["ok"] for x in d["cast_rounding"].values()),
                                  old_check_ratio=d["expert"]["old_check"]["ratio"])
    # ---- decision
    allc = cases + [hist]
    impl_error = [f"{c['config']} L{c['layer']} e{c['expert_id']}" for c in cases
                  if not (c["solver_recomputed_cast_equals_stored_file"] and all(c["installed_bitwise_equal_cast"].values())
                          and c["expert"]["module_bitwise_equal_independent_reference"] and c["inputs_exactly_bf16"])]
    impl_error += [] if (hist["expert"]["module_bitwise_equal_independent_reference"] and hist["inputs_exactly_bf16"]) else ["historical"]
    not_reproduced = [f"{c['config']} L{c['layer']} e{c['expert_id']}" for c in allc if not c["old_check_reproduced_exactly"]]
    fe = fail["expert"]
    # total <= weight_repr + arith (triangle inequality on Y_bf16 - Y_solver = (Y_cast - Y_solver) + (Y_bf16 - Y_cast)),
    # so an exceedance of tol = weight_repr + 1.5 x orig_arith requires arith > 1.5 x orig_arith: the BF16 arithmetic
    # deviation with the quantized weights exceeds the 1.5x margin calibrated on the ORIGINAL weights
    explained = bool(fail["old_check_reproduced_exactly"] and fail["integrity_pass"]
                     and fe["total_rel"] <= fe["weight_repr_rel"] + fe["arith_rel"] + 1e-12
                     and fe["arith_rel"] > 1.5 * fe["orig_weights_arith_rel"])
    controls_ok = all(v["detected"] for v in neg.values())
    if impl_error:
        decision = "AUDIT-FOUND-IMPLEMENTATION-ERROR"
    elif not_reproduced or not explained or not controls_ok:
        decision = "AUDIT-INCONCLUSIVE"
    else:
        decision = "AUDIT-CONFIRMED-BF16-REPRESENTATION"
    ar = np.array([c["expert"]["arith_ratio_quantized_vs_original"] for c in allc])
    ratios = np.array([c["expert"]["old_check"]["ratio"] for c in allc])
    qe = np.array([c["expert"]["quant_effect_rel"] for c in allc])
    summ = dict(
        n_cases=len(allc), n_configs=len(audit_configs()) + 1,
        all_installed_bitwise_equal_cast=all(all(c["installed_bitwise_equal_cast"].values()) for c in allc),
        all_solver_recomputation_equal_stored_files=all(c["solver_recomputed_cast_equals_stored_file"] is True for c in cases),
        all_module_bitwise_equal_independent_reference=all(c["expert"]["module_bitwise_equal_independent_reference"] for c in allc),
        all_inputs_exactly_bf16=all(c["inputs_exactly_bf16"] for c in allc),
        all_cast_rounding_within_bf16_unit_roundoff=all(all(v["ok"] for v in c["cast_rounding"].values()) for c in allc),
        all_old_checks_reproduced_exactly=not not_reproduced, n_old_check_failures=int((ratios > 1).sum()),
        old_ratio_max=float(ratios.max()), old_ratio_p50=float(np.median(ratios)),
        arith_ratio_quantized_vs_original=dict(min=float(ar.min()), median=float(np.median(ar)), max=float(ar.max())),
        corr_old_ratio_vs_quant_effect=float(np.corrcoef(ratios, qe)[0, 1]),
        corr_old_ratio_vs_arith_ratio=float(np.corrcoef(ratios, ar)[0, 1]),
        integrity_pass_all=all(c["integrity_pass"] for c in allc))
    out = dict(decision=decision, failing_case=fail, historical_max_case=hist, negative_controls=neg, summary=summ,
               implementation_errors=impl_error, not_reproduced=not_reproduced, failure_explained=explained,
               validation_input_sha256=xsha, cases=cases, seconds=time.time() - t0,
               earlier_decision_unchanged="runs_frozen_routing_robustness/summary.json decision RUN-BLOCKED and the frozen "
                                          "tolerance check are not modified")
    jdump(out, p("installation_audit.json"))
    print(f"[audit] {decision}: failing case total {fe['total_rel']:.6f} = weight repr {fe['weight_repr_rel']:.6f} (+) "
          f"arith {fe['arith_rel']:.6f}; arith ratio q/orig {fe['arith_ratio_quantized_vs_original']:.2f}; controls {controls_ok}", flush=True)
    if decision != "AUDIT-CONFIRMED-BF16-REPRESENTATION":
        raise SystemExit(f"audit {decision}: act-order experiment stopped")


def freeze_protocol(audit):
    fpr = FRD.protocol()
    man = json.load(open(p("input_manifest.json")))
    body = dict(
        experiment="GPTQ act-order ablation after frozen-routing calibration (FR s0 / FC s0, layers 1/8/16/23, five groupings)",
        question="Are GPTQ's Hessian-dependent act-order permutation and its induced quantization-group membership "
                 "responsible for a substantial fraction of the residual weight / PPL sensitivity under frozen routing?",
        model=fpr["model"], model_snapshot=fpr["model_snapshot"], gptq=man["gptq"], B=128, allocation_seed=0,
        quant_layers=LAYERS, n_quantized_experts=240, n_projection_tensors=720,
        selections={c: dict(ids_hash=v["ids_hash"], K=v["K"]) for c, v in man["selections"].items()},
        groupings=man["grouping_manifest_sha256"], routing_original_sha256=man["routing_original_sha256"],
        conditions=dict(OO="GPTQ(H_O), natural act-order (frozen reference)", AO="GPTQ(H_AO,g), act-order from H_AO,g (reused)",
                        AOF="GPTQ(H_AO,g) with actorder_override = captured ORIGINAL permutation P_O (new, 10 configs)"),
        identification="AOF fixes the permutation AND the induced column->group membership jointly; scales, zeros, "
                       "damping, Cholesky factor, error feedback are recomputed from H_AO,g",
        evaluation=dict(windows_sha256=man["windows_sha256"], batching=f"EP.layerwise_nll batch {EP.BATCH}", repeats=2),
        installation_audit=dict(decision=audit["decision"], file_sha256=sha256_file(p("installation_audit.json"))),
        integrity_protocol=AA.INTEGRITY_PROTOCOL, frozen_old_tolerance="reported for every AOF configuration, unchanged, not gating",
        decision_rule=AA.DECISION_RULE, code_sha256=code_hashes(),
        hard_constraints="no re-selection / new groupings / layers / strategies; GPTQ bits, group size, damping unchanged; "
                         "scales and zeros never frozen; no forced routing at inference; earlier RUN-BLOCKED unchanged")
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        assert old["protocol_sha256"] == AA.obj_sha(body), "protocol changed after freezing"
        return
    body["protocol_sha256"] = AA.obj_sha(body)
    body["created"] = time.strftime("%F %T")
    jdump(body, p("protocol.json"))
    print(f"[audit] protocol frozen sha256 {body['protocol_sha256'][:16]}")


def protocol():
    pr = json.load(open(p("protocol.json")))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert AA.obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    now = code_hashes()
    changed = [f for f, s in pr["code_sha256"].items() if now.get(f) != s and not f.startswith("diagnostics/report_")]
    assert not changed, f"frozen computation code changed: {changed}"
    return pr


# ---------------------------------------------------------------- one E3 quantization pass (OO capture / AO / AOF)
def load_original_perms(c):
    man = json.load(open(p("original_actorder_manifest.json")))
    assert sha256_file(p("original_actorder.pt")) == man["file_sha256"]
    t = torch.load(p("original_actorder.pt"))[c]
    for k, v in t.items():
        assert AA.perm_sha(v) == man["strategies"][c][k]["sha256"], k
    return t


@contextmanager
def install_spy(log):
    """I1 / I3 at write-back for every expert: installed parameter == bf16(W_solver), correctly rounded."""
    from moe_calib import model_utils as MU
    orig = MU.set_expert_weights

    def spy(model, L, e, wq):
        orig(model, L, e, wq)
        mod = MU.expert_module(model, L, e)
        for k, w in wq.items():
            inst = getattr(mod, k).weight.detach()
            if w.dtype == torch.float32:
                ok_r, r = AA.cast_rounding_ok(w.to(inst.device), inst)
                log.append(((L, e, k), bool(torch.equal(inst, w.to(inst.device, inst.dtype))), ok_r, r,
                            bool(torch.isfinite(inst).all())))
    MU.set_expert_weights = spy
    try:
        yield log
    finally:
        MU.set_expert_weights = orig


def quant_pass(c, mode, g):
    """mode: 'OO' (capture natural permutations + natural-override regression), 'AO' (regression vs stored AO),
    'AOF' (frozen ORIGINAL permutation). Returns (model, record dict, q)."""
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    import e3_allocation as E3
    cfg, sel, _, union = MRD.selections()
    test_ids = list(range(*cfg["split"]["test"]))
    gq, K = cfg["gptq"], sel[c]["K"]
    tables = FRD.load_tables("O")
    if mode == "OO":
        cdir, chosen, r2s, ecache = OCACHE, list(sel[c]["chosen"]), {s: s for s in union}, None
        Href = torch.load(oo_hessian_path(c))
    else:
        MRD.verify_cache(g)
        cdir, chosen, r2s, ecache = MRD.gdir(g), MRD.MG.remap(sel[c]["chosen"], union), dict(enumerate(union)), OCACHE
        Href = torch.load(XR.ao_hessian_path(c, g))
    oo_perm = load_original_perms(c) if mode == "AOF" else None
    ao_ref = None
    if mode in ("AOF", "AO"):                                   # AO reference tensors of the check experts
        wd = XR.ao_weights_dir(c, g)
        ao_ref = {}
        for L in LAYERS:
            Wf = torch.load(os.path.join(wd, f"L{L}.pt"))
            for e in (0, 30):
                ao_ref[(L, e)] = {k: Wf[f"{e}.{k}"].clone() for k in AA.PROJ}
            del Wf
    model, t_load = DB.load_model()
    pdig = json.load(open(os.path.join(ROOT, FRD.FD.OUT, "pristine_digests.json")))
    tg0 = PQ.target_tensor_digests(model, LAYERS)
    assert tg0 == {k: pdig["pristine_targets"][k] for k in tg0}, "targets not pristine"
    u0 = PQ.untouched_digests(model, LAYERS)
    router_dig = {kk: v for kk, v in u0.items() if ".mlp.gate." in kk}
    orig_w = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()} for L, e in CHECK}
    torch.cuda.reset_peak_memory_stats()
    rec = dict(perm={}, perm_ok=[], natural_override_bitwise=[], gate_up_perm_equal=[], hess_sha_equal=[], tokens={},
               ao_check_cast_equal_stored=[], quantizer={}, col_err2={}, n_dead={}, aof_spec=[])
    orig_q = E3.quantize_from_acc_or_fallback

    def qw(model_, acc, L, e, bits, group_size, percdamp):
        infos = {}
        if mode == "AOF":
            perms = {k: oo_perm[f"{L}.{e}.{k}"].long() for k in AA.PROJ}
            Wq, fb = orig_q(model_, acc, L, e, bits, group_size, percdamp, perms=perms, infos=infos)
            rec["perm_ok"].append(all(torch.equal(infos[k]["perm"], perms[k]) for k in AA.PROJ))
            if (L, e) in CHECK:                                 # AO default path on the SAME Hessian: == stored AO
                ia = {}
                Wao, _ = orig_q(model_, acc, L, e, bits, group_size, percdamp, infos=ia)
                rec["ao_check_cast_equal_stored"].append(all(torch.equal(Wao[k].to(torch.bfloat16).cpu(), ao_ref[(L, e)][k]) for k in AA.PROJ))
                rec["quantizer"][f"AO.{L}.{e}"] = {k: dict(scale=ia[k]["scale"], zero=ia[k]["zero"], perm=ia[k]["perm"].to(torch.int16),
                                                           col_err2=ia[k]["col_err2"]) for k in AA.PROJ}
                rec["aof_spec"].append(dict(L=L, e=e, weights_differ_from_ao={k: not torch.equal(Wq[k], Wao[k]) for k in AA.PROJ}))
        else:
            Wq, fb = orig_q(model_, acc, L, e, bits, group_size, percdamp, infos=infos)
            nat = {k: infos[k]["perm"].long() for k in AA.PROJ}
            W2, _ = orig_q(model_, acc, L, e, bits, group_size, percdamp, perms=nat)     # Regression B
            rec["natural_override_bitwise"].append(all(torch.equal(Wq[k], W2[k]) for k in AA.PROJ))
            if mode == "OO":
                Hp = Href["perm"]
                rec["perm_ok"].append(torch.equal(nat["gate_proj"], Hp[f"{L}.{e}.gu"].long())
                                      and torch.equal(nat["down_proj"], Hp[f"{L}.{e}.down"].long()))
            else:
                rec["perm_ok"].append(True)
                if (L, e) in CHECK:
                    rec["ao_check_cast_equal_stored"].append(all(torch.equal(Wq[k].to(torch.bfloat16).cpu(), ao_ref[(L, e)][k]) for k in AA.PROJ))
        rec["gate_up_perm_equal"].append(torch.equal(infos["gate_proj"]["perm"], infos["up_proj"]["perm"]))
        for k in AA.PROJ:
            rec["perm"][f"{L}.{e}.{k}"] = infos[k]["perm"].to(torch.int16)
            rec["col_err2"][f"{L}.{e}.{k}"] = infos[k]["col_err2"].float()
            rec["n_dead"][f"{L}.{e}.{k}"] = infos[k]["n_dead"]
        if (L, e) in CHECK:
            rec["quantizer"][f"{mode}.{L}.{e}"] = {k: dict(scale=infos[k]["scale"], zero=infos[k]["zero"], perm=infos[k]["perm"].to(torch.int16),
                                                           col_err2=infos[k]["col_err2"]) for k in AA.PROJ}
        return Wq, fb

    def on_hessian(L, e, acc):
        for pj in ("gu", "down"):
            H = acc.H_sum((L, e, pj), list(range(acc.K)))
            rec["hess_sha_equal"].append(FRT.tensor_sha(H) == Href["sha"][f"{L}.{e}.{pj}"])
            rec["tokens"][f"{L}.{e}.{pj}"] = acc.tokens((L, e, pj))

    log = []
    t1 = time.time()
    E3.quantize_from_acc_or_fallback = qw
    try:
        with install_spy(log), FRT.forced_e3(E3, tables, r2s, expect_natural=(mode == "OO"), on_hessian=on_hessian) as fstats:
            q = PQ.quantize_install(model, cdir, LAYERS, chosen, test_ids, K, gq["bits"], gq["group_size"], gq["percdamp"],
                                    capture=CHECK, eval_cache_dir=ecache)
    finally:
        E3.quantize_from_acc_or_fallback = orig_q
    assert E3.quantize_from_acc_or_fallback is orig_q
    rec["quantize_s"] = time.time() - t1
    FRT.assert_no_override(model)
    after = PQ.target_tensor_digests(model, LAYERS)
    u1 = PQ.untouched_digests(model, LAYERS)
    exp_tok = {f"{L}.{e}": float(FRT.expert_token_counts(tables[L], sel[c]["chosen"], 60)[e]) for L in LAYERS for e in range(60)}
    rec.update(
        model_load_s=t_load, routing_stats={str(L): fstats[L].stats() for L in LAYERS},
        n_quantized_experts=q["counts"]["n_experts_quantized_total"], n_projection_tensors=len(after),
        rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"],
        projections_unchanged_from_bf16=sum(after[k] == tg0[k] for k in after), nonfinite_weights=len(PQ.nonfinite_targets(model, LAYERS)),
        untouched_params_unchanged=u1 == u0, router_weights_unchanged={k: u1[k] for k in router_dig} == router_dig,
        token_counts_equal_R_O=all(rec["tokens"][f"{L}.{e}.gu"] == exp_tok[f"{L}.{e}"] == rec["tokens"][f"{L}.{e}.down"]
                                   for L in LAYERS for e in range(60)),
        I1_all_installed_equal_cast=len(log) == 720 and all(x[1] for x in log),
        I3_all_cast_correctly_rounded=len(log) == 720 and all(x[2] for x in log),
        I3_max_err_over_bound=max(x[3] for x in log) if log else None, I4_all_finite=all(x[4] for x in log), n_installs_logged=len(log),
        installed_weights_sha256=PQ.installed_digest(model, LAYERS),
        recon=float(np.mean([r["mean"] for r in q["results"]])),
        recon_per_layer={str(L): float(np.mean([r["mean"] for r in q["results"] if r["layer"] == L])) for L in LAYERS},
        per_expert=[dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"]) for r in q["results"]],
        hessian_s=q["hessian_s"], gptq_s=q["gptq_s"])
    # I2 + the frozen (old) check on the 8 check experts, frozen validation inputs
    from moe_calib.data_utils import load_layer_cache
    integ, old = [], []
    for L in LAYERS:
        cache = load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]]
        for e in (0, 30):
            Xc = PQ.expert_inputs(model, L, e, cache)
            dec = AA.precision_decomposition(MU.expert_module(model, L, e), q["captured"][(L, e)], orig_w[(L, e)], Xc, MU.GPU)
            dec.update(layer=L, expert_id=e, integrity_pass=AA.integrity_pass(dec))
            integ.append(dec)
            ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], orig_w[(L, e)], Xc)
            ck["tolerance"] = EP.INSTALL_TOL(ck)
            ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                              and ck["installed_vs_e3_rel"] <= ck["tolerance"])
            old.append(ck)
    rec["integrity_checks"] = [{k: v for k, v in d.items() if k != "per_projection"} | dict(per_projection=d["per_projection"]) for d in integ]
    rec["I2_all_module_equal_reference"] = all(d["expert"]["module_bitwise_equal_independent_reference"] for d in integ)
    rec["integrity_pass"] = bool(rec["I1_all_installed_equal_cast"] and rec["I3_all_cast_correctly_rounded"] and rec["I4_all_finite"]
                                 and all(d["integrity_pass"] for d in integ))
    rec["frozen_old_check"] = old
    rec["frozen_old_check_all_pass"] = all(ck["pass"] for ck in old)
    rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    rec["peak_cpu_gb"] = FRD.FD.peak_cpu_gb()
    return model, rec, q


def base_ok(rec):
    return bool(rec["n_quantized_experts"] == 240 and rec["n_projection_tensors"] == 720 and rec["rtn_fallbacks"] == 0
                and rec["projections_unchanged_from_bf16"] == 0 and rec["nonfinite_weights"] == 0
                and rec["untouched_params_unchanged"] and rec["router_weights_unchanged"] and rec["token_counts_equal_R_O"]
                and rec["integrity_pass"] and all(rec["gate_up_perm_equal"]) and len(rec["perm_ok"]) == 240 and all(rec["perm_ok"])
                and len(rec["hess_sha_equal"]) == 480 and all(rec["hess_sha_equal"]))


def save_quantizer(rec, name):
    os.makedirs(os.path.join(ACACHE, "quantizer"), exist_ok=True)
    path = os.path.join(ACACHE, "quantizer", f"{name}.pt")
    torch.save(dict(check_experts=rec["quantizer"], col_err2=rec["col_err2"], n_dead=rec["n_dead"]), path)
    return path, sha256_file(path)


def slim(rec):
    return {k: v for k, v in rec.items() if k not in ("perm", "quantizer", "col_err2", "tokens", "n_dead")}


# ---------------------------------------------------------------- stages: capture, merge, regression, AOF
def stage_capture_order(a):
    c = a.strategy
    path = p("order_capture", f"{c}.json")
    if load_json(path) and load_json(path)["passed"]:
        print(f"[capture_order] {c} done"); return
    assert (load_json(p("installation_audit.json")) or {}).get("decision") == "AUDIT-CONFIRMED-BF16-REPRESENTATION"
    model, rec, q = quant_pass(c, "OO", None)
    ref = json.load(open(XR.oo_result_path(c)))
    ref_pe = [(r["layer"], r["expert"], r["mean"], r["tokens"]) for r in ref["per_expert"]]
    res = slim(rec) | dict(
        strategy=c, regression_A=dict(weights_equal_frozen_OO=rec["installed_weights_sha256"] == ref["installed_weights_sha256"],
                                      recon_equal=rec["recon"] == ref["recon"],
                                      per_expert_recon_equal=[(d["layer"], d["expert"], d["mean"], d["tokens"]) for d in rec["per_expert"]] == ref_pe,
                                      hessians_equal_frozen_OO=f"{sum(rec['hess_sha_equal'])}/{len(rec['hess_sha_equal'])}",
                                      perms_equal_frozen_OO_hessian_actorder=f"{sum(rec['perm_ok'])}/{len(rec['perm_ok'])}"),
        regression_B_natural_override_bitwise=f"{sum(rec['natural_override_bitwise'])}/{len(rec['natural_override_bitwise'])}")
    res["passed"] = bool(base_ok(rec) and all(rec["natural_override_bitwise"]) and len(rec["natural_override_bitwise"]) == 240
                         and res["regression_A"]["weights_equal_frozen_OO"] and res["regression_A"]["recon_equal"]
                         and res["regression_A"]["per_expert_recon_equal"])
    os.makedirs(p("order_capture"), exist_ok=True)
    torch.save(rec["perm"], p("order_capture", f"{c}_perm.pt"))
    res["quantizer_file"], res["quantizer_file_sha256"] = save_quantizer(rec, f"{c}_OO")
    res["perm_file_sha256"] = sha256_file(p("order_capture", f"{c}_perm.pt"))
    hf = torch.load(oo_hessian_path(c))
    res["hessian_sha"] = {k: hf["sha"][k] for k in hf["sha"]}
    jdump(res, path)
    print(f"[capture_order] {c}: weights==OO {res['regression_A']['weights_equal_frozen_OO']} recon {res['regression_A']['recon_equal']} "
          f"hessians {res['regression_A']['hessians_equal_frozen_OO']} perms {res['regression_A']['perms_equal_frozen_OO_hessian_actorder']} "
          f"natural-override {res['regression_B_natural_override_bitwise']} integrity {rec['integrity_pass']} -> passed {res['passed']} "
          f"({rec['quantize_s']:.0f} s)", flush=True)
    if not res["passed"]:
        raise SystemExit(f"capture_order {c} FAILED")


def stage_merge_order(a):
    if os.path.exists(p("original_actorder_manifest.json")):
        print("[merge_order] done"); return
    gq = json.load(open(p("input_manifest.json")))["gptq"]
    allp, man = {}, dict(strategies={}, gptq=gq, gptq_config_sha256=AA.obj_sha(gq),
                         source="permutation applied inside moe_calib.gptq.gptq_quantize (info['perm']) during the OO "
                                "re-run whose installed weights equal the frozen OO bitwise",
                         gate_up_shared="gate_proj and up_proj use the same H_gu; equality of their permutations verified per expert")
    for c in STRATS:
        r = json.load(open(p("order_capture", f"{c}.json")))
        assert r["passed"]
        assert sha256_file(p("order_capture", f"{c}_perm.pt")) == r["perm_file_sha256"]
        t = torch.load(p("order_capture", f"{c}_perm.pt"))
        assert len(t) == 720
        allp[c] = t
        man["strategies"][c] = {k: dict(layer=int(k.split(".")[0]), expert=int(k.split(".")[1]), projection=k.split(".")[2],
                                        length=int(v.numel()), sha256=AA.perm_sha(v),
                                        hessian_sha256=r["hessian_sha"][f"{k.rsplit('.', 1)[0]}.{AA.PROJ_H[k.split('.')[2]]}"],
                                        is_permutation=bool(torch.equal(torch.sort(v.long()).values, torch.arange(v.numel()))))
                                for k, v in t.items()}
        assert all(v["is_permutation"] for v in man["strategies"][c].values())
    torch.save(allp, p("original_actorder.pt"))
    man["file_sha256"] = sha256_file(p("original_actorder.pt"))
    jdump(man, p("original_actorder_manifest.json"))
    print(f"[merge_order] {sum(len(v) for v in allp.values())} permutations saved")


def stage_solver_regression(a):
    if load_json(p("solver_regression.json")) and load_json(p("solver_regression.json"))["passed"]:
        print("[solver_regression] done"); return
    audit = load_json(p("installation_audit.json")) or {}
    assert audit.get("decision") == "AUDIT-CONFIRMED-BF16-REPRESENTATION"
    c, g = "FR", "G20261111"
    perms = load_original_perms(c)                        # tensors load and match the manifest
    model, rec, q = quant_pass(c, "AO", g)
    ref = json.load(open(XR.ao_result_path(c, g)))
    oc = {c2: json.load(open(p("order_capture", f"{c2}.json"))) for c2 in STRATS}
    res = slim(rec) | dict(
        strategy=c, grouping=g,
        regression_A_default_on_AO_hessian=dict(weights_equal_stored_AO=rec["installed_weights_sha256"] == ref["installed_weights_sha256"],
                                                recon_equal=rec["recon"] == ref["recon"],
                                                hessians_equal_stored_AO=f"{sum(rec['hess_sha_equal'])}/{len(rec['hess_sha_equal'])}"),
        regression_B_natural_override_bitwise_AO=f"{sum(rec['natural_override_bitwise'])}/{len(rec['natural_override_bitwise'])}",
        regression_B_natural_override_bitwise_OO={c2: oc[c2]["regression_B_natural_override_bitwise"] for c2 in STRATS},
        regression_A_OO={c2: oc[c2]["regression_A"] for c2 in STRATS},
        original_perms_loaded=dict(n=len(perms), manifest_sha_verified=True),
        regression_D_note="AOF specificity (permutation == ORIGINAL, Hessian == AO, AO default path on the same Hessian "
                          "== stored AO, weights not copied from OO) is verified inside every AOF configuration")
    res["passed"] = bool(base_ok(rec) and all(rec["natural_override_bitwise"]) and len(rec["natural_override_bitwise"]) == 240
                         and res["regression_A_default_on_AO_hessian"]["weights_equal_stored_AO"]
                         and res["regression_A_default_on_AO_hessian"]["recon_equal"]
                         and all(oc[c2]["passed"] for c2 in STRATS))
    jdump(res, p("solver_regression.json"))
    print(f"[solver_regression] AO default == stored {res['regression_A_default_on_AO_hessian']} natural override "
          f"{res['regression_B_natural_override_bitwise_AO']} -> passed {res['passed']}", flush=True)
    if not res["passed"]:
        raise SystemExit("solver regression FAILED: AOF blocked")
    freeze_protocol(audit)                    # integrity protocol + decision rule + code, before any AOF run


def stage_aof(a):
    from moe_calib import partial_model_quant as PQ
    c, g = a.strategy, a.grouping
    rpath = aof_result_path(c, g)
    if os.path.exists(rpath):
        print(f"[aof] {aof_name(c, g)} done"); return
    pr = protocol()
    assert (load_json(p("solver_regression.json")) or {}).get("passed"), "solver regression not passed: AOF blocked"
    model, rec, q = quant_pass(c, "AOF", g)
    name = aof_name(c, g)
    wd = os.path.join(ACACHE, "weights", name)
    os.makedirs(wd, exist_ok=True)
    files = {}
    for L in LAYERS:
        path = os.path.join(wd, f"L{L}.pt")
        PQ.save_layer(model, L, path)
        files[str(L)] = dict(sha256=sha256_file(path), layer_sha256=PQ.layer_digest(model, L))
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == pr["evaluation"]["windows_sha256"]
    FRT.assert_no_override(model)
    evals = []
    for _ in range(2):
        t2 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        evals.append(EP.ppl_record(nll, cnt, time.time() - t2))
    ev = evals[0]
    qf, qsha = save_quantizer(rec, name)
    oo_digest = json.load(open(XR.oo_result_path(c)))["installed_weights_sha256"]
    out = slim(rec) | dict(
        strategy=c, grouping=g, condition="AOF", name=name, weights_dir=wd, weight_files=files, quantizer_file=qf,
        quantizer_file_sha256=qsha, perm_equal_original=f"{sum(rec['perm_ok'])}/{len(rec['perm_ok'])}",
        hessians_equal_AO=f"{sum(rec['hess_sha_equal'])}/{len(rec['hess_sha_equal'])}",
        ao_default_on_same_hessian_equals_stored_AO=f"{sum(rec['ao_check_cast_equal_stored'])}/{len(rec['ao_check_cast_equal_stored'])}",
        weights_not_copied_from_OO=rec["installed_weights_sha256"] != oo_digest,
        repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
        repeat_eval_per_window_identical=evals[0]["per_window_nll"] == evals[1]["per_window_nll"], **ev)
    out["valid"] = bool(base_ok(rec) and all(rec["ao_check_cast_equal_stored"]) and len(rec["ao_check_cast_equal_stored"]) == 8
                        and out["weights_not_copied_from_OO"] and out["repeat_eval_per_window_identical"]
                        and np.isfinite(ev["per_window_nll"]).all())
    jdump(out, rpath)
    print(f"[aof] {name}: PPL {ev['ppl']:.6f} recon {rec['recon']:.7f} perms {out['perm_equal_original']} hessians "
          f"{out['hessians_equal_AO']} integrity {rec['integrity_pass']} old-check {rec['frozen_old_check_all_pass']} "
          f"valid {out['valid']} ({rec['quantize_s']:.0f} s)", flush=True)
    if not out["valid"]:
        raise SystemExit(f"{name} failed its correctness checks")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "audit", "capture_order", "merge_order",
                                                       "solver_regression", "aof", "report"])
    ap.add_argument("--strategy", choices=STRATS)
    ap.add_argument("--grouping", choices=G)
    a = ap.parse_args(argv)
    for d in ("logs", "figures", "results"):
        os.makedirs(p(d), exist_ok=True)
    if a.stage == "report":
        import report_actorder_ablation as R
        R.main(); return
    dict(preflight=stage_preflight, audit=stage_audit, capture_order=stage_capture_order, merge_order=stage_merge_order,
         solver_regression=stage_solver_regression, aof=stage_aof)[a.stage](a)


if __name__ == "__main__":
    main()
