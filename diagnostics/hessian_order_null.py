"""FP32 Hessian accumulation-order null experiment (FR s0 / FC s0, layers 1/8/16/23, ORIGINAL activations,
ORIGINAL routing, ORIGINAL act-order P_O). Only the order in which the 128 per-sequence contributions enter the
FP32 block accumulation changes (moe_calib/hessian_order_null.py documents the mechanism).

Stages (--stage):
  preflight                          inputs, references, resources -> input_manifest.json
  baseline_regression --strategy S   original order through the new pathway (OrderedForcedProfiler, override P_O):
                                     Hessians / act-order / weights / reconstruction / PPL / per-window NLL == frozen OO
  order_manifests                    five prespecified orders + hashes; freezes protocol.json (decision rule, code)
  hessian_preflight --strategy S     original + 5 orders, Hessians only (no GPTQ): differences vs the original order
  fp64_control --strategy S          FP64 vs FP32 block accumulation, original / 20261121 / 20261122 (diagnostic)
  quant --strategy S --order SEED [--repeat]   one GPTQ + PPL configuration (fresh process); refuses if every
                                     Hessian was order-invariant (early stop)
  report
"""
from __future__ import annotations
import argparse, gc, json, os, shutil, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import actorder_ablation as XA          # noqa: E402
import batch_sensitivity as DB          # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import frozen_routing as FRD            # noqa: E402
import frozen_routing_robustness as XR  # noqa: E402
import multigroup_robustness as MRD     # noqa: E402
from moe_calib import actorder_ablation as AA          # noqa: E402
from moe_calib import frozen_routing as FRT            # noqa: E402
from moe_calib import hessian_order_null as HN         # noqa: E402

OUT = "runs_hessian_order_null"
HCACHE = os.path.expanduser(os.environ.get("HESSIAN_ORDER_CACHE", "~/moe_cache_hessian_order_null"))
LAYERS, STRATS, OCACHE = FRD.LAYERS, list(MRD.STRATS), FRD.OCACHE
CHECK = [tuple(x) for x in EP.CHECK_EXPERTS]
DIAG_KEYS = FRT.DIAG_KEYS
SEL_PREFIX = dict(FR="ec355fb8d866e4ff", FC="0353229b6d3090db")
OO_PPL = dict(FR=9.267851290916612, FC=9.277908190612935)
FP64_CHUNK = 15
jdump, sha256_file = FRD.jdump, FRD.sha256_file
CODE_FILES = ["moe_calib/hessian_order_null.py", "diagnostics/hessian_order_null.py", "moe_calib/gptq.py",
              "moe_calib/quant_eval.py", "moe_calib/profiler.py", "moe_calib/hessian.py", "moe_calib/frozen_routing.py",
              "moe_calib/actorder_ablation.py", "diagnostics/actorder_ablation.py", "e3_allocation.py",
              "moe_calib/partial_model_quant.py", "diagnostics/eval_partial_ppl.py",
              "diagnostics/report_hessian_order_null.py"]


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def oname(seed):
    return "O0" if seed is None else f"O{seed}"


def cname(c, seed, repeat=False):
    return f"{c}_{oname(seed)}" + ("_repeat" if repeat else "")


def hess_file(c, seed):
    return os.path.join(HCACHE, "hessians", f"{c}_{oname(seed)}.pt")


def code_hashes():
    return {f: sha256_file(os.path.join(ROOT, f)) for f in CODE_FILES if os.path.exists(os.path.join(ROOT, f))}


def resources():
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    return dict(host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, gpu_free_gb=free / 1e9,
                gpu_total_gb=tot / 1e9, disk_free_tb=shutil.disk_usage(os.path.expanduser("~")).free / 1e12)


# ---------------------------------------------------------------- preflight
def stage_preflight(a):
    cfg, sel, checks, union = MRD.selections()
    mpr = MRD.protocol()
    FRD.protocol()
    XR.protocol()
    XA.protocol()                                   # act-order study: protocol + frozen computation code unchanged
    for c in STRATS:
        assert sel[c]["ids_hash"] == SEL_PREFIX[c] and len(set(sel[c]["chosen"])) == 128
    cache_files = {}
    for L in LAYERS:
        h = sha256_file(os.path.join(OCACHE, f"L{L}.pt"))
        assert h == mpr["references"]["original_cache_files"][str(L)], f"original cache L{L} modified"
        cache_files[str(L)] = h
    tables = FRD.load_tables("O")                   # verifies routing_original.pt sha
    rman = json.load(open(FRD.p("routing_original.json")))
    perms = {c: XA.load_original_perms(c) for c in STRATS}       # verifies all 720 permutation hashes per strategy
    aman = json.load(open(XA.p("original_actorder_manifest.json")))
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    assert abs(base["ppl"] - 9.237090) < 1e-6 and prot["scored_targets_total"] == 65408 and tuple(win.shape) == (128, 512)
    assert EP.BATCH == 8
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True) and cfg["layers"] == LAYERS
    assert sel["FR"]["K"] == sel["FC"]["K"] == 4
    refs = {}
    for c in STRATS:
        r = json.load(open(XR.oo_result_path(c)))
        assert r["ppl"] == OO_PPL[c] and r["valid"] and r["rtn_fallbacks"] == 0
        hf = XA.oo_hessian_path(c)
        refs[c] = dict(result=os.path.relpath(XR.oo_result_path(c), ROOT), result_sha256=sha256_file(XR.oo_result_path(c)),
                       ppl=r["ppl"], recon=r["recon"], installed_weights_sha256=r["installed_weights_sha256"],
                       weights_dir=r["weights_dir"], weight_files={L: v["sha256"] for L, v in r["weight_files"].items()},
                       hessian_file=hf, hessian_file_sha256=sha256_file(hf))
        for L, v in r["weight_files"].items():
            assert sha256_file(os.path.join(r["weights_dir"], f"L{L}.pt")) == v["sha256"]
    res = resources()
    pre = dict(selections=checks, environment=DB.env_record(), created=time.strftime("%F %T"), **res,
               feasible=bool(res["gpu_free_gb"] > 12 and res["disk_free_tb"] > 0.1))
    jdump(pre, p("logs", "preflight.json"))
    jdump(dict(selections={c: dict(ids_hash=sel[c]["ids_hash"], selection_sha256=HN.selection_sha(sel[c]["chosen"]),
                                   K=sel[c]["K"], n=128, unique=len(set(sel[c]["chosen"])), replay_order=sel[c]["chosen"])
                           for c in STRATS},
               original_cache_dir=OCACHE, original_cache_files=cache_files,
               routing_original=dict(file=FRD.p("routing_original.pt"), file_sha256=rman["file_sha256"],
                                     n_layers=len(tables)),
               original_actorder=dict(file=XA.p("original_actorder.pt"), file_sha256=aman["file_sha256"],
                                      n_permutations={c: len(perms[c]) for c in STRATS}),
               windows_sha256=prot["windows_sha256"], eval_batch=EP.BATCH, bf16_ppl=base["ppl"], gptq=cfg["gptq"],
               layers=LAYERS, oo_references=refs,
               earlier_protocols=dict(actorder=json.load(open(XA.p("protocol.json")))["protocol_sha256"],
                                      frozen_routing=json.load(open(FRD.p("protocol.json")))["protocol_sha256"])),
          p("input_manifest.json"))
    print(f"preflight ok: GPU free {res['gpu_free_gb']:.1f} GB, host {res['host_mem_available_gb']:.0f} GB")
    if not pre["feasible"]:
        raise SystemExit("preflight: insufficient resources")


# ---------------------------------------------------------------- one quantization pass (ordered Hessians, P_O)
def quant_pass(c, seed, ref_sha):
    """GPTQ of all 240 experts with Hessians accumulated in order `seed` (None = original) and act-order fixed to
    P_O. ref_sha: {key: sha256} the regenerated Hessians must equal. Returns (model, rec, q)."""
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    import e3_allocation as E3
    cfg, sel, _, union = MRD.selections()
    test_ids = list(range(*cfg["split"]["test"]))
    gq, K = cfg["gptq"], sel[c]["K"]
    chosen = list(sel[c]["chosen"])
    order, blocks, _ = HN.order_spec(chosen, seed, K)
    tables = FRD.load_tables("O")
    P = XA.load_original_perms(c)
    model, t_load = DB.load_model()
    pdig = json.load(open(os.path.join(ROOT, FRD.FD.OUT, "pristine_digests.json")))
    tg0 = PQ.target_tensor_digests(model, LAYERS)
    assert tg0 == {k: pdig["pristine_targets"][k] for k in tg0}, "targets not pristine"
    u0 = PQ.untouched_digests(model, LAYERS)
    router_dig = {kk: v for kk, v in u0.items() if ".mlp.gate." in kk}
    orig_w = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()} for L, e in CHECK}
    torch.cuda.reset_peak_memory_stats()
    rec = dict(perm_ok=[], gate_up_perm_equal=[], hess_sha_equal=[], tokens={}, block_tokens={}, natural_vs_PO={},
               quantizer={}, col_err2={}, codes_exact=[])
    orig_q = E3.quantize_from_acc_or_fallback

    def qw(model_, acc, L, e, bits, group_size, percdamp):
        perms = {k: P[f"{L}.{e}.{k}"].long() for k in AA.PROJ}
        infos = {}
        Wq, fb = orig_q(model_, acc, L, e, bits, group_size, percdamp, perms=perms, infos=infos)
        rec["perm_ok"].append(all(torch.equal(infos[k]["perm"], perms[k]) for k in AA.PROJ))
        rec["gate_up_perm_equal"].append(torch.equal(infos["gate_proj"]["perm"], infos["up_proj"]["perm"]))
        for k in AA.PROJ:
            rec["col_err2"][f"{L}.{e}.{k}"] = infos[k]["col_err2"].float()
        if (L, e) in CHECK:
            ent = {}
            for k in AA.PROJ:
                codes, exact = HN.codes_from(Wq[k], perms[k], infos[k]["scale"], infos[k]["zero"], group_size)
                rec["codes_exact"].append(exact)
                ent[k] = dict(scale=infos[k]["scale"], zero=infos[k]["zero"], codes=codes)
            rec["quantizer"][f"{L}.{e}"] = ent
        return Wq, fb

    def on_hessian(L, e, acc):
        for pj in ("gu", "down"):
            H = acc.H_sum((L, e, pj), list(range(acc.K)))
            key = f"{L}.{e}.{pj}"
            rec["hess_sha_equal"].append(FRT.tensor_sha(H) == ref_sha[key])
            rec["tokens"][key] = acc.tokens((L, e, pj))
            rec["block_tokens"][key] = acc.raw_n[(L, e, pj)].tolist()
            pk = "gate_proj" if pj == "gu" else "down_proj"
            rec["natural_vs_PO"][key] = AA.perm_compare(HN.natural_perm(H).cpu(), P[f"{L}.{e}.{pk}"])

    log = []
    t1 = time.time()
    E3.quantize_from_acc_or_fallback = qw
    try:
        with XA.install_spy(log), HN.ordered_forced_e3(E3, tables, {s: s for s in union}, chosen, order, blocks,
                                                       on_hessian=on_hessian) as fstats:
            q = PQ.quantize_install(model, OCACHE, LAYERS, chosen, test_ids, K, gq["bits"], gq["group_size"],
                                    gq["percdamp"], capture=CHECK, eval_cache_dir=None)
    finally:
        E3.quantize_from_acc_or_fallback = orig_q
    assert E3.quantize_from_acc_or_fallback is orig_q
    rec["quantize_s"] = time.time() - t1
    FRT.assert_no_override(model)
    after = PQ.target_tensor_digests(model, LAYERS)
    u1 = PQ.untouched_digests(model, LAYERS)
    exp_tok = {f"{L}.{e}": float(FRT.expert_token_counts(tables[L], chosen, 60)[e]) for L in LAYERS for e in range(60)}
    rec.update(
        order_seed=seed, replay_order=order, blocks=blocks, model_load_s=t_load,
        routing_stats={str(L): fstats[L].stats() for L in LAYERS},
        routing_equal_R_O_all_tokens=all(fstats[L].n_ordered_diff == 0 and fstats[L].n_seq == 128 for L in LAYERS),
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
    from moe_calib.data_utils import load_layer_cache
    integ, old = [], []
    for L in LAYERS:
        cache = load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]]
        for e in (0, 30):
            Xc = PQ.expert_inputs(model, L, e, cache)
            dec = AA.precision_decomposition(MU.expert_module(model, L, e), q["captured"][(L, e)], orig_w[(L, e)], Xc, MU.GPU)
            dec.update(layer=L, expert_id=e, integrity_pass=AA.integrity_pass(dec))
            integ.append({k: v for k, v in dec.items() if k != "per_projection"})
            ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], orig_w[(L, e)], Xc)
            ck["tolerance"] = EP.INSTALL_TOL(ck)
            ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                              and ck["installed_vs_e3_rel"] <= ck["tolerance"])
            old.append(ck)
    rec["integrity_checks"] = integ
    rec["I2_all_module_equal_reference"] = all(d["expert"]["module_bitwise_equal_independent_reference"] for d in integ)
    rec["integrity_pass"] = bool(rec["I1_all_installed_equal_cast"] and rec["I3_all_cast_correctly_rounded"] and rec["I4_all_finite"]
                                 and all(d["integrity_pass"] for d in integ))
    rec["frozen_old_check"] = old
    rec["frozen_old_check_all_pass"] = all(ck["pass"] for ck in old)
    rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    rec["peak_cpu_gb"] = FRD.FD.peak_cpu_gb()
    return model, rec, q


def base_ok(rec):
    return bool(XA.base_ok(rec) and rec["routing_equal_R_O_all_tokens"] and all(rec["codes_exact"])
                and len(rec["codes_exact"]) == 24)


def evaluate(model, pr_windows_sha):
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == pr_windows_sha
    FRT.assert_no_override(model)
    evals = []
    for _ in range(2):
        t2 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        evals.append(EP.ppl_record(nll, cnt, time.time() - t2))
    return evals


def save_outputs(model, rec, name):
    from moe_calib import partial_model_quant as PQ
    wd = os.path.join(HCACHE, "weights", name)
    os.makedirs(wd, exist_ok=True)
    files = {}
    for L in LAYERS:
        path = os.path.join(wd, f"L{L}.pt")
        PQ.save_layer(model, L, path)
        files[str(L)] = dict(sha256=sha256_file(path), layer_sha256=PQ.layer_digest(model, L))
    qd = os.path.join(HCACHE, "quantizer")
    os.makedirs(qd, exist_ok=True)
    qf = os.path.join(qd, f"{name}.pt")
    torch.save(dict(check_experts=rec["quantizer"], col_err2=rec["col_err2"]), qf)
    return wd, files, qf, sha256_file(qf)


def slim(rec):
    return {k: v for k, v in rec.items() if k not in ("quantizer", "col_err2", "tokens", "block_tokens", "natural_vs_PO")}


def perm_summary(nv):
    v = list(nv.values())
    return dict(n=len(v), n_identical=sum(x["identical"] for x in v),
                mean_frac_positions_changed=float(np.mean([x["frac_positions_changed"] for x in v])),
                mean_frac_columns_changed_group=float(np.mean([x["frac_columns_changed_group"] for x in v])))


# ---------------------------------------------------------------- baseline regression
def stage_baseline_regression(a):
    c = a.strategy
    path = p("baseline_regression", f"{c}.json")
    if (load_json(path) or {}).get("passed"):
        print(f"[baseline] {c} done"); return
    man = json.load(open(p("input_manifest.json")))
    hf = torch.load(XA.oo_hessian_path(c))
    assert sha256_file(XA.oo_hessian_path(c)) == man["oo_references"][c]["hessian_file_sha256"]
    model, rec, q = quant_pass(c, None, hf["sha"])
    evals = evaluate(model, man["windows_sha256"])
    ev = evals[0]
    name = cname(c, None)
    wd, files, qf, qsha = save_outputs(model, rec, name)
    ref = json.load(open(XR.oo_result_path(c)))
    ref_pe = [(r["layer"], r["expert"], r["mean"], r["tokens"]) for r in ref["per_expert"]]
    reg = dict(
        hessians_equal_frozen_OO=f"{sum(rec['hess_sha_equal'])}/{len(rec['hess_sha_equal'])}",
        natural_actorder_equal_P_O=f"{sum(x['identical'] for x in rec['natural_vs_PO'].values())}/{len(rec['natural_vs_PO'])}",
        applied_perm_equal_P_O=f"{sum(rec['perm_ok'])}/{len(rec['perm_ok'])}",
        replay_order_equal_original=rec["replay_order"] == list(MRD.selections()[1][c]["chosen"]),
        blocks_equal_positional=rec["blocks"] == HN.positional_blocks(128, 4),
        token_counts_equal_R_O=rec["token_counts_equal_R_O"], routing_equal_R_O=rec["routing_equal_R_O_all_tokens"],
        weights_equal_frozen_OO=rec["installed_weights_sha256"] == ref["installed_weights_sha256"],
        weight_files_equal=all(files[L]["sha256"] == v["sha256"] for L, v in ref["weight_files"].items()),
        recon_equal=rec["recon"] == ref["recon"],
        per_expert_recon_equal=[(d["layer"], d["expert"], d["mean"], d["tokens"]) for d in rec["per_expert"]] == ref_pe,
        ppl_equal=ev["ppl"] == ref["ppl"], per_window_nll_equal=ev["per_window_nll"] == ref["per_window_nll"],
        repeat_eval_identical=evals[0]["per_window_nll"] == evals[1]["per_window_nll"])
    out = slim(rec) | dict(strategy=c, name=name, regression=reg, weights_dir=wd, weight_files=files, quantizer_file=qf,
                           quantizer_file_sha256=qsha, oo_ppl=ref["ppl"], **ev)
    out["passed"] = bool(base_ok(rec) and all(v is True or (isinstance(v, str) and v.split("/")[0] == v.split("/")[1])
                                              for v in reg.values()))
    os.makedirs(p("baseline_regression"), exist_ok=True)
    jdump(out, path)
    print(f"[baseline] {c}: {reg} integrity {rec['integrity_pass']} -> passed {out['passed']} ({rec['quantize_s']:.0f} s)", flush=True)
    if not out["passed"]:
        raise SystemExit(f"baseline regression {c} FAILED: RUN-BLOCKED")


# ---------------------------------------------------------------- order manifests + protocol freeze
def stage_order_manifests(a):
    for c in STRATS:
        assert (load_json(p("baseline_regression", f"{c}.json")) or {}).get("passed"), f"baseline {c} not passed"
    man = json.load(open(p("input_manifest.json")))
    os.makedirs(p("order_manifests"), exist_ok=True)
    idx = {}
    for seed in HN.ORDER_SEEDS:
        path = p("order_manifests", f"O{seed}.json")
        body = dict(seed=seed, generator="numpy.random.default_rng(seed).permutation(128)",
                    index_permutation=HN.index_permutation(seed), strategies={})
        for c in STRATS:
            chosen = man["selections"][c]["replay_order"]
            order, blocks, perm = HN.order_spec(chosen, seed, 4)
            wb = HN.within_block_orders(order, blocks, 4)
            ob = HN.within_block_orders(chosen, HN.positional_blocks(128, 4), 4)
            body["strategies"][c] = dict(replay_order=order, blocks=blocks,
                                         block_members_equal_original={str(k): sorted(wb[k]) == sorted(ob[k]) for k in wb},
                                         within_block_order_changed={str(k): wb[k] != ob[k] for k in wb},
                                         multiset_equal=sorted(order) == sorted(chosen))
        body["manifest_sha256"] = HN.obj_sha(body)
        if os.path.exists(path):
            assert json.load(open(path))["manifest_sha256"] == body["manifest_sha256"], f"{path} changed"
        else:
            jdump(body, path)
        idx[str(seed)] = dict(file=os.path.relpath(path, ROOT), manifest_sha256=body["manifest_sha256"],
                              file_sha256=sha256_file(path))
    jdump(dict(seeds=list(HN.ORDER_SEEDS), manifests=idx), p("order_manifests", "index.json"))
    if os.path.exists(p("protocol.json")):
        protocol(); print("[order_manifests] protocol already frozen"); return
    br = {c: load_json(p("baseline_regression", f"{c}.json")) for c in STRATS}
    body = dict(
        experiment="FP32 Hessian accumulation-order null experiment (FR s0 / FC s0, layers 1/8/16/23)",
        question="Can changing only the FP32 accumulation order of the same expert Hessian contributions produce "
                 "different GPTQ solutions and measurable model-level PPL variation?",
        inputs=dict(input_manifest_sha256=sha256_file(p("input_manifest.json"))),
        mechanism="per-sequence contribution Xf.T @ Xf (FP32 GEMM, unchanged) added with FP32 += into the block of "
                  "the sequence's ORIGINAL replay position (K=4 contiguous blocks of 32); blocks summed over dim 0; "
                  "intervention permutes the replay order, keeping every sequence's block, so only the FP32 "
                  "addition order inside each block changes",
        orders=idx, fp64_control_seeds=list(HN.FP64_SEEDS),
        fixed=dict(selections={c: man["selections"][c]["selection_sha256"] for c in STRATS},
                   activations=man["original_cache_files"], routing=man["routing_original"]["file_sha256"],
                   actorder=man["original_actorder"]["file_sha256"], gptq=man["gptq"], windows_sha256=man["windows_sha256"],
                   eval_batch=man["eval_batch"]),
        baseline_regression={c: dict(passed=br[c]["passed"], installed_weights_sha256=br[c]["installed_weights_sha256"])
                             for c in STRATS},
        integrity_protocol=AA.INTEGRITY_PROTOCOL, frozen_old_tolerance="reported, not gating (installation audit)",
        decision_rule=HN.DECISION_RULE, tau_ppl=HN.TAU_PPL, diag_keys=[list(k) for k in DIAG_KEYS],
        code_sha256=code_hashes(),
        hard_constraints="no change of selections, caches, routing, act-order, group membership, GPTQ "
                         "hyperparameters, precision, installation, evaluation; FP64 only as a diagnostic; "
                         "no adaptive orders; earlier results untouched")
    body["protocol_sha256"] = HN.obj_sha(body)
    body["created"] = time.strftime("%F %T")
    jdump(body, p("protocol.json"))
    print(f"[order_manifests] 5 orders saved; protocol frozen sha256 {body['protocol_sha256'][:16]}")


def protocol():
    pr = json.load(open(p("protocol.json")))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert HN.obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    now = code_hashes()
    changed = [f for f, s in pr["code_sha256"].items() if now.get(f) != s and not f.startswith("diagnostics/report_")]
    assert not changed, f"frozen computation code changed: {changed}"
    for s, v in pr["orders"].items():
        assert sha256_file(os.path.join(ROOT, v["file"])) == v["file_sha256"], f"order manifest {s} modified"
    return pr


# ---------------------------------------------------------------- Hessian preflight (no GPTQ)
def stage_hessian_preflight(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    c = a.strategy
    path = p("hessian_preflight", f"{c}.json")
    if load_json(path):
        print(f"[hessian] {c} done"); aggregate_hessian(); return
    pr = protocol()
    man = json.load(open(p("input_manifest.json")))
    chosen = man["selections"][c]["replay_order"]
    K = man["selections"][c]["K"]
    P = XA.load_original_perms(c)
    oo = torch.load(XA.oo_hessian_path(c))
    tables = FRD.load_tables("O")
    gq = man["gptq"]
    model, _ = DB.load_model()
    seeds = [None] + list(HN.ORDER_SEEDS)
    files = {oname(s): dict(full={}, diag={}, sha={}, tokens={}, block_tokens={}) for s in seeds}
    keys_m = {oname(s): {} for s in seeds}
    members, contrib, routing, tsec = {}, {}, {}, {}
    torch.cuda.reset_peak_memory_stats()
    for L in LAYERS:
        cache = load_layer_cache(OCACHE, L)
        base = {}
        for s in seeds:
            order, blocks, _ = HN.order_spec(chosen, s, K)
            rec = {}
            t0 = time.time()
            prof = HN.OrderedForcedProfiler(model, L, tables[L], {x: x for x in chosen}, True, chosen, order, blocks,
                                            record=rec, fp_keys=[k for k in DIAG_KEYS if k[0] == L], K=K)
            prof.run(cache, chosen, weights=[1.0] * len(chosen))
            tsec[f"{oname(s)}.{L}"] = time.time() - t0
            st = prof.stats()
            routing[f"{oname(s)}.{L}"] = st | dict(equal_R_O=st["n_ordered_diff"] == 0 and st["n_seq"] == 128)
            members[f"{oname(s)}.{L}"] = {f"{k[0]}.{k[1]}.{k[2]}.b{b}": sorted(v) for (k, b), v in rec["members"].items()}
            contrib[f"{oname(s)}.{L}"] = {f"{k[0]}.{k[1]}.{k[2]}.s{sid}": v for (k, sid), v in rec["contrib"].items()}
            f = files[oname(s)]
            for e in range(MU.num_experts(model)):
                for pj in ("gu", "down"):
                    key = (L, e, pj)
                    sk = f"{L}.{e}.{pj}"
                    H = prof.acc.H_sum(key, list(range(K)))
                    f["sha"][sk] = FRT.tensor_sha(H)
                    f["diag"][sk] = torch.diag(H).cpu().clone()
                    f["tokens"][sk] = prof.acc.tokens(key)
                    f["block_tokens"][sk] = prof.acc.raw_n[key].tolist()
                    pk = "gate_proj" if pj == "gu" else "down_proj"
                    m = dict(natural_vs_PO=AA.perm_compare(HN.natural_perm(H).cpu(), P[f"{L}.{e}.{pk}"]))
                    if key in DIAG_KEYS:
                        f["full"][sk] = H.cpu().clone()
                    if s is None:
                        base[key] = H
                        m.update(sha_equal_frozen_OO=f["sha"][sk] == oo["sha"][sk], nonfinite=int((~torch.isfinite(H)).sum()))
                    else:
                        m.update(HN.hdiff(H, base[key]))
                        m["block_tokens_equal"] = f["block_tokens"][sk] == files["O0"]["block_tokens"][sk]
                        if key in DIAG_KEYS:
                            m["factorization"] = HN.factorization_sensitivity(H, base[key], P[f"{L}.{e}.{pk}"], gq["percdamp"])
                        del H
                    keys_m[oname(s)][sk] = m
            prof.free()
            del prof
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[hessian] {c} L{L} {oname(s)}: {tsec[f'{oname(s)}.{L}']:.0f} s", flush=True)
        del base, cache
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(HCACHE, "hessians"), exist_ok=True)
    hsha = {}
    for s in seeds:
        hp = hess_file(c, s)
        torch.save(files[oname(s)], hp)
        hsha[oname(s)] = sha256_file(hp)
    # ---- invariance of everything except the addition order
    o0 = "O0"
    checks = dict(
        baseline_hessians_equal_frozen_OO=sum(m["sha_equal_frozen_OO"] for m in keys_m[o0].values()),
        baseline_nonfinite=sum(m["nonfinite"] for m in keys_m[o0].values()),
        routing_equal_R_O_all=all(v["equal_R_O"] for v in routing.values()),
        block_members_equal=all(members[f"{oname(s)}.{L}"] == members[f"O0.{L}"] for s in seeds for L in LAYERS),
        token_counts_equal=all(files[oname(s)]["tokens"] == files[o0]["tokens"] for s in seeds),
        block_token_counts_equal=all(files[oname(s)]["block_tokens"] == files[o0]["block_tokens"] for s in seeds),
        diag_contributions_identical=all(contrib[f"{oname(s)}.{L}"] == contrib[f"O0.{L}"] for s in seeds for L in LAYERS),
        n_diag_contributions=sum(len(contrib[f"O0.{L}"]) for L in LAYERS),
        nonfinite_any=sum(m.get("nonfinite", 0) for s in seeds for m in keys_m[oname(s)].values()),
        baseline_natural_actorder_equal_P_O=sum(m["natural_vs_PO"]["identical"] for m in keys_m[o0].values()))
    checks["passed"] = bool(checks["baseline_hessians_equal_frozen_OO"] == 480 and checks["baseline_nonfinite"] == 0
                            and checks["routing_equal_R_O_all"] and checks["block_members_equal"]
                            and checks["token_counts_equal"] and checks["block_token_counts_equal"]
                            and checks["diag_contributions_identical"] and checks["nonfinite_any"] == 0
                            and checks["baseline_natural_actorder_equal_P_O"] == 480)
    per_order = {oname(s): summarize_order(keys_m[oname(s)]) for s in HN.ORDER_SEEDS}
    out = dict(strategy=c, protocol_sha256=pr["protocol_sha256"], checks=checks, per_order=per_order, keys=keys_m,
               hessian_files={k: dict(path=hess_file(c, None if k == "O0" else int(k[1:])), sha256=v) for k, v in hsha.items()},
               members_sha256={k: HN.obj_sha(v) for k, v in members.items()},
               diag_contribution_fingerprints={k: v for k, v in contrib.items() if k.startswith("O0.")},
               replay_seconds=tsec, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FRD.FD.peak_cpu_gb())
    os.makedirs(p("hessian_preflight"), exist_ok=True)
    jdump(out, path)
    print(f"[hessian] {c}: checks passed {checks['passed']}; per order changed Hessians "
          f"{ {k: v['n_hessians_changed'] for k, v in per_order.items()} }", flush=True)
    aggregate_hessian()
    if not checks["passed"]:
        raise SystemExit(f"Hessian preflight {c}: invariance check FAILED: RUN-BLOCKED")


def summarize_order(km):
    def agg(sel):
        v = [km[k] for k in sel]
        rf = np.array([x["rel_fro"] for x in v])
        return dict(n_hessians=len(v), n_hessians_changed=sum(not x["bitwise_identical"] for x in v),
                    frac_hessians_changed=float(np.mean([not x["bitwise_identical"] for x in v])),
                    n_elements=int(sum(x["n"] for x in v)), n_elements_changed=int(sum(x["n_diff"] for x in v)),
                    frac_elements_changed=float(sum(x["n_diff"] for x in v) / sum(x["n"] for x in v)),
                    max_abs=float(max(x["max_abs"] for x in v)), mean_abs=float(np.mean([x["mean_abs"] for x in v])),
                    rel_fro_median=float(np.median(rf)), rel_fro_min=float(rf.min()), rel_fro_max=float(rf.max()),
                    rel_diag_median=float(np.median([x["rel_diag"] for x in v])),
                    trace_rel_diff_max_abs=float(max(abs(x["trace_rel_diff"]) for x in v)),
                    max_ulp=int(max(x["max_ulp"] for x in v)), nonfinite=int(sum(x["nonfinite"] for x in v)),
                    natural_actorder_identical_P_O=sum(x["natural_vs_PO"]["identical"] for x in v),
                    natural_actorder_mean_frac_positions_changed=float(np.mean([x["natural_vs_PO"]["frac_positions_changed"] for x in v])),
                    natural_actorder_mean_frac_group_changed=float(np.mean([x["natural_vs_PO"]["frac_columns_changed_group"] for x in v])),
                    block_tokens_equal=all(x["block_tokens_equal"] for x in v))
    allk = list(km)
    out = agg(allk) | dict(gu=agg([k for k in allk if k.endswith(".gu")]), down=agg([k for k in allk if k.endswith(".down")]))
    experts = {k.rsplit(".", 1)[0] for k in allk}
    out["n_experts_any_hessian_changed"] = sum(any(not km[f"{x}.{pj}"]["bitwise_identical"] for pj in ("gu", "down")) for x in experts)
    out["diag_subset"] = {k: {kk: vv for kk, vv in km[k].items()} for k in [f"{L}.{e}.{pj}" for L, e, pj in DIAG_KEYS]}
    return out


def aggregate_hessian():
    pre = {c: load_json(p("hessian_preflight", f"{c}.json")) for c in STRATS}
    if not all(pre.values()):
        return None
    any_diff = {f"{c}.{s}": pre[c]["per_order"][f"O{s}"]["n_hessians_changed"] > 0 for c in STRATS for s in HN.ORDER_SEEDS}
    ok = all(pre[c]["checks"]["passed"] for c in STRATS)
    out = dict(checks={c: pre[c]["checks"] for c in STRATS}, checks_passed=ok,
               per_order={c: {k: {kk: vv for kk, vv in v.items() if kk != "diag_subset"} for k, v in pre[c]["per_order"].items()} for c in STRATS},
               diag_subset={c: {k: v["diag_subset"] for k, v in pre[c]["per_order"].items()} for c in STRATS},
               hessians_differ=any_diff, any_hessian_differs=any(any_diff.values()),
               early_stop=(not any(any_diff.values())),
               early_stop_decision="NULL-ORDER-INVARIANT" if ok and not any(any_diff.values()) else None,
               repeat_config=HN.repeat_config({(c, s): any_diff[f"{c}.{s}"] for c in STRATS for s in HN.ORDER_SEEDS}))
    jdump(out, p("hessian_differences.json"))
    return out


# ---------------------------------------------------------------- FP64 accumulation control (diagnostic)
def stage_fp64_control(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    c = a.strategy
    path = p("fp64_control", f"{c}.json")
    if load_json(path):
        print(f"[fp64] {c} done"); return
    protocol()
    man = json.load(open(p("input_manifest.json")))
    chosen, K = man["selections"][c]["replay_order"], man["selections"][c]["K"]
    hp = {oname(s): torch.load(hess_file(c, s))["sha"] for s in (None,) + HN.FP64_SEEDS}
    tables = FRD.load_tables("O")
    model, _ = DB.load_model()
    E = MU.num_experts(model)
    d_gu, d_down = MU.expert_dims(model)
    dims = dict(gu=d_gu, down=d_down)
    seeds = [None] + list(HN.FP64_SEEDS)
    res, t0 = {}, time.time()
    torch.cuda.reset_peak_memory_stats()
    for L in LAYERS:
        cache = load_layer_cache(OCACHE, L)
        for c0 in range(0, E, FP64_CHUNK):
            keys = [(L, e, pj) for e in range(c0, min(c0 + FP64_CHUNK, E)) for pj in ("gu", "down")]
            base = {}
            for s in seeds:
                order, blocks, _ = HN.order_spec(chosen, s, K)
                prof = HN.OrderedForcedProfiler(model, L, tables[L], {x: x for x in chosen}, True, chosen, order, blocks,
                                                K=K, collect_hessian=False)
                prof.acc = HN.Fp64Accumulator(keys, dims, K, MU.GPU)
                prof.run(cache, chosen, weights=[1.0] * len(chosen))
                assert prof.n_ordered_diff == 0
                for key in keys:
                    sk = f"{key[0]}.{key[1]}.{key[2]}"
                    if prof.acc.raw_n[key].sum() == 0:
                        continue
                    H32, H64 = prof.acc.sums(key)
                    r = res.setdefault(sk, {})
                    r[f"fp32_path_equals_preflight_{oname(s)}"] = FRT.tensor_sha(H32) == hp[oname(s)][sk]
                    a64 = H64.abs()
                    # FP32 accumulation error relative to the FP64 accumulation of the same contributions
                    r[f"fp32_vs_fp64_rel_fro_{oname(s)}"] = float((H32.double() - H64).norm() / H64.norm())
                    if s is None:
                        base[key] = (H32, H64)
                    else:
                        b32, b64 = base[key]
                        d64 = H64 - b64
                        c32 = H64.float()
                        r[f"O{s}"] = dict(
                            fp64_n_diff=int((H64 != b64).sum()), fp64_frac_diff=float((H64 != b64).double().mean()),
                            fp64_rel_fro=float(d64.norm() / b64.norm()), fp64_max_abs=float(d64.abs().max()),
                            fp64_cast_fp32_n_diff=int((c32 != b64.float()).sum()),
                            fp64_cast_fp32_frac_diff=float((c32 != b64.float()).double().mean()),
                            fp32_n_diff=int((H32 != b32).sum()), fp32_frac_diff=float((H32 != b32).double().mean()),
                            fp32_rel_fro=float((H32.double() - b32.double()).norm() / b32.double().norm()),
                            fp32_max_abs=float((H32.double() - b32.double()).abs().max()), n=H64.numel(),
                            max_abs_value=float(a64.max()))
                prof.acc = None
                del prof
                gc.collect()
                torch.cuda.empty_cache()
            del base
            torch.cuda.empty_cache()
            print(f"[fp64] {c} L{L} experts {c0}-{min(c0 + FP64_CHUNK, E) - 1} ({time.time() - t0:.0f} s)", flush=True)
        del cache
    summ = {}
    for s in HN.FP64_SEEDS:
        v = [r[f"O{s}"] for r in res.values() if f"O{s}" in r]
        n = sum(x["n"] for x in v)
        summ[f"O{s}"] = dict(
            n_hessians=len(v), fp32_n_hessians_changed=sum(x["fp32_n_diff"] > 0 for x in v),
            fp64_n_hessians_changed=sum(x["fp64_n_diff"] > 0 for x in v),
            fp64_cast_n_hessians_changed=sum(x["fp64_cast_fp32_n_diff"] > 0 for x in v),
            fp32_frac_elements_changed=sum(x["fp32_n_diff"] for x in v) / n,
            fp64_frac_elements_changed=sum(x["fp64_n_diff"] for x in v) / n,
            fp64_cast_frac_elements_changed=sum(x["fp64_cast_fp32_n_diff"] for x in v) / n,
            fp32_rel_fro_median=float(np.median([x["fp32_rel_fro"] for x in v])),
            fp64_rel_fro_median=float(np.median([x["fp64_rel_fro"] for x in v])),
            fp32_rel_fro_max=float(max(x["fp32_rel_fro"] for x in v)), fp64_rel_fro_max=float(max(x["fp64_rel_fro"] for x in v)),
            ratio_median_fp64_over_fp32=float(np.median([x["fp64_rel_fro"] / x["fp32_rel_fro"] for x in v if x["fp32_rel_fro"] > 0])))
    allr = list(res.values())
    summ["fp32_path_equals_preflight_all"] = all(v for r in allr for k, v in r.items() if k.startswith("fp32_path_equals"))
    summ["fp32_accumulation_error_vs_fp64_rel_fro_median"] = float(np.median([r["fp32_vs_fp64_rel_fro_O0"] for r in allr]))
    out = dict(strategy=c, summary=summ, keys=res, seconds=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
               design="the same per-sequence contribution outer = Xf.T @ Xf (FP32 GEMM, as in BlockHessianAccumulator.add) "
                      "accumulated per original block in FP32 (H32) and FP64 (H64), blocks summed in the same precision; "
                      "only accumulation precision differs. Diagnostic only, never used for GPTQ.")
    os.makedirs(p("fp64_control"), exist_ok=True)
    jdump(out, path)
    print(f"[fp64] {c}: {summ}", flush=True)


# ---------------------------------------------------------------- one shuffled-order configuration
def stage_quant(a):
    c, s = a.strategy, a.order
    name = cname(c, s, a.repeat)
    rpath = p("results", f"{name}.json")
    if os.path.exists(rpath):
        print(f"[quant] {name} done"); return
    pr = protocol()
    hd = load_json(p("hessian_differences.json"))
    assert hd is not None and hd["checks_passed"], "Hessian preflight incomplete or failed"
    if hd["early_stop"]:
        raise SystemExit("all shuffled-order Hessians are bitwise identical: NULL-ORDER-INVARIANT, no GPTQ")
    if a.repeat:
        assert list(hd["repeat_config"]) == [c, s], f"prespecified repeatability configuration is {hd['repeat_config']}"
        assert os.path.exists(p("results", f"{cname(c, s)}.json")), "run the primary configuration first"
    hf = hess_file(c, s)
    pre = load_json(p("hessian_preflight", f"{c}.json"))
    assert sha256_file(hf) == pre["hessian_files"][oname(s)]["sha256"]
    ref_sha = torch.load(hf)["sha"]
    model, rec, q = quant_pass(c, s, ref_sha)
    evals = evaluate(model, pr["fixed"]["windows_sha256"])
    ev = evals[0]
    wd, files, qf, qsha = save_outputs(model, rec, name)
    oo_digest = json.load(open(XR.oo_result_path(c)))["installed_weights_sha256"]
    out = slim(rec) | dict(
        strategy=c, order_seed=s, name=name, repeat=a.repeat, weights_dir=wd, weight_files=files, quantizer_file=qf,
        quantizer_file_sha256=qsha, perm_equal_original=f"{sum(rec['perm_ok'])}/{len(rec['perm_ok'])}",
        hessians_equal_preflight=f"{sum(rec['hess_sha_equal'])}/{len(rec['hess_sha_equal'])}",
        natural_actorder=perm_summary(rec["natural_vs_PO"]),
        weights_equal_OO=rec["installed_weights_sha256"] == oo_digest,
        repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
        repeat_eval_per_window_identical=evals[0]["per_window_nll"] == evals[1]["per_window_nll"], **ev)
    out["valid"] = bool(base_ok(rec) and out["repeat_eval_per_window_identical"] and np.isfinite(ev["per_window_nll"]).all())
    os.makedirs(p("results"), exist_ok=True)
    jdump(out, rpath)
    print(f"[quant] {name}: PPL {ev['ppl']:.6f} (OO {OO_PPL[c]:.6f}) recon {rec['recon']:.7f} weights==OO {out['weights_equal_OO']} "
          f"perms {out['perm_equal_original']} hessians {out['hessians_equal_preflight']} integrity {rec['integrity_pass']} "
          f"valid {out['valid']} ({rec['quantize_s']:.0f} s)", flush=True)
    if not out["valid"]:
        raise SystemExit(f"{name} failed its correctness checks")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "baseline_regression", "order_manifests",
                                                       "hessian_preflight", "fp64_control", "quant", "report"])
    ap.add_argument("--strategy", choices=STRATS)
    ap.add_argument("--order", type=int, choices=list(HN.ORDER_SEEDS))
    ap.add_argument("--repeat", action="store_true")
    a = ap.parse_args(argv)
    for d in ("logs", "figures", "results"):
        os.makedirs(p(d), exist_ok=True)
    if a.stage == "report":
        import report_hessian_order_null as R
        R.main(); return
    dict(preflight=stage_preflight, baseline_regression=stage_baseline_regression, order_manifests=stage_order_manifests,
         hessian_preflight=stage_hessian_preflight, fp64_control=stage_fp64_control, quant=stage_quant)[a.stage](a)


if __name__ == "__main__":
    main()
