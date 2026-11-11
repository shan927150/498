"""Frozen-routing 2x2 factorial ablation of MoE GPTQ calibration (FR s0 / FC s0, four layers, ORIGINAL vs G20261111).

Factors of the calibration-Hessian construction (moe_calib.frozen_routing): activation source X in {O, A} and routing
source R in {O, A}. OO / AA are the natural conditions (must reproduce the frozen multigroup results bitwise);
AO (alternative activations, frozen ORIGINAL routing) and OA (original activations, ALTERNATIVE routing) are the
interventions. Inference routing, router weights, caches, GPTQ, reconstruction and PPL evaluation are unchanged.

Stages (--stage): preflight | capture | regression --strategy S --cond OO|AA | hybrid --strategy S --cond AO|OA |
                  report
Every quantized configuration runs in a fresh process from the pristine checkpoint.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB          # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402
import multigroup_robustness as MRD     # noqa: E402
from moe_calib import batch_sensitivity as BS        # noqa: E402
from moe_calib import frozen_routing as FRT          # noqa: E402

OUT = "runs_frozen_routing"
FCACHE = os.path.expanduser(os.environ.get("FROZEN_ROUTING_CACHE", "~/moe_cache_frozen_routing"))
OCACHE = MRD.OCACHE
GROUPING = "G20261111"                       # first of the prespecified multigroup seeds; fixed before any result
LAYERS = MRD.LAYERS
STRATS = MRD.STRATS
jdump, sha256_file = FD.jdump, FD.sha256_file


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ref_result(c, cond):
    """Frozen natural-condition results (runs_multigroup_robustness): OO = CTRL ORIGINAL, AA = G20261111."""
    name = f"CTRL_{c}_ORIGINAL" if cond == "OO" else f"{c}_{GROUPING}"
    path = os.path.join(ROOT, MRD.OUT, "results", f"{name}.json")
    return path, json.load(open(path))


def weights_dir(c, cond):
    if cond in FRT.NATURAL:
        return ref_result(c, cond)[1]["weights_dir"]
    return os.path.join(FCACHE, "weights", f"{c}_{cond}")


def act_source(cond, union, grouping=GROUPING):
    """(cache_dir, row_to_sid) of a condition's activation source (alternative = multigroup cache of `grouping`,
    whose row i holds union_calibration_ids[i] for every prespecified grouping)."""
    a = FRT.CONDITIONS[cond][0]
    if a == "O":
        return OCACHE, None                  # identity: ~/moe_cache row = original sequence id
    return MRD.gdir(grouping), list(union)


# ---------------------------------------------------------------- preflight + frozen protocol
def stage_preflight(a):
    cfg, sel, checks, union = MRD.selections()
    mpr = MRD.protocol()                                         # re-verifies multigroup protocol + manifests
    MRD.verify_cache(GROUPING)
    for L in LAYERS:
        assert sha256_file(os.path.join(OCACHE, f"L{L}.pt")) == mpr["references"]["original_cache_files"][str(L)]
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    assert abs(base["ppl"] - 9.237090) < 1e-6 and prot["scored_targets_total"] == 65408 and tuple(win.shape) == (128, 512)
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    refs = {}
    for c in STRATS:
        for cond in FRT.NATURAL:
            path, r = ref_result(c, cond)
            assert r["valid"] and r["rtn_fallbacks"] == 0 and r["n_projection_tensors"] == 720
            wd = r["weights_dir"]
            for L in LAYERS:
                assert os.path.exists(os.path.join(wd, f"L{L}.pt")), (wd, L)
            refs[f"{c}_{cond}"] = dict(path=os.path.relpath(path, ROOT), result_sha256=sha256_file(path), ppl=r["ppl"],
                                       mean_nll=r["mean_nll"], recon=r["recon"],
                                       installed_weights_sha256=r["installed_weights_sha256"], weights_dir=wd,
                                       weight_files={L: v["sha256"] for L, v in r["weight_files"].items()})
    assert refs["FR_OO"]["ppl"] == 9.267851290916612 and refs["FC_OO"]["ppl"] == 9.277908190612935
    gman = json.load(open(os.path.join(ROOT, MRD.OUT, "grouping_manifests", f"{GROUPING}.json")))
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    pre = dict(selections=checks, n_union=len(union), bf16_ppl=base["ppl"], environment=DB.env_record(),
               host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, gpu_free_gb=free / 1e9,
               gpu_total_gb=tot / 1e9, disk_free_tb=shutil.disk_usage(os.path.expanduser("~")).free / 1e12,
               created=time.strftime("%F %T"))
    os.makedirs(p("logs"), exist_ok=True)
    jdump(pre, p("logs", "preflight.json"))
    jdump(dict(strategies={c: dict(chosen=sel[c]["chosen"], **checks[c]) for c in STRATS},
               union_calibration_ids=union, n_union=len(union)), p("selected_sets.json"))
    body = dict(
        experiment="Frozen-routing 2x2 factorial ablation of MoE GPTQ calibration (FR s0 / FC s0, four layers)",
        question="Does freezing the expert assignments used to build calibration Hessians suppress the GPTQ weight and "
                 "PPL sensitivity caused by changing the BF16 calibration batch composition?",
        model=mpr["model"], model_snapshot=mpr["model_snapshot"], gptq=cfg["gptq"], B=128, allocation_seed=0,
        quant_layers=LAYERS, n_quantized_experts=240, n_projection_tensors=720,
        selections={c: dict(ids_hash=checks[c]["ids_hash"], K=checks[c]["K"]) for c in STRATS},
        union_calibration_ids=union, n_union=len(union),
        grouping=dict(name=GROUPING, manifest_sha256=gman["manifest_sha256"],
                      reason="first seed of the prespecified multigroup list; fixed before any factorial result",
                      cache_files=json.load(open(os.path.join(ROOT, MRD.OUT, "cache_manifest.json")))["groupings"][GROUPING]["files"]),
        original_cache_files=mpr["references"]["original_cache_files"],
        conditions={k: dict(activation=v[0], routing=v[1]) for k, v in FRT.CONDITIONS.items()},
        alignment="(original sequence id, token position): ORIGINAL cache row = sid; G20261111 cache row i = "
                  "union_calibration_ids[i]; every sequence is 512 tokens in both caches",
        routing_capture="ordered top-k of the model's own GPU softmax(float)+topk inside the batch-1 layer replay of "
                        "each cache (the E3 Profiler path); validated against the unmodified Profiler's per-sequence "
                        "routed-token counts",
        hessian_semantics="unmodified E3 Profiler + BlockHessianAccumulator: per expert UNWEIGHTED sums over routed tokens "
                          "of x x^T (gate/up input = MoE-block input) and z z^T (z = act(gate x)*up x computed by the "
                          "expert with its ORIGINAL BF16 weights), K blocks by chosen order, summed; router weights "
                          "scale only the expert OUTPUT and are not part of the Hessian",
        intervention="selected_experts replaced inside a verbatim copy of Qwen2MoeSparseMoeBlock.forward, ONLY during "
                     "the Hessian-building Profiler run; reconstruction Profiler, inference and PPL use the class "
                     "forward; the block output under forced routing is discarded (non-sequential calibration)",
        reconstruction="E3 held-out relative reconstruction on ~/moe_cache test rows 2128-2255 (eval_cache_dir), natural routing",
        evaluation=dict(windows_sha256=prot["windows_sha256"], batching=f"EP.layerwise_nll batch {EP.BATCH}",
                        bf16_ppl=base["ppl"], repeats_per_config=2),
        natural_references=refs,
        regression_requirements=["identical routed-token counts (manifest vs unmodified Profiler)",
                                 "bitwise-identical gate/up and down Hessians (forced path vs unmodified Profiler, all "
                                 "480) and identical act-order permutations",
                                 "installed effective-weight sha256 == frozen reference",
                                 "reconstruction (mean and per expert) == frozen reference",
                                 "PPL: implied by bitwise weights + the verified deterministic evaluator (not re-run)"],
        diagnostic_subset=dict(layers=list(FRT.DIAG_LAYERS), experts=list(FRT.DIAG_EXPERTS), projections=["gu", "down"],
                               note="prespecified (= the partial-PPL expert-check subset); act-order and diag "
                                    "statistics are recorded for ALL 240 experts"),
        decision_rule=FRT.DECISION_RULE, tau=FRT.TAU,
        hard_constraints="no re-selection, seed 0, grouping G20261111 only, 4 layers, frozen GPTQ / windows / "
                         "evaluation, no forced routing at inference, no additional groupings or seeds after results")
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        assert old["protocol_sha256"] == FRT.obj_sha(body), "protocol changed after freezing"
        print(f"preflight ok; protocol already frozen (sha256 {old['protocol_sha256'][:16]})")
        return
    body["protocol_sha256"] = FRT.obj_sha(body)
    body["created"] = time.strftime("%F %T")
    body["environment"] = DB.env_record()
    jdump(body, p("protocol.json"))
    print(f"preflight ok: union {len(union)}; GPU free {free / 1e9:.1f} GB; protocol sha256 {body['protocol_sha256'][:16]}")


def protocol():
    pr = json.load(open(p("protocol.json")))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created", "environment")}
    assert FRT.obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    return pr


# ---------------------------------------------------------------- routing capture
def stage_capture(a):
    from moe_calib import model_utils as MU
    from moe_calib.profiler import Profiler
    pr = protocol()
    union = pr["union_calibration_ids"]
    MRD.verify_cache(GROUPING)
    if os.path.exists(p("routing_validation.json")) and load_json(p("routing_validation.json"))["valid"]:
        print("[capture] done"); return
    model, t_load = DB.load_model()
    E, k = MU.num_experts(model), MU.top_k(model)
    srcs = {"O": (OCACHE, list(union)), "A": (MRD.gdir(GROUPING), list(range(len(union))))}
    tables = {s: {} for s in srcs}
    val, t0 = {}, time.time()
    for L in LAYERS:
        val[str(L)] = {}
        for s, (cdir, rows) in srcs.items():
            cache = torch.load(os.path.join(cdir, f"L{L}.pt"), mmap=True)
            with MU.layer_on_gpu(model, L):
                sel, prob = FRT.capture_routing(model, L, cache, rows, union)
            # independent check: the UNMODIFIED Profiler (class forward + its own gate-hook topk), routing only
            prof = Profiler(model, L, collect_hessian=False)
            prof.run(cache, rows, routing_only=True)
            ref_counts = {union[i]: {e: c for (_, e), c in prof.seq_counts[r].items()} for i, r in enumerate(rows)}
            got = FRT.expert_seq_counts(sel, E)
            mism = [sid for sid in union if got[sid] != ref_counts[sid]]
            T = int(cache.shape[1])
            tables[s][L] = dict(sel={sid: sel[sid].to(torch.int8) for sid in union},
                                prob={sid: prob[sid] for sid in union})
            val[str(L)][s] = dict(n_seq=len(union), T=T, k=int(sel[union[0]].shape[1]),
                                  counts_match_unmodified_profiler=not mism, n_mismatched_sequences=len(mism),
                                  routing_sha256=FRT.routing_sha(sel),
                                  expert_token_counts=FRT.expert_token_counts(sel, union, E).tolist())
            del cache
        cmp = BS.routing_compare(torch.cat([tables["A"][L]["sel"][sid] for sid in union]).long(),
                                 torch.cat([tables["O"][L]["sel"][sid] for sid in union]).long(), n_experts=E)
        cmp.pop("per_expert_count_abs_diff")
        val[str(L)]["A_vs_O"] = cmp
        for c in STRATS:
            ch = json.load(open(p("selected_sets.json")))["strategies"][c]["chosen"]
            val[str(L)][f"A_vs_O_{c}"] = {kk: v for kk, v in BS.routing_compare(
                torch.cat([tables["A"][L]["sel"][sid] for sid in ch]).long(),
                torch.cat([tables["O"][L]["sel"][sid] for sid in ch]).long(), n_experts=E).items()
                if kk != "per_expert_count_abs_diff"}
        print(f"[capture] L{L}: set changed {cmp['frac_any_expert_changed'] * 100:.2f}% of tokens; counts match "
              f"O {val[str(L)]['O']['counts_match_unmodified_profiler']} A {val[str(L)]['A']['counts_match_unmodified_profiler']}",
              flush=True)
    for s, name in (("O", "original"), ("A", "alternative")):
        path = p(f"routing_{name}.pt")
        torch.save({L: tables[s][L] for L in LAYERS}, path)
        jdump(dict(source=name, cache=srcs[s][0], layers=LAYERS, union_calibration_ids=union,
                   format=f"routing_{name}.pt: {{layer: {{'sel': {{sid: int8 [512, {k}] ordered top-k}}, "
                          f"'prob': {{sid: fp32 [512, {k}] softmax prob of the selected experts}}}}}}",
                   file_sha256=sha256_file(path),
                   per_layer={str(L): {kk: val[str(L)][s][kk] for kk in ("routing_sha256", "T", "k", "expert_token_counts")}
                              for L in LAYERS}), p(f"routing_{name}.json"))
    ok = all(val[str(L)][s]["counts_match_unmodified_profiler"] for L in LAYERS for s in srcs)
    jdump(dict(valid=ok, per_layer=val, capture_s=time.time() - t0, model_load_s=t_load,
               alignment=pr["alignment"]), p("routing_validation.json"))
    print(f"[capture] valid {ok} ({time.time() - t0:.0f} s)")
    if not ok:
        raise SystemExit("routing capture does not reproduce the unmodified Profiler: stop")


def load_tables(src):
    name = dict(O="original", A="alternative")[src]
    man = json.load(open(p(f"routing_{name}.json")))
    path = p(f"routing_{name}.pt")
    assert sha256_file(path) == man["file_sha256"], f"{path} modified"
    t = torch.load(path)
    return {L: {sid: s.long() for sid, s in t[L]["sel"].items()} for L in LAYERS}


# ---------------------------------------------------------------- one quantization (natural regression or hybrid)
def run_config(a, kind, grouping=GROUPING, result_path=None, wd=None, hpath=None):
    """kind 'regression' (natural OO / AA) or 'hybrid' (AO / OA). grouping / result_path / wd / hpath default to the
    factorial study (G20261111, runs_frozen_routing/results, ~/moe_cache_frozen_routing); the robustness study
    (diagnostics/frozen_routing_robustness.py) passes other prespecified groupings and its own output paths."""
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.profiler import Profiler
    from moe_calib.data_utils import load_layer_cache
    import e3_allocation as E3
    c, cond = a.strategy, a.cond
    assert (cond in FRT.NATURAL) == (kind == "regression"), (kind, cond)
    name = f"{c}_{cond}"
    result_path = result_path or p("results", f"{name}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    assert kind == "hybrid" or grouping == GROUPING, "natural regression is defined for the factorial grouping only"
    if os.path.exists(result_path):
        print(f"[{kind}] {name} done"); return
    pr = protocol()
    assert load_json(p("routing_validation.json"))["valid"], "routing capture not validated"
    if kind == "hybrid":
        for cc in STRATS:
            for nc in FRT.NATURAL:
                r = load_json(p("results", f"{cc}_{nc}.json"))
                assert r and r["regression_passed"], f"natural regression {cc}_{nc} not passed: hybrids blocked"
    cfg, sel, _, union = MRD.selections()
    assert sel[c]["ids_hash"] == pr["selections"][c]["ids_hash"]
    test_ids = list(range(*cfg["split"]["test"]))
    MRD.verify_cache(grouping)
    act, rout = FRT.CONDITIONS[cond]
    cdir, row_to_sid = act_source(cond, union, grouping)
    if row_to_sid is None:
        chosen, r2s = list(sel[c]["chosen"]), {s: s for s in union}
    else:
        chosen, r2s = MRD.MG.remap(sel[c]["chosen"], union), dict(enumerate(union))
        assert [union[i] for i in chosen] == sel[c]["chosen"]
    tables = load_tables(rout)
    ecache = None if cdir == OCACHE else OCACHE
    model, t_load = DB.load_model()
    E = MU.num_experts(model)
    pdig = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    tg0 = PQ.target_tensor_digests(model, LAYERS)
    assert tg0 == {k: pdig["pristine_targets"][k] for k in tg0}, "targets not pristine"
    u0 = PQ.untouched_digests(model, LAYERS)
    assert all(u0[kk] == pdig["pristine_untouched"][kk] for kk in u0 if kk in pdig["pristine_untouched"])
    router_dig = {kk: v for kk, v in u0.items() if ".mlp.gate." in kk}
    assert len(router_dig) == 24
    gq, K = cfg["gptq"], sel[c]["K"]
    torch.cuda.reset_peak_memory_stats()
    # ---- reference pass (natural conditions only): the UNMODIFIED Profiler on the same cache rows, pristine weights
    plain = {}
    t_plain = 0.0
    if kind == "regression":
        t0 = time.time()
        for L in LAYERS:
            cache = load_layer_cache(cdir, L)
            pf = Profiler(model, L, K=K).run(cache, chosen, weights=[1.0] * len(chosen))
            for e in range(E):
                for pj in ("gu", "down"):
                    plain[(L, e, pj)] = (FRT.tensor_sha(pf.acc.H_sum((L, e, pj), list(range(K)))),
                                         pf.acc.tokens((L, e, pj)))
            pf.free(); del pf, cache
        t_plain = time.time() - t0
    # ---- forced-path quantization (E3 quantize_and_score, unchanged except the Hessian Profiler's routing source)
    hess, hfull = {}, {}

    def on_hessian(L, e, acc):
        for pj in ("gu", "down"):
            H = acc.H_sum((L, e, pj), list(range(acc.K)))
            perm = FRT.actorder_perm(H)
            hess[(L, e, pj)] = dict(sha=FRT.tensor_sha(H), tokens=acc.tokens((L, e, pj)),
                                    diag=torch.diag(H).cpu().clone(), perm=perm.cpu().to(torch.int16))
            if (L, e, pj) in FRT.DIAG_KEYS:
                hfull[f"{L}.{e}.{pj}"] = H.cpu().clone()
    orig = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()}
            for L, e in EP.CHECK_EXPERTS}
    t1 = time.time()
    with FRT.forced_e3(E3, tables, r2s, expect_natural=(kind == "regression"), on_hessian=on_hessian) as fstats:
        q = PQ.quantize_install(model, cdir, LAYERS, chosen, test_ids, K, gq["bits"], gq["group_size"], gq["percdamp"],
                                capture=EP.CHECK_EXPERTS, eval_cache_dir=ecache)
    t_q = time.time() - t1
    FRT.assert_no_override(model)
    routing_stats = {str(L): fstats[L].stats() for L in LAYERS}
    # forced-path token counts must equal the routing table's counts over the chosen sequences
    counts_ok = all(hess[(L, e, "gu")]["tokens"] == float(FRT.expert_token_counts(tables[L], sel[c]["chosen"], E)[e])
                    for L in LAYERS for e in range(E))
    after = PQ.target_tensor_digests(model, LAYERS)
    unchanged = [kk for kk in after if after[kk] == tg0[kk]]
    nonfin = PQ.nonfinite_targets(model, LAYERS)
    u1 = PQ.untouched_digests(model, LAYERS)
    untouched_ok = u1 == u0
    router_ok = {kk: u1[kk] for kk in router_dig} == router_dig
    dig = PQ.installed_digest(model, LAYERS)
    recon = float(np.mean([r["mean"] for r in q["results"]]))
    per_expert = [dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"]) for r in q["results"]]
    checks = []
    for L, e in EP.CHECK_EXPERTS:
        Xc = PQ.expert_inputs(model, L, e, load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]])
        ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], orig[(L, e)], Xc)
        ck["tolerance"] = EP.INSTALL_TOL(ck)
        ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                          and ck["installed_vs_e3_rel"] <= ck["tolerance"])
        checks.append(ck)
    hpath = hpath or os.path.join(FCACHE, "hessians", f"{name}.pt")
    os.makedirs(os.path.dirname(hpath), exist_ok=True)
    torch.save(dict(full=hfull, diag={f"{L}.{e}.{pj}": v["diag"] for (L, e, pj), v in hess.items()},
                    perm={f"{L}.{e}.{pj}": v["perm"] for (L, e, pj), v in hess.items()},
                    tokens={f"{L}.{e}.{pj}": v["tokens"] for (L, e, pj), v in hess.items()},
                    sha={f"{L}.{e}.{pj}": v["sha"] for (L, e, pj), v in hess.items()}), hpath)
    base_ok = (q["counts"]["n_zero_token_quantization_fallback"] == 0 and q["counts"]["n_experts_quantized_total"] == 240
               and len(after) == 720 and not unchanged and not nonfin and untouched_ok and router_ok and counts_ok
               and all(ck["pass"] for ck in checks))
    out = dict(strategy=c, condition=cond, kind=kind, grouping=grouping, activation_source=act, routing_source=rout, name=name,
               ids_hash=sel[c]["ids_hash"], activation_cache=cdir, reconstruction_cache=OCACHE, gptq=gq, K=K,
               layers=LAYERS, n_quantized_experts=q["counts"]["n_experts_quantized_total"],
               n_projection_tensors=len(after), rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"],
               n_experts_scored=q["counts"]["n_experts_test_observed"], recon=recon,
               recon_per_layer={str(L): float(np.mean([r["mean"] for r in q["results"] if r["layer"] == L])) for L in LAYERS},
               per_expert=per_expert, projections_unchanged_from_bf16=len(unchanged), nonfinite_weights=len(nonfin),
               untouched_params_unchanged=untouched_ok, router_weights_unchanged=router_ok,
               forced_token_counts_match_table=counts_ok, forced_routing_stats=routing_stats, expert_checks=checks,
               installed_weights_sha256=dig, hessian_file=hpath, hessian_file_sha256=sha256_file(hpath),
               model_load_s=t_load, reference_profiler_s=t_plain, quantize_s=t_q, hessian_s=q["hessian_s"],
               gptq_s=q["gptq_s"], peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb())
    if kind == "regression":
        rpath, ref = ref_result(c, cond)
        h_eq = sum(hess[k]["sha"] == plain[k][0] for k in plain)
        tok_eq = all(hess[k]["tokens"] == plain[k][1] for k in plain)
        ref_pe = [(r["layer"], r["expert"], r["mean"], r["tokens"]) for r in ref["per_expert"]]
        reg = dict(reference=os.path.relpath(rpath, ROOT),
                   routing_identical_to_natural=all(v["n_ordered_diff"] == 0 for v in routing_stats.values()),
                   token_counts_identical=tok_eq,
                   hessians_bitwise_identical=f"{h_eq}/{len(plain)}",
                   actorder_identical="implied by bitwise Hessians" if h_eq == len(plain) else "not verified",
                   weights_exact=dig == ref["installed_weights_sha256"],
                   recon_exact=recon == ref["recon"],
                   per_expert_recon_exact=[(d["layer"], d["expert"], d["mean"], d["tokens"]) for d in per_expert] == ref_pe,
                   reference_ppl=ref["ppl"], reference_mean_nll=ref["mean_nll"],
                   ppl_note="weights bitwise identical to the frozen configuration and the evaluator is deterministic "
                            "(repeat |dPPL| = 0 in every earlier run): PPL identical by construction, not re-evaluated")
        reg_ok = (reg["routing_identical_to_natural"] and tok_eq and h_eq == len(plain) and reg["weights_exact"]
                  and reg["recon_exact"] and reg["per_expert_recon_exact"])
        out.update(regression=reg, regression_passed=bool(reg_ok and base_ok), valid=bool(base_ok),
                   ppl=ref["ppl"], mean_nll=ref["mean_nll"], total_nll=ref["total_nll"], total_tokens=ref["total_tokens"],
                   per_window_nll=ref["per_window_nll"], per_window_tokens=ref["per_window_tokens"],
                   weights_dir=ref["weights_dir"])
        jdump(out, result_path)
        print(f"[regression] {name}: hessians {reg['hessians_bitwise_identical']} weights {reg['weights_exact']} "
              f"recon {reg['recon_exact']} routing {reg['routing_identical_to_natural']} -> passed "
              f"{out['regression_passed']} ({t_q:.0f} s)", flush=True)
        if not out["regression_passed"]:
            raise SystemExit(f"natural regression {name} FAILED: hybrids must not be interpreted")
        return
    # ---- hybrid: save weights, evaluate PPL with the unmodified model forward (twice)
    wd = wd or weights_dir(c, cond)
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
    rec = evals[0]
    ok = bool(base_ok and np.isfinite(rec["per_window_nll"]).all() and evals[0]["ppl"] == evals[1]["ppl"]
              and routing_stats[str(LAYERS[0])]["n_tokens"] > 0)
    out.update(weight_files=files, weights_dir=wd, repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
               repeat_eval_per_window_identical=evals[0]["per_window_nll"] == evals[1]["per_window_nll"],
               valid=ok, **rec)
    jdump(out, result_path)
    print(f"[hybrid] {name}: PPL {rec['ppl']:.6f} recon {recon:.7f} sha {dig[:16]} forced-set-diff "
          + " ".join(f"L{L} {routing_stats[str(L)]['frac_set_diff'] * 100:.2f}%" for L in LAYERS)
          + f" valid {ok} ({t_q:.0f} s)", flush=True)
    if not ok:
        raise SystemExit(f"{name} failed its correctness checks")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "capture", "regression", "hybrid", "report"])
    ap.add_argument("--strategy", choices=list(STRATS))
    ap.add_argument("--cond", choices=list(FRT.CONDITIONS))
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "report":
        import report_frozen_routing as R
        R.main(); return
    if a.stage in ("regression", "hybrid"):
        return run_config(a, a.stage)
    dict(preflight=stage_preflight, capture=stage_capture)[a.stage](a)


if __name__ == "__main__":
    main()
