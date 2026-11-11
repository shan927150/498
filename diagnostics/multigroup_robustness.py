"""Multi-grouping robustness of the FR s0 vs FC s0 four-layer GPTQ PPL comparison.

Only ONE thing varies: the BF16 forward-batch grouping used to build the calibration layer-input cache.
Five prespecified groupings (moe_calib.multigroup_robustness.GROUPING_SEEDS) of ONLY the 217 calibration
rows (sorted FR s0 u FC s0), 27 x 8 + 1; the same grouping is used for FR and FC (paired design).
Frozen: selections, B=128 seed 0, GPTQ (4-bit, g128, percdamp 1.0, act-order), layers 1/8/16/23, the
reconstruction rows (always ~/moe_cache test rows 2128-2255 via E3's eval_cache_dir), WikiText-2 windows
and evaluation batching. ORIGINAL / COMPACT / SHUFFLED (runs_batch_sensitivity) are historical references.

Stages (--stage): preflight | manifest | cache | diagnostics | quant --strategy FR|FC --grouping G|ORIGINAL |
                  report
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
from moe_calib import batch_sensitivity as BS        # noqa: E402
from moe_calib import multigroup_robustness as MG    # noqa: E402

OUT = "runs_multigroup_robustness"
OCACHE = os.path.expanduser("~/moe_cache")
MCACHE = os.path.expanduser(os.environ.get("MULTIGROUP_CACHE", "~/moe_cache_multigroup_robustness"))
BSW = os.path.expanduser("~/moe_cache_batch_sensitivity/weights")       # ORIGINAL weights (read only)
LAYERS = [1, 8, 16, 23]
STRATS = DB.STRATS
GROUPS = [MG.gname(s) for s in MG.GROUPING_SEEDS]
N_LAYERS = 24
jdump, sha256_file, tensor_sha = FD.jdump, FD.sha256_file, DB.tensor_sha


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def gdir(g):
    return os.path.join(MCACHE, g)


def wdir(name):
    return os.path.join(MCACHE, "weights", name)


def rname(c, g):
    return f"{c}_{g}" if g != "ORIGINAL" else f"CTRL_{c}_ORIGINAL"


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def selections():
    """FD.load_selections re-verifies source file hashes, ids and warm-start hashes against the partial-PPL manifest."""
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    man, sel = FD.load_selections()
    pil = json.load(open(os.path.join(ROOT, FD.OUT, "selected_sets.json")))
    bsp = json.load(open(os.path.join(ROOT, DB.OUT, "protocol.json")))
    warm = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))["seeds"]["0"]["ids"]
    banned = set(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))
    checks = {}
    for c in STRATS:
        ch = sel[c]["chosen"]
        checks[c] = dict(n=len(ch), unique=len(set(ch)), ids_hash=sel[c]["ids_hash"], K=sel[c]["K"],
                         source=sel[c]["source"], source_sha256=sel[c]["source_sha256"],
                         ids_hash_matches_batch_sensitivity=sel[c]["ids_hash"] == bsp["selections"][c]["ids_hash"],
                         chosen_matches_pilot=ch == pil[c]["chosen"], ordered_warm_start_is_frozen=ch[:32] == warm,
                         all_in_pool=all(0 <= x < cfg["split"]["pool"][1] for x in ch),
                         no_tune_or_test_leakage=not (set(ch) & banned))
        assert checks[c]["n"] == 128 == checks[c]["unique"] and all(
            v for v in checks[c].values() if isinstance(v, bool)), checks[c]
    union = sorted(set(sel["FR"]["chosen"]) | set(sel["FC"]["chosen"]))
    assert len(union) == MG.N_UNION and union == bsp["union_calibration_ids"]
    return cfg, sel, checks, union


# ---------------------------------------------------------------- preflight
def stage_preflight(a):
    cfg, sel, checks, union = selections()
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    assert abs(base["ppl"] - 9.237090) < 1e-6 and prot["scored_targets_total"] == 65408 and tuple(win.shape) == (128, 512)
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    out = dict(selections=checks, n_union=len(union), bf16_ppl=base["ppl"], windows_sha256=prot["windows_sha256"],
               environment=DB.env_record(), host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6,
               gpu_free_gb=free / 1e9, gpu_total_gb=tot / 1e9, cuda_available=torch.cuda.is_available(),
               disk_free_tb=shutil.disk_usage(os.path.expanduser("~")).free / 1e12, created=time.strftime("%F %T"))
    jdump(out, p("logs", "preflight.json"))
    print(f"preflight ok: union {len(union)}; GPU {out['environment']['gpu']} free {free / 1e9:.1f} GB; "
          f"host avail {out['host_mem_available_gb']:.0f} GB")


# ---------------------------------------------------------------- manifest + frozen protocol
def stage_manifest(a):
    cfg, sel, checks, union = selections()
    test_ids = list(range(*cfg["split"]["test"]))
    forbidden = list(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))      # tune + test rows
    os.makedirs(p("grouping_manifests"), exist_ok=True)
    mans = {}
    for s in MG.GROUPING_SEEDS:
        bt = MG.grouping_batches(union, s)
        v = MG.validate_grouping(bt, union, forbidden)
        assert all(v.values()), (s, v)
        m = MG.manifest(s, bt, union, {c: sel[c]["chosen"] for c in STRATS})
        m["validation"] = v
        path = p("grouping_manifests", f"{MG.gname(s)}.json")
        old = load_json(path)
        if old is not None:
            assert old["manifest_sha256"] == m["manifest_sha256"], f"{path} changed: frozen manifests are never rewritten"
        else:
            jdump(m, path)
        mans[MG.gname(s)] = m
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    pilot = json.load(open(os.path.join(ROOT, FD.OUT, "protocol.json")))
    ref = {c: json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{c}_s0.json"))) for c in STRATS}
    hist = json.load(open(os.path.join(ROOT, DB.OUT, "results.json")))
    body = dict(
        experiment="Multi-grouping robustness of MoE GPTQ calibration (FR s0 vs FC s0, four layers)",
        question="How stable is the FR vs FC PPL comparison under different valid BF16 calibration batch groupings, "
                 "with selections and GPTQ fixed?",
        model=pilot["model"], model_snapshot=pilot["model_snapshot"], tokenizer=prot.get("tokenizer"),
        gptq=cfg["gptq"], B=128, allocation_seed=0, quant_layers=LAYERS, n_quantized_experts=240,
        n_projection_tensors=720, projections=["gate_proj", "up_proj", "down_proj"],
        precision="BF16 model and activations; fake-quantized BF16 effective weights; everything else BF16",
        calibration="non-sequential: Hessians of layer L from the grouping's cached PRISTINE-BF16 inputs of layer L "
                    "(E3 Profiler, one sequence at a time, chosen ids in their frozen order)",
        reconstruction="E3 held-out relative expert reconstruction on ~/moe_cache test rows 2128-2255 (eval_cache_dir) "
                       "for every configuration; never the grouping cache",
        evaluation=dict(windows="runs_partial_ppl/windows.pt (128 x 512, 65,408 targets)", windows_sha256=prot["windows_sha256"],
                        batching=f"EP.layerwise_nll batch {EP.BATCH}", loss="FP32 logits -> FP64 token-weighted NLL",
                        bf16_ppl=base["ppl"], repeats_per_config=2),
        selections=checks, union_calibration_ids=union, n_union=len(union),
        groupings={g: dict(seed=m["seed"], manifest_sha256=m["manifest_sha256"], batches=m["batches"],
                           batch_size_pattern=m["batch_size_pattern"]) for g, m in mans.items()},
        grouping_policy="calibration-only: sorted 217-row union, default_rng(seed).permutation, consecutive batches of 8 "
                        "(27 x 8 + 1); NO evaluation / held-out rows; same grouping for FR and FC",
        cache_row_semantics="row i of every grouping cache = original sequence id union_calibration_ids[i]",
        historical_references=dict(
            policy="ORIGINAL: cache_inputs.py batches (176 x 8 consecutive original rows); COMPACT / SHUFFLED: 345 rows "
                   "(217 union + 128 old-test rows), 43 x 8 + 1 -- a DIFFERENT batch-generation policy; context only, "
                   "never pooled with the 5 new groupings",
            ppl={k: v["ppl"] for k, v in hist.items()}),
        determinism_controls=dict(
            original_reproduction="fresh-process FR / FC quantization from the ORIGINAL cache (~/moe_cache) must "
                                  "reproduce the partial-PPL installed-weight sha256, PPL and reconstruction bitwise "
                                  "(also = repeated quantization of an identical cache)",
            repeat_ppl="every configuration evaluated twice in-process: |dPPL| must be 0",
            execution_order="every grouping's batches also executed in reversed order: the reassembled layer inputs "
                            "must be bitwise identical at layers 1/8/16/23"),
        references=dict(partial_ppl={c: dict(ppl=ref[c]["ppl"], installed_weights_sha256=ref[c]["installed_weights_sha256"],
                                             recon=ref[c]["recon_recomputed"]) for c in STRATS},
                        delta_original=MG.DELTA_ORIGINAL,
                        original_cache_files={str(L): sha256_file(os.path.join(OCACHE, f"L{L}.pt")) for L in LAYERS},
                        original_ids_sha256=sha256_file(os.path.join(OCACHE, "ids.pt"))),
        diagnostics=dict(layers=LAYERS, rows="the 217 calibration rows", reference="ORIGINAL (~/moe_cache rows)",
                         metrics="relative L2, % bitwise-equal elements, p50/p99 token rel. L2, fraction of tokens with "
                                 "changed top-4 SET, routing-distribution L1; routing by GPU softmax+topk of the "
                                 "Profiler-path router logits (one sequence at a time)"),
        bootstrap=dict(n_boot=FD.N_BOOT, seed=FD.BOOT_SEED, method="FD.paired_bootstrap (joint window resampling)",
                       note=MG.BOOTSTRAP_NOTE),
        decision_rule=MG.DECISION_RULE,
        hard_constraints="no re-selection, B=128 seed 0, 4 layers, frozen GPTQ / precision / windows / evaluation batching, "
                         "exactly these 5 grouping seeds, no test rows in groupings, no extension after results")
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        assert old["protocol_sha256"] == MG.obj_sha(body), "protocol changed after freezing"
        print(f"protocol already frozen; consistent (sha256 {old['protocol_sha256'][:16]})"); return
    body["protocol_sha256"] = MG.obj_sha(body)
    body["created"] = time.strftime("%F %T")
    body["environment"] = DB.env_record()
    jdump(body, p("protocol.json"))
    print(f"manifest ok: 5 groupings {[m['batch_size_pattern'][-1] for m in mans.values()]}; protocol sha256 "
          f"{body['protocol_sha256'][:16]}")


def protocol():
    pr = json.load(open(p("protocol.json")))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created", "environment")}
    assert MG.obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    for g in GROUPS:
        m = json.load(open(p("grouping_manifests", f"{g}.json")))
        assert MG.obj_sha(MG.strip_sha({k: v for k, v in m.items() if k != "validation"})) == m["manifest_sha256"] \
            == pr["groupings"][g]["manifest_sha256"], f"manifest {g} modified"
    return pr


# ---------------------------------------------------------------- cache
def stage_cache(a):
    pr = protocol()
    union = pr["union_calibration_ids"]
    out_path = p("cache_manifest.json")
    man = load_json(out_path) or dict(groupings={}, cache_root=MCACHE)
    alltok = DB.all_tokens()
    model, t_load = DB.load_model()
    if "model" not in man:
        man["model"] = dict(pristine_all_parameters=DB.verify_pristine_full(model), config=DB.model_config_record(model),
                            model_snapshot=pr["model_snapshot"], environment=DB.env_record(), load_s=t_load)
        man["token_sha256_union"] = tensor_sha(alltok[union])
        jdump(man, out_path)
    assert man["model"]["pristine_all_parameters"], "model weights differ from the checkpoint state"
    assert not man["model"]["config"]["training"] and not man["model"]["config"]["dropout_modules_p_gt_0"]
    for g in GROUPS:
        if man["groupings"].get(g, {}).get("complete"):
            continue
        bt = pr["groupings"][g]["batches"]
        d = gdir(g); os.makedirs(d, exist_ok=True)
        files, keep = {}, {}

        def save(L, X):
            if L not in LAYERS:
                return
            path = os.path.join(d, f"L{L}.pt")
            torch.save(X.clone(), path + ".tmp"); os.replace(path + ".tmp", path)
            keep[L] = X.clone()
            files[str(L)] = dict(shape=list(X.shape), dtype=str(X.dtype), sha256=sha256_file(path),
                                 tensor_sha256=tensor_sha(X), bytes=os.path.getsize(path),
                                 nan=int(torch.isnan(X.float()).sum()), inf=int(torch.isinf(X.float()).sum()))
        t1 = time.time()
        tsha = DB.forward_plan(model, alltok, bt, union, save)
        t_build = time.time() - t1
        neg = {}

        def cmp(L, X):
            if L in LAYERS:
                neg[str(L)] = dict(bitwise_equal=bool(torch.equal(X, keep[L])), max_abs=float((X.float() - keep[L].float()).abs().max()))
        t2 = time.time()
        tsha_r = DB.forward_plan(model, alltok, bt[::-1], union, cmp)                 # execution order only
        ent = dict(seed=pr["groupings"][g]["seed"], manifest_sha256=pr["groupings"][g]["manifest_sha256"],
                   cache_dir=d, layers=LAYERS, sequence_ids=union, token_sha256=tsha, files=files,
                   storage_bytes=sum(f["bytes"] for f in files.values()), generation_s=t_build,
                   negative_control=dict(plan="same batches, reversed execution order", token_sha256=tsha_r,
                                         per_layer=neg, passed=all(x["bitwise_equal"] for x in neg.values()) and len(neg) == 4,
                                         seconds=time.time() - t2),
                   token_ids_identical=tsha == man["token_sha256_union"] == tsha_r, complete=True)
        man["groupings"][g] = ent
        jdump(man, out_path)
        print(f"[cache] {g}: {t_build:.0f} s; reversed-order control bitwise {ent['negative_control']['passed']}; "
              f"tokens identical {ent['token_ids_identical']}", flush=True)
        if not ent["negative_control"]["passed"] or not ent["token_ids_identical"]:
            raise SystemExit(f"{g}: determinism / token control FAILED; investigate before proceeding")
        del keep
    man["semantics"] = dict(
        token_ids_identical=all(e["token_ids_identical"] for e in man["groupings"].values()),
        execution_order_controls_passed=all(e["negative_control"]["passed"] for e in man["groupings"].values()),
        position_ids_and_mask_identical="by construction: one set of batch-1 decoder-layer kwargs (causal mask, "
                                        "positions, rotary) broadcast over every batch (model.config.layer_kwargs)",
        no_padding_no_cross_sequence_attention="every row is a full 512-token chunk; per-sequence causal attention "
                                               "(FP64 CPU test: every plan equal to 1e-12)",
        no_token_dropping="Qwen2-MoE routing has no capacity limit", same_model_weights=man["model"]["pristine_all_parameters"],
        dtype=man["model"]["config"]["dtype"], attn_implementation=man["model"]["config"]["attn_implementation"],
        eval_mode=not man["model"]["config"]["training"])
    man["valid"] = bool(man["semantics"]["token_ids_identical"] and man["semantics"]["execution_order_controls_passed"]
                        and man["semantics"]["same_model_weights"] and man["semantics"]["eval_mode"]
                        and all(f["nan"] == 0 == f["inf"] for e in man["groupings"].values() for f in e["files"].values()))
    man["storage_bytes_total"] = sum(e["storage_bytes"] for e in man["groupings"].values())
    jdump(man, out_path)
    print(f"[cache] valid {man['valid']}; {man['storage_bytes_total'] / 1e9:.1f} GB")


def verify_cache(g):
    man = json.load(open(p("cache_manifest.json")))
    assert man["valid"]
    for L in LAYERS:
        assert sha256_file(os.path.join(gdir(g), f"L{L}.pt")) == man["groupings"][g]["files"][str(L)]["sha256"], (g, L)


# ---------------------------------------------------------------- activation / routing diagnostics
def stage_diagnostics(a):
    from moe_calib import model_utils as MU
    pr = protocol()
    union = pr["union_calibration_ids"]
    for g in GROUPS:
        verify_cache(g)
    chosen = FD.load_selections()[1]
    sub = {c: [i for i, r in enumerate(union) if r in set(chosen[c]["chosen"])] for c in STRATS}   # union-row index
    model, t_load = DB.load_model()
    E, k = MU.num_experts(model), MU.top_k(model)
    out, t0 = {}, time.time()
    for L in LAYERS:
        R = torch.load(os.path.join(OCACHE, f"L{L}.pt"), mmap=True)[union].clone()
        X = {"ORIGINAL": R, **{g: torch.load(os.path.join(gdir(g), f"L{L}.pt")) for g in GROUPS}}
        T = R.shape[1]
        with MU.layer_on_gpu(model, L):
            lg = {v: DB.replay(model, L, x, False)[0] for v, x in X.items()}
        sels = {}
        for v in X:
            sels[v] = torch.cat([BS.topk_route(lg[v][i].to(MU.GPU), k)[1].cpu() for i in range(lg[v].shape[0])])
        res = {}
        for g in GROUPS:
            dr = BS.activation_drift(X[g], R)
            rc = BS.routing_compare(sels[g], sels["ORIGINAL"], n_experts=E)
            per_c = {}
            for c, idx in sub.items():
                ti = (torch.tensor(idx)[:, None] * T + torch.arange(T)[None]).flatten()
                per_c[c] = BS.routing_compare(sels[g][ti], sels["ORIGINAL"][ti], n_experts=E)["frac_any_expert_changed"]
            res[g] = dict(relative_l2=dr["relative_l2"], pct_bitwise_equal=100 * dr["frac_elements_exactly_equal"],
                          token_rel_l2_p50=dr["token_rel_l2_p50"], token_rel_l2_p99=dr["token_rel_l2_p99"],
                          max_abs=dr["max_abs"], nan=dr["nan_count"], inf=dr["inf_count"],
                          frac_tokens_top4_set_changed=rc["frac_any_expert_changed"],
                          frac_tokens_reordered_same_set=rc["frac_reordered_same_set"],
                          mean_topk_overlap=rc["mean_topk_overlap"], routing_distribution_l1=rc["routing_distribution_l1"],
                          frac_set_changed_by_strategy_rows=per_c, n_tokens=rc["n_tokens"])
        # between-grouping spread (are new groupings as far from each other as from ORIGINAL?)
        g0 = GROUPS[0]
        res["pairwise_vs_" + g0] = {g: dict(relative_l2=BS.activation_drift(X[g], X[g0])["relative_l2"],
                                            frac_tokens_top4_set_changed=BS.routing_compare(sels[g], sels[g0], n_experts=E)
                                            ["frac_any_expert_changed"]) for g in GROUPS[1:]}
        out[str(L)] = res
        del X, lg, sels, R
        print(f"[diag] L{L}: rel L2 " + " ".join(f"{res[g]['relative_l2']:.2e}" for g in GROUPS) + "; set changed " +
              " ".join(f"{res[g]['frac_tokens_top4_set_changed']:.4f}" for g in GROUPS) + f" ({time.time() - t0:.0f} s)",
              flush=True)
    jdump(dict(reference="ORIGINAL = ~/moe_cache rows of the 217 calibration ids", rows=len(union),
               routing="GPU softmax(float)+topk(4) of Profiler-path router logits, one sequence at a time",
               per_layer=out, seconds=time.time() - t0, model_load_s=t_load), p("activation_diagnostics.json"))


# ---------------------------------------------------------------- quantization (one config per fresh process)
def stage_quant(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    c, g = a.strategy, a.grouping
    name = rname(c, g)
    os.makedirs(p("results"), exist_ok=True)
    if os.path.exists(p("results", f"{name}.json")):
        print(f"[quant] {name} done"); return
    pr = protocol()
    cfg, sel, _, union = selections()
    test_ids = list(range(*cfg["split"]["test"]))
    if g == "ORIGINAL":
        for L in LAYERS:
            assert sha256_file(os.path.join(OCACHE, f"L{L}.pt")) == pr["references"]["original_cache_files"][str(L)]
        cdir, chosen, ecache = OCACHE, sel[c]["chosen"], None
    else:
        ctl = [load_json(p("results", f"{rname(s, 'ORIGINAL')}.json")) for s in STRATS]
        assert all(r and r["determinism"]["original_reproduced_bitwise"] for r in ctl), "ORIGINAL controls not passed"
        verify_cache(g)
        cdir, chosen, ecache = gdir(g), MG.remap(sel[c]["chosen"], union), OCACHE
        assert [union[i] for i in chosen] == sel[c]["chosen"]
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == pr["evaluation"]["windows_sha256"]
    model, t_load = DB.load_model()
    pdig = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    t0 = time.time()
    tg0 = PQ.target_tensor_digests(model, LAYERS)
    assert tg0 == {k: pdig["pristine_targets"][k] for k in tg0}, "targets not pristine"
    u0 = PQ.untouched_digests(model, LAYERS)
    pu = pdig["pristine_untouched"]
    n_untouched_verified = sum(1 for kk in u0 if kk in pu)
    assert all(u0[kk] == pu[kk] for kk in u0 if kk in pu) and n_untouched_verified > 0, "untouched parameters not pristine"
    t_verify = time.time() - t0
    orig = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()}
            for L, e in EP.CHECK_EXPERTS}
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    gq = cfg["gptq"]
    q = PQ.quantize_install(model, cdir, LAYERS, chosen, test_ids, sel[c]["K"], gq["bits"], gq["group_size"],
                            gq["percdamp"], capture=EP.CHECK_EXPERTS, eval_cache_dir=ecache)
    t_q = time.time() - t1
    checks = []
    for L, e in EP.CHECK_EXPERTS:
        Xc = PQ.expert_inputs(model, L, e, load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]])
        ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], orig[(L, e)], Xc)
        ck["tolerance"] = EP.INSTALL_TOL(ck)
        ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                          and ck["installed_vs_e3_rel"] <= ck["tolerance"])
        checks.append(ck)
    after = PQ.target_tensor_digests(model, LAYERS)
    unchanged = [kk for kk in after if after[kk] == tg0[kk]]
    nonfin = PQ.nonfinite_targets(model, LAYERS)
    untouched_ok = PQ.untouched_digests(model, LAYERS) == u0
    dig = PQ.installed_digest(model, LAYERS)
    os.makedirs(wdir(name), exist_ok=True)
    files, wcmp = {}, {}
    for L in LAYERS:
        path = os.path.join(wdir(name), f"L{L}.pt")
        PQ.save_layer(model, L, path)
        files[str(L)] = dict(sha256=sha256_file(path), layer_sha256=PQ.layer_digest(model, L))
        ref_path = os.path.join(BSW, f"{c}_ORIGINAL", f"L{L}.pt")              # weight change vs ORIGINAL
        if g != "ORIGINAL" and os.path.exists(ref_path):
            Wo, Wv = torch.load(ref_path), torch.load(path)
            n = dif = 0; d2 = o2 = 0.0
            for kk in Wo:
                o, w = Wo[kk].float(), Wv[kk].float()
                n += o.numel(); dif += int((o != w).sum()); d2 += float(((w - o) ** 2).sum()); o2 += float((o ** 2).sum())
            wcmp[str(L)] = dict(frac_differ=dif / n, rel_l2=(d2 / o2) ** .5, n=n, d2=d2, o2=o2)
    evals = []
    for _ in range(2):
        t2 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        evals.append(EP.ppl_record(nll, cnt, time.time() - t2))
    rec = evals[0]
    per_layer = {str(L): float(np.mean([r["mean"] for r in q["results"] if r["layer"] == L])) for L in LAYERS}
    ok = (q["counts"]["n_zero_token_quantization_fallback"] == 0 and q["counts"]["n_experts_quantized_total"] == 240
          and len(after) == 720 and all(ck["pass"] for ck in checks) and not unchanged and not nonfin and untouched_ok
          and bool(np.isfinite(rec["per_window_nll"]).all()) and evals[0]["ppl"] == evals[1]["ppl"])
    out = dict(strategy=c, grouping=g, name=name, ids_hash=sel[c]["ids_hash"], calibration_cache=cdir,
               reconstruction_cache=ecache or OCACHE, gptq=gq, layers=LAYERS,
               n_quantized_experts=q["counts"]["n_experts_quantized_total"], n_projection_tensors=len(after),
               rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"],
               n_experts_scored=q["counts"]["n_experts_test_observed"],
               recon=float(np.mean([r["mean"] for r in q["results"]])), recon_per_layer=per_layer,
               per_expert=[dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"]) for r in q["results"]],
               projections_unchanged_from_bf16=len(unchanged), nonfinite_weights=len(nonfin),
               untouched_params_unchanged=untouched_ok, n_untouched_params=len(u0),
               n_untouched_verified_vs_checkpoint=n_untouched_verified, expert_checks=checks,
               installed_weights_sha256=dig, weight_files=files, weights_dir=wdir(name),
               weight_change_vs_original=wcmp, repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
               repeat_eval_per_window_identical=evals[0]["per_window_nll"] == evals[1]["per_window_nll"],
               model_load_s=t_load, pristine_verify_s=t_verify, quantize_s=t_q, hessian_s=q["hessian_s"],
               gptq_s=q["gptq_s"], peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb(),
               valid=bool(ok), **rec)
    if g == "ORIGINAL":
        ref = pr["references"]["partial_ppl"][c]
        out["determinism"] = dict(weights_exact=dig == ref["installed_weights_sha256"], ppl_exact=rec["ppl"] == ref["ppl"],
                                  recon_exact=out["recon"] == ref["recon"], reference=ref)
        out["determinism"]["original_reproduced_bitwise"] = all(out["determinism"][k] for k in
                                                                 ("weights_exact", "ppl_exact", "recon_exact"))
    jdump(out, p("results", f"{name}.json"))
    print(f"[quant] {name}: PPL {rec['ppl']:.6f} recon {out['recon']:.7f} sha {dig[:16]} valid {ok} ({t_q:.0f} s)"
          + (f" reproduced {out['determinism']['original_reproduced_bitwise']}" if g == "ORIGINAL" else ""), flush=True)
    if not ok:
        raise SystemExit(f"{name} failed its correctness checks")
    if g == "ORIGINAL" and not out["determinism"]["original_reproduced_bitwise"]:
        raise SystemExit(f"{name}: ORIGINAL not reproduced; investigate before proceeding")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "manifest", "cache", "diagnostics", "quant", "report"])
    ap.add_argument("--strategy", choices=list(STRATS))
    ap.add_argument("--grouping", choices=GROUPS + ["ORIGINAL"])
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "report":
        import report_multigroup_robustness as R
        R.main(); return
    dict(preflight=stage_preflight, manifest=stage_manifest, cache=stage_cache, diagnostics=stage_diagnostics,
         quant=stage_quant)[a.stage](a)


if __name__ == "__main__":
    main()
