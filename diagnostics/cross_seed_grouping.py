"""Cross-seed x calibration-grouping robustness of the FR vs FC four-layer GPTQ PPL comparison.

Allocation seeds 0/1/2 (frozen matched-warm-start FR / FC selections) x the 5 grouping seeds of the
multi-grouping study (same policy: sorted U_s, default_rng(g).permutation, batches of 8, last batch short),
paired FR/FC within each (s, g). Seed 0 = the 10 runs of runs_multigroup_robustness, REUSED after integrity
validation; seeds 1/2 = 20 new fresh-process configurations. Everything else frozen (GPTQ, 4 layers, windows,
evaluation batching, reconstruction rows 2128-2255 of ~/moe_cache).

Stages (--stage): preflight | manifest | cache | diagnostics | quant --seed S --strategy C --grouping G [--control] |
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
import multigroup_robustness as MR      # noqa: E402
from moe_calib import batch_sensitivity as BS        # noqa: E402
from moe_calib import cross_seed_grouping as CS      # noqa: E402
from moe_calib import multigroup_robustness as MG    # noqa: E402

OUT = "runs_cross_seed_grouping"
OCACHE = MR.OCACHE
XCACHE = os.path.expanduser(os.environ.get("CROSS_SEED_CACHE", "~/moe_cache_cross_seed_grouping"))
LAYERS = MR.LAYERS
STRATS = MR.STRATS
GSEEDS = list(MG.GROUPING_SEEDS)
NEW_SEEDS = (1, 2)
jdump, sha256_file, tensor_sha = FD.jdump, FD.sha256_file, DB.tensor_sha


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def gname(g):
    return MG.gname(g)


def cache_dir(s, g):
    return MR.gdir(gname(g)) if s == 0 else os.path.join(XCACHE, f"s{s}", gname(g))


def wdir(name):
    return os.path.join(XCACHE, "weights", name)


def rname(c, s, g):
    return f"{c}_s{s}_{gname(g)}"


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ---------------------------------------------------------------- selections (any allocation seed)
def selections(s):
    import e3_allocation as E3
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    man = json.load(open(os.path.join(ROOT, EP.OUT, "selected_set_manifest.json")))
    warm = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))["seeds"][str(s)]["ids"]
    banned = set(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))
    sel, checks = {}, {}
    for c in STRATS:
        ent = man["configs"][f"{c}_s{s}"]
        src = os.path.join(ROOT, ent["source"])
        r = json.load(open(src))
        ch = r["chosen"]
        checks[c] = dict(n=len(ch), unique=len(set(ch)), ids_hash=ent["ids_hash"], K=ent["K"], source=ent["source"],
                         source_sha256=ent["source_sha256"], source_sha256_matches=sha256_file(src) == ent["source_sha256"],
                         chosen_matches_manifest=ch == ent["chosen"], warm_hash_matches=E3.warm_hash(ch) == ent["ids_hash"],
                         manifest_entry_valid=bool(ent["valid"] and all(ent["checks"].values())),
                         ordered_warm_start_is_frozen=ch[:32] == warm,
                         all_in_pool=all(0 <= x < cfg["split"]["pool"][1] for x in ch),
                         no_tune_or_test_leakage=not (set(ch) & banned), strategy=ent["strategy"] == c, seed=ent["seed"] == s)
        assert checks[c]["n"] == 128 == checks[c]["unique"] and all(
            v for v in checks[c].values() if isinstance(v, bool)), (s, c, checks[c])
        sel[c] = dict(chosen=ch, K=ent["K"], ids_hash=ent["ids_hash"])
    assert sel["FR"]["chosen"][:32] == sel["FC"]["chosen"][:32], "warm start not shared"
    union = CS.union_of(sel["FR"]["chosen"], sel["FC"]["chosen"])
    return cfg, sel, checks, union


# ---------------------------------------------------------------- seed-0 reuse validation
def validate_seed0():
    """The 10 seed-0 runs may be reused iff manifests, selections, caches, GPTQ, windows and results all check out."""
    pr = MR.protocol()                                         # protocol + manifest sha integrity (asserts)
    cfg, sel, checks, union = selections(0)
    win, prot = EP.load_windows()
    out = dict(protocol_sha256=pr["protocol_sha256"], checks={})
    ck = out["checks"]
    ck["union_matches"] = pr["union_calibration_ids"] == union
    ck["selection_hashes_match"] = all(pr["selections"][c]["ids_hash"] == sel[c]["ids_hash"] for c in STRATS)
    ck["groupings_regenerate_identically"] = all(pr["groupings"][gname(g)]["batches"] == MG.grouping_batches(union, g)
                                                 for g in GSEEDS)
    ck["gptq_identical"] = pr["gptq"] == cfg["gptq"]
    ck["windows_identical"] = pr["evaluation"]["windows_sha256"] == prot["windows_sha256"]
    cm = json.load(open(os.path.join(ROOT, MR.OUT, "cache_manifest.json")))
    ck["cache_manifest_valid"] = bool(cm["valid"])
    cache_ok = {}
    for g in GSEEDS:
        try:
            MR.verify_cache(gname(g)); cache_ok[gname(g)] = True
        except AssertionError:
            cache_ok[gname(g)] = False
    ck["cache_files_sha256_match"] = all(cache_ok.values())
    res_ok = {}
    for c in STRATS:
        for g in GSEEDS:
            r = json.load(open(os.path.join(ROOT, MR.OUT, "results", f"{c}_{gname(g)}.json")))
            nll = np.array(r["per_window_nll"]); cnt = np.array(r["per_window_tokens"])
            res_ok[f"{c}_{gname(g)}"] = bool(
                r["valid"] and r["ids_hash"] == sel[c]["ids_hash"] and r["gptq"] == cfg["gptq"] and r["layers"] == LAYERS
                and r["n_quantized_experts"] == 240 and r["n_projection_tensors"] == 720 and r["rtn_fallbacks"] == 0
                and r["projections_unchanged_from_bf16"] == 0 and r["nonfinite_weights"] == 0 and r["untouched_params_unchanged"]
                and r["repeat_eval_abs_diff_ppl"] == 0 and int(cnt.sum()) == 65408
                and abs(float(np.exp(nll.sum() / cnt.sum())) - r["ppl"]) < 1e-12
                and r["calibration_cache"] == MR.gdir(gname(g)) and r["reconstruction_cache"] == OCACHE
                and all(sha256_file(os.path.join(r["weights_dir"], f"L{L}.pt")) == r["weight_files"][str(L)]["sha256"]
                        for L in LAYERS))
    ck["results_integrity"] = all(res_ok.values())
    out["per_result"] = res_ok
    out["per_cache"] = cache_ok
    out["valid"] = all(ck.values())
    return out


# ---------------------------------------------------------------- preflight
def stage_preflight(a):
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    win, prot = EP.load_windows()
    assert abs(base["ppl"] - 9.237090) < 1e-6 and prot["scored_targets_total"] == 65408 and EP.BATCH == 8
    seeds = {}
    for s in CS.ALLOC_SEEDS:
        cfg, sel, checks, union = selections(s)
        seeds[s] = dict(checks=checks, n_union=len(union), batch_pattern=CS.expected_sizes(len(union)))
    assert seeds[0]["n_union"] == 217
    t0 = time.time()
    reuse = validate_seed0()
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    out = dict(seeds=seeds, seed0_reuse=reuse, reuse_check_s=time.time() - t0, bf16_ppl=base["ppl"],
               windows_sha256=prot["windows_sha256"], environment=DB.env_record(),
               host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, gpu_free_gb=free / 1e9,
               disk_free_tb=shutil.disk_usage(os.path.expanduser("~")).free / 1e12, created=time.strftime("%F %T"))
    jdump(out, p("logs", "preflight.json"))
    print(f"preflight: unions {[seeds[s]['n_union'] for s in seeds]}; seed-0 reuse valid {reuse['valid']}; "
          f"GPU free {free / 1e9:.1f} GB")
    if not reuse["valid"]:
        raise SystemExit(f"seed-0 reuse validation failed: {reuse['checks']}")


# ---------------------------------------------------------------- manifests + frozen protocol
def stage_manifest(a):
    os.makedirs(p("grouping_manifests"), exist_ok=True)
    pre = json.load(open(p("logs", "preflight.json")))
    assert pre["seed0_reuse"]["valid"]
    seeds, groupings, selman = {}, {}, {}
    for s in CS.ALLOC_SEEDS:
        cfg, sel, checks, union = selections(s)
        forbidden = list(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))
        selman[f"s{s}"] = {c: dict(ids_hash=sel[c]["ids_hash"], K=sel[c]["K"], chosen=sel[c]["chosen"],
                                   checks=checks[c]) for c in STRATS}
        selman[f"s{s}"]["union"] = union
        selman[f"s{s}"]["union_sha256"] = MG.obj_sha(union)
        seeds[s] = dict(union_sha256=MG.obj_sha(union), n_union=len(union),
                        selection_hashes={c: sel[c]["ids_hash"] for c in STRATS})
        for g in GSEEDS:
            bt = MG.grouping_batches(union, g)
            v = MG.validate_grouping(bt, union, forbidden)
            assert all(v.values()), (s, g, v)
            m = MG.manifest(g, bt, union, {c: sel[c]["chosen"] for c in STRATS})
            m.update(allocation_seed=s, union_sha256=MG.obj_sha(union), selection_hashes=seeds[s]["selection_hashes"])
            m["manifest_sha256"] = MG.obj_sha(MG.strip_sha(m))
            m["validation"] = v
            if s == 0:
                old = json.load(open(os.path.join(ROOT, MR.OUT, "grouping_manifests", f"{gname(g)}.json")))
                assert old["batches"] == m["batches"] and old["manifest_sha256"] == pre_seed0_sha(old), "seed-0 manifest mismatch"
                m["reused_from"] = dict(path=f"{MR.OUT}/grouping_manifests/{gname(g)}.json",
                                        manifest_sha256=old["manifest_sha256"])
            path = p("grouping_manifests", f"s{s}_{gname(g)}.json")
            old = load_json(path)
            if old is not None:
                assert old["manifest_sha256"] == m["manifest_sha256"], f"{path} changed: frozen manifests are never rewritten"
            else:
                jdump(m, path)
            groupings[CS.cname(s, g)] = dict(allocation_seed=s, grouping_seed=g, manifest_sha256=m["manifest_sha256"],
                                             batches=m["batches"], batch_size_pattern=m["batch_size_pattern"])
    if not os.path.exists(p("selected_set_manifest.json")):
        jdump(selman, p("selected_set_manifest.json"))
    else:
        assert json.load(open(p("selected_set_manifest.json"))) == json.loads(json.dumps(selman)), "selections changed"
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    pilot = json.load(open(os.path.join(ROOT, FD.OUT, "protocol.json")))
    mrp = json.load(open(os.path.join(ROOT, MR.OUT, "protocol.json")))
    body = dict(
        experiment="Cross-seed x calibration-grouping robustness of FR vs FC four-layer GPTQ PPL",
        model=pilot["model"], model_snapshot=pilot["model_snapshot"], tokenizer=prot.get("tokenizer"),
        gptq=json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))["gptq"], B=128,
        allocation_seeds=list(CS.ALLOC_SEEDS), grouping_seeds=GSEEDS, quant_layers=LAYERS, n_quantized_experts=240,
        n_projection_tensors=720, seeds=seeds, groupings=groupings,
        grouping_policy=mrp["grouping_policy"] + " (applied to each seed's own union U_s; groupings nested within seeds)",
        evaluation=dict(protocol_sha256=sha256_file(os.path.join(ROOT, EP.OUT, "evaluation_protocol.json")),
                        windows_sha256=prot["windows_sha256"], batching=f"EP.layerwise_nll batch {EP.BATCH}",
                        bf16_ppl=base["ppl"], repeats_per_config=2),
        reconstruction="E3 held-out relative expert reconstruction on ~/moe_cache test rows 2128-2255 (eval_cache_dir), "
                       "identical for every configuration",
        seed0=dict(reused_from=MR.OUT, multigroup_protocol_sha256=mrp["protocol_sha256"], reuse_validation=pre["seed0_reuse"]),
        determinism_controls=dict(
            execution_order="every new (s,g) cache rebuilt with reversed batch execution order must match bitwise",
            repeat_quant="seed-0 FR G20261111 re-quantized in a fresh process from the reused cache must equal the "
                         "reused weights sha256 and PPL exactly", repeat_ppl="2 in-process evaluations, |dPPL| = 0"),
        hierarchical_bootstrap=dict(n_boot=CS.HBOOT_N, seed=CS.HBOOT_SEED), window_bootstrap=dict(n_boot=FD.N_BOOT, seed=FD.BOOT_SEED),
        uncertainty_levels=CS.UNCERTAINTY_LEVELS, decision_rule=CS.DECISION_RULE, environment=mrp["environment"],
        hard_constraints="no re-selection, seeds 0/1/2 only, 5 grouping seeds only, same policy, B=128, 4 layers, frozen "
                         "GPTQ / precision / windows / batching, no test rows, no extension after results")
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        assert old["protocol_sha256"] == MG.obj_sha(body), "protocol changed after freezing"
        print(f"protocol already frozen; consistent ({old['protocol_sha256'][:16]})"); return
    body["protocol_sha256"] = MG.obj_sha(body)
    body["created"] = time.strftime("%F %T")
    jdump(body, p("protocol.json"))
    print(f"manifest ok: unions {[seeds[s]['n_union'] for s in seeds]}; protocol sha256 {body['protocol_sha256'][:16]}")


def pre_seed0_sha(old):
    return MG.obj_sha(MG.strip_sha({k: v for k, v in old.items() if k != "validation"}))


def protocol():
    pr = json.load(open(p("protocol.json")))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert MG.obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    for key, x in pr["groupings"].items():
        m = json.load(open(p("grouping_manifests", f"s{x['allocation_seed']}_{gname(x['grouping_seed'])}.json")))
        assert MG.obj_sha(MG.strip_sha({k: v for k, v in m.items() if k not in ("validation", "reused_from")})) \
            == m["manifest_sha256"] == x["manifest_sha256"], f"manifest {key} modified"
    return pr


# ---------------------------------------------------------------- cache (new seeds only)
def stage_cache(a):
    pr = protocol()
    out_path = p("cache_manifest.json")
    man = load_json(out_path) or dict(entries={}, cache_root=XCACHE)
    alltok = DB.all_tokens()
    model, t_load = DB.load_model()
    if "model" not in man:
        man["model"] = dict(pristine_all_parameters=DB.verify_pristine_full(model), config=DB.model_config_record(model),
                            environment=DB.env_record(), load_s=t_load)
        jdump(man, out_path)
    assert man["model"]["pristine_all_parameters"] and not man["model"]["config"]["training"] \
        and not man["model"]["config"]["dropout_modules_p_gt_0"]
    for s in NEW_SEEDS:
        union = selections(s)[3]
        assert MG.obj_sha(union) == pr["seeds"][str(s)]["union_sha256"]
        tsha_ref = tensor_sha(alltok[union])
        for g in GSEEDS:
            key = CS.cname(s, g)
            if man["entries"].get(key, {}).get("complete"):
                continue
            bt = pr["groupings"][key]["batches"]
            d = cache_dir(s, g); os.makedirs(d, exist_ok=True)
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
                    neg[str(L)] = bool(torch.equal(X, keep[L]))
            t2 = time.time()
            tsha_r = DB.forward_plan(model, alltok, bt[::-1], union, cmp)
            ent = dict(allocation_seed=s, grouping_seed=g, manifest_sha256=pr["groupings"][key]["manifest_sha256"],
                       cache_dir=d, layers=LAYERS, n_rows=len(union), sequence_ids=union, token_sha256=tsha, files=files,
                       storage_bytes=sum(f["bytes"] for f in files.values()), generation_s=t_build,
                       negative_control=dict(per_layer_bitwise=neg, passed=len(neg) == 4 and all(neg.values()),
                                             seconds=time.time() - t2),
                       token_ids_identical=tsha == tsha_ref == tsha_r,
                       finite=all(f["nan"] == 0 == f["inf"] for f in files.values()), complete=True)
            man["entries"][key] = ent
            jdump(man, out_path)
            print(f"[cache] {key}: {t_build:.0f} s; reversed-order bitwise {ent['negative_control']['passed']}; tokens "
                  f"identical {ent['token_ids_identical']}", flush=True)
            if not (ent["negative_control"]["passed"] and ent["token_ids_identical"] and ent["finite"]):
                raise SystemExit(f"{key}: cache control FAILED; not quantizing corrupted inputs")
            del keep
    E = man["entries"]
    man["semantics"] = dict(token_ids_identical=all(e["token_ids_identical"] for e in E.values()),
                            execution_order_controls_passed=all(e["negative_control"]["passed"] for e in E.values()),
                            finite=all(e["finite"] for e in E.values()),
                            same_model_weights=man["model"]["pristine_all_parameters"],
                            dtype=man["model"]["config"]["dtype"], attn_implementation=man["model"]["config"]["attn_implementation"],
                            eval_mode=not man["model"]["config"]["training"],
                            positions_mask="batch-1 decoder-layer kwargs broadcast over every batch (identical semantics)",
                            no_padding_no_token_dropping="full 512-token rows; Qwen2-MoE has no capacity limit")
    man["valid"] = bool(len(E) == 10 and all(v for k, v in man["semantics"].items() if isinstance(v, bool)))
    man["seed0"] = "reused: runs_multigroup_robustness/cache_manifest.json (validated in preflight)"
    man["storage_bytes_total_new"] = sum(e["storage_bytes"] for e in E.values())
    jdump(man, out_path)
    print(f"[cache] valid {man['valid']}; new storage {man['storage_bytes_total_new'] / 1e9:.1f} GB")


def verify_cache(s, g):
    if s == 0:
        return MR.verify_cache(gname(g))
    man = json.load(open(p("cache_manifest.json")))
    assert man["valid"]
    e = man["entries"][CS.cname(s, g)]
    for L in LAYERS:
        assert sha256_file(os.path.join(cache_dir(s, g), f"L{L}.pt")) == e["files"][str(L)]["sha256"], (s, g, L)


# ---------------------------------------------------------------- diagnostics (new seeds; seed 0 reused)
def stage_diagnostics(a):
    from moe_calib import model_utils as MU
    protocol()
    if os.path.exists(p("activation_diagnostics.json")):
        print("[diag] done"); return
    model, t_load = DB.load_model()
    E, k = MU.num_experts(model), MU.top_k(model)
    out, t0 = {}, time.time()
    for s in NEW_SEEDS:
        union = selections(s)[3]
        for g in GSEEDS:
            verify_cache(s, g)
        out[str(s)] = {}
        for L in LAYERS:
            R = torch.load(os.path.join(OCACHE, f"L{L}.pt"), mmap=True)[union].clone()
            X = {"ORIGINAL": R, **{gname(g): torch.load(os.path.join(cache_dir(s, g), f"L{L}.pt")) for g in GSEEDS}}
            with MU.layer_on_gpu(model, L):
                lg = {v: DB.replay(model, L, x, False)[0] for v, x in X.items()}
            sels = {v: torch.cat([BS.topk_route(lg[v][i].to(MU.GPU), k)[1].cpu() for i in range(lg[v].shape[0])]) for v in X}
            res = {}
            for g in GSEEDS:
                G = gname(g)
                dr = BS.activation_drift(X[G], R)
                rc = BS.routing_compare(sels[G], sels["ORIGINAL"], n_experts=E)
                res[G] = dict(relative_l2=dr["relative_l2"], pct_bitwise_equal=100 * dr["frac_elements_exactly_equal"],
                              token_rel_l2_p50=dr["token_rel_l2_p50"], token_rel_l2_p99=dr["token_rel_l2_p99"],
                              nan=dr["nan_count"], inf=dr["inf_count"],
                              frac_tokens_top4_set_changed=rc["frac_any_expert_changed"],
                              routing_distribution_l1=rc["routing_distribution_l1"], n_tokens=rc["n_tokens"])
            out[str(s)][str(L)] = res
            del X, lg, sels, R
            print(f"[diag] s{s} L{L}: rel L2 " + " ".join(f"{res[gname(g)]['relative_l2']:.2e}" for g in GSEEDS) +
                  "; set changed " + " ".join(f"{res[gname(g)]['frac_tokens_top4_set_changed']:.4f}" for g in GSEEDS) +
                  f" ({time.time() - t0:.0f} s)", flush=True)
    s0 = json.load(open(os.path.join(ROOT, MR.OUT, "activation_diagnostics.json")))["per_layer"]
    out["0"] = {L: {G: {k_: x[G][k_] for k_ in ("relative_l2", "pct_bitwise_equal", "token_rel_l2_p50", "token_rel_l2_p99",
                                                 "nan", "inf", "frac_tokens_top4_set_changed", "routing_distribution_l1", "n_tokens")}
                    for G in x if G.startswith("G")} for L, x in s0.items()}
    jdump(dict(reference="ORIGINAL = ~/moe_cache rows of each seed's union (seed-specific original extraction)",
               routing="GPU softmax(float)+topk(4) of Profiler-path router logits, one sequence at a time",
               seed0="reused from runs_multigroup_robustness/activation_diagnostics.json", per_seed=out,
               seconds=time.time() - t0, model_load_s=t_load), p("activation_diagnostics.json"))


# ---------------------------------------------------------------- quantization (one config per fresh process)
def stage_quant(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    s, c, g = a.seed, a.strategy, a.grouping
    name = rname(c, s, g) + ("_CTRL_repeat" if a.control else "")
    os.makedirs(p("results"), exist_ok=True)
    if os.path.exists(p("results", f"{name}.json")):
        print(f"[quant] {name} done"); return
    if s == 0 and not a.control:
        raise SystemExit("seed-0 configurations are reused, not re-run (use --control for the determinism check)")
    pr = protocol()
    cfg, sel, _, union = selections(s)
    assert MG.obj_sha(union) == pr["seeds"][str(s)]["union_sha256"]
    assert sel[c]["ids_hash"] == pr["seeds"][str(s)]["selection_hashes"][c]
    if s != 0:
        cm = json.load(open(p("cache_manifest.json")))
        assert cm["valid"], "cache controls not passed"
        ctl = load_json(p("results", f"{rname('FR', 0, GSEEDS[0])}_CTRL_repeat.json"))
        assert ctl and ctl["determinism"]["reproduced_bitwise"], "seed-0 repeat-quantization control not passed"
    verify_cache(s, g)
    test_ids = list(range(*cfg["split"]["test"]))
    cdir, chosen = cache_dir(s, g), MG.remap(sel[c]["chosen"], union)
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
    n_uv = sum(1 for kk in u0 if kk in pu)
    assert all(u0[kk] == pu[kk] for kk in u0 if kk in pu) and n_uv > 0, "untouched parameters not pristine"
    t_verify = time.time() - t0
    orig = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()}
            for L, e in EP.CHECK_EXPERTS}
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    gq = cfg["gptq"]
    q = PQ.quantize_install(model, cdir, LAYERS, chosen, test_ids, sel[c]["K"], gq["bits"], gq["group_size"],
                            gq["percdamp"], capture=EP.CHECK_EXPERTS, eval_cache_dir=OCACHE)
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
    files = {}
    if not a.control:                                         # control: hash only (no duplicate weight artifact)
        os.makedirs(wdir(name), exist_ok=True)
        for L in LAYERS:
            path = os.path.join(wdir(name), f"L{L}.pt")
            PQ.save_layer(model, L, path)
            files[str(L)] = dict(sha256=sha256_file(path), layer_sha256=PQ.layer_digest(model, L))
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
    out = dict(strategy=c, allocation_seed=s, grouping=gname(g), grouping_seed=g, name=name, ids_hash=sel[c]["ids_hash"],
               manifest_sha256=pr["groupings"][CS.cname(s, g)]["manifest_sha256"], calibration_cache=cdir,
               reconstruction_cache=OCACHE, gptq=gq, layers=LAYERS,
               n_quantized_experts=q["counts"]["n_experts_quantized_total"], n_projection_tensors=len(after),
               rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"], n_experts_scored=q["counts"]["n_experts_test_observed"],
               recon=float(np.mean([r["mean"] for r in q["results"]])), recon_per_layer=per_layer,
               per_expert=[dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"]) for r in q["results"]],
               projections_unchanged_from_bf16=len(unchanged), nonfinite_weights=len(nonfin),
               untouched_params_unchanged=untouched_ok, n_untouched_params=len(u0), n_untouched_verified_vs_checkpoint=n_uv,
               expert_checks=checks, installed_weights_sha256=dig, weight_files=files, weights_dir=wdir(name) if files else None,
               repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
               repeat_eval_per_window_identical=evals[0]["per_window_nll"] == evals[1]["per_window_nll"],
               model_load_s=t_load, pristine_verify_s=t_verify, quantize_s=t_q, hessian_s=q["hessian_s"], gptq_s=q["gptq_s"],
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb(), valid=bool(ok), **rec)
    if a.control:
        ref = json.load(open(os.path.join(ROOT, MR.OUT, "results", f"{c}_{gname(g)}.json")))
        out["determinism"] = dict(reference=f"{MR.OUT}/results/{c}_{gname(g)}.json", weights_exact=dig == ref["installed_weights_sha256"],
                                  ppl_exact=rec["ppl"] == ref["ppl"], recon_exact=out["recon"] == ref["recon"],
                                  per_window_exact=rec["per_window_nll"] == ref["per_window_nll"])
        out["determinism"]["reproduced_bitwise"] = all(out["determinism"][k] for k in
                                                       ("weights_exact", "ppl_exact", "recon_exact", "per_window_exact"))
    jdump(out, p("results", f"{name}.json"))
    print(f"[quant] {name}: PPL {rec['ppl']:.6f} recon {out['recon']:.7f} sha {dig[:16]} valid {ok} ({t_q:.0f} s)"
          + (f" reproduced {out['determinism']['reproduced_bitwise']}" if a.control else ""), flush=True)
    if not ok:
        raise SystemExit(f"{name} failed its correctness checks")
    if a.control and not out["determinism"]["reproduced_bitwise"]:
        raise SystemExit(f"{name}: repeat quantization not reproduced; investigate before proceeding")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "manifest", "cache", "diagnostics", "quant", "report"])
    ap.add_argument("--seed", type=int, choices=list(CS.ALLOC_SEEDS))
    ap.add_argument("--strategy", choices=list(STRATS))
    ap.add_argument("--grouping", type=int, choices=GSEEDS)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "report":
        import report_cross_seed_grouping as R
        R.main(); return
    dict(preflight=stage_preflight, manifest=stage_manifest, cache=stage_cache, diagnostics=stage_diagnostics,
         quant=stage_quant)[a.stage](a)


if __name__ == "__main__":
    main()
