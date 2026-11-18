"""Full-E3 validation of Shared Covariance-aware calibration (SC) vs FR / TR / TC (B = 128, 240 experts, 4 layers).

SC = one shared set of 128 pool sequences (the frozen 32-sequence matched warm start + 96 optimized slots) minimizing
the equal-expert mean 64-d covariance-sketch mismatch over all 240 routed experts (moe_calib/shared_covariance_selection).
FR / TR / TC = the frozen matched-warm-start selections (runs_partial_ppl/selected_set_manifest.json).

Stages (--stage):
  preflight   validate baselines / warm starts / pool / evaluation fingerprints; compute the cost estimate
  protocol    freeze settings + decision rule (immutable; sha256 recorded)
  sketch      [--layer L]: full-pool per-(seq, expert, proj) sketches + full-dim pool Hessians, one layer per call
  mvpcheck    the new L16 sketches reproduce the 8-expert MVP sketches and its SC covariance scores
  select      --seed s : SC selection (GPU), twice (determinism), validation, objective of every baseline set
  freeze      code + selection hashes, before any SC reconstruction
  evalA       --configs SC:0 FR:0 ... : ORIGINAL extraction: test recon + PPL + confirmation recon per config
  diag        covariance (sketch + full-dim, gu / down), routing, exposure, diversity diagnostics for the 12 sets
  gmanifest   Stage B: union of the 4 strategies per seed, 5 groupings x 3 seeds, frozen manifests
  gcache      --seed s : grouping caches (full model, reversed-order control)
  gquant      --seed s --grouping g [--control] : 4 strategies quantized from the grouping cache, recon + PPL
  report      -> diagnostics/report_shared_covariance_e3.py
"""
from __future__ import annotations

import argparse, hashlib, json, os, resource, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import expertwise_calibration as EW          # noqa: E402
from moe_calib import shared_covariance_selection as SCS    # noqa: E402

OUT = "runs_shared_covariance_e3"
STORE = os.path.expanduser(os.environ.get("SC_E3_STORE", "~/moe_cache_shared_covariance_e3"))
OCACHE = os.path.expanduser("~/moe_cache")
CCACHE = os.path.expanduser("~/moe_cache_confirmation")
LAYERS = [1, 8, 16, 23]
E = 60
POOL_N = 2000
SEEDS = [0, 1, 2]
BASE = ("FR", "TR", "TC")
STRATS = ("FR", "TR", "TC", "SC")
GSEEDS = [20261111, 20261112, 20261113, 20261114, 20261115]
B, WARM = 128, 32
SETTINGS = dict(B=B, warm_size=WARM, layers=LAYERS, n_experts=240, sketch_dim=64, projection_seed=20261116,
                greedy_sample_frac=0.1, swap_max_passes=5, swap_rel_tol=1e-9, eps=SCS.EPS, rng_stream="SeedSequence([seed, 1])",
                term_weight="1/480 per (layer, expert, proj) = 1/240 per expert x 1/2 per proj; 1/4 per layer",
                storage="sketch fp32 (svech, k(k+1)/2 = 2080), every exact objective accumulated in fp64",
                K=4, gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True), threshold=SCS.THRESHOLD,
                grouping_seeds=GSEEDS, grouping_batch=8)
ROUTING_NPZ = "runs_typicality_matched/routing_profile_v2.npz"
MVP_DIR = "runs_expertwise_mvp"
MVP_STORE = os.path.expanduser("~/moe_cache_expertwise_mvp/sketch.pt")
FROZEN_CODE = ("moe_calib/gptq.py", "moe_calib/quant_eval.py", "moe_calib/profiler.py", "moe_calib/hessian.py",
               "moe_calib/model_utils.py", "e3_allocation.py", "moe_calib/partial_model_quant.py",
               "diagnostics/eval_partial_ppl.py", "moe_calib/multigroup_robustness.py", "diagnostics/batch_sensitivity.py",
               "configs/frozen_b4.json")
EXP_CODE = ("moe_calib/shared_covariance_selection.py", "moe_calib/expertwise_calibration.py",
            "diagnostics/shared_covariance_e3.py")


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def st(*a):
    return os.path.join(STORE, *a)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path)))


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hashes(files):
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def ids_hash(ids):
    import e3_allocation as E3
    return E3.warm_hash(ids)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def gpu_peak():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def forbidden():
    return list(range(2000, 2256))


# ===================================================================== baselines
def baselines():
    """{(C, s): dict(chosen, K, ids_hash, test_src, conf_src, ppl_src)} from the frozen partial-PPL manifest, re-validated."""
    man = jload("runs_partial_ppl/selected_set_manifest.json")
    ws = jload("runs_typicality_matched/warm_starts.json")
    out = {}
    for c in BASE:
        for s in SEEDS:
            ent = man["configs"][f"{c}_s{s}"]
            src = os.path.join(ROOT, ent["source"])
            r = json.load(open(src))
            ch = r["chosen"]
            W = ws["seeds"][str(s)]["ids"]
            ck = dict(manifest_valid=bool(ent["valid"] and all(ent["checks"].values())),
                      source_sha256=sha256_file(src) == ent["source_sha256"], chosen_matches=ch == ent["chosen"],
                      hash=ids_hash(ch) == ent["ids_hash"], n128=len(ch) == 128 == len(set(ch)),
                      warm=ch[:WARM] == W and r["warm_start"] == W,
                      warm_hash=ws["seeds"][str(s)]["hash"] == ids_hash(W),
                      pool=all(0 <= x < POOL_N for x in ch), no_leak=not (set(ch) & set(forbidden())), K=ent["K"] == 4)
            conf = jload(f"runs_aggregate_constraint/eval/{c}_s{s}.json")
            pp = jload(f"runs_partial_ppl/ppl_{c}_s{s}.json")
            ck.update(conf_same_ids=conf["ids_hash"] == ent["ids_hash"], ppl_same_ids=pp["ids_hash"] == ent["ids_hash"])
            assert all(ck.values()), (c, s, ck)
            out[(c, s)] = dict(chosen=ch, K=ent["K"], ids_hash=ent["ids_hash"], test_src=ent["source"],
                               conf_src=f"runs_aggregate_constraint/eval/{c}_s{s}.json",
                               ppl_src=f"runs_partial_ppl/ppl_{c}_s{s}.json", checks=ck, warm=W)
    for s in SEEDS:
        assert len({tuple(out[(c, s)]["chosen"][:WARM]) for c in BASE}) == 1, "warm start not shared"
    return out


def selection(c, s):
    if c == "SC":
        d = jload(p("selections", f"seed{s}.json"))
        return dict(chosen=d["chosen"], K=4, ids_hash=d["ids_hash"])
    b = baselines()[(c, s)]
    return dict(chosen=b["chosen"], K=b["K"], ids_hash=b["ids_hash"])


# ===================================================================== preflight (+ estimate)
def stage_preflight(a):
    t0 = time.time()
    bl = baselines()
    z = np.load(os.path.join(ROOT, ROUTING_NPZ))
    assert z["layers"].tolist() == LAYERS and z["pool_ids"].tolist() == list(range(POOL_N)) and str(z["rule"]) == "model_topk"
    C = z["C"]
    pools = jload("runs_typicality/pools.json")
    meta = jload(os.path.join(OCACHE, "meta.json"))
    lock = jload("runs_aggregate_constraint/locked_protocol.json")["confirmation"]
    prov = jload(os.path.join(CCACHE, "provenance.json"))
    ids = torch.load(os.path.join(OCACHE, "ids.pt"))
    alltok = torch.cat([ids["pool"], ids["holdout"]])
    fp = dict(pool_token_sha256=hashlib.sha256(alltok[:POOL_N].numpy().tobytes()).hexdigest(),
              test_token_sha256=hashlib.sha256(alltok[2128:2256].numpy().tobytes()).hexdigest(),
              tune_token_sha256=hashlib.sha256(alltok[2000:2128].numpy().tobytes()).hexdigest(),
              ocache_files={str(L): dict(bytes=os.path.getsize(os.path.join(OCACHE, f"L{L}.pt")),
                                         mtime=os.path.getmtime(os.path.join(OCACHE, f"L{L}.pt"))) for L in LAYERS},
              confirmation_fingerprint=prov["confirmation_fingerprint"],
              confirmation_matches_locked=prov["confirmation_fingerprint"] == lock["fingerprint"],
              confirmation_layer_sha256_locked=lock["layer_cache_sha256"],
              routing_profile_sha256=sha256_file(os.path.join(ROOT, ROUTING_NPZ)),
              typicality_pools=pools.get("fingerprints", None))
    if a.hash_caches:
        fp["confirmation_layer_sha256_now"] = {str(L): sha256_file(os.path.join(CCACHE, f"L{L}.pt")) for L in LAYERS}
        fp["confirmation_cache_unchanged"] = fp["confirmation_layer_sha256_now"] == lock["layer_cache_sha256"]
    tok = C.reshape(POOL_N, 4, E)
    V = 64 * 65 // 2
    est = dict(
        sketch_storage_gb=dict(per_layer=E * 2 * POOL_N * V * 4 / 1e9, total=4 * E * 2 * POOL_N * V * 4 / 1e9,
                               note="fp32 svech sketches, disk (torch.save, mmap load); per-seq counts negligible"),
        full_dim_pool_reference_gb=4 * E * (2048 ** 2 + 1408 ** 2) * 4 / 1e9,
        sketch_pass=dict(layer_replays=4 * POOL_N, full_forward_equiv_seqs=4 * POOL_N / 24,
                         full_forward_equiv_s=4 * POOL_N / 24 * meta["forward_s_per_seq"],
                         peak_gpu_gb_est="~1.2 GB layer + 1.5 GB full-dim pool Hessians (60 experts) + buffers ~ 3-4 GB",
                         peak_cpu_gb_est="~2 GB sketch tensor + partial model ~5 GB"),
        selection=dict(terms=480, stacked_tensor_gb=480 * POOL_N * V * 4 / 1e9, device="GPU (fp32 storage)",
                       per_update_matvec_gmac=480 * POOL_N * V / 1e9, greedy_updates=96,
                       swap_matvecs_per_pass=96, max_passes=5,
                       peak_gpu_gb_est="~8 GB sketches + ~1.5 GB fp64 work"),
        pool_tokens_per_expert=dict(min=int(tok.sum(0).min()), max=int(tok.sum(0).max())),
        stage_a=dict(new_configs=["SC_s0", "SC_s1", "SC_s2"], validation_reruns=["FR_s0", "TR_s1", "TC_s2"],
                     per_config_s_est="quantize+test ~270 s, PPL ~45 s, confirmation re-quantize ~400 s",
                     peak_cpu_gb_est="~64-68 GB (full model + snapshots; cap 72 GB -> strictly one process at a time)"),
        stage_b=dict(configs=4 * 3 * 5, caches=15, per_config_s_est="~330 s (quantize ~270 + PPL ~45)",
                     cache_gb_est="~3.4 GB per (seed, grouping) for a ~410-row union x 4 layers"),
        existing_costs=dict(pool_cache_s=meta["cache_seconds"], forward_s_per_seq=meta["forward_s_per_seq"],
                            routing_profile_s=357.0))
    out = dict(baselines={f"{c}_s{s}": dict(ids_hash=v["ids_hash"], K=v["K"], checks=v["checks"], test_src=v["test_src"],
                                            conf_src=v["conf_src"], ppl_src=v["ppl_src"]) for (c, s), v in bl.items()},
               warm_starts={str(s): dict(ids=bl[("FR", s)]["warm"], hash=ids_hash(bl[("FR", s)]["warm"])) for s in SEEDS},
               fingerprints=fp, estimate=est, seconds=time.time() - t0, created=time.strftime("%F %T"))
    jdump(out, p("preflight.json"))
    print(json.dumps(est, indent=1))
    print("[preflight] 9 baselines valid; warm starts shared; confirmation locked:", fp["confirmation_matches_locked"])


# ===================================================================== protocol
def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        pr = jload(p("protocol.json"))
        body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
        assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
        print("[protocol] already frozen", pr["protocol_sha256"][:16]); return
    pre = jload(p("preflight.json"))
    mvp = jload(os.path.join(MVP_DIR, "protocol.json"))
    body = dict(
        experiment="Full E3 validation of Shared Covariance-aware calibration (SC) vs FR / TR / TC",
        model="Qwen/Qwen1.5-MoE-A2.7B", settings=SETTINGS, seeds=SEEDS, strategies=list(STRATS),
        experts="all 60 routed experts of layers 1/8/16/23 (240); no expert subset, no per-expert sets or quotas",
        candidate_pool=dict(rows=[0, POOL_N], cache=OCACHE, token_sha256=pre["fingerprints"]["pool_token_sha256"]),
        baselines=pre["baselines"], warm_starts=pre["warm_starts"],
        objective="J(S) = 1/240 sum_{L,e} 1/2 sum_{p in gu,down} ||H_Lep(S) - H_Lep^pool||_F / (||H_Lep^pool||_F + 1e-12); "
                  "H = token-normalized 64-d sketch second moment, z = R_p^T x, R_p N(0,1/64) from seed 20261116 "
                  "(identical to the MVP); pool reference over rows 0-1999 only; empty-token rule H(S) := 0; "
                  "W (the seed's ordered 32-sequence warm start) is contained in S and never removed",
        optimizer="MVP optimizer: stochastic greedy from W (argmin over a seeded uniform 10 % sample of the remaining "
                  "pool, 96 steps), then <= 5 best-improvement swap passes over the 96 non-warm slots (whole pool as "
                  "candidates), accepted only if the exact fp64 objective decreases by > 1e-9 relative",
        mvp_reference=dict(protocol=f"{MVP_DIR}/protocol.json", projection_seed=mvp["settings"]["projection_seed"],
                           sketch_dim=mvp["settings"]["sketch_dim"], greedy_sample_frac=mvp["settings"]["greedy_sample_frac"],
                           swap_max_passes=mvp["settings"]["swap_max_passes"],
                           differences=["fixed 32-sequence warm start (E3 matched-baseline design)",
                                        "swap passes skip the warm slots", "240 experts / 4 layers instead of 8 / L16",
                                        "stacked fp32 storage with fp64 exact acceptance (MVP: fp64 per term)",
                                        "no token matching (original E3 calibration problem)"]),
        evaluation=dict(
            stage_a="ORIGINAL extraction (~/moe_cache): e3_allocation.quantize_and_score on the frozen test rows 2128-2255 "
                    "(primary) and on the locked confirmation split (secondary, eval_cache_dir); partial-model WikiText-2 "
                    "PPL with the runs_partial_ppl windows / installer / checks",
            baseline_reuse="FR/TR/TC stored results are reused ONLY if FR s0, TR s1 and TC s2 re-run through this "
                           "pipeline reproduce their stored test per-expert errors, confirmation per-expert errors, "
                           "installed-weight sha256 and PPL bitwise; otherwise all 9 are re-run",
            stage_b="for each allocation seed: union of the 4 strategies' ids, 5 groupings (grouping seeds 20261111-5, "
                    "moe_calib.multigroup_robustness.grouping_batches, batches of 8), grouping caches by "
                    "batch_sensitivity.forward_plan (reversed-order control), all 4 strategies quantized from the same "
                    "grouping cache; reconstruction on ~/moe_cache test rows; PPL on the same windows",
            fingerprints=pre["fingerprints"]),
        primary=["SC vs FR", "SC vs TR", "SC vs TC"], decision_rule=SCS.DECISION_RULE,
        statistics="paired by allocation seed (3 units); experts are never treated as replicates; Stage B nested "
                   "(seed > grouping), hierarchical bootstrap 20,000 resamples seed 20261120, no flat pooling",
        frozen_code_sha256=hashes(FROZEN_CODE))
    body["protocol_sha256"] = obj_sha(body)
    body["created"] = time.strftime("%F %T")
    jdump(body, p("protocol.json"))
    print("[protocol] frozen", body["protocol_sha256"][:16])


def protocol():
    pr = jload(p("protocol.json"))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    assert hashes(FROZEN_CODE) == pr["frozen_code_sha256"], "frozen solver / evaluation / E3 code changed"
    return pr


# ===================================================================== sketch
def load_partial():
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, LAYERS))
    from moe_calib import model_utils as MU
    t0 = time.time()
    model, _ = MU.load_model()
    return model, time.time() - t0


@torch.no_grad()
def stage_sketch(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib import partial_model_quant as PQ
    protocol()
    layers = [a.layer] if a.layer is not None else LAYERS
    todo = [L for L in layers if not os.path.exists(p("profiling", f"sketch_L{L}.json"))]
    if not todo:
        print("[sketch] done"); return
    os.makedirs(STORE, exist_ok=True)
    model, t_load = load_partial()
    z = np.load(os.path.join(ROOT, ROUTING_NPZ))["C"]
    dims = {"gu": model.config.hidden_size, "down": model.config.moe_intermediate_size}
    k = SETTINGS["sketch_dim"]
    Rp = {q: r.to(MU.GPU) for q, r in EW.projection_matrices(dims, k, SETTINGS["projection_seed"]).items()}
    iu, w = EW.svech_index(k, MU.GPU)
    V = len(w)
    experts = list(range(E))
    for L in todo:
        dig0 = PQ.layer_digest(model, L)
        cache = load_layer_cache(OCACHE, L)
        M = torch.zeros(E, 2, POOL_N, V, dtype=torch.float32)
        n = torch.zeros(E, POOL_N, dtype=torch.int32)
        bufM = torch.zeros(E, 2, V, dtype=torch.float32, device=MU.GPU)
        bufn = torch.zeros(E, dtype=torch.int32, device=MU.GPU)
        Hf = {(e, q): torch.zeros(dims[q], dims[q], dtype=torch.float32, device=MU.GPU) for e in experts for q in EW.PROJS}

        def cb(sid, e, Xg, Xd):
            bufn[e] = Xg.shape[0]
            if Xg.shape[0] == 0:
                return
            for j, (q, X) in enumerate((("gu", Xg), ("down", Xd))):
                bufM[e, j] = EW.svech(EW.sketch_outer(X, Rp[q]), iu, w).float()
                Xf = X.float()
                Hf[(e, q)] += Xf.T @ Xf
        torch.cuda.reset_peak_memory_stats()
        from moe_calib.expertwise_calibration import ExpertCapture
        with MU.layer_on_gpu(model, L):
            plain = [MU.run_layer(model, L, cache[i:i + 1].to(MU.GPU)).cpu() for i in range(8)]
            cap = ExpertCapture(MU.moe_block(model, L), experts, MU.top_k(model), cb).attach()
            sync(); t0 = time.time()
            try:
                for sid in range(POOL_N):
                    cap.cur_seq = sid
                    bufM.zero_(); bufn.zero_()
                    MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
                    cap.finish_sequence()
                    M[:, :, sid] = bufM.cpu()
                    n[:, sid] = bufn.cpu()
            finally:
                cap.detach()
            sync(); t_sk = time.time() - t0
            hooked = [MU.run_layer(model, L, cache[i:i + 1].to(MU.GPU)).cpu() for i in range(8)]
        li = LAYERS.index(L)
        counts_ok = bool(np.array_equal(n.numpy().T, z[:, li * E:(li + 1) * E].astype(np.int64)))
        # direct-accumulation identity: sum_s sketch == svech(R^T H_full R) per (e, p)
        ident = 0.0
        for e in experts:
            for j, q in enumerate(EW.PROJS):
                lhs = M[e, j].double().sum(0)
                RH = EW.svech(Rp[q].double().T @ Hf[(e, q)].double() @ Rp[q].double(), iu, w).cpu()
                ident = max(ident, float(torch.linalg.norm(lhs - RH) / torch.linalg.norm(RH)))
        torch.save(dict(layer=L, M=M, n=n, svech_dim=V, sketch_dim=k, projection_seed=SETTINGS["projection_seed"],
                        dims=dims), st(f"sketch_L{L}.pt"))
        torch.save(dict(layer=L, H={f"{e}.{q}": Hf[(e, q)].cpu() for e in experts for q in EW.PROJS},
                        N={str(e): int(n[e].sum()) for e in experts}), st(f"hfull_L{L}.pt"))
        val = dict(layer=L, n_alignment_checks=cap.n_checks, expected_checks=POOL_N * E,
                   outputs_identical_with_hooks=all(torch.equal(x, y) for x, y in zip(plain, hooked)),
                   counts_equal_routing_cache=counts_ok, sketch_vs_projected_full_hessian_max_rel=ident,
                   sketch_reproduces_direct_accumulation=ident < 1e-4,
                   weights_unchanged=PQ.layer_digest(model, L) == dig0, sketch_finite=bool(torch.isfinite(M).all()),
                   all_experts_have_pool_tokens=bool((n.sum(1) > 0).all()),
                   pool_tokens=dict(min=int(n.sum(1).min()), max=int(n.sum(1).max())),
                   seconds=t_sk, s_per_seq=t_sk / POOL_N, peak_gpu_gb=gpu_peak(), peak_rss_gb=rss_gb(),
                   sketch_file_gb=os.path.getsize(st(f"sketch_L{L}.pt")) / 1e9,
                   hfull_file_gb=os.path.getsize(st(f"hfull_L{L}.pt")) / 1e9, model_load_s=t_load,
                   sketch_sha256=sha256_file(st(f"sketch_L{L}.pt")), created=time.strftime("%F %T"))
        val["passed"] = bool(val["n_alignment_checks"] == val["expected_checks"] and val["outputs_identical_with_hooks"]
                             and counts_ok and val["sketch_reproduces_direct_accumulation"] and val["weights_unchanged"]
                             and val["sketch_finite"] and val["all_experts_have_pool_tokens"])
        jdump(val, p("profiling", f"sketch_L{L}.json"))
        print(f"[sketch] L{L}: passed={val['passed']} {t_sk:.0f}s identity {ident:.1e} counts {counts_ok} "
              f"gpu {val['peak_gpu_gb']:.1f} GB", flush=True)
        if not val["passed"]:
            raise SystemExit(f"sketch L{L} failed checks")
        del M, Hf, cache
        torch.cuda.empty_cache()


def load_stacked(device):
    """[480, N, V] fp32 (term order layer, expert, proj), n [480, N], weights [480]."""
    Ms, ns = [], []
    for L in LAYERS:
        d = torch.load(st(f"sketch_L{L}.pt"), mmap=True)
        Ms.append(d["M"].reshape(E * 2, POOL_N, -1).to(device))
        ns.append(d["n"].double().repeat_interleave(2, dim=0))
    M = torch.cat(Ms)
    n = torch.cat(ns).to(device)
    return M, n, torch.full((M.shape[0],), 1.0 / M.shape[0], dtype=torch.float64)


# ===================================================================== MVP reproduction
def stage_mvpcheck(a):
    out = p("profiling", "mvp_check.json")
    if os.path.exists(out):
        print("[mvpcheck] done"); return
    mvp = torch.load(MVP_STORE)
    ex = mvp["experts"]
    d = torch.load(st("sketch_L16.pt"), mmap=True)
    bit = all(torch.equal(d["M"][e, j], mvp["M"][i, j].float()) for i, e in enumerate(ex) for j in range(2))
    cnt = all(torch.equal(d["n"][e].long(), mvp["n"][i].long()) for i, e in enumerate(ex))
    maxrel = max(float((d["M"][e, j].double() - mvp["M"][i, j]).abs().max() / mvp["M"][i, j].abs().max())
                 for i, e in enumerate(ex) for j in range(2))
    Mn = {(e, q): d["M"][e, j].double() for e in ex for j, q in enumerate(EW.PROJS)}
    nn = {e: d["n"][e].double() for e in ex}
    sel = jload(os.path.join(MVP_DIR, "selections", "seed0.json"))
    rec = {}
    for c in EW.CONDITIONS:
        for e in ex:
            new = EW.expert_mismatch(Mn, nn, e, sel["selection"][c][str(e)])
            old = sel["sketch_mismatch_pre_matching"][c][str(e)]
            rec[f"{c}.{e}"] = dict(mvp=old, new=new, rel=abs(new - old) / old)
    # the same 8-expert objective through the stacked implementation
    T = torch.stack([Mn[(e, q)] for e in ex for q in EW.PROJS])
    nT = torch.stack([nn[e] for e in ex for q in EW.PROJS])
    obj = SCS.StackedObjective(T, nT, torch.full((16,), 1 / 16, dtype=torch.float64))
    sc_mvp = float(np.mean([sel["sketch_mismatch_pre_matching"]["SC"][str(e)] for e in ex]))
    res = dict(experts=ex, sketches_bitwise_equal_to_mvp_fp32=bool(bit), counts_equal=bool(cnt),
               sketch_max_rel_diff_vs_mvp_fp64=maxrel,
               mvp_covariance_scores_max_rel_diff=max(v["rel"] for v in rec.values()), per_set=rec,
               stacked_sc_objective=obj.exact(sel["selection"]["SC"][str(ex[0])]), mvp_sc_objective=sc_mvp)
    res["stacked_vs_mvp_rel"] = abs(res["stacked_sc_objective"] - sc_mvp) / sc_mvp
    res["passed"] = bool(cnt and maxrel < 1e-6 and res["mvp_covariance_scores_max_rel_diff"] < 1e-6
                         and res["stacked_vs_mvp_rel"] < 1e-6)
    jdump(res, out)
    print(f"[mvpcheck] passed={res['passed']} bitwise={bit} max rel {maxrel:.1e} scores "
          f"{res['mvp_covariance_scores_max_rel_diff']:.1e} stacked {res['stacked_vs_mvp_rel']:.1e}")


# ===================================================================== select
def stage_select(a):
    out = p("selections", f"seed{a.seed}.json")
    if os.path.exists(out):
        print(f"[select] seed {a.seed} done"); return
    protocol()
    assert all(jload(p("profiling", f"sketch_L{L}.json"))["passed"] for L in LAYERS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bl = baselines()
    warm = bl[("FR", a.seed)]["warm"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    M, n, w = load_stacked(dev)
    t_load = time.time() - t0
    t0 = time.time()
    obj = SCS.StackedObjective(M, n, w)
    t_init = time.time() - t0
    S = SETTINGS
    t0 = time.time()
    sel, info = SCS.select_shared(obj, warm, B, a.seed, S["greedy_sample_frac"], S["swap_max_passes"], S["swap_rel_tol"])
    t_sel = time.time() - t0
    sel2, info2 = SCS.select_shared(SCS.StackedObjective(M, n, w), warm, B, a.seed, S["greedy_sample_frac"],
                                    S["swap_max_passes"], S["swap_rel_tol"])
    SCS.validate_shared(sel, warm, B, POOL_N, forbidden())
    det = sel == sel2 and info["trace"] == info2["trace"]
    mono = all(y <= x for x, y in zip(info["trace"], info["trace"][1:]))
    pt = {}
    sets = {c: bl[(c, a.seed)]["chosen"] for c in BASE}
    sets["SC"] = sel
    sets["WARM"] = warm
    for c, ids in sets.items():
        d = obj.per_term(ids).cpu().numpy().reshape(4, E, 2)
        pt[c] = dict(objective=float(d.mean()), per_layer={str(L): float(d[i].mean()) for i, L in enumerate(LAYERS)},
                     gu=float(d[..., 0].mean()), down=float(d[..., 1].mean()),
                     per_expert=[float(x) for x in d.mean(2).reshape(-1)])
    ov = {c: SCS.overlap(sel, sets[c]) for c in BASE}
    res = dict(seed=a.seed, chosen=sel, ids_hash=ids_hash(sel), warm=warm, warm_hash=ids_hash(warm),
               deterministic_rerun_identical=bool(det), objective_monotone=bool(mono),
               n_post_warm_from_typical_half=None, optimizer={k: v for k, v in info.items() if k != "greedy_order"},
               greedy_order=info["greedy_order"], sketch_objective=pt, overlap_with_baselines=ov,
               cost=dict(load_s=t_load, init_s=t_init, select_s=t_sel, greedy_s=info["greedy_s"], swap_s=info["swap_s"],
                         peak_gpu_gb=gpu_peak(), peak_rss_gb=rss_gb(), device=dev),
               created=time.strftime("%F %T"))
    typ = set(jload("runs_typicality/pool_typical_half.json")["ids"])
    res["n_post_warm_from_typical_half"] = int(sum(x in typ for x in sel[WARM:]))
    jdump(res, out)
    print(f"[select] seed {a.seed}: det={det} mono={mono} J warm {info['warm_objective']:.4f} greedy "
          f"{info['greedy_objective']:.4f} final {info['final_objective']:.4f} swaps {info['accepted']} "
          f"({t_sel:.0f}s); J FR {pt['FR']['objective']:.4f} TR {pt['TR']['objective']:.4f} "
          f"TC {pt['TC']['objective']:.4f}; overlap FR {ov['FR']['intersection']}")


# ===================================================================== freeze
def stage_freeze(a):
    if os.path.exists(p("freeze.json")):
        print("[freeze] already frozen"); return
    protocol()
    sel = {str(s): jload(p("selections", f"seed{s}.json")) for s in SEEDS}
    assert all(v["deterministic_rerun_identical"] and v["objective_monotone"] for v in sel.values())
    assert all(jload(p("profiling", f"sketch_L{L}.json"))["passed"] for L in LAYERS)
    assert jload(p("profiling", "mvp_check.json"))["passed"]
    jdump(dict(experiment_code_sha256=hashes(EXP_CODE), frozen_code_sha256=hashes(FROZEN_CODE),
               sc_selection_hashes={s: v["ids_hash"] for s, v in sel.items()},
               selection_files_sha256={s: sha256_file(p("selections", f"seed{s}.json")) for s in sel},
               note="frozen after sketch validation and SC selection, BEFORE any SC reconstruction / PPL",
               created=time.strftime("%F %T")), p("freeze.json"))
    print("[freeze] ok")


def check_frozen():
    pr = protocol()
    fz = jload(p("freeze.json"))
    assert hashes(EXP_CODE) == fz["experiment_code_sha256"], "experiment code changed after freeze"
    for s in SEEDS:
        assert sha256_file(p("selections", f"seed{s}.json")) == fz["selection_files_sha256"][str(s)]
    return pr, fz


# ===================================================================== Stage A (ORIGINAL extraction)
def load_full():
    os.environ.pop("MOE_LOAD_LAYERS", None)
    import eval_partial_ppl as EP
    return EP.load_full_model()


def install_and_ppl(model, snap, untouched0, cache_dir, chosen, K, test_ids, win, eval_cache_dir=None):
    """E3 quantize (all 240 experts) + install + checks + PPL + restore. Returns a record."""
    import eval_partial_ppl as EP
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    g = SETTINGS["gptq"]
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot, "model not pristine before quantization"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    q = PQ.quantize_install(model, cache_dir, LAYERS, chosen, test_ids, K, g["bits"], g["group_size"], g["percdamp"],
                            capture=EP.CHECK_EXPERTS, eval_cache_dir=eval_cache_dir)
    t_q = time.time() - t0
    checks = []
    for L, e in EP.CHECK_EXPERTS:
        X = PQ.expert_inputs(model, L, e, load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]])
        ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], snap[(L, e)], X)
        ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                          and ck["installed_vs_e3_rel"] <= EP.INSTALL_TOL(ck))
        checks.append(ck)
    eq_after, _ = PQ.experts_match(model, snap)
    untouched = PQ.untouched_digests(model, LAYERS)
    n_changed = sum(untouched[k] != untouched0[k] for k in untouched0)
    nonfin = PQ.nonfinite_targets(model, LAYERS)
    dig = PQ.installed_digest(model, LAYERS)
    t1 = time.time()
    nll, cnt = EP.layerwise_nll(model, win)
    rec = EP.ppl_record(nll, cnt, time.time() - t1)
    peak = gpu_peak()
    PQ.restore_experts(model, snap)
    eq_rest, tot = PQ.experts_match(model, snap)
    ok = (q["counts"]["n_zero_token_quantization_fallback"] == 0 and q["counts"]["n_experts_quantized_total"] == 240
          and all(ck["pass"] for ck in checks) and eq_after == 0 and n_changed == 0 and not nonfin and eq_rest == tot
          and bool(np.isfinite(rec["per_window_nll"]).all()) and rec["total_tokens"] == EP.N_WIN * (EP.SEQLEN - 1))
    per = [dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"], p90=r["p90"]) for r in q["results"]]
    return dict(per_expert_test=per, recon_test=float(np.mean([r["mean"] for r in per])),
                eval_universe=q["counts"]["eval_universe"], rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"],
                n_quantized=q["counts"]["n_experts_quantized_total"], expert_checks=checks,
                experts_still_pristine_after_install=eq_after, untouched_params_changed=n_changed,
                nonfinite_targets=len(nonfin), restored_pristine=eq_rest == tot, installed_weights_sha256=dig,
                hessian_s=q["hessian_s"], gptq_s=q["gptq_s"], quantize_wall_s=t_q, peak_gpu_gb=peak,
                install_ok=bool(ok), **rec)


def confirm_scores(model, snap, chosen, K):
    import e3_allocation as E3
    from moe_calib import partial_model_quant as PQ
    g = SETTINGS["gptq"]
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot
    t0 = time.time()
    results, counts, _, t_h, t_q = E3.quantize_and_score(
        model, OCACHE, LAYERS, chosen, {}, list(range(256)), K, g["bits"], g["group_size"], p=None, write_back=False,
        percdamp=g["percdamp"], eval_cache_dir=CCACHE)
    return dict(per_expert_conf=[dict(layer=r["layer"], expert=r["expert"], mean=r["mean"]) for r in results],
                recon_conf=float(np.mean([r["mean"] for r in results])), conf_universe=counts["eval_universe"],
                conf_wall_s=time.time() - t0, conf_hessian_s=t_h, conf_gptq_s=t_q)


def stage_evalA(a):
    import eval_partial_ppl as EP
    from moe_calib import partial_model_quant as PQ
    pr, fz = check_frozen()
    win, prot = EP.load_windows()
    todo = []
    for x in a.configs:
        c, s = x.split(":")[0], int(x.split(":")[1])
        if not os.path.exists(p("results", f"A_{c}_s{s}.json")):
            todo.append((c, s))
    if not todo:
        print("[evalA] done"); return
    model, t_load = load_full()
    snap = PQ.snapshot_experts(model, LAYERS)
    untouched0 = PQ.untouched_digests(model, LAYERS)
    test_ids = list(range(2128, 2256))
    for c, s in todo:
        sel = selection(c, s)
        assert ids_hash(sel["chosen"]) == sel["ids_hash"]
        if c == "SC":
            assert sel["ids_hash"] == fz["sc_selection_hashes"][str(s)]
        SCS.validate_shared(sel["chosen"], baselines()[("FR", s)]["warm"], B, POOL_N, forbidden())
        t0 = time.time()
        r = install_and_ppl(model, snap, untouched0, OCACHE, sel["chosen"], sel["K"], test_ids, win)
        r.update(confirm_scores(model, snap, sel["chosen"], sel["K"]))
        r.update(strategy=c, seed=s, ids_hash=sel["ids_hash"], K=sel["K"], extraction="ORIGINAL",
                 windows_sha256=prot["windows_sha256"], model_load_s=t_load, wall_s=time.time() - t0, peak_cpu_gb=rss_gb())
        if c in BASE:                                          # bitwise reproduction of the stored results
            b = baselines()[(c, s)]
            tst = {(x["layer"], x["expert"]): x["mean"] for x in jload(b["test_src"])["per_expert"]}
            cnf = {(x["layer"], x["expert"]): x["mean"] for x in jload(b["conf_src"])["per_expert"]}
            pp = jload(b["ppl_src"])
            r["reproduction"] = dict(
                test_per_expert_bitwise=tst == {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert_test"]},
                conf_per_expert_bitwise=cnf == {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert_conf"]},
                installed_weights_sha256_equal=pp["installed_weights_sha256"] == r["installed_weights_sha256"],
                ppl_equal=pp["ppl"] == r["ppl"], per_window_nll_equal=pp["per_window_nll"] == r["per_window_nll"])
            r["reproduction"]["all"] = all(r["reproduction"].values())
        jdump(r, p("results", f"A_{c}_s{s}.json"))
        print(f"[evalA] {c} s{s}: test {r['recon_test']:.7f} conf {r['recon_conf']:.7f} PPL {r['ppl']:.6f} "
              f"install_ok {r['install_ok']}" + (f" reproduced {r['reproduction']['all']}" if c in BASE else "")
              + f" ({r['wall_s']:.0f}s)", flush=True)
        if not r["install_ok"]:
            raise SystemExit(f"{c} s{s} failed installation checks")


def stage_a_results():
    """{(C, s): dict(per_expert_test, per_expert_conf, recon_test, recon_conf, ppl, per_window_nll, source)}: SC from this
    run; baselines from their stored artifacts only if the validation re-runs reproduced them bitwise, else this run."""
    val = {k: jload(p("results", f"A_{k}.json")) for k in ("FR_s0", "TR_s1", "TC_s2") if os.path.exists(p("results", f"A_{k}.json"))}
    reuse = len(val) == 3 and all(v["reproduction"]["all"] for v in val.values())
    out = {}
    bl = baselines()
    for c in STRATS:
        for s in SEEDS:
            f = p("results", f"A_{c}_s{s}.json")
            if c == "SC" or not reuse:
                if not os.path.exists(f):
                    continue
                r = jload(f)
                out[(c, s)] = dict(per_expert_test=r["per_expert_test"], per_expert_conf=r["per_expert_conf"],
                                   recon_test=r["recon_test"], recon_conf=r["recon_conf"], ppl=r["ppl"],
                                   per_window_nll=r["per_window_nll"], source=os.path.relpath(f, ROOT), cost=r)
            else:
                b = bl[(c, s)]
                t, cf, pp = jload(b["test_src"]), jload(b["conf_src"]), jload(b["ppl_src"])
                out[(c, s)] = dict(per_expert_test=t["per_expert"], per_expert_conf=cf["per_expert"],
                                   recon_test=t["summary"]["recon_mean"], recon_conf=cf["recon_mean"], ppl=pp["ppl"],
                                   per_window_nll=pp["per_window_nll"],
                                   source=f"{b['test_src']} | {b['conf_src']} | {b['ppl_src']} (reused, validated)",
                                   cost=dict(hessian_s=pp["hessian_s"], gptq_s=pp["gptq_s"], eval_s=pp["eval_s"],
                                             peak_gpu_gb=pp["peak_gpu_gb"]))
    return out, reuse


# ===================================================================== diagnostics
@torch.no_grad()
def stage_diag(a):
    out = p("diagnostics", "sets.json")
    if os.path.exists(out):
        print("[diag] done"); return
    check_frozen()
    import report_typicality as RT
    from moe_calib import typicality as TY
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    z = np.load(os.path.join(ROOT, ROUTING_NPZ))
    C = z["C"].astype(float)
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    ref = TY.pool_reference(C, 4, E)
    scores = TY.sequence_scores(C, 4, E, ref)
    sets = {(c, s): selection(c, s)["chosen"] for c in STRATS for s in SEEDS}
    res = {f"{c}_s{s}": {} for c, s in sets}
    for (c, s), ch in sets.items():
        idx = [row[x] for x in ch]
        cnt = C[idx].sum(0)
        m = RT.set_metrics(dict(chosen=ch, selected_counts=cnt.tolist()), C, row, ref, scores, LAYERS, E)
        X = C[idx].reshape(len(idx), 4, E)
        tot = X.sum(0)                                         # [4, E] tokens per expert
        m.update(tokens_per_expert=dict(median=float(np.median(tot)), p10=float(np.quantile(tot, .1)),
                                        p05=float(np.quantile(tot, .05)), min=float(tot.min()), mean=float(tot.mean())),
                 gini_by_layer={str(L): SCS.gini(tot[i]) for i, L in enumerate(LAYERS)},
                 distinct_sources_per_expert_mean=float((X > 0).sum(0).mean()),
                 set_routing_l1_mean=float(np.mean(list(m["set_routing_l1_by_layer"].values()))),
                 d_post_warm_mean=float(scores["d"][idx[WARM:]].mean()))
        res[f"{c}_s{s}"]["routing"] = m
    # sketch covariance mismatch (all sets) from the selection files
    for s in SEEDS:
        so = jload(p("selections", f"seed{s}.json"))["sketch_objective"]
        for c in STRATS:
            res[f"{c}_s{s}"]["sketch"] = so[c]
    # full-dimensional relative Frobenius mismatch per (L, e, p) vs the full-pool reference
    model, _ = load_partial()
    t0 = time.time()
    for L in LAYERS:
        cache = load_layer_cache(OCACHE, L)
        hf = torch.load(st(f"hfull_L{L}.pt"), mmap=True)
        for (c, s), ch in sets.items():
            pr = Profiler(model, L, K=1).run(cache, ch)
            v = {q: [] for q in EW.PROJS}
            for e in range(E):
                for q in EW.PROJS:
                    v[q].append(EW.rel_frob(pr.acc.H[(L, e, q)][0].cpu(), pr.acc.tokens((L, e, q)),
                                            hf["H"][f"{e}.{q}"], float(hf["N"][str(e)])))
            pr.free()
            d = res[f"{c}_s{s}"].setdefault("full", {})
            d[str(L)] = dict(gu=float(np.mean(v["gu"])), down=float(np.mean(v["down"])),
                             per_expert=[0.5 * (x + y) for x, y in zip(v["gu"], v["down"])])
        del cache, hf
        torch.cuda.empty_cache()
        print(f"[diag] full-dim L{L} done ({time.time() - t0:.0f}s)", flush=True)
    for k, v in res.items():
        f = v["full"]
        v["full_summary"] = dict(objective=float(np.mean([0.5 * (f[str(L)]["gu"] + f[str(L)]["down"]) for L in LAYERS])),
                                 gu=float(np.mean([f[str(L)]["gu"] for L in LAYERS])),
                                 down=float(np.mean([f[str(L)]["down"] for L in LAYERS])),
                                 per_layer={str(L): 0.5 * (f[str(L)]["gu"] + f[str(L)]["down"]) for L in LAYERS})
    jdump(dict(sets=res, full_dim_seconds=time.time() - t0, created=time.strftime("%F %T")), out)
    print("[diag] written")


# ===================================================================== Stage B: grouping robustness
def gname(g):
    return f"G{int(g)}"


def gcache_dir(s, g):
    return st("grouping", f"s{s}", gname(g))


def stage_gmanifest(a):
    from moe_calib import multigroup_robustness as MG
    check_frozen()
    path = p("grouping_robustness", "protocol.json")
    seeds, groupings = {}, {}
    for s in SEEDS:
        sel = {c: selection(c, s) for c in STRATS}
        union = sorted(set().union(*[set(sel[c]["chosen"]) for c in STRATS]))
        seeds[str(s)] = dict(union=union, n_union=len(union), union_sha256=MG.obj_sha(union),
                             selection_hashes={c: sel[c]["ids_hash"] for c in STRATS})
        for g in GSEEDS:
            bt = MG.grouping_batches(union, g)
            v = MG.validate_grouping(bt, union, forbidden())
            assert all(v.values()), (s, g, v)
            m = MG.manifest(g, bt, union, {c: sel[c]["chosen"] for c in STRATS})
            m.update(allocation_seed=s, union_sha256=MG.obj_sha(union), selection_hashes=seeds[str(s)]["selection_hashes"],
                     construction="sorted(FR u TR u TC u SC of seed s) -> numpy.random.default_rng(g).permutation -> "
                                  "consecutive batches of 8 (moe_calib.multigroup_robustness.grouping_batches)")
            m["manifest_sha256"] = MG.obj_sha(MG.strip_sha(m))
            m["validation"] = v
            mp = p("grouping_robustness", "manifests", f"s{s}_{gname(g)}.json")
            if os.path.exists(mp):
                assert jload(mp)["manifest_sha256"] == m["manifest_sha256"], f"{mp} changed"
            else:
                jdump(m, mp)
            groupings[f"s{s}_{gname(g)}"] = dict(batches=bt, manifest_sha256=m["manifest_sha256"])
    body = dict(seeds=seeds, groupings=groupings, grouping_seeds=GSEEDS, strategies=list(STRATS),
                selections_frozen_before_stage_b=True, freeze_sha256=sha256_file(p("freeze.json")),
                reconstruction="E3 test rows 2128-2255 of ~/moe_cache (eval_cache_dir) for every configuration",
                ppl="runs_partial_ppl windows, EP.layerwise_nll batch 8")
    if os.path.exists(path):
        old = jload(path)
        assert old["body_sha256"] == obj_sha(body), "Stage B protocol changed"
        print("[gmanifest] already frozen"); return
    jdump(dict(body, body_sha256=obj_sha(body), created=time.strftime("%F %T")), path)
    print("[gmanifest] unions", [seeds[str(s)]["n_union"] for s in SEEDS])


def gprotocol():
    pr = jload(p("grouping_robustness", "protocol.json"))
    body = {k: v for k, v in pr.items() if k not in ("body_sha256", "created")}
    assert obj_sha(body) == pr["body_sha256"], "Stage B protocol modified"
    return pr


def stage_gcache(a):
    import batch_sensitivity as DB
    check_frozen()
    gp = gprotocol()
    s = a.seed
    union = gp["seeds"][str(s)]["union"]
    manp = p("grouping_robustness", f"cache_manifest_s{s}.json")
    man = jload(manp) if os.path.exists(manp) else dict(entries={})
    todo = [g for g in GSEEDS if not man["entries"].get(gname(g), {}).get("complete")]
    if not todo:
        print(f"[gcache] s{s} done"); return
    alltok = DB.all_tokens()
    model, t_load = DB.load_model()
    tsha_ref = DB.tensor_sha(alltok[union])
    for g in todo:
        bt = gp["groupings"][f"s{s}_{gname(g)}"]["batches"]
        d = gcache_dir(s, g)
        os.makedirs(d, exist_ok=True)
        files, keep = {}, {}

        def save(L, X):
            if L not in LAYERS:
                return
            path = os.path.join(d, f"L{L}.pt")
            torch.save(X.clone(), path + ".tmp"); os.replace(path + ".tmp", path)
            keep[L] = X.clone()
            files[str(L)] = dict(shape=list(X.shape), sha256=sha256_file(path), bytes=os.path.getsize(path),
                                 finite=bool(torch.isfinite(X.float()).all()))
        t0 = time.time()
        tsha = DB.forward_plan(model, alltok, bt, union, save)
        t_b = time.time() - t0
        neg = {}

        def cmp(L, X):
            if L in LAYERS:
                neg[str(L)] = bool(torch.equal(X, keep[L]))
        t1 = time.time()
        tsha_r = DB.forward_plan(model, alltok, bt[::-1], union, cmp)
        ent = dict(cache_dir=d, n_rows=len(union), files=files, generation_s=t_b, reversed_control_s=time.time() - t1,
                   reversed_order_bitwise=neg, token_ids_identical=tsha == tsha_ref == tsha_r,
                   manifest_sha256=gp["groupings"][f"s{s}_{gname(g)}"]["manifest_sha256"],
                   storage_bytes=sum(f["bytes"] for f in files.values()))
        ent["complete"] = bool(len(neg) == 4 and all(neg.values()) and ent["token_ids_identical"]
                               and all(f["finite"] for f in files.values()))
        man["entries"][gname(g)] = ent
        man["model_load_s"] = t_load
        jdump(man, manp)
        print(f"[gcache] s{s} {gname(g)}: {t_b:.0f}s, reversed bitwise {all(neg.values())}, tokens "
              f"{ent['token_ids_identical']}", flush=True)
        if not ent["complete"]:
            raise SystemExit("grouping cache control failed")
        del keep


def stage_gquant(a):
    import eval_partial_ppl as EP
    from moe_calib import multigroup_robustness as MG
    from moe_calib import partial_model_quant as PQ
    check_frozen()
    gp = gprotocol()
    s, g = a.seed, a.grouping
    union = gp["seeds"][str(s)]["union"]
    ent = jload(p("grouping_robustness", f"cache_manifest_s{s}.json"))["entries"][gname(g)]
    assert ent["complete"]
    d = gcache_dir(s, g)
    for L in LAYERS:
        assert sha256_file(os.path.join(d, f"L{L}.pt")) == ent["files"][str(L)]["sha256"], "grouping cache changed"
    strats = [a.strategy] if a.strategy else list(STRATS)
    tag = "_CTRL" if a.control else ""
    todo = [c for c in strats if not os.path.exists(p("grouping_robustness", "results", f"{c}_s{s}_{gname(g)}{tag}.json"))]
    if not todo:
        print(f"[gquant] s{s} {gname(g)} done"); return
    win, prot = EP.load_windows()
    model, t_load = load_full()
    snap = PQ.snapshot_experts(model, LAYERS)
    untouched0 = PQ.untouched_digests(model, LAYERS)
    test_ids = list(range(2128, 2256))
    for c in todo:
        sel = selection(c, s)
        assert sel["ids_hash"] == gp["seeds"][str(s)]["selection_hashes"][c]
        chosen = MG.remap(sel["chosen"], union)
        assert [union[i] for i in chosen] == sel["chosen"]
        t0 = time.time()
        r = install_and_ppl(model, snap, untouched0, d, chosen, sel["K"], test_ids, win, eval_cache_dir=OCACHE)
        r.update(strategy=c, seed=s, grouping=gname(g), grouping_seed=g, ids_hash=sel["ids_hash"],
                 manifest_sha256=ent["manifest_sha256"], calibration_cache=d, wall_s=time.time() - t0,
                 model_load_s=t_load, peak_cpu_gb=rss_gb())
        r.pop("per_window_tokens", None)
        if a.control:
            ref = jload(p("grouping_robustness", "results", f"{c}_s{s}_{gname(g)}.json"))
            r["determinism"] = dict(weights_exact=r["installed_weights_sha256"] == ref["installed_weights_sha256"],
                                    ppl_exact=r["ppl"] == ref["ppl"], recon_exact=r["recon_test"] == ref["recon_test"],
                                    per_window_exact=r["per_window_nll"] == ref["per_window_nll"])
            r["determinism"]["reproduced_bitwise"] = all(r["determinism"].values())
        jdump(r, p("grouping_robustness", "results", f"{c}_s{s}_{gname(g)}{tag}.json"))
        print(f"[gquant] {c} s{s} {gname(g)}{tag}: recon {r['recon_test']:.7f} PPL {r['ppl']:.6f} ok {r['install_ok']}"
              + (f" reproduced {r['determinism']['reproduced_bitwise']}" if a.control else "")
              + f" ({r['wall_s']:.0f}s)", flush=True)
        if not r["install_ok"]:
            raise SystemExit("installation checks failed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "protocol", "sketch", "mvpcheck", "select", "freeze",
                                                       "evalA", "diag", "gmanifest", "gcache", "gquant", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--grouping", type=int, default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--configs", nargs="+", default=[])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--hash-caches", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "report":
        import report_shared_covariance_e3 as R
        R.main(); return
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
