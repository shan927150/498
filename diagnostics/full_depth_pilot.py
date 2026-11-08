"""Full-depth routed-expert GPTQ pilot: Full Random s0 vs Full Coverage-LP s0.

All 60 routed experts (gate/up/down) of all 24 MoE layers = 1,440 experts /
4,320 projections are GPTQ-quantized with E3's unchanged solver and frozen
settings (4-bit, g128, percdamp 1.0, act-order), fake-quantized weights stored
in BF16. Attention, routers, shared experts, embeddings, norms and lm_head stay
BF16. This is NOT fully 4-bit LLM quantization.

Calibration is NON-SEQUENTIAL exactly as in E3: every layer's expert Hessians
come from that layer's inputs in the ORIGINAL BF16 model (a compact cache of
only the rows needed: the union of the two calibration sets + the 128 test rows
used for reconstruction scoring). Evaluation is a SEQUENTIAL 24-layer forward on
the frozen 128 x 512 WikiText-2 windows of runs_partial_ppl.

Stages (--stage):
  preflight   resources, model config, selections, windows, pristine digests,
              frozen decision rule -> protocol.json, selected_sets.json
  cache       BF16 layer inputs for all 24 layers -> ~/moe_cache_full_depth_pilot,
              cache_manifest.json (incl. comparison with ~/moe_cache for L1/8/16/23)
  regression  four-layer FR/FC s0 through the updated pipeline, with the original
              and with the new cache -> regression_checks.json
  quant       --strategy FR|FC: fresh process, all 24 layers, resumable per layer,
              PPL -> full_random_s0.json / full_coverage_s0.json
  report      summary.{md,json}, figures
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, resource, shutil, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP          # noqa: E402  (windows, scorer, manifest, tolerance)

OUT = "runs_full_depth_pilot"
# v2: built with the ORIGINAL cache_inputs.py batch composition (batches of 8 consecutive original rows
# from row 0), which reproduces ~/moe_cache bitwise; v1 (compact batches) is kept only as evidence
# (logs/regression_attempt1_FAILED_v1cache.json).
PCACHE = os.path.expanduser(os.environ.get("FULL_DEPTH_CACHE", "~/moe_cache_full_depth_pilot_v2"))
OCACHE = os.path.expanduser("~/moe_cache")
STRATS = {"FR": "full_random_s0", "FC": "full_coverage_s0"}
SEED = 0
OLD_LAYERS = [1, 8, 16, 23]
CHECK_EXPERTS = [(0, 0), (0, 30), (11, 0), (11, 30), (23, 0), (23, 30)]     # early / middle / late
N_BOOT, BOOT_SEED = 5000, 20261109
CACHE_BATCH = 8
DECISION_RULE = {
    "statistic": "Delta_24 = PPL(FC24) - PPL(FR24); paired window bootstrap (5,000 joint resamples of the 128 "
                 "windows, seed 20261109) of the token-weighted mean-NLL difference FC24 - FR24",
    "Delta_4": "PPL(FC4) - PPL(FR4) from runs_partial_ppl (seed 0)",
    "tau": "1e-4 x PPL_BF16 (numerical floor; earlier evaluations were bitwise repeatable)",
    "FULL-DEPTH-UNSTABLE": "any non-finite weight / NLL, or PPL(FR24) or PPL(FC24) > 2 x PPL_BF16",
    "FULL-DEPTH-PENALTY-AMPLIFIED": "Delta_24 > tau AND bootstrap 95 % CI of the NLL difference > 0 AND "
                                    "Delta_24 >= 2 x Delta_4",
    "FULL-DEPTH-PENALTY-PERSISTS": "Delta_24 > tau AND bootstrap CI > 0 AND Delta_24 < 2 x Delta_4",
    "FULL-DEPTH-PENALTY-NOT-OBSERVED": "Delta_24 <= tau OR the bootstrap CI includes 0 (negligible or reversed)",
    "RUN-BLOCKED": "any stage fails a correctness check or a configuration is missing",
}


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def jdump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------- statistics
def paired_bootstrap(nll_a, nll_b, tok, n_boot=N_BOOT, seed=BOOT_SEED):
    """Joint resampling of windows; statistic = token-weighted mean NLL(a) - mean NLL(b)."""
    a, b, t = (np.asarray(x, float) for x in (nll_a, nll_b, tok))
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(a), size=(n_boot, len(a)))
    T = t[idx].sum(1)
    d = a[idx].sum(1) / T - b[idx].sum(1) / T
    point = a.sum() / t.sum() - b.sum() / t.sum()
    return dict(point=float(point), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                n_boot=n_boot, seed=seed)


def decide(bf16, fr24, fc24, d4, ci, finite):
    tau = 1e-4 * bf16
    if not finite or fr24 > 2 * bf16 or fc24 > 2 * bf16:
        return "FULL-DEPTH-UNSTABLE"
    d24 = fc24 - fr24
    if d24 <= tau or ci[0] <= 0:
        return "FULL-DEPTH-PENALTY-NOT-OBSERVED"
    return "FULL-DEPTH-PENALTY-AMPLIFIED" if d24 >= 2 * d4 else "FULL-DEPTH-PENALTY-PERSISTS"


# ---------------------------------------------------------------- shared loading
def load_selections():
    man = json.load(open(os.path.join(ROOT, EP.OUT, "selected_set_manifest.json")))
    import e3_allocation as E3
    sel = {}
    for c in STRATS:
        ent = man["configs"][f"{c}_s{SEED}"]
        src = os.path.join(ROOT, ent["source"])
        r = json.load(open(src))
        assert sha256_file(src) == ent["source_sha256"], f"{src} changed since the partial-PPL manifest"
        assert r["chosen"] == ent["chosen"] and E3.warm_hash(r["chosen"]) == ent["ids_hash"]
        assert ent["valid"] and all(ent["checks"].values())
        sel[c] = dict(ent, chosen=r["chosen"], warm_start=r["warm_start"])
    return man, sel


def compact_rows(sel, test_ids):
    """Sorted union of the two calibration sets, then the test rows."""
    union = sorted(set(sel["FR"]["chosen"]) | set(sel["FC"]["chosen"]))
    rows = union + list(test_ids)
    assert len(set(rows)) == len(rows)
    return rows


def load_model_full():
    os.environ.pop("MOE_LOAD_LAYERS", None)
    from moe_calib import model_utils as MU
    t0 = time.time()
    model, tok = MU.load_model()
    return model, time.time() - t0


# ---------------------------------------------------------------- stage: preflight
def stage_preflight(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    os.makedirs(p("logs"), exist_ok=True)
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    man, sel = load_selections()
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    assert base["protocol_windows_sha256"] == prot["windows_sha256"]
    free_disk = shutil.disk_usage(os.path.dirname(PCACHE)).free
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    avail_gb = int(mem["MemAvailable"].split()[0]) / 1e6
    gpu_free, gpu_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    model, t_load = load_model_full()
    layers = PQ.resolve_layers("all", model)
    E, K = MU.num_experts(model), MU.top_k(model)
    assert len(layers) == MU.num_layers(model) == 24 and E == 60 and K == 4
    t0 = time.time()
    pristine_targets = PQ.target_tensor_digests(model, layers)
    pristine_untouched = PQ.untouched_digests(model, layers)
    t_digest = time.time() - t0
    assert len(pristine_targets) == 4320
    split = cfg["split"]
    test_ids = list(range(*split["test"]))
    rows = compact_rows(sel, test_ids)
    n_bytes_layer = len(rows) * split["seqlen"] * MU.expert_dims(model)[0] * 2
    jdump(dict(pristine_targets=pristine_targets, pristine_untouched=pristine_untouched), p("pristine_digests.json"))
    selected = {c: dict(name=STRATS[c], ids_hash=s["ids_hash"], chosen=s["chosen"], warm_start=s["warm_start"],
                        n=len(s["chosen"]), unique=len(set(s["chosen"])), source=s["source"],
                        source_sha256=s["source_sha256"], checks=s["checks"], recon_test=s["recon_test"],
                        recon_confirmation=s["recon_confirmation"], K=s["K"]) for c, s in sel.items()}
    jdump(selected, p("selected_sets.json"))
    proto = dict(
        experiment="Full-depth routed-expert GPTQ pilot (NOT fully 4-bit LLM quantization)",
        model=MU.MODEL_NAME, model_snapshot=os.path.basename(MU._snapshot_dir(MU.MODEL_NAME)),
        gptq=cfg["gptq"], B=128, seed=SEED, seqlen=split["seqlen"], quant_layers=layers,
        n_experts=len(layers) * E, n_projections=len(pristine_targets),
        untouched="attention, routers (mlp.gate), shared experts (+ gate), embeddings, norms, lm_head: BF16",
        calibration="non-sequential: Hessians of layer L from ORIGINAL BF16 inputs of layer L (compact cache)",
        evaluation="sequential 24-layer forward; frozen runs_partial_ppl windows",
        windows_sha256=prot["windows_sha256"], bf16_ppl=base["ppl"],
        partial_reference={c: json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{c}_s0.json")))["ppl"] for c in STRATS},
        compact_rows=rows, n_rows=len(rows), check_experts=CHECK_EXPERTS,
        install_tolerance="installed_vs_e3_rel <= weight_rounding_only_rel + 1.5 x bf16_compute_noise_rel "
                          "(partial-PPL rule) and weights bitwise BF16(W_q)",
        decision_rule=DECISION_RULE,
        resources=dict(host_mem_available_gb=avail_gb, gpu_free_gb=gpu_free / 1e9, gpu_total_gb=gpu_total / 1e9,
                       disk_free_tb=free_disk / 1e12, model_load_s=t_load, digest_s=t_digest,
                       estimates=dict(cache_gb=24 * n_bytes_layer / 1e9,
                                      installed_weights_gb_per_strategy=4320 * 2048 * 1408 * 2 / 1e9,
                                      model_cpu_gb=28.6, gptq_gpu_gb="~8 (one layer + one expert's Hessians)")),
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        assert old["decision_rule"] == proto["decision_rule"] and old["compact_rows"] == rows, "protocol changed"
        print("protocol already frozen; consistent"); return
    jdump(proto, p("protocol.json"))
    print(f"preflight ok: 24 MoE layers x 60 experts; {len(rows)} cache rows ({24 * n_bytes_layer / 1e9:.1f} GB); "
          f"RAM avail {avail_gb:.0f} GB, GPU free {gpu_free / 1e9:.1f} GB, disk {free_disk / 1e12:.0f} TB")


# ---------------------------------------------------------------- stage: cache
def stage_cache(a):
    from moe_calib import model_utils as MU
    proto = json.load(open(p("protocol.json")))
    rows = proto["compact_rows"]
    os.makedirs(PCACHE, exist_ok=True)
    man_path = p("cache_manifest.json")
    if os.path.exists(man_path):
        m = json.load(open(man_path))
        ok = all(os.path.exists(os.path.join(PCACHE, f"L{L}.pt")) for L in range(24)) and m.get("complete")
        if ok:
            print("cache complete (verify with --verify-cache)"); return
    ids = torch.load(os.path.join(OCACHE, "ids.pt"))
    meta = json.load(open(os.path.join(OCACHE, "meta.json")))
    P = meta["pool_size"]
    alltok = torch.cat([ids["pool"], ids["holdout"]])
    B = meta.get("batch", CACHE_BATCH)                      # cache_inputs.py default: 8
    # every needed row is forwarded inside its ORIGINAL batch (rows B*g .. B*g+B-1), as cache_inputs.py did
    groups = sorted({r // B for r in rows})
    grows = [g * B + j for g in groups for j in range(B) if g * B + j < len(alltok)]
    pos = {r: i for i, r in enumerate(grows)}
    take = torch.tensor([pos[r] for r in rows])
    tok = alltok[grows]
    model, t_load = load_model_full()
    N, T = tok.shape
    t0 = time.time()
    hs = torch.empty(N, T, MU.expert_dims(model)[0], dtype=MU.DTYPE)
    for i in range(0, N, B):
        hs[i:i + B] = MU.embed(model, tok[i:i + B]).cpu()
    files = {}
    for L in range(MU.num_layers(model)):
        path = os.path.join(PCACHE, f"L{L}.pt")
        tmp = path + ".tmp"
        sub = hs[take].clone()
        torch.save(sub, tmp)
        os.replace(tmp, path)
        files[str(L)] = dict(shape=list(sub.shape), dtype=str(sub.dtype), sha256=sha256_file(path))
        if L < MU.num_layers(model) - 1:
            with MU.layer_on_gpu(model, L):
                for i in range(0, N, B):
                    hs[i:i + B] = MU.run_layer(model, L, hs[i:i + B].to(MU.GPU)).cpu()
        print(f"[cache] layer {L:2d} saved and forwarded ({time.time() - t0:.0f} s)", flush=True)
    secs = time.time() - t0
    # agreement with the frozen E3 cache for the four original layers (same original row ids)
    comp = {}
    for L in OLD_LAYERS:
        old = torch.load(os.path.join(OCACHE, f"L{L}.pt"), mmap=True)
        new = torch.load(os.path.join(PCACHE, f"L{L}.pt"), mmap=True)
        mx, eq, rel = 0.0, 0, []
        for j, r in enumerate(rows):
            o, n_ = old[r].float(), new[j].float()
            mx = max(mx, float((o - n_).abs().max()))
            eq += bool(torch.equal(old[r], new[j]))
            rel.append(float((o - n_).norm() / o.norm()))
        comp[str(L)] = dict(rows_bitwise_equal=eq, rows=len(rows), max_abs_diff=mx, max_rel_fro=max(rel),
                            mean_rel_fro=float(np.mean(rel)))
        print(f"[cache] L{L} vs ~/moe_cache: {eq}/{len(rows)} rows bitwise equal, max abs {mx:.3g}, "
              f"max rel {max(rel):.3g}", flush=True)
    jdump(dict(cache_dir=PCACHE, construction="original batch composition: each needed row forwarded in its "
               f"original batch of {B} consecutive rows (as cache_inputs.py); {len(groups)} batches, {N} rows forwarded",
               rows=rows, row_semantics="original ~/moe_cache row ids (pool 0-1999, holdout "
               "2000-2255); token ids taken from ~/moe_cache/ids.pt", n_rows=len(rows),
               layers=list(range(24)), files=files, batch=B, seconds=secs, model_load_s=t_load,
               model_snapshot=proto["model_snapshot"], comparison_with_original_cache=comp,
               calib_hashes={c: s["ids_hash"] for c, s in json.load(open(p("selected_sets.json"))).items()},
               disk_gb=sum(os.path.getsize(os.path.join(PCACHE, f"L{L}.pt")) for L in range(24)) / 1e9,
               complete=True), man_path)


def verify_cache():
    m = json.load(open(p("cache_manifest.json")))
    for L, f in m["files"].items():
        assert sha256_file(os.path.join(PCACHE, f"L{L}.pt")) == f["sha256"], f"cache L{L} corrupted"
    return m


# ---------------------------------------------------------------- quantization helpers
def remap(ids, rows):
    r = {s: i for i, s in enumerate(rows)}
    return [r[s] for s in ids]


def quant_layers(model, layers, chosen, eval_ids, cache_dir, K, cfg, capture=()):
    from moe_calib import partial_model_quant as PQ
    g = cfg["gptq"]
    return PQ.quantize_install(model, cache_dir, layers, chosen, eval_ids, K, g["bits"], g["group_size"],
                               g["percdamp"], capture=capture)


# ---------------------------------------------------------------- stage: regression
def stage_regression(a):
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_meta, split_ids
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    proto = json.load(open(p("protocol.json")))
    sel = json.load(open(p("selected_sets.json")))
    rows = proto["compact_rows"]
    win, prot = EP.load_windows()
    verify_cache()
    model, t_load = load_model_full()
    snap = PQ.snapshot_experts(model, OLD_LAYERS)
    meta = load_meta(OCACHE)
    test_ids = split_ids(meta, cfg["split"]["tune"][1] - cfg["split"]["tune"][0])["test"]
    out = dict(layers=OLD_LAYERS, configs={})
    for c, name in STRATS.items():
        ref = json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{c}_s0.json")))
        res = {}
        for which, cdir, ch, ev in (("original_cache", OCACHE, sel[c]["chosen"], test_ids),
                                    ("pilot_cache", PCACHE, remap(sel[c]["chosen"], rows), remap(test_ids, rows))):
            assert PQ.experts_match(model, snap)[0] == len(OLD_LAYERS) * 60 * 3
            t0 = time.time()
            q = quant_layers(model, OLD_LAYERS, ch, ev, cdir, sel[c]["K"], cfg)
            tq = time.time() - t0
            dig = PQ.installed_digest(model, OLD_LAYERS)
            nll, cnt = EP.layerwise_nll(model, win)
            rec = EP.ppl_record(nll, cnt, 0.0)
            recon = float(np.mean([r["mean"] for r in q["results"]]))
            res[which] = dict(installed_sha256=dig, digest_equals_partial_ppl=dig == ref["installed_weights_sha256"],
                              ppl=rec["ppl"], ppl_partial_reference=ref["ppl"], abs_diff_ppl=abs(rec["ppl"] - ref["ppl"]),
                              recon=recon, recon_reference=ref["recon_recomputed"], quantize_s=tq,
                              per_window_nll=rec["per_window_nll"], rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"])
            PQ.restore_experts(model, snap)
            print(f"[regression] {c} {which}: PPL {rec['ppl']:.6f} (ref {ref['ppl']:.6f}), weights identical to "
                  f"partial-PPL run: {res[which]['digest_equals_partial_ppl']}", flush=True)
        out["configs"][c] = res
    d4_ref = out["configs"]["FC"]["original_cache"]["ppl_partial_reference"] - out["configs"]["FR"]["original_cache"]["ppl_partial_reference"]
    d4_new = out["configs"]["FC"]["pilot_cache"]["ppl"] - out["configs"]["FR"]["pilot_cache"]["ppl"]
    orig_exact = all(out["configs"][c]["original_cache"]["digest_equals_partial_ppl"] and
                     out["configs"][c]["original_cache"]["abs_diff_ppl"] == 0.0 for c in STRATS)
    pilot_ok = all(out["configs"][c]["pilot_cache"]["abs_diff_ppl"] <= 0.1 * abs(d4_ref) for c in STRATS)
    out.update(delta4_reference=d4_ref, delta4_pilot_cache=d4_new, original_cache_exact=orig_exact,
               pilot_cache_tolerance="|PPL(pilot cache) - PPL(original cache)| <= 0.1 x |Delta_4| per strategy",
               pilot_cache_ok=pilot_ok, passed=orig_exact and pilot_ok)
    jdump(out, p("regression_checks.json"))
    if not out["passed"]:
        raise SystemExit("four-layer regression FAILED; not proceeding to full depth")
    print(f"regression passed: Delta_4 reference {d4_ref:+.6f}, with pilot cache {d4_new:+.6f}")


# ---------------------------------------------------------------- stage: quant (full depth)
def stage_quant(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    c = a.strategy
    name = STRATS[c]
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    proto = json.load(open(p("protocol.json")))
    reg = json.load(open(p("regression_checks.json")))
    assert reg["passed"], "four-layer regression has not passed"
    sel = json.load(open(p("selected_sets.json")))[c]
    pr = json.load(open(p("pristine_digests.json")))
    rows = proto["compact_rows"]
    chosen = remap(sel["chosen"], rows)
    test_ids = list(range(*cfg["split"]["test"]))
    ev = remap(test_ids, rows)
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == proto["windows_sha256"]
    verify_cache()
    wdir = os.path.join(PCACHE, "weights", name)
    ldir = p("layers", name)
    os.makedirs(wdir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    model, t_load = load_model_full()
    layers = PQ.resolve_layers("all", model)
    assert layers == proto["quant_layers"]
    # pristine verification (fresh process: nothing quantized yet)
    t0 = time.time()
    tg = PQ.target_tensor_digests(model, layers)
    assert tg == pr["pristine_targets"], "target weights differ from the pretrained checkpoint"
    assert PQ.untouched_digests(model, layers) == pr["pristine_untouched"], "untouched params differ"
    t_verify = time.time() - t0
    orig_check = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()}
                  for L, e in CHECK_EXPERTS}
    torch.cuda.reset_peak_memory_stats()
    per_layer, t_q0 = [], time.time()
    for L in layers:
        mpath, wpath = os.path.join(ldir, f"L{L}.json"), os.path.join(wdir, f"L{L}.pt")
        if os.path.exists(mpath) and os.path.exists(wpath):
            m = json.load(open(mpath))
            good = (m.get("complete") and m["strategy"] == c and m["ids_hash"] == sel["ids_hash"] and
                    m["gptq"] == cfg["gptq"] and sha256_file(wpath) == m["file_sha256"])
            if good:
                PQ.install_layer_file(model, L, wpath)
                assert PQ.layer_digest(model, L) == m["layer_sha256"], f"resumed layer {L} digest mismatch"
                per_layer.append(dict(m, resumed=True)); print(f"[quant {c}] L{L:2d} resumed (verified)", flush=True)
                continue
            print(f"[quant {c}] L{L:2d} artifact invalid; recomputing", flush=True)
        cap = [(LL, e) for LL, e in CHECK_EXPERTS if LL == L]
        t1 = time.time()
        q = quant_layers(model, [L], chosen, ev, PCACHE, sel["K"], cfg, capture=cap)
        tq = time.time() - t1
        n = q["counts"]
        assert n["n_experts_quantized_total"] == 60, n
        checks = []
        if cap:
            cache = load_layer_cache(PCACHE, L)
            for LL, e in cap:
                X = PQ.expert_inputs(model, LL, e, cache[ev[:EP.N_CHECK_SEQS]])
                ck = PQ.expert_output_check(model, LL, e, q["captured"][(LL, e)], orig_check[(LL, e)], X)
                ck["tolerance"] = EP.INSTALL_TOL(ck)
                ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                                  and ck["installed_vs_e3_rel"] <= ck["tolerance"])
                checks.append(ck)
        tokens = float(sum(r["tokens"] for r in q["results"]))
        if n["n_experts_test_observed"] == 60:              # every routed token of every calibration sequence
            assert tokens == len(chosen) * proto["seqlen"] * 4, (L, tokens)
        ok = (n["n_zero_token_quantization_fallback"] == 0 and all(ck["pass"] for ck in checks)
              and not PQ.nonfinite_targets(model, [L]))
        PQ.save_layer(model, L, wpath)
        m = dict(strategy=c, seed=SEED, ids_hash=sel["ids_hash"], model_snapshot=proto["model_snapshot"],
                 layer=L, gptq=cfg["gptq"], n_experts=60, n_projections=180,
                 rtn_fallbacks=n["n_zero_token_quantization_fallback"], n_experts_scored=n["n_experts_test_observed"],
                 recon_mean=float(np.mean([r["mean"] for r in q["results"]])) if q["results"] else None,
                 per_expert=[dict(expert=r["expert"], tokens=r["tokens"], mean=r["mean"]) for r in q["results"]],
                 calib_routed_tokens_scored_experts=tokens, expert_checks=checks,
                 layer_sha256=PQ.layer_digest(model, L), file_sha256=sha256_file(wpath),
                 hessian_s=q["hessian_s"], gptq_s=q["gptq_s"], wall_s=tq, complete=bool(ok))
        jdump(m, mpath)
        if not ok:
            raise SystemExit(f"layer {L} failed its checks; run invalid")
        per_layer.append(m)
        print(f"[quant {c}] L{L:2d}: recon {m['recon_mean']:.6f}, {tq:.0f} s, fallbacks 0, checks "
              f"{len(checks)} ok", flush=True)
    t_quant = time.time() - t_q0
    # post-quantization verification of the whole model
    after = PQ.target_tensor_digests(model, layers)
    unchanged = [k for k in after if after[k] == pr["pristine_targets"][k]]
    nonfin = PQ.nonfinite_targets(model, layers)
    untouched_ok = PQ.untouched_digests(model, layers) == pr["pristine_untouched"]
    dig = PQ.installed_digest(model, layers)
    assert not unchanged and not nonfin and untouched_ok, (len(unchanged), nonfin[:3], untouched_ok)
    evals = []
    for r in range(2):                                       # determinism check
        t1 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        evals.append(EP.ppl_record(nll, cnt, time.time() - t1))
    rec = evals[0]
    finite = bool(np.isfinite(rec["per_window_nll"]).all())
    out = dict(strategy=c, name=name, seed=SEED, ids_hash=sel["ids_hash"],
               scope="full-depth routed-expert GPTQ (fake-quantized BF16 weights); attention/routers/shared "
                     "experts/embeddings/norms/lm_head BF16", quant_layers=layers, n_quantized_layers=len(layers),
               n_quantized_experts=60 * len(per_layer), n_quantized_projections=len(after),
               projections_unchanged_from_bf16=len(unchanged), nonfinite_weights=len(nonfin),
               untouched_params_unchanged=untouched_ok, n_untouched_params=len(pr["pristine_untouched"]),
               rtn_fallbacks=sum(m["rtn_fallbacks"] for m in per_layer),
               expert_checks=[ck for m in per_layer for ck in m["expert_checks"]],
               installed_weights_sha256=dig, windows_sha256=prot["windows_sha256"], finite_nll=finite,
               repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
               recon_by_layer={str(m["layer"]): m["recon_mean"] for m in per_layer},
               model_load_s=t_load, pristine_verify_s=t_verify, quantize_total_s=t_quant,
               hessian_total_s=sum(m["hessian_s"] for m in per_layer), gptq_total_s=sum(m["gptq_s"] for m in per_layer),
               resumed_layers=sum(1 for m in per_layer if "resumed" in m),
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=peak_cpu_gb(),
               weight_artifacts_gb=sum(os.path.getsize(os.path.join(wdir, f"L{L}.pt")) for L in layers) / 1e9,
               **rec)
    jdump(out, p(f"{name}.json"))
    print(f"[quant {c}] full depth: PPL {rec['ppl']:.6f} (repeat |d| {out['repeat_eval_abs_diff_ppl']:.1e}); "
          f"quantize {t_quant / 60:.1f} min; peak GPU {out['peak_gpu_gb']:.1f} GB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "cache", "regression", "quant", "report"])
    ap.add_argument("--strategy", choices=list(STRATS))
    ap.add_argument("--quant-layers", default="all", help="only 'all' is used by this pilot")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "report":
        import report_full_depth_pilot as R
        R.main(); return
    dict(preflight=stage_preflight, cache=stage_cache, regression=stage_regression, quant=stage_quant)[a.stage](a)


if __name__ == "__main__":
    main()
