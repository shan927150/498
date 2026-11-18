"""Family D (calibration batch grouping) frozen-routing causal intervention.

Reuses the validated Family A machinery (moe_calib/inference_routing_override.py via
diagnostics/inference_routing_amplification.py) and the Family A BF16 reference routing, unchanged.

Design (from the repository, runs_cross_seed_grouping + runs_multigroup_robustness):
  15 comparisons = 3 allocation seeds {0,1,2} x 5 groupings G20261111..15; FR and FC paired by (seed, grouping);
  groupings are nested within seeds (historical analysis rejects flat pooling).
  Seed-0 ORIGINAL-extraction pair (CTRL_*_ORIGINAL = Family A FR/FC_ORIGINAL) is reused from Family A.
  Seed-1/2 ORIGINAL-extraction pairs (runs_partial_ppl) have no stored weights -> natural sign only.

Stages: audit | protocol | gpu | weights | analyze   (python diagnostics/grouping_routing_amplification.py --stage S)
"""
from __future__ import annotations
import argparse, gc, hashlib, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP                                    # noqa: E402
import inference_routing_amplification as FA                     # noqa: E402  (Family A driver, reused)

OUT = "runs_grouping_routing_amplification"
CACHE = os.path.expanduser(os.environ.get("GRA_CACHE", "~/moe_cache_grouping_routing"))
SEEDS = (0, 1, 2)
GROUPINGS = ("G20261111", "G20261112", "G20261113", "G20261114", "G20261115")
STRATS = ("FR", "FC")
PARITY_CONFIGS = ("FR_s0_G20261111", "FC_s2_G20261111")          # prespecified: first FR config, first FC of last seed
LAYERS_Q, PROJ = FA.LAYERS_Q, FA.PROJ
FA_REF = os.path.expanduser("~/moe_cache_inference_routing/bf16_reference_routing.pt")
FA_REF_MAN = "runs_inference_routing_amplification/bf16_routing_manifest.json"
jdump, jload, sha256_file, obj_sha = FA.jdump, FA.jload, FA.sha256_file, FA.obj_sha


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def c(*a):
    return os.path.join(CACHE, *a)


def cname(s, g):
    return f"s{s}_{g}"


def name(st, s, g):
    return f"{st}_s{s}_{g}"


def result_src(st, s, g):
    if s == 0:
        return f"runs_multigroup_robustness/results/{st}_{g}.json"
    return f"runs_cross_seed_grouping/results/{st}_s{s}_{g}.json"


CONFIGS = [name(st, s, g) for s in SEEDS for g in GROUPINGS for st in STRATS]


def run_file(n, mode):
    return c("runs", f"{n}_{mode}.pt")


def res_file(n, mode):
    return p("results", f"{n}_{mode}.json")


# ---------------------------------------------------------------- audit
def audit_one(n, st, s, g, bf16, sel, prot_cs):
    t0 = time.time()
    src = result_src(st, s, g)
    r = jload(os.path.join(ROOT, src))
    ck = dict(result_exists=r is not None)
    if r is None:
        return dict(name=n, valid=False, checks=ck, missing=src)
    wd = r["weights_dir"]
    ck.update(result_valid_flag=bool(r.get("valid")), strategy=r.get("strategy") == st,
              grouping=r.get("grouping") == g, gptq=r.get("gptq") == dict(bits=4, group_size=128, percdamp=1.0, act_order=True),
              layers=r.get("layers") == LAYERS_Q, n_quantized_experts_240=r.get("n_quantized_experts") == 240,
              n_projection_tensors_720=r.get("n_projection_tensors") == 720, rtn_fallbacks_0=r.get("rtn_fallbacks") == 0,
              selection_ids_hash=r.get("ids_hash") == sel[f"s{s}"][st]["ids_hash"],
              tokens_65408=r.get("total_tokens") == 65408 and len(r.get("per_window_nll", [])) == 128)
    if s > 0:
        ck["allocation_seed"] = r.get("allocation_seed") == s
        ck["grouping_manifest_sha256"] = r.get("manifest_sha256") == prot_cs["groupings"][cname(s, g)]["manifest_sha256"]
    with ThreadPoolExecutor(4) as ex:
        reads = dict(zip(LAYERS_Q, ex.map(FA._read, [os.path.join(wd, f"L{L}.pt") for L in LAYERS_Q])))
    files, n_t, n_shape, n_bf, n_fin, n_diff = {}, 0, 0, 0, 0, 0
    h = hashlib.sha256()
    for L in LAYERS_Q:
        fsha, b = reads[L]
        d = torch.load(io.BytesIO(b))
        del b
        exp = r["weight_files"][str(L)]
        keys_ok = set(d) == {f"{e}.{k}" for e in range(60) for k in PROJ}
        for e in range(60):
            for k in PROJ:
                t = d[f"{e}.{k}"]
                n_t += 1
                n_shape += tuple(t.shape) == FA.SHAPES[k]
                n_bf += t.dtype == torch.bfloat16
                n_fin += bool(torch.isfinite(t.float()).all())
                n_diff += not torch.equal(t, bf16[(L, e, k)])
                h.update(t.contiguous().view(torch.int16).numpy().tobytes())
        lsha = FA.layer_sha_of(d)
        del d
        files[str(L)] = dict(path=os.path.join(wd, f"L{L}.pt"), sha256=fsha, sha256_match=fsha == exp["sha256"],
                             layer_sha256=lsha, layer_sha256_match=lsha == exp["layer_sha256"], keys_complete=keys_ok)
    inst = h.hexdigest()
    ck.update(file_sha256_all_match=all(f["sha256_match"] for f in files.values()),
              layer_sha256_all_match=all(f["layer_sha256_match"] for f in files.values()),
              keys_complete_240_experts=all(f["keys_complete"] for f in files.values()),
              installed_weights_sha256_match=inst == r["installed_weights_sha256"], n_tensors_720=n_t == 720,
              shapes_ok_720=n_shape == 720, dtype_bf16_720=n_bf == 720, finite_720=n_fin == 720,
              all_720_differ_from_bf16_checkpoint=n_diff == 720)
    gc.collect()
    return dict(name=n, strategy=st, allocation_seed=s, grouping=g, result=src, result_sha256=sha256_file(os.path.join(ROOT, src)),
                weights_dir=wd, files=files, installed_weights_sha256=inst, ids_hash=r.get("ids_hash"),
                historical_ppl=r["ppl"], historical_per_window_nll=r["per_window_nll"], checks=ck,
                valid=all(ck.values()), audit_s=time.time() - t0)


def stage_audit(a):
    os.makedirs(p("logs"), exist_ok=True)
    bf16, snap = FA._bf16_checkpoint_experts()
    sel = jload(os.path.join(ROOT, "runs_cross_seed_grouping/selected_set_manifest.json"))
    prot_cs = jload(os.path.join(ROOT, "runs_cross_seed_grouping/protocol.json"))
    inv = {}
    for s in SEEDS:
        for g in GROUPINGS:
            for st in STRATS:
                n = name(st, s, g)
                inv[n] = audit_one(n, st, s, g, bf16, sel, prot_cs)
                print(f"[audit] {n}: valid={inv[n]['valid']} ({inv[n].get('audit_s', 0):.0f} s) "
                      f"{[k for k, v in inv[n]['checks'].items() if not v]}", flush=True)
    # cross-config integrity: no duplicates / substitution, historical paired table reproduces
    shas = [v["installed_weights_sha256"] for v in inv.values() if v.get("valid")]
    hist_pairs = jload(os.path.join(ROOT, "runs_cross_seed_grouping/paired_differences.json"))["pairs"]
    hist_ppl = jload(os.path.join(ROOT, "runs_cross_seed_grouping/ppl_results.json"))["configs"]
    pair_ok = {}
    for s in SEEDS:
        for g in GROUPINGS:
            fr, fc = inv[name("FR", s, g)], inv[name("FC", s, g)]
            hp = hist_pairs[cname(s, g)]
            pair_ok[cname(s, g)] = dict(
                delta_reproduces=fc["historical_ppl"] - fr["historical_ppl"] == hp["delta_ppl"],
                same_seed_grouping=fr["allocation_seed"] == fc["allocation_seed"] and fr["grouping"] == fc["grouping"],
                ppl_results_sha_match=all(hist_ppl[k]["installed_weights_sha256"] == inv[k]["installed_weights_sha256"]
                                          for k in (name("FR", s, g), name("FC", s, g)) if k in hist_ppl))
    fa_inv = jload(os.path.join(ROOT, "runs_inference_routing_amplification/artifact_inventory.json"))
    fa_orig = {st: fa_inv["configs"][f"{st}_ORIGINAL"] for st in STRATS}
    refs = dict(
        s0_ORIGINAL=dict(available=True, source="Family A runs_inference_routing_amplification (FR/FC_ORIGINAL natural + "
                         "frozen, bitwise-verified there)", installed_weights_sha256={st: fa_orig[st]["installed_weights_sha256"]
                                                                                     for st in STRATS},
                         audit_valid=all(fa_orig[st]["valid"] for st in STRATS)),
        s1_ORIGINAL=dict(available=False, reason="runs_partial_ppl quantized in-process; no weights_dir; installed sha "
                         "not found in any stored weight set (EBSS FR_s1/FC_s1 dirs are a different extraction policy)"),
        s2_ORIGINAL=dict(available=False, reason="as s1"))
    for s in (1, 2):
        for st in STRATS:
            rp = jload(os.path.join(ROOT, f"runs_partial_ppl/ppl_{st}_s{s}.json"))
            refs[f"s{s}_ORIGINAL"][f"{st}_historical_ppl"] = rp["ppl"]
            refs[f"s{s}_ORIGINAL"][f"{st}_installed_weights_sha256"] = rp["installed_weights_sha256"]
    out = dict(checkpoint=dict(model="Qwen/Qwen1.5-MoE-A2.7B", snapshot=snap), configs={k: {x: y for x, y in v.items()
                                                                                          if x != "historical_per_window_nll"}
                                                                                      for k, v in inv.items()},
               n_configs=len(inv), n_valid=sum(v["valid"] for v in inv.values()),
               missing=[k for k, v in inv.items() if not v["checks"].get("result_exists")],
               duplicates=len(shas) != len(set(shas)), pairs=pair_ok, references=refs,
               excluded=["FR_s0_G20261111_CTRL_repeat (determinism control, identical weights)"],
               all_valid=all(v["valid"] for v in inv.values()) and len(shas) == len(set(shas)) and
                         all(all(x.values()) for x in pair_ok.values()),
               regenerated_weights=0, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(out, p("artifact_inventory.json"))
    wm = {k: dict(strategy=v["strategy"], allocation_seed=v["allocation_seed"], grouping=v["grouping"], kind="grouping",
                  weights_dir=v["weights_dir"], installed_weights_sha256=v["installed_weights_sha256"],
                  files={L: dict(sha256=f["sha256"], layer_sha256=f["layer_sha256"]) for L, f in v["files"].items()},
                  historical_result=v["result"], historical_ppl=v["historical_ppl"],
                  historical_per_window_nll=v["historical_per_window_nll"], order=None) for k, v in inv.items()}
    jdump(dict(configs=wm, manifest_sha256=obj_sha(wm)), p("weight_manifest.json"))
    print(f"[audit] all valid {out['all_valid']} ({out['n_valid']}/{out['n_configs']}), duplicates {out['duplicates']}")


# ---------------------------------------------------------------- protocol
def code_hashes():
    fs = ["moe_calib/inference_routing_override.py", "moe_calib/frozen_routing.py", "moe_calib/model_utils.py",
          "moe_calib/partial_model_quant.py", "diagnostics/eval_partial_ppl.py",
          "diagnostics/inference_routing_amplification.py", "diagnostics/grouping_routing_amplification.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in fs}


def stage_protocol(a):
    inv, wm = jload(p("artifact_inventory.json")), jload(p("weight_manifest.json"))
    assert inv and inv["all_valid"], "audit must pass"
    _, eprot = EP.load_windows()
    fam = jload(os.path.join(ROOT, FA_REF_MAN))
    body = dict(
        study="Family D (calibration batch grouping) frozen-routing causal intervention",
        model=dict(name="Qwen/Qwen1.5-MoE-A2.7B", snapshot=inv["checkpoint"]["snapshot"], quantized_layers=LAYERS_Q,
                   gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True), fake_quant="BF16 effective weights"),
        design=dict(comparisons={cname(s, g): dict(FR=name("FR", s, g), FC=name("FC", s, g), allocation_seed=s, grouping=g)
                                 for s in SEEDS for g in GROUPINGS},
                    structure="3 allocation seeds x 5 groupings; FR/FC paired by (seed, grouping); groupings nested in "
                              "seeds (not 15 independent samples)",
                    references=dict(s0_ORIGINAL="Family A FR/FC_ORIGINAL natural + frozen runs (reused, not re-run)",
                                    s1_s2_ORIGINAL="historical natural PPL only (weights not stored): frozen not evaluable"),
                    excluded=inv["excluded"]),
        weight_manifest_sha256=wm["manifest_sha256"],
        weight_sets={k: v["installed_weights_sha256"] for k, v in wm["configs"].items()},
        reference_routing=dict(file=FA_REF, file_sha256=fam["file_sha256"], idx_sha256=fam["idx_sha256"],
                               source="Family A bf16_routing_manifest.json (reused exactly; not regenerated)"),
        evaluation=dict(windows_sha256=eprot["windows_sha256"], n_windows=128, seqlen=512, scored_targets=65408,
                        batch=EP.BATCH, forward="historical sequential 24-layer layer-wise evaluator"),
        intervention="identical to Family A protocol: selected_experts <- BF16 reference indices in all 24 MoE layers; "
                     "current router softmax probabilities gathered at forced indices; norm_topk_prob=False; "
                     "FRT.make_forced_forward via moe_calib.inference_routing_override",
        parity=dict(A_natural="every natural evaluation reproduces historical per-window NLL bitwise",
                    B_record_only=f"{PARITY_CONFIGS}: recorder vs historical evaluator per-window bitwise; per-layer and "
                                  "final hidden-state hashes equal to the hook-free evaluator",
                    C_self_forcing=f"{PARITY_CONFIGS}: forced with own natural indices -> bitwise identical",
                    D_bf16_self_forcing="BF16 forced with reference -> bitwise identical (per-window, per-token, layer hashes)",
                    E_frozen_dispatch="every frozen run: dispatch counts == BF16 reference; probabilities finite and "
                                      "model-dependent",
                    F_restoration="after each config: installed digest unchanged during eval, router weights unchanged, "
                                  "targets restored to pristine, no hooks/overrides; full non-target digest at start/end",
                    tolerance="exact; no loosening"),
        metrics=dict(
            delta="Delta[s,g] = PPL(FC,s,g) - PPL(FR,s,g) (positive = FR better); delta_g = Delta_nat - Delta_fro",
            primary="R_SD = pooled within-seed SD(Delta_fro) / pooled within-seed SD(Delta_nat); pooled = "
                    "sqrt(mean_s SD_s^2), SD_s ddof=1 over the 5 groupings (historical moe_calib.cross_seed_grouping."
                    "variance_summary definition)",
            secondary=["per-seed SD ratios", "flat SD over 15 (descriptive only)", "FR/FC PPL pooled within-seed SDs",
                       "means, ranges, FR-vs-FC counts", "historical decision rule CS.decide under both regimes",
                       "hierarchical bootstrap CS.hierarchical_bootstrap (historical n/seed)"],
            reversal_R1="historical decision-rule definition: fraction of the 15 Delta with sign opposite to sign(M), "
                        "M = mean of seed means, computed within each regime",
            reversal_R2="fp64 mechanism-table definition: Delta[s,g] sign vs the same seed's ORIGINAL-extraction Delta "
                        "sign; frozen: seed 0 vs frozen ORIGINAL; seeds 1/2 frozen ORIGINAL unavailable -> reported as "
                        "vs natural ORIGINAL sign, labelled mixed-regime",
            window_ci="full_depth_pilot.paired_bootstrap(FC, FR, tokens), n_boot 5000, seed 20261109, mean-NLL "
                      "difference; same windows/resampling for both regimes; across-window != across-grouping",
            routing="set disagreement vs BF16; within-seed pairwise between groupings (10 pairs per seed x strategy); "
                    "seed-0 vs ORIGINAL (Family A-compatible); margins p4-p5 as Family A",
            weights="pairwise within-seed groupings and seed-0 vs ORIGINAL: changed-element fraction, mean/median abs "
                    "diff, rel Frobenius, per-layer/per-expert distribution, BF16 ULP distance; Family A order vs "
                    "ORIGINAL with the same function for calibration"),
        decision_rules={
            "BLOCKED": "any mandatory parity (A-F) fails",
            "STRONG-COLLAPSE (A)": "R_SD <= 0.5 AND frozen R1 reversal count <= half the natural R1 count",
            "WEAK-COLLAPSE (B)": "R_SD >= 0.9 AND frozen R1 count > half the natural R1 count",
            "PARTIAL-COLLAPSE (C)": "otherwise",
            "note": "R_SD <= 0.5 is a descriptive screening threshold (as Family A), not a significance test; no "
                    "additive variance decomposition"},
        code_sha256_at_freeze=code_hashes())
    body["protocol_sha256"] = obj_sha(body)
    if os.path.exists(p("protocol.json")):
        print("protocol already frozen; not overwriting"); return
    body["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(body, p("protocol.json"))
    print(f"[protocol] frozen {body['protocol_sha256'][:16]}")


def check_protocol():
    pr = jload(p("protocol.json"))
    b = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(b) == pr["protocol_sha256"], "protocol.json modified"
    return pr


# ---------------------------------------------------------------- GPU
def save_run(n, mode, sess, r, extra):
    os.makedirs(c("runs"), exist_ok=True)
    os.makedirs(p("results"), exist_ok=True)
    d = dict(token_nll=torch.from_numpy(r["token_nll"]), dispatch=sess.dispatch.clone(), idx=sess.idx, pk=sess.pk,
             p_next=sess.p_next, p_forced=sess.p_forced)
    tmp = run_file(n, mode) + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, run_file(n, mode))
    rec = EP.ppl_record(r["nll"], r["cnt"], extra.pop("eval_s"))
    rec.update(config=n, mode=mode, final_hidden_sha256=r["final_sha"], run_file=run_file(n, mode),
               run_file_sha256=sha256_file(run_file(n, mode)), **extra)
    jdump(rec, res_file(n, mode))
    return rec


def no_hooks(model):
    from moe_calib import model_utils as MU
    for L in range(MU.num_layers(model)):
        b = MU.moe_block(model, L)
        if "forward" in b.__dict__ or b.gate._forward_hooks or any(x._forward_pre_hooks for x in b.experts):
            return False
    return True


def stage_gpu(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib import inference_routing_override as IRO
    pr = check_protocol()
    wm = jload(p("weight_manifest.json"))
    assert wm["manifest_sha256"] == pr["weight_manifest_sha256"]
    win, eprot = EP.load_windows()
    assert eprot["windows_sha256"] == pr["evaluation"]["windows_sha256"]
    assert sha256_file(FA_REF) == pr["reference_routing"]["file_sha256"], "BF16 reference routing changed"
    ref = torch.load(FA_REF)
    assert hashlib.sha256(ref["idx"].numpy().tobytes()).hexdigest() == pr["reference_routing"]["idx_sha256"]
    ref_idx = ref["idx"]
    cost = jload(p("logs", "gpu_cost_raw.json")) or dict(runs={}, loads=[])
    val = jload(p("routing_override_validation.json")) or {}
    t0 = time.time()
    model, _ = MU.load_model()
    cost["loads"].append(dict(model_load_s=time.time() - t0))
    snap = PQ.snapshot_experts(model, LAYERS_Q)
    t1 = time.time()
    untouched0 = PQ.untouched_digests(model, LAYERS_Q)
    cost["untouched_digest_s"] = time.time() - t1
    gates0 = FA.gate_digest(model)
    hb = jload(os.path.join(ROOT, FA.HIST_BF16))

    def log_cost(k, s, pk):
        cost["runs"][k] = dict(eval_s=s, peak_gpu_gb=pk, peak_cpu_gb=FA.peak_cpu_gb())
        jdump(cost, p("logs", "gpu_cost_raw.json"))

    def save_val():
        jdump(val, p("routing_override_validation.json"))

    # ---------- D: BF16 self-forcing (and BF16 natural parity) in this process
    if not val.get("D_bf16_self_forcing", {}).get("passed"):
        _, rp, s, pk = FA.evaluate(model, win, None, plain=True)
        log_cost("BF16_plain", s, pk)
        _, r0, s, pk = FA.evaluate(model, win, None, layer_hashes=True)
        log_cost("BF16_nohook_hashes", s, pk)
        sf, rf, s, pk = FA.evaluate(model, win, "force", ref=ref_idx.clone(), layer_hashes=True)
        log_cost("BF16_self_forced", s, pk)
        v = dict(plain_vs_historical=FA.nll_equal(rp["nll"], hb["runs"][0]["per_window_nll"]),
                 forced_vs_plain=FA.nll_equal(rf["nll"], rp["nll"]),
                 forced_vs_nohook_token_bitwise=bool(np.array_equal(rf["token_nll"], r0["token_nll"])),
                 layer_hashes_equal=rf["layer_sha"] == r0["layer_sha"], final_hidden_equal=rf["final_sha"] == r0["final_sha"],
                 natural_choice_equals_reference=bool(torch.equal(sf.idx, ref_idx)),
                 forced_probs_equal_reference_topk=bool(torch.equal(sf.p_forced, ref["pk"])),
                 dispatch_equal_reference=bool(torch.equal(sf.dispatch, ref["dispatch"])),
                 forced_calls_per_layer=sf.n_forced_calls.tolist(), no_hooks_after=no_hooks(model))
        v["passed"] = all(x for x in v.values() if isinstance(x, bool)) and v["plain_vs_historical"]["bitwise"] and \
            v["forced_vs_plain"]["bitwise"] and all(x == 16 for x in v["forced_calls_per_layer"])
        val["D_bf16_self_forcing"] = v
        save_val()
        print(f"[parity D] BF16 self-forcing passed={v['passed']}", flush=True)
        del sf, rf, r0, rp
        if not v["passed"]:
            raise SystemExit("BF16 parity failed -> BLOCKED")

    def run_cfg(n, parity=False):
        done = all((jload(res_file(n, m)) or {}).get("valid") for m in ("natural", "frozen"))
        if done and not parity:
            return
        t_inst = FA.install(model, n, wm, snap)
        cost["runs"][f"{n}_install"] = dict(install_s=t_inst)
        ent = wm["configs"][n]
        hist = ent["historical_per_window_nll"]
        common = dict(installed_weights_sha256=ent["installed_weights_sha256"], protocol_sha256=pr["protocol_sha256"],
                      strategy=ent["strategy"], allocation_seed=ent["allocation_seed"], grouping=ent["grouping"])
        if parity:
            _, rp, s, pk = FA.evaluate(model, win, None, plain=True)
            log_cost(f"{n}_plain", s, pk)
            sn, rn, s, pk = FA.evaluate(model, win, "natural", layer_hashes=True)
            log_cost(f"{n}_natural", s, pk)
            _, r0, s0_, pk0 = FA.evaluate(model, win, None, layer_hashes=True)
            log_cost(f"{n}_nohook_hashes", s0_, pk0)
            sc, rc, s2, pk2 = FA.evaluate(model, win, "force", ref=sn.idx.clone(), layer_hashes=True)
            log_cost(f"{n}_self_forced", s2, pk2)
            B = dict(record_vs_plain=FA.nll_equal(rn["nll"], rp["nll"]), plain_vs_historical=FA.nll_equal(rp["nll"], hist),
                     layer_hashes_equal=rn["layer_sha"] == r0["layer_sha"], final_hidden_equal=rn["final_sha"] == r0["final_sha"])
            B["passed"] = B["record_vs_plain"]["bitwise"] and B["plain_vs_historical"]["bitwise"] and \
                B["layer_hashes_equal"] and B["final_hidden_equal"]
            C = dict(per_window=FA.nll_equal(rc["nll"], rn["nll"]), token_nll_bitwise=bool(np.array_equal(rc["token_nll"], rn["token_nll"])),
                     layer_hashes_equal=rc["layer_sha"] == rn["layer_sha"], final_hidden_equal=rc["final_sha"] == rn["final_sha"],
                     selections_equal=bool(torch.equal(sc.idx, sn.idx)), router_probs_equal=bool(torch.equal(sc.pk, sn.pk)),
                     forced_probs_equal_natural_topk=bool(torch.equal(sc.p_forced, sn.pk)),
                     dispatch_equal=bool(torch.equal(sc.dispatch, sn.dispatch)),
                     ppl_equal=FA.ppl_of(rc["nll"]) == FA.ppl_of(rn["nll"]),
                     forced_calls_all_16=all(x == 16 for x in sc.n_forced_calls.tolist()))
            C["passed"] = all(x for x in C.values() if isinstance(x, bool)) and C["per_window"]["bitwise"]
            val.setdefault("B_record_only", {})[n] = B
            val.setdefault("C_self_forcing", {})[n] = C
            save_val()
            print(f"[parity B/C] {n}: B {B['passed']} C {C['passed']}", flush=True)
            del sc, rc, r0, rp
            if not (B["passed"] and C["passed"]):
                FA.restore(model, snap)
                raise SystemExit(f"parity failed on {n} -> BLOCKED")
        else:
            sn, rn, s, pk = FA.evaluate(model, win, "natural")
            log_cost(f"{n}_natural", s, pk)
        reg = FA.nll_equal(rn["nll"], hist)
        rec = save_run(n, "natural", sn, rn, dict(eval_s=s, peak_gpu_gb=pk, historical_regression=reg,
                                                  valid=bool(reg["bitwise"]), **common))
        del sn, rn
        fs, fr, s, pk = FA.evaluate(model, win, "force", ref=ref_idx)
        log_cost(f"{n}_frozen", s, pk)
        E = dict(dispatch_equals_bf16=bool(torch.equal(fs.dispatch, ref["dispatch"])),
                 forced_calls_all_16=all(x == 16 for x in fs.n_forced_calls.tolist()),
                 p_forced_finite=bool(torch.isfinite(fs.p_forced).all()), pk_finite=bool(torch.isfinite(fs.pk).all()),
                 token_nll_finite=bool(np.isfinite(fr["token_nll"]).all()),
                 p_forced_model_dependent=bool(not torch.equal(fs.p_forced, ref["pk"])),
                 frac_forced_probs_differ_from_bf16=float((fs.p_forced != ref["pk"]).float().mean()))
        recf = save_run(n, "frozen", fs, fr, dict(eval_s=s, peak_gpu_gb=pk, checks=E,
                                                  valid=all(v for v in E.values() if isinstance(v, bool)), **common))
        del fs, fr
        F = dict(installed_unchanged_after_eval=PQ.installed_digest(model, LAYERS_Q) == ent["installed_weights_sha256"],
                 router_weights_unchanged=FA.gate_digest(model) == gates0, no_hooks_or_overrides=no_hooks(model))
        FA.restore(model, snap)
        F["targets_restored_pristine"] = True                  # FA.restore asserts experts_match == total
        val.setdefault("E_frozen_dispatch", {})[n] = E
        val.setdefault("F_restoration", {})[n] = F
        val.setdefault("A_natural_parity", {})[n] = reg
        save_val()
        print(f"[{n}] natural {rec['ppl']:.6f} (hist bitwise {reg['bitwise']})  frozen {recf['ppl']:.6f}  "
              f"E {recf['valid']}  F {all(F.values())}", flush=True)
        if not (reg["bitwise"] and recf["valid"] and all(F.values())):
            raise SystemExit(f"integrity failed on {n} -> BLOCKED")
        gc.collect()

    for n in PARITY_CONFIGS:
        if not (val.get("C_self_forcing", {}).get(n, {}).get("passed")):
            run_cfg(n, parity=True)
    for n in CONFIGS:
        run_cfg(n)
    untouched1 = PQ.untouched_digests(model, LAYERS_Q)
    eq, tot = PQ.experts_match(model, snap)
    _, rend, s, pk = FA.evaluate(model, win, None, plain=True)
    cost["end_checks"] = dict(untouched_params_unchanged=untouched1 == untouched0, n_untouched=len(untouched0),
                              targets_pristine=eq == tot, gates_unchanged=FA.gate_digest(model) == gates0,
                              no_hooks=no_hooks(model),
                              bf16_end_per_window_bitwise=FA.nll_equal(rend["nll"], hb["runs"][0]["per_window_nll"])["bitwise"])
    cost["peak_cpu_gb"] = FA.peak_cpu_gb()
    jdump(cost, p("logs", "gpu_cost_raw.json"))
    print(f"[end] {cost['end_checks']}", flush=True)


# ---------------------------------------------------------------- weight differences (CPU)
def load_weights(wdir):
    out = {}
    for L in LAYERS_Q:
        d = torch.load(os.path.join(wdir, f"L{L}.pt"))
        for e in range(60):
            for k in PROJ:
                out[(L, e, k)] = d[f"{e}.{k}"]
    return out


@torch.no_grad()
def weight_diff(A, B):
    """A vs B (B = reference). Exact counts; median over a fixed stride-7 subsample."""
    n = ch = 0
    sabs = sd2 = sb2 = 0.0
    per_layer = {L: [0, 0] for L in LAYERS_Q}
    per_proj = {k: [0, 0] for k in PROJ}
    per_expert = {}
    ulp = np.zeros(5, dtype=np.int64)                       # 1, 2, 3-7, >=8 same sign, sign change
    sub = []
    for key in sorted(A):
        a, b = A[key], B[key]
        L, e, k = key
        neq = a != b
        c_ = int(neq.sum())
        nn = a.numel()
        n += nn; ch += c_
        per_layer[L][0] += c_; per_layer[L][1] += nn
        per_proj[k][0] += c_; per_proj[k][1] += nn
        pe_ = per_expert.setdefault((L, e), [0, 0])
        pe_[0] += c_; pe_[1] += nn
        d = (a.float() - b.float())
        sabs += float(d.abs().sum(dtype=torch.float64))
        sd2 += float((d.double() ** 2).sum())
        sb2 += float((b.double() ** 2).sum())
        sub.append(d.abs().flatten()[::7])
        if c_:
            ia, ib = a[neq].view(torch.int16).int(), b[neq].view(torch.int16).int()
            same = (ia < 0) == (ib < 0)
            dist = (ia - ib).abs()
            ulp[0] += int((same & (dist == 1)).sum()); ulp[1] += int((same & (dist == 2)).sum())
            ulp[2] += int((same & (dist >= 3) & (dist <= 7)).sum()); ulp[3] += int((same & (dist >= 8)).sum())
            ulp[4] += int((~same).sum())
    sub = torch.cat(sub)
    pe = np.array([x[0] / x[1] for x in per_expert.values()])
    return dict(frac_changed=ch / n, n_changed=ch, n_elements=n, mean_abs_diff=sabs / n,
                median_abs_diff_subsample=float(sub.median()), rel_frobenius=float(np.sqrt(sd2 / sb2)),
                frac_changed_per_layer={str(L): v[0] / v[1] for L, v in per_layer.items()},
                frac_changed_per_projection={k: v[0] / v[1] for k, v in per_proj.items()},
                per_expert_frac_changed_quantiles={str(q): float(np.quantile(pe, q)) for q in (0, .1, .5, .9, 1)},
                ulp_share_of_changed=dict(zip(["1", "2", "3-7", ">=8", "sign_change"], (ulp / max(ch, 1)).tolist())))


def stage_weights(a):
    wm = jload(p("weight_manifest.json"))["configs"]
    fa_inv = jload(os.path.join(ROOT, "runs_inference_routing_amplification/artifact_inventory.json"))["configs"]
    out = dict(within_seed_pairs={}, s0_vs_original={}, family_a_vs_original={})
    t0 = time.time()
    for st in STRATS:
        orig = load_weights(fa_inv[f"{st}_ORIGINAL"]["weights_dir"])
        fa = load_weights(fa_inv[f"{st}_O20261121"]["weights_dir"])
        out["family_a_vs_original"][f"{st}_O20261121"] = weight_diff(fa, orig)
        del fa
        for s in SEEDS:
            W = {g: load_weights(wm[name(st, s, g)]["weights_dir"]) for g in GROUPINGS}
            for i, g in enumerate(GROUPINGS):
                if s == 0:
                    out["s0_vs_original"][name(st, s, g)] = weight_diff(W[g], orig)
                for g2 in GROUPINGS[i + 1:]:
                    out["within_seed_pairs"][f"{st}_s{s}_{g}__{g2}"] = weight_diff(W[g], W[g2])
            print(f"[weights] {st} s{s} done ({time.time() - t0:.0f} s)", flush=True)
            del W
            gc.collect()
        del orig
    out["note"] = ("frac_changed counts BF16 effective weight elements whose bits differ; ULP distance from the int16 "
                   "bit pattern (same sign); median abs diff from a fixed 1/7 subsample")
    jdump(out, p("logs", "weight_differences_raw.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "protocol", "gpu", "weights", "analyze"])
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "analyze":
        import report_grouping_routing_amplification as R
        return R.analyze()
    dict(audit=stage_audit, protocol=stage_protocol, gpu=stage_gpu, weights=stage_weights)[a.stage](a)


if __name__ == "__main__":
    main()
