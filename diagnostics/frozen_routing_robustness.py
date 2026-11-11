"""Frozen-routing robustness across the five prespecified calibration batch groupings (FR s0 / FC s0, four layers).

AO_g = GPTQ(X_g, R_O) for g in G20261111..G20261115 (G20261111 reused from runs_frozen_routing), compared with the
natural AA_g = GPTQ(X_g, R_g) (runs_multigroup_robustness) and the single ORIGINAL reference OO. The forced-routing
quantization is the validated diagnostics/frozen_routing.run_config (unchanged apart from grouping / output paths).

Stages (--stage):
  preflight                      selections, manifests, caches, references, frozen protocol (decision rule)
  capture                        R_O verification vs selections and all five caches; expected AO token counts;
                                 natural AA_g Hessians (unmodified Profiler, no GPTQ) for the Hessian diagnostics
  regression                     G20261111 AO: forced Hessians recomputed bitwise, stored weights re-installed and
                                 PPL re-evaluated exactly; fingerprints of every reused OO / AA / AO artifact
  ao --strategy S --grouping G   one new AO configuration (fresh process, pristine checkpoint)
  report
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB          # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import frozen_routing as FRD            # noqa: E402
import multigroup_robustness as MRD     # noqa: E402
from moe_calib import frozen_routing as FRT              # noqa: E402
from moe_calib import frozen_routing_robustness as RB    # noqa: E402

OUT = "runs_frozen_routing_robustness"
RCACHE = os.path.join(FRD.FCACHE, "robustness")          # new AO weights / Hessians, AA_g diagnostic Hessians
LAYERS, STRATS, OCACHE = FRD.LAYERS, list(RB.STRATS), FRD.OCACHE
jdump, sha256_file = FRD.jdump, FRD.sha256_file


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ao_name(c, g):
    return f"{c}_AO_{g}"


def ao_result_path(c, g):
    if g in RB.REUSED_AO:
        return FRD.p("results", f"{c}_AO.json")
    return p("results", f"{ao_name(c, g)}.json")


def ao_weights_dir(c, g):
    return FRD.weights_dir(c, "AO") if g in RB.REUSED_AO else os.path.join(RCACHE, "weights", ao_name(c, g))


def ao_hessian_path(c, g):
    return os.path.join(FRD.FCACHE, "hessians", f"{c}_AO.pt") if g in RB.REUSED_AO \
        else os.path.join(RCACHE, "hessians", f"{ao_name(c, g)}.pt")


def aa_hessian_path(c, g):
    return os.path.join(RCACHE, "hessians", f"{c}_AA_{g}.pt")


def aa_result_path(c, g):
    return os.path.join(ROOT, MRD.OUT, "results", f"{c}_{g}.json")


def oo_result_path(c):
    return os.path.join(ROOT, MRD.OUT, "results", f"CTRL_{c}_ORIGINAL.json")


def hess_record(acc, L, e, K):
    out = {}
    for pj in ("gu", "down"):
        H = acc.H_sum((L, e, pj), list(range(K)))
        out[pj] = dict(H=H, sha=FRT.tensor_sha(H), tokens=acc.tokens((L, e, pj)), diag=torch.diag(H).cpu().clone(),
                       perm=FRT.actorder_perm(H).cpu().to(torch.int16))
    return out


def save_hessians(recs, path):
    """Same layout as the factorial study's hessian files."""
    d = dict(full={}, diag={}, perm={}, tokens={}, sha={})
    for (L, e, pj), v in recs.items():
        k = f"{L}.{e}.{pj}"
        if (L, e, pj) in FRT.DIAG_KEYS:
            d["full"][k] = v["H"].cpu().clone()
        d["diag"][k], d["perm"][k], d["tokens"][k], d["sha"][k] = v["diag"], v["perm"], v["tokens"], v["sha"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)
    return sha256_file(path)


# ---------------------------------------------------------------- preflight + frozen protocol
def stage_preflight(a):
    cfg, sel, checks, union = MRD.selections()                 # re-verifies source hashes, warm start, pool, leakage
    mpr = MRD.protocol()                                       # multigroup protocol + all five manifests intact
    fpr = FRD.protocol()                                       # factorial protocol intact
    for c in STRATS:
        assert sel[c]["ids_hash"] == RB.SELECTION_HASHES[c] == fpr["selections"][c]["ids_hash"], c
    assert tuple(mpr["groupings"]) == RB.GROUPINGS and [mpr["groupings"][g]["seed"] for g in RB.GROUPINGS] == list(RB.GROUPING_SEEDS)
    cman = json.load(open(os.path.join(ROOT, MRD.OUT, "cache_manifest.json")))
    for g in RB.GROUPINGS:
        MRD.verify_cache(g)
        assert cman["groupings"][g]["sequence_ids"] == union, f"{g}: cache rows are not union_calibration_ids"
    win, prot = EP.load_windows()
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    assert abs(base["ppl"] - 9.237090) < 1e-6 and prot["scored_targets_total"] == 65408 and tuple(win.shape) == (128, 512)
    assert prot["windows_sha256"] == fpr["evaluation"]["windows_sha256"]
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True) == fpr["gptq"]
    refs = {}
    for c in STRATS:
        for kind, g, path in [("OO", "ORIGINAL", oo_result_path(c))] + [("AA", g, aa_result_path(c, g)) for g in RB.GROUPINGS] \
                + [("AO", g, ao_result_path(c, g)) for g in RB.REUSED_AO]:
            r = json.load(open(path))
            assert r["valid"] and r["rtn_fallbacks"] == 0 and r["n_projection_tensors"] == 720 and r["ids_hash"] == sel[c]["ids_hash"]
            refs[f"{c}_{kind}_{g}"] = dict(path=os.path.relpath(path, ROOT), result_sha256=sha256_file(path), ppl=r["ppl"],
                                           mean_nll=r["mean_nll"], recon=r["recon"], weights_dir=r["weights_dir"],
                                           installed_weights_sha256=r["installed_weights_sha256"],
                                           weight_files={L: v["sha256"] for L, v in r["weight_files"].items()}
                                           if "weight_files" in r else None,
                                           hessian_file_sha256=r.get("hessian_file_sha256"))
    # the natural references of the factorial study are the same files (OO = CTRL ORIGINAL, AA_G20261111)
    for c in STRATS:
        assert fpr["natural_references"][f"{c}_OO"]["result_sha256"] == refs[f"{c}_OO_ORIGINAL"]["result_sha256"]
        assert fpr["natural_references"][f"{c}_AA"]["result_sha256"] == refs[f"{c}_AA_G20261111"]["result_sha256"]
    fppl = json.load(open(FRD.p("ppl_results.json")))["configs"]
    for c in STRATS:
        assert refs[f"{c}_AO_G20261111"]["ppl"] == fppl[c]["AO"]["ppl"]
    assert round(refs["FR_AO_G20261111"]["ppl"], 6) == 9.265861 and round(refs["FC_AO_G20261111"]["ppl"], 6) == 9.277881
    rorig = json.load(open(FRD.p("routing_original.json")))
    gm = {g: json.load(open(os.path.join(ROOT, MRD.OUT, "grouping_manifests", f"{g}.json")))["manifest_sha256"]
          for g in RB.GROUPINGS}
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    disk = shutil.disk_usage(os.path.expanduser("~")).free
    pre = dict(selections=checks, n_union=len(union), bf16_ppl=base["ppl"], environment=DB.env_record(),
               host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, gpu_free_gb=free / 1e9,
               gpu_total_gb=tot / 1e9, disk_free_tb=disk / 1e12, created=time.strftime("%F %T"),
               feasible=bool(free > 12e9 and disk > 50e9))
    jdump(pre, p("logs", "preflight.json"))
    if not pre["feasible"]:
        raise SystemExit("preflight: insufficient GPU memory or disk")
    jdump(dict(selections={c: dict(ids_hash=sel[c]["ids_hash"], K=sel[c]["K"], source=sel[c]["source"],
                                   source_sha256=sel[c]["source_sha256"], chosen=sel[c]["chosen"]) for c in STRATS},
               union_calibration_ids=union, grouping_manifest_sha256=gm,
               grouping_cache_files={g: cman["groupings"][g]["files"] for g in RB.GROUPINGS},
               original_cache_files=mpr["references"]["original_cache_files"],
               routing_original=dict(path=os.path.relpath(FRD.p("routing_original.pt"), ROOT), file_sha256=rorig["file_sha256"],
                                     per_layer_routing_sha256={L: v["routing_sha256"] for L, v in rorig["per_layer"].items()}),
               factorial_protocol_sha256=fpr["protocol_sha256"], multigroup_protocol_sha256=mpr["protocol_sha256"],
               windows_sha256=prot["windows_sha256"], references=refs), p("input_manifest.json"))
    body = dict(
        experiment="Frozen-routing robustness across the five prespecified calibration batch groupings (FR s0 / FC s0, "
                   "layers 1/8/16/23)",
        question="Does freezing the ORIGINAL expert assignments used to build GPTQ calibration Hessians consistently "
                 "reduce model-level PPL sensitivity to BF16 calibration batch composition?",
        model=fpr["model"], model_snapshot=fpr["model_snapshot"], gptq=cfg["gptq"], B=128, allocation_seed=0,
        quant_layers=LAYERS, n_quantized_experts=240, n_projection_tensors=720,
        selections={c: dict(ids_hash=sel[c]["ids_hash"], K=sel[c]["K"]) for c in STRATS},
        union_calibration_ids=union, groupings={g: dict(seed=s, manifest_sha256=gm[g]) for g, s in zip(RB.GROUPINGS, RB.GROUPING_SEEDS)},
        conditions=dict(OO="GPTQ(X_O, R_O): CTRL ORIGINAL, one reference per strategy",
                        AA="GPTQ(X_g, R_g): natural multigroup result (reused)",
                        AO="GPTQ(X_g, R_O): frozen ORIGINAL routing (G20261111 reused, four groupings new)"),
        reused_ao=list(RB.REUSED_AO), new_ao=[ao_name(c, g) for g in RB.NEW_AO for c in STRATS],
        routing_original_sha256=rorig["file_sha256"],
        alignment="(original sequence id, token position): grouping cache row i = union_calibration_ids[i] (verified "
                  "from each cache manifest); R_O keyed by sequence id; every sequence 512 tokens",
        intervention=fpr["intervention"], hessian_semantics=fpr["hessian_semantics"],
        reconstruction=fpr["reconstruction"],
        evaluation=dict(windows_sha256=prot["windows_sha256"], batching=f"EP.layerwise_nll batch {EP.BATCH}",
                        bf16_ppl=base["ppl"], repeats_per_new_config=2),
        invariants=["forced routed-token count per expert == R_O count over the chosen sequences (every AO config)",
                    "AO routed-token count vector identical across the five groupings for each strategy",
                    "G20261111 AO: forced Hessians bitwise == stored (480), re-installed weights digest == stored, "
                    "re-evaluated PPL == stored exactly"],
        diagnostic_subset=fpr["diagnostic_subset"],
        aa_hessians="natural AA_g Hessians are recomputed with the UNMODIFIED Profiler (no GPTQ) for the Hessian "
                    "diagnostics only; G20261111 must equal the factorial study's AA Hessians bitwise",
        decision_rule=RB.DECISION_RULE,
        hard_constraints="no re-selection, seed 0, the five prespecified groupings only, 4 layers, frozen GPTQ / windows / "
                         "evaluation / reconstruction, no OA runs, no forced routing at inference, no additional runs "
                         "after results")
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        assert old["protocol_sha256"] == RB.obj_sha(body), "protocol changed after freezing"
        print(f"preflight ok; protocol already frozen (sha256 {old['protocol_sha256'][:16]})")
        return
    body["protocol_sha256"] = RB.obj_sha(body)
    body["created"] = time.strftime("%F %T")
    body["environment"] = DB.env_record()
    jdump(body, p("protocol.json"))
    print(f"preflight ok: GPU free {free / 1e9:.1f} GB, disk {disk / 1e12:.1f} TB; protocol sha256 {body['protocol_sha256'][:16]}")


def protocol():
    pr = json.load(open(p("protocol.json")))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created", "environment")}
    assert RB.obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    return pr


# ---------------------------------------------------------------- routing preparation + AA_g Hessians
def stage_capture(a):
    from moe_calib import model_utils as MU
    from moe_calib.profiler import Profiler
    from moe_calib.data_utils import load_layer_cache
    pr = protocol()
    union = pr["union_calibration_ids"]
    out_path = p("routing_preparation.json")
    rp = load_json(out_path) or {}
    cfg, sel, _, _ = MRD.selections()
    if "R_O" not in rp:
        tables = FRD.load_tables("O")                          # sha-checked against routing_original.json
        rorig = json.load(open(FRD.p("routing_original.json")))
        per = {}
        for L in LAYERS:
            t = tables[L]
            assert sorted(t) == union, f"L{L}: R_O keys != union"
            shp = {tuple(v.shape) for v in t.values()}
            per[str(L)] = dict(routing_sha256=FRT.routing_sha(t), matches_manifest=FRT.routing_sha(t) == rorig["per_layer"][str(L)]["routing_sha256"],
                               shapes=[list(s) for s in shp], experts_in_range=all(int(v.min()) >= 0 and int(v.max()) < 60 for v in t.values()),
                               distinct_topk=all(bool((v.sort(-1).values.diff(dim=-1) > 0).all()) for v in t.values()))
        cman = json.load(open(os.path.join(ROOT, MRD.OUT, "cache_manifest.json")))
        caches = {}
        for g in RB.GROUPINGS:
            MRD.verify_cache(g)
            caches[g] = {str(L): dict(shape=cman["groupings"][g]["files"][str(L)]["shape"],
                                      rows_equal_union=cman["groupings"][g]["sequence_ids"] == union,
                                      tokens_per_row_equal_routing=cman["groupings"][g]["files"][str(L)]["shape"][1] == 512)
                         for L in LAYERS}
        expected = {}
        for c in STRATS:
            expected[c] = {str(L): FRT.expert_token_counts(tables[L], sel[c]["chosen"], 60).tolist() for L in LAYERS}
        # the factorial AO (G20261111) Hessians must carry exactly these counts
        for c in STRATS:
            hf = torch.load(ao_hessian_path(c, "G20261111"))
            got = {str(L): [hf["tokens"][f"{L}.{e}.gu"] for e in range(60)] for L in LAYERS}
            assert all(float(x) == float(y) for L in got for x, y in zip(got[L], expected[c][L])), c
        ok = all(v["matches_manifest"] and v["shapes"] == [[512, 4]] and v["experts_in_range"] and v["distinct_topk"]
                 for v in per.values()) and all(x["rows_equal_union"] and x["tokens_per_row_equal_routing"] and
                                                 x["shape"][0] == len(union) for g in caches for x in caches[g].values())
        rp = dict(R_O=per, caches=caches, expected_ao_token_counts=expected, G20261111_AO_counts_match=True, valid=bool(ok),
                  note="R_O is the factorial study's ORIGINAL routing table (not recomputed); the expected AO "
                       "routed-token counts depend only on R_O and the chosen sequence ids, so they must be identical "
                       "for every grouping")
        jdump(rp, out_path)
        print(f"[capture] R_O verified: valid {ok}", flush=True)
        if not ok:
            raise SystemExit("R_O / cache alignment invalid")
    todo = [(c, g) for g in RB.GROUPINGS for c in STRATS if not os.path.exists(aa_hessian_path(c, g))]
    rp.setdefault("aa_hessians", {})
    if todo:
        model, t_load = DB.load_model()
        E = MU.num_experts(model)
        for c, g in todo:
            t0 = time.time()
            chosen = MRD.MG.remap(sel[c]["chosen"], union)
            recs = {}
            for L in LAYERS:
                cache = load_layer_cache(MRD.gdir(g), L)
                pf = Profiler(model, L, K=sel[c]["K"]).run(cache, chosen, weights=[1.0] * len(chosen))
                for e in range(E):
                    for pj, v in hess_record(pf.acc, L, e, sel[c]["K"]).items():
                        recs[(L, e, pj)] = v
                pf.free(); del pf, cache
            fsha = save_hessians(recs, aa_hessian_path(c, g))
            ent = dict(file=aa_hessian_path(c, g), file_sha256=fsha, seconds=time.time() - t0)
            if g == "G20261111":                                   # must equal the factorial study's AA Hessians
                ref = torch.load(os.path.join(FRD.FCACHE, "hessians", f"{c}_AA.pt"))["sha"]
                n_eq = sum(ref[f"{L}.{e}.{pj}"] == v["sha"] for (L, e, pj), v in recs.items())
                ent["bitwise_equal_factorial_AA"] = f"{n_eq}/{len(recs)}"
                if n_eq != len(recs):
                    raise SystemExit(f"{c} {g}: AA Hessians differ from the factorial study")
            rp["aa_hessians"][f"{c}_{g}"] = ent
            jdump(rp, out_path)
            print(f"[capture] AA Hessians {c} {g}: {ent['seconds']:.0f} s {ent.get('bitwise_equal_factorial_AA', '')}", flush=True)
    print("[capture] done")


# ---------------------------------------------------------------- regression of the reused artifacts
def stage_regression(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    pr = protocol()
    out_path = p("regression.json")
    if load_json(out_path) and load_json(out_path)["passed"]:
        print("[regression] done"); return
    assert load_json(p("routing_preparation.json"))["valid"]
    man = json.load(open(p("input_manifest.json")))
    cfg, sel, _, union = MRD.selections()
    t0 = time.time()
    fp = {}
    for name, r in man["references"].items():                 # fingerprints of every reused artifact
        ok = sha256_file(os.path.join(ROOT, r["path"])) == r["result_sha256"]
        wf = {L: sha256_file(os.path.join(r["weights_dir"], f"L{L}.pt")) == s for L, s in r["weight_files"].items()} \
            if r["weight_files"] else {}
        fp[name] = dict(result_json=ok, weight_files=all(wf.values()) if wf else "not stored", n_weight_files=len(wf))
    t_fp = time.time() - t0
    tables = FRD.load_tables("O")
    model, t_load = DB.load_model()
    pdig = json.load(open(os.path.join(ROOT, FRD.FD.OUT, "pristine_digests.json")))
    reg = {}
    win, prot = EP.load_windows()
    for c in STRATS:
        rpath = ao_result_path(c, "G20261111")
        r = json.load(open(rpath))
        stored = torch.load(r["hessian_file"])
        assert sha256_file(r["hessian_file"]) == r["hessian_file_sha256"]
        # (1) forced Hessians recomputed through the same ForcedProfiler path the parameterized run_config uses
        chosen, r2s = MRD.MG.remap(sel[c]["chosen"], union), dict(enumerate(union))
        cdir, _ = FRD.act_source("AO", union, "G20261111")
        n_eq = n = 0
        tok_ok = True
        for L in LAYERS:
            cache = load_layer_cache(cdir, L)
            fpf = FRT.ForcedProfiler(model, L, tables[L], r2s, False, K=sel[c]["K"]).run(cache, chosen, weights=[1.0] * len(chosen))
            for e in range(60):
                for pj, v in hess_record(fpf.acc, L, e, sel[c]["K"]).items():
                    n += 1
                    n_eq += v["sha"] == stored["sha"][f"{L}.{e}.{pj}"]
                    tok_ok &= v["tokens"] == stored["tokens"][f"{L}.{e}.{pj}"]
            fpf.free(); del fpf, cache
        FRT.assert_no_override(model)
        # (2) stored weights re-installed into the pristine model -> installed digest + PPL exactly as stored
        tg0 = PQ.target_tensor_digests(model, LAYERS)
        assert tg0 == {k: pdig["pristine_targets"][k] for k in tg0}, "targets not pristine"
        snap = PQ.snapshot_experts(model, LAYERS)
        for L in LAYERS:
            PQ.install_layer_file(model, L, os.path.join(r["weights_dir"], f"L{L}.pt"))
        dig = PQ.installed_digest(model, LAYERS)
        t2 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        rec = EP.ppl_record(nll, cnt, time.time() - t2)
        PQ.restore_experts(model, snap)
        assert PQ.experts_match(model, snap)
        reg[c] = dict(result=os.path.relpath(rpath, ROOT), hessians_bitwise=f"{n_eq}/{n}", token_counts_identical=bool(tok_ok),
                      installed_digest_equal=dig == r["installed_weights_sha256"], ppl_stored=r["ppl"], ppl_reevaluated=rec["ppl"],
                      ppl_exact=rec["ppl"] == r["ppl"], per_window_exact=rec["per_window_nll"] == r["per_window_nll"],
                      mean_nll_exact=rec["mean_nll"] == r["mean_nll"], eval_s=rec["eval_s"])
        reg[c]["passed"] = bool(n_eq == n == 480 and tok_ok and reg[c]["installed_digest_equal"] and reg[c]["ppl_exact"]
                                and reg[c]["per_window_exact"])
        print(f"[regression] {c} AO G20261111: hessians {n_eq}/{n} digest {reg[c]['installed_digest_equal']} "
              f"PPL {rec['ppl']:.6f} exact {reg[c]['ppl_exact']}", flush=True)
    passed = all(v["passed"] for v in reg.values()) and all(v["result_json"] and v["weight_files"] is True for v in fp.values())
    jdump(dict(passed=bool(passed), ao_G20261111=reg, fingerprints=fp, fingerprint_s=t_fp, model_load_s=t_load,
               note="G20261111 AO PPL compared with the stored JSON floats (9.265860908067534 / FC stored value), not "
                    "rounded numbers"), out_path)
    print(f"[regression] passed {passed} ({time.time() - t0:.0f} s)")
    if not passed:
        raise SystemExit("regression FAILED: new AO configurations blocked")


# ---------------------------------------------------------------- one new AO configuration
def stage_ao(a):
    c, g = a.strategy, a.grouping
    assert g in RB.NEW_AO, f"{g} is not a new AO grouping"
    rpath = ao_result_path(c, g)
    if os.path.exists(rpath):
        print(f"[ao] {ao_name(c, g)} done"); return
    protocol()
    assert (load_json(p("regression.json")) or {}).get("passed"), "regression not passed: new AO runs blocked"
    rp = load_json(p("routing_preparation.json"))
    assert rp and rp["valid"]
    FRD.run_config(SimpleNamespace(strategy=c, cond="AO"), "hybrid", grouping=g, result_path=rpath,
                   wd=ao_weights_dir(c, g), hpath=ao_hessian_path(c, g))
    r = json.load(open(rpath))
    hf = torch.load(r["hessian_file"])
    got = {str(L): [float(hf["tokens"][f"{L}.{e}.gu"]) for e in range(60)] for L in LAYERS}
    exp = {L: [float(x) for x in v] for L, v in rp["expected_ao_token_counts"][c].items()}
    down_ok = all(hf["tokens"][f"{L}.{e}.down"] == hf["tokens"][f"{L}.{e}.gu"] for L in LAYERS for e in range(60))
    inv = dict(token_counts_equal_expected_R_O=got == exp, down_counts_equal_gu=bool(down_ok),
               n_experts_with_tokens=sum(x > 0 for L in got for x in got[L]))
    r["robustness_invariants"] = inv
    r["valid"] = bool(r["valid"] and inv["token_counts_equal_expected_R_O"] and down_ok)
    jdump(r, rpath)
    print(f"[ao] {ao_name(c, g)}: invariants {inv} valid {r['valid']}", flush=True)
    if not r["valid"]:
        raise SystemExit(f"{ao_name(c, g)} failed its invariants")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "capture", "regression", "ao", "report"])
    ap.add_argument("--strategy", choices=STRATS)
    ap.add_argument("--grouping", choices=list(RB.NEW_AO))
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True); os.makedirs(p("results"), exist_ok=True)
    if a.stage == "report":
        import report_frozen_routing_robustness as R
        R.main(); return
    dict(preflight=stage_preflight, capture=stage_capture, regression=stage_regression, ao=stage_ao)[a.stage](a)


if __name__ == "__main__":
    main()
