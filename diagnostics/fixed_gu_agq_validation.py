"""Fixed GU-AGQ robustness study: AGQ gate/up + GPTQ down, fixed for every expert / layer / seed.

Layers 1/8/16/23 (240 routed experts) x C4 Random FR seeds 0/1/2 (CANONICAL-B1, GPTQ 4-bit g128 percdamp 1.0
act-order). Every configuration is assembled ONLY from the two existing validated candidate sets of the SAME seed:
    GPTQ  ~/moe_cache_ebss_official_validation/weights/FR_s{s}
    AGQ   ~/moe_cache_agq_only/weights/AGQ_FR_s{s}
Primary methods: GPTQ, FULL_AGQ, GU_AGQ. DOWN_AGQ is evaluated as a diagnostic only (2x2 Shapley attribution).
Evaluation reuses diagnostics/projection_selective_agq.evaluate_layer (complete mixed-weight expert outputs, frozen BF16
routing, LayerEvalAccumulator cross-check, native BF16 block forward, D + C decomposition).

Stages: audit | protocol | regression | evaluate --split {confirm, oldtest} | tests [--full] | report
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import full_depth_pilot as FD            # noqa: E402
import official_ebss_validation as OV    # noqa: E402
import projection_selective_agq as PA    # noqa: E402  (validated mixed-weight evaluator)
from moe_calib import projection_select as PS  # noqa: E402

OUT = "runs_fixed_gu_agq"
LAYERS = [1, 8, 16, 23]
SEEDS = (0, 1, 2)
E = 60
PRIMARY_CONFIGS = ("GPTQ", "FULL_AGQ", "GU_AGQ")
DIAG_CONFIGS = ("DOWN_AGQ",)
ALL_CONFIGS = ("GPTQ", "GU_AGQ", "DOWN_AGQ", "FULL_AGQ")
SPLITS = PA.SPLITS                                    # confirm (primary), oldtest (secondary)
RULES = dict(tau_rel=0.005, max_layer_regression_rel=0.005, cost_ratio_max=1.10)
HIST_REL = {0: -0.0151, 1: -0.0103, 2: -0.0116}       # reported AGQ vs GPTQ 4-layer means (references only)
PREV_INSPECTED = [(0, 8), (0, 16)]                    # seed/layer cells that motivated GU-AGQ


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


def peak_cpu_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def hist(method, s):
    path = dict(GPTQ=f"runs_ebss_official_validation/results/FR_s{s}.json",
                AGQ=f"runs_agq_only_validation/results/AGQ_FR_s{s}.json")[method]
    return path, json.load(open(os.path.join(ROOT, path)))


def load_candidates(s, L):
    """{0: GPTQ files, 1: AGQ files} of seed s, layer L; file sha256 verified against the historical record."""
    out = {}
    for q, nm in PS.SOURCE.items():
        _, r = hist(nm, s)
        path = os.path.join(r["weights_dir"], f"L{L}.pt")
        assert sha256_file(path) == r["weight_files"][str(L)]["sha256"], f"{nm} s{s} L{L} file changed"
        out[q] = torch.load(path)
    return out


def load_partial():
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, LAYERS))
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    t0 = time.time()
    model, _ = MU.load_model()
    pr = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    tg = PQ.target_tensor_digests(model, LAYERS)
    assert tg == {k: pr["pristine_targets"][k] for k in tg}, "expert weights not pristine"
    return model, time.time() - t0


def code_hashes():
    files = ["diagnostics/fixed_gu_agq_validation.py", "diagnostics/projection_selective_agq.py",
             "moe_calib/projection_select.py", "moe_calib/agq.py", "moe_calib/gptq.py", "moe_calib/profiler.py",
             "moe_calib/hessian.py", "moe_calib/model_utils.py", "tests/test_fixed_gu_agq.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))}


# ============================================================================ stage: audit
def stage_audit(a):
    cm_agq = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "calibration_manifest.json")))
    inv = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), layers=LAYERS, seeds=list(SEEDS), seeds_detail={})
    cman = dict(configs={c: PS.CONFIGS[c] for c in ALL_CONFIGS}, primary=list(PRIMARY_CONFIGS),
                diagnostic=list(DIAG_CONFIGS), seeds={})
    for s in SEEDS:
        sd = dict(candidates={}, layers={})
        rows = {}
        for nm in ("GPTQ", "AGQ"):
            rp, r = hist(nm, s)
            rows[nm] = r["cache_rows"]
            sd["candidates"][nm] = dict(result_json=rp, weights_dir=r["weights_dir"], gptq=r["gptq"],
                                        hessian_weighting=r.get("hessian_weighting", "none"), policy=r.get("policy"),
                                        valid=r["valid"], rtn_fallbacks=r["rtn_fallbacks"],
                                        installed_weights_sha256=r["installed_weights_sha256"],
                                        cache_rows_sha256=hashlib.sha256(json.dumps(r["cache_rows"]).encode()).hexdigest())
        sd["same_calibration_rows"] = rows["GPTQ"] == rows["AGQ"] == cm_agq["configs"][f"FR_s{s}"]["cache_rows"]
        sd["calibration_ids_hash"] = cm_agq["configs"][f"FR_s{s}"]["ids_hash"]
        sd["same_gptq_config"] = sd["candidates"]["GPTQ"]["gptq"] == sd["candidates"]["AGQ"]["gptq"]
        cman["seeds"][str(s)] = {}
        for L in LAYERS:
            lay = {}
            tsha = {}
            for q, nm in PS.SOURCE.items():
                _, r = hist(nm, s)
                path = os.path.join(r["weights_dir"], f"L{L}.pt")
                rec = dict(path=path, exists=os.path.exists(path))
                if rec["exists"]:
                    f = torch.load(path)
                    rec["file_sha256"] = sha256_file(path)
                    rec["file_sha256_matches_record"] = rec["file_sha256"] == r["weight_files"][str(L)]["sha256"]
                    rec["missing"] = [f"{e}.{k}" for e in range(E) for k in PS.PROJ if f"{e}.{k}" not in f]
                    rec["n_experts"] = len({k.split(".")[0] for k in f}); rec["n_tensors"] = len(f)
                    rec["shapes"] = {k: list(f[f"0.{k}"].shape) for k in PS.PROJ}
                    rec["shapes_uniform"] = all(list(f[f"{e}.{k}"].shape) == rec["shapes"][k]
                                                for e in range(E) for k in PS.PROJ)
                    rec["all_bf16"] = all(t.dtype == torch.bfloat16 for t in f.values())
                    rec["all_finite"] = all(bool(torch.isfinite(t).all()) for t in f.values())
                    tsha[q] = {key: OV.tensor_sha(t) for key, t in f.items()}
                    del f
                rec["valid"] = bool(rec["exists"] and rec.get("file_sha256_matches_record") and not rec.get("missing")
                                    and rec.get("n_experts") == E and rec.get("n_tensors") == 3 * E
                                    and rec.get("shapes_uniform") and rec.get("all_bf16") and rec.get("all_finite"))
                lay[nm] = rec
            lay["valid"] = lay["GPTQ"]["valid"] and lay["AGQ"]["valid"]
            lay["candidates_differ_all_experts"] = all(any(tsha[0][f"{e}.{k}"] != tsha[1][f"{e}.{k}"] for k in PS.PROJ)
                                                       for e in range(E)) if lay["valid"] else False
            sd["layers"][str(L)] = lay
            cman["seeds"][str(s)][str(L)] = {}
            for c in ALL_CONFIGS:
                _, man = PS.build_projection_config(tsha, c, E)       # values are hashes: manifest of sources
                for r_ in man:
                    r_["tensor_sha256"] = tsha[r_["q"]][f"{r_['expert']}.{r_['projection']}"]
                    r_["source_file"] = lay[PS.SOURCE[r_["q"]]]["path"]
                cman["seeds"][str(s)][str(L)][c] = dict(manifest_sha256=PS.manifest_hash(man), projections=man,
                                                         n_from_agq=sum(x["q"] == 1 for x in man))
            print(f"[audit] s{s} L{L}: valid {lay['valid']}", flush=True)
        inv["seeds_detail"][str(s)] = sd
    inv["all_valid"] = all(sd["same_calibration_rows"] and sd["same_gptq_config"] and
                           all(v["valid"] and v["candidates_differ_all_experts"] for v in sd["layers"].values())
                           for sd in inv["seeds_detail"].values())
    inv["new_gptq_solves_required"] = 0 if inv["all_valid"] else None
    jdump(inv, p("artifact_inventory.json"))
    jdump(cman, p("configuration_manifest.json"))
    print(f"[audit] all valid: {inv['all_valid']}")


# ============================================================================ stage: protocol
def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        print("[protocol] already frozen"); return
    inv = jload("artifact_inventory.json")
    assert inv["all_valid"]
    prot = dict(
        experiment="Fixed GU-AGQ (AGQ gate/up + GPTQ down) across 4 MoE layers and 3 matched C4 Random seeds",
        nature="retrospective robustness / mechanism study; GU-AGQ was motivated by L8/L16 seed-0 results and both "
               "evaluation splits were inspected before; previously inspected cells: " + str(PREV_INSPECTED),
        model="Qwen/Qwen1.5-MoE-A2.7B (BF16)", layers=LAYERS, seeds=list(SEEDS),
        gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True, capture="CANONICAL-B1 non-sequential BF16",
                  new_solves=0),
        methods=dict(primary={c: PS.CONFIGS[c] for c in PRIMARY_CONFIGS},
                     diagnostic={c: PS.CONFIGS[c] for c in DIAG_CONFIGS},
                     fixed_rule="GU_AGQ for every expert, layer and seed; no selection of any kind",
                     source="the two candidate sets of the SAME seed only"),
        candidate_sha256={str(s): {nm: inv["seeds_detail"][str(s)]["candidates"][nm]["installed_weights_sha256"]
                                   for nm in ("GPTQ", "AGQ")} for s in SEEDS},
        configuration_manifest_sha256=sha256_file(p("configuration_manifest.json")),
        splits=dict(primary=dict(SPLITS["confirm"]), secondary=dict(SPLITS["oldtest"])),
        metric="E_L = sum ||y_BF16 - y_q||^2 / (sum ||y_BF16||^2 + 1e-30) (AGQ-only validation definition: fp32 "
               "complete expert outputs, float64 sums, frozen BF16 routing fp32 softmax top-4 c without renorm, "
               "shared expert included)",
        aggregate="E_4L = mean of the four per-layer ratios (pooled ratio reported as secondary diagnostic only)",
        comparisons=dict(primary="Delta_s = (E4L_GU - E4L_FULL_AGQ) / E4L_FULL_AGQ per seed; mean over seeds",
                         secondary="(E4L_GU - E4L_GPTQ) / E4L_GPTQ", replication_unit="calibration seed"),
        decision_rule=dict(
            rules=RULES,
            BLOCKED="any artifact / regression / correctness check fails",
            PROMISING="all: (1) mean_s Delta_s <= -tau (confirm); (2) Delta_s < 0 for all 3 seeds; (3) for each layer, "
                      "seed-averaged E_L(GU) vs seed-averaged E_L(FULL_AGQ) <= +0.5 %; (4) old-test mean_s Delta_s < 0; "
                      "(5) correctness; (6) estimated fresh-deployment cost of a direct GU-AGQ pass <= 1.10x Full AGQ",
            UNSTABLE="not promising and either (per-seed Delta_s signs differ and max_s Delta_s - min_s Delta_s >= tau) "
                     "or (old-test mean Delta has the opposite sign to confirm mean Delta and |old-test mean Delta| >= "
                     "tau)",
            SMALL_GAIN="not above; Delta_s < 0 for all seeds, old-test mean Delta < 0, mean Delta > -tau",
            LAYER_SPECIFIC="not above; some layer's seed-averaged GU vs FULL_AGQ <= -tau and another's >= +tau",
            NO_GAIN="otherwise (GU-AGQ equal to or worse than Full AGQ, or no consistent improvement)",
            recommendation=dict(A="PROMISING", C="SMALL_GAIN", B="all other verdicts"),
            implementation="diagnostics/fixed_gu_agq_validation.classify (unit-tested)"),
        diagnostics=["DOWN_AGQ for exact 2-group Shapley attribution (not a candidate)", "per-expert diagonal gain "
                     "concentration", "L16 expert 40 across seeds (post hoc)", "D + C decomposition",
                     "native BF16 ranking"],
        conditional="direct single-pass GU-AGQ path implemented and timed ONLY if PROMISING",
        regression=dict(split="oldtest", configs=["GPTQ", "FULL_AGQ"], all_seeds_layers=True,
                        reference="runs_agq_only_validation/results/eval_FR_s{s}.json per-window values",
                        also="confirm GPTQ vs runs_legacy_metric_reeval/results/confirm (FR_s{s})"),
        code_sha256=code_hashes(), frozen_before_any_new_result=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(prot, p("protocol.json"))
    print("[protocol] frozen")


# ============================================================================ decision (pure)
def classify(rules, conf_delta: dict, old_delta: dict, layer_rel_seedavg: dict, correctness_ok: bool,
             cost_ratio: float) -> dict:
    """conf_delta / old_delta = {seed: (E4L_GU - E4L_AGQ)/E4L_AGQ}; layer_rel_seedavg = {L: rel of seed-averaged
    E_L(GU) vs E_L(FULL_AGQ)}; cost_ratio = estimated direct GU-AGQ prep cost / Full AGQ prep cost."""
    if not correctness_ok:
        return dict(verdict="FIXED-GU-BLOCKED", criteria={})
    tau = rules["tau_rel"]
    d = np.array(list(conf_delta.values())); o = np.array(list(old_delta.values()))
    md, mo = float(d.mean()), float(o.mean())
    crit = {"1_mean_delta_le_minus_tau": md <= -tau, "2_all_seeds_improve": bool((d < 0).all()),
            "3_no_layer_regression_gt_threshold": max(layer_rel_seedavg.values()) <= rules["max_layer_regression_rel"],
            "4_oldtest_direction_agrees": mo < 0, "5_correctness": True,
            "6_cost_ratio_ok": cost_ratio <= rules["cost_ratio_max"]}
    signs_mixed = len(set(np.sign(d))) > 1
    unstable = (signs_mixed and float(d.max() - d.min()) >= tau) or (np.sign(mo) != np.sign(md) and abs(mo) >= tau)
    lay = list(layer_rel_seedavg.values())
    if all(crit.values()):
        v = "FIXED-GU-PROMISING"
    elif unstable:
        v = "FIXED-GU-UNSTABLE"
    elif (d < 0).all() and mo < 0 and md > -tau:
        v = "FIXED-GU-SMALL-GAIN"
    elif min(lay) <= -tau and max(lay) >= tau:
        v = "FIXED-GU-LAYER-SPECIFIC"
    else:
        v = "FIXED-GU-NO-GAIN"
    rec = {"FIXED-GU-PROMISING": "A", "FIXED-GU-SMALL-GAIN": "C"}.get(v, "B")
    return dict(verdict=v, recommendation=rec, criteria=crit, mean_delta_confirm=md, mean_delta_oldtest=mo,
                median_delta_confirm=float(np.median(d)), seed_spread=float(d.max() - d.min()),
                seeds_improved=int((d < 0).sum()), unstable=bool(unstable), layer_rel_seedavg=layer_rel_seedavg,
                cost_ratio=cost_ratio)


# ============================================================================ evaluation runs
def _run(split_key, split_name, configs, accumulator_check):
    """Evaluate configs for all seeds x layers; results under results/{split_key}/s{s}/L{L}.json."""
    model, t_load = load_partial()
    torch.cuda.reset_peak_memory_stats()
    cost = dict(model_load_s=t_load, cells={})
    for s in SEEDS:
        for L in LAYERS:
            outp = p("results", split_key, f"s{s}", f"L{L}.json")
            if os.path.exists(outp):
                print(f"[{split_key}] s{s} L{L} done"); continue
            t0 = time.time()
            cands = load_candidates(s, L)
            tl = time.time() - t0
            out, mats, meta = PA.evaluate_layer(model, L, split_name, configs, cands, native=True,
                                                accumulator_check=accumulator_check)
            meta.update(seed=s, candidate_load_s=tl, total_s=time.time() - t0)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            np.savez(p("results", split_key, f"s{s}", f"expert_mats_L{L}.npz"), **mats)
            jdump(dict(configs=out, meta=meta), outp)
            cost["cells"][f"s{s}_L{L}"] = dict(seconds=time.time() - t0, candidate_load_s=tl, lean_s=meta["lean_s"],
                                               native_s=meta["native_s"])
            print(f"[{split_key}] s{s} L{L}: " + "  ".join(f"{c} {out[c]['moe_layer_output_rel_err']:.6e}" for c in out)
                  + f" ({time.time() - t0:.0f} s)", flush=True)
            del cands
    cost.update(peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=peak_cpu_gb())
    jdump(cost, p("logs", f"cost_{split_key}.json"))


def stage_regression(a):
    assert os.path.exists(p("protocol.json"))
    _run("regression_oldtest", "oldtest", ("GPTQ", "FULL_AGQ"), accumulator_check=True)
    res = dict(cells={}, note="old-test split; GPTQ and FULL_AGQ only, before any GU/DOWN result")
    for s in SEEDS:
        ev = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "results", f"eval_FR_s{s}.json")))
        for L in LAYERS:
            r = json.load(open(p("results", "regression_oldtest", f"s{s}", f"L{L}.json")))
            cell = {}
            for c, hn in (("GPTQ", "GPTQ"), ("FULL_AGQ", "AGQ")):
                h, o = ev["layers"][str(L)][hn], r["configs"][c]
                cell[c] = dict(value=o["moe_layer_output_rel_err"], historical=h["moe_layer_output_rel_err"],
                               rel_diff=abs(o["moe_layer_output_rel_err"] / h["moe_layer_output_rel_err"] - 1),
                               per_window_exact=[x / (y + 1e-30) for x, y in zip(o["per_seq"]["num"],
                                                                                 o["per_seq"]["den"])]
                               == h["per_seq_moe_layer_output_rel_err"],
                               native=o["native_rel_err"], native_historical=h["moe_layer_output_native_rel_err"],
                               native_exact=o["native_rel_err"] == h["moe_layer_output_native_rel_err"])
            m = r["meta"]
            cell["accumulator_check"] = m["accumulator_check"]
            cell["passed"] = bool(all(cell[c]["rel_diff"] <= 1e-9 and cell[c]["per_window_exact"] and
                                      cell[c]["native_exact"] for c in ("GPTQ", "FULL_AGQ"))
                                  and all(v["abs_diff"] <= 1e-12 and v["per_seq_equal"]
                                          for v in m["accumulator_check"].values())
                                  and m["pristine_restored"] and m["install_matches_config"]
                                  and m["nontarget_unchanged_during_install"] and m["reinstall_bitwise_identical"])
            res["cells"][f"s{s}_L{L}"] = cell
            print(f"[regression] s{s} L{L}: passed {cell['passed']}", flush=True)
        g = np.mean([res["cells"][f"s{s}_L{L}"]["GPTQ"]["value"] for L in LAYERS])
        d = np.mean([res["cells"][f"s{s}_L{L}"]["FULL_AGQ"]["value"] for L in LAYERS])
        res[f"seed{s}_agq_vs_gptq_4layer"] = dict(measured=float((d - g) / g), reported=HIST_REL[s],
                                                  matches_reported_to_0_01pt=bool(abs((d - g) / g - HIST_REL[s]) < 1e-4))
    res["passed"] = all(v["passed"] for v in res["cells"].values()) and all(
        res[f"seed{s}_agq_vs_gptq_4layer"]["matches_reported_to_0_01pt"] for s in SEEDS)
    jdump(res, p("candidate_regression.json"))
    print(f"[regression] passed={res['passed']}")
    if not res["passed"]:
        raise SystemExit("historical regression FAILED -> FIXED-GU-BLOCKED")


def stage_evaluate(a):
    assert jload("candidate_regression.json")["passed"]
    if a.split == "confirm":
        _run("confirm", "confirm", ALL_CONFIGS, accumulator_check=True)
    else:   # GPTQ / FULL_AGQ on old-test are the regression results (identical windows and code)
        _run("oldtest", "oldtest", ("GU_AGQ", "DOWN_AGQ"), accumulator_check=False)


def stage_tests(a):
    env = dict(os.environ, MOE_DEVICE="cpu")
    env.pop("MOE_LOAD_LAYERS", None)
    for tag, args in (("fixed_gu", ["tests/test_fixed_gu_agq.py", "tests/test_projection_selective_agq.py"]),
                      ("all", ["tests"])):
        if tag == "all" and not a.full:
            continue
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, env=env, capture_output=True,
                           text=True)
        open(p("logs", f"pytest_{tag}.log"), "w").write(r.stdout + r.stderr)
        print(f"[tests] {tag}: rc={r.returncode} :: {r.stdout.strip().splitlines()[-1] if r.stdout else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage == "report":
        import report_fixed_gu_agq as R
        return R.main()
    dict(audit=stage_audit, protocol=stage_protocol, regression=stage_regression, evaluate=stage_evaluate,
         tests=stage_tests)[a.stage](a)


if __name__ == "__main__":
    main()
