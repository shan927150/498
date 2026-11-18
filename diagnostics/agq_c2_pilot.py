"""EXPLORATORY small pilot: affinity exponent q in {0, 1, 2} for MoE GPTQ Hessians, H_q = sum_i c_i^q x_i x_i^T
(q = 0 standard GPTQ, q = 1 published MoEQuant AGQ, q = 2 experimental c^2 = diagonal of the MoE-output objective).

Frozen scope: Qwen1.5-MoE-A2.7B, FR s0 (C4 Random B = 128 x 512, CANONICAL-B1 cache), layers 8 and 16 (outcome-informed),
6 experts per layer chosen by FR s0 calibration routed-token counts only (2 lowest / 2 median / 2 highest), GPTQ 4-bit g128
percdamp 1.0 act-order, solver unchanged. Evaluation on ~/moe_cache rows 2128-2255 with frozen BF16 routing; aggregated
layer output via SUBSET REPLACEMENT (54 other experts fixed at the historical q = 0 weights).

Stages: freeze | correctness | quantize | evaluate | report
"""
from __future__ import annotations

import argparse, json, os, resource, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import agq_only_validation as AV    # noqa: E402
from moe_calib import agq as AG     # noqa: E402

OUT = "runs_agq_c2_pilot"
WDIR = os.path.expanduser(os.environ.get("AGQ_C2_WEIGHTS", "~/moe_cache_agq_c2_pilot"))
LAYERS = [8, 16]
CONFIG = "FR_s0"
QS = (0, 1, 2)
E = AV.E
PROJ = (("gate_proj", "gu"), ("up_proj", "gu"), ("down_proj", "down"))


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = AV.jdump, AV.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


def code_hashes():
    fs = ["moe_calib/agq.py", "moe_calib/profiler.py", "moe_calib/hessian.py", "moe_calib/gptq.py",
          "diagnostics/agq_c2_pilot.py", "diagnostics/agq_only_validation.py", "tests/test_agq_c2.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in fs}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ===================================================================== freeze
def select_experts():
    cnt = AV.calib_counts(CONFIG)                                   # [240] FR s0 calibration routed tokens
    out = {}
    for L in LAYERS:
        li = AV.LAYERS.index(L)
        c = cnt[li * E:(li + 1) * E]
        order = sorted(range(E), key=lambda e: (int(c[e]), e))      # ascending count, expert id tie-break
        out[str(L)] = dict(low=order[:2], median=order[E // 2 - 1:E // 2 + 1], high=order[-2:],
                           counts={str(e): int(c[e]) for e in range(E)})
    return out


def stage_freeze(a):
    os.makedirs(p("logs"), exist_ok=True)
    if os.path.exists(p("protocol.json")):
        print("[freeze] already frozen"); return
    sel = select_experts()
    jdump(dict(rule="FR s0 calibration routed-token counts (runs_ebss_official_validation capture_routing_nll.npz, "
                    "rows of FR_s0) within each layer; sort ascending by (count, expert id); low = ranks 0-1, "
                    "median = ranks 29-30, high = ranks 58-59",
               not_used=["AGQ improvement", "held-out error", "c^2 performance", "PPL"], layers=sel),
          p("selected_experts.json"))
    g = AV.jload("protocol.json")["gptq"]
    proto = dict(
        experiment="EXPLORATORY c^2-weighted Hessian pilot (novel modification of AGQ, not MoEQuant)",
        hessian="H_q = sum_i c_i^q x_i x_i^T, q in {0,1,2}; implemented as rows scaled by c^(q/2) (q=0: standard path)",
        c="fp32 softmax(router logits)[selected top-4], no renormalization (runs_agq_only_validation)",
        objective_note="c^2 is only the diagonal of the aggregated MoE output-error objective; cross-expert terms ignored",
        model="Qwen/Qwen1.5-MoE-A2.7B @ 1a758c50", gptq=g, calibration=CONFIG + " (CANONICAL-B1 cache, 128 x 512)",
        layers=LAYERS, layer_selection="outcome-informed (AGQ L8 slightly worse, L16 -3.1 %); exploratory, NOT confirmatory",
        experts="selected_experts.json (12 experts, 36 projections per q)",
        baselines=dict(q0="must reproduce historical FR_s0 GPTQ weights bitwise for the 12 experts",
                       q1="must reproduce the AGQ_FR_s0 weights (runs_agq_only_validation) bitwise"),
        evaluation=dict(
            heldout="~/moe_cache rows 2128-2255, all routed tokens, frozen BF16 routing, fp32 math",
            expert_metrics="E0/E1/E2 = sum c^j ||DeltaE||^2 / sum c^j ||E||^2 on the full routed expert MLP output; "
                           "reported pooled over tokens of the 12 experts and as mean of per-expert ratios",
            projection_metrics="sum c^j ||(W - W_q) x_p||^2 / sum c^j ||W x_p||^2 per projection, j in {0,1,2}",
            subset_layer_output="A: all 60 experts historical q=0; B: 6 selected -> q=1; C: 6 selected -> q=2; "
                                "error = sum ||y_BF16 - y||^2 / sum ||y_BF16||^2 (shared expert BF16 weights in y)",
            decomposition="||Delta y||^2 = D + C, D = sum_k c_k^2 ||DeltaE_k||^2, C = 2 sum_{a<b} c_a c_b <DeltaE_a, DeltaE_b>"),
        decision_rule=dict(
            surrogate_gain_rel=0.01, max_layer_regression_rel=0.005,
            cost=dict(max_time_increase_vs_q0=0.25, max_peak_gpu_increase_vs_q0=0.10,
                      measured_as="Profiler Hessian pass + 36 GPTQ solves per layer"),
            C2_PILOT_PROMISING="pooled E2(q2) <= -1 % vs q1 AND layer output (C vs B) improves in >= 1 of L8/L16 AND no "
                               "layer regresses by > 0.5 % vs q1 AND correctness AND cost modest",
            C2_REGRESSION="any layer output (C vs B) > +0.5 %, or unstable quantization (non-finite / dead-column change)",
            C2_SURROGATE_ONLY="pooled E2 criterion met but layer-output criteria not",
            C2_NO_GAIN="otherwise", C2_BLOCKED="correctness / stability / missing baseline",
            note="screening rules, not significance tests; frozen before any q = 2 quality result"),
        limits=["one seed", "two layers", "12 experts", "q in {0,1,2} only", "no PPL", "no damping change",
                "no extrapolation from subset to full layer"],
        code_sha256=code_hashes(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(proto, p("protocol.json"))
    print("[freeze] experts:", {L: (v["low"], v["median"], v["high"]) for L, v in sel.items()})


def experts_of(L):
    s = jload("selected_experts.json")["layers"][str(L)]
    return [(g, e) for g in ("low", "median", "high") for e in s[g]]


# ===================================================================== correctness
@torch.no_grad()
def stage_correctness(a):
    """pytest (AGQ + c^2 suites) and a real-data q = 2 oracle: production Profiler(router_affinity, power 2) Hessians vs
    float64 X^T diag(c^2) X from the same forward (layer 8, 4 calibration rows, all 60 experts, gu and down)."""
    env = dict(os.environ, MOE_DEVICE="cpu"); env.pop("MOE_LOAD_LAYERS", None)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_agq.py", "tests/test_agq_c2.py"], cwd=ROOT,
                       env=env, capture_output=True, text=True)
    open(p("logs", "pytest_agq_c2.log"), "w").write(r.stdout + r.stderr)
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    model, _ = AV.load_partial()
    rows, _ = AV.config_rows(CONFIG)
    rows = rows[:4]
    L = 8
    cache = load_layer_cache(AV.ECACHE, L)
    blk = MU.moe_block(model, L)
    X = {e: {"gu": [], "down": []} for e in range(E)}
    hooks = []
    for e in range(E):
        hooks.append(blk.experts[e].gate_proj.register_forward_pre_hook(
            lambda m, a_, e=e: X[e]["gu"].append(a_[0].detach().double().cpu())))
        hooks.append(blk.experts[e].down_proj.register_forward_pre_hook(
            lambda m, a_, e=e: X[e]["down"].append(a_[0].detach().double().cpu())))
    prs = {q: Profiler(model, L, K=1, hessian_weighting="router_affinity", collect_affinity=True, affinity_power=q)
           for q in (1, 2)}
    try:
        prs[2].run(cache, rows)
    finally:
        for h in hooks:
            h.remove()
    prs[1].run(cache, rows)
    std = Profiler(model, L, K=1).run(cache, rows)
    res, worst = {}, 0.0
    for e in range(E):
        lst = prs[2].affinity.get((L, e), [])
        if not lst:
            continue
        c = torch.cat([c_ for *_, c_ in lst]).double().numpy()
        for proj in ("gu", "down"):
            Xe = torch.cat(X[e][proj]).numpy()
            ref2 = AG.weighted_hessian_reference_q(Xe, c, 2)
            got2 = prs[2].acc.H[(L, e, proj)][0].double().cpu().numpy()
            got1 = prs[1].acc.H[(L, e, proj)][0].double().cpu().numpy()
            got0 = std.acc.H[(L, e, proj)][0].double().cpu().numpy()
            n2 = np.linalg.norm(ref2)
            rec = dict(rows=int(c.size), rel_err_q2_vs_oracle=float(np.linalg.norm(got2 - ref2) / n2),
                       rel_err_q1_vs_oracle=float(np.linalg.norm(got1 - AG.weighted_hessian_reference_q(Xe, c, 1)) /
                                                  np.linalg.norm(AG.weighted_hessian_reference_q(Xe, c, 1))),
                       rel_err_q0_vs_oracle=float(np.linalg.norm(got0 - Xe.T @ Xe) / np.linalg.norm(Xe.T @ Xe)),
                       rel_dist_q2_to_c4=float(np.linalg.norm(got2 - AG.weighted_hessian_reference_q(Xe, c, 4)) / n2),
                       rel_dist_q2_to_c1=float(np.linalg.norm(got2 - AG.weighted_hessian_reference_q(Xe, c, 1)) / n2),
                       symmetric=bool(np.array_equal(got2, got2.T)),
                       min_diag_q2=float(np.diag(got2).min()), n_zero_diag_q0=int((np.diag(got0) == 0).sum()),
                       n_zero_diag_q2=int((np.diag(got2) == 0).sum()))
            res[f"{e}.{proj}"] = rec
            worst = max(worst, rec["rel_err_q2_vs_oracle"], rec["rel_err_q1_vs_oracle"], rec["rel_err_q0_vs_oracle"])
    tol = 1e-5
    oracle = dict(layer=L, rows=rows, tolerance=tol, worst_rel_err=worst, experts=res,
                  min_dist_to_c4=min(v["rel_dist_q2_to_c4"] for v in res.values()),
                  min_dist_to_c1=min(v["rel_dist_q2_to_c1"] for v in res.values()),
                  dead_columns_unchanged=all(v["n_zero_diag_q0"] == v["n_zero_diag_q2"] for v in res.values()),
                  all_symmetric=all(v["symmetric"] for v in res.values()))
    oracle["passed"] = bool(worst <= tol and oracle["min_dist_to_c4"] > 100 * tol and oracle["min_dist_to_c1"] > 100 * tol
                            and oracle["dead_columns_unchanged"] and oracle["all_symmetric"])
    jdump(oracle, p("oracle_results.json"))
    out = dict(pytest_rc=r.returncode, pytest=r.stdout.strip().splitlines()[-1] if r.stdout else "",
               real_data_oracle_passed=oracle["passed"],
               prior_agq_gates=AV.jload("decision.json")["gates"])
    out["passed"] = bool(r.returncode == 0 and oracle["passed"] and all(out["prior_agq_gates"].values()))
    jdump(out, p("correctness_results.json"))
    print(f"[correctness] pytest rc {r.returncode} ({out['pytest']}); oracle worst {worst:.2e} passed {oracle['passed']}")
    if not out["passed"]:
        raise SystemExit("correctness FAILED")


# ===================================================================== quantize
@torch.no_grad()
def stage_quantize(a):
    from moe_calib import gptq as GQ
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import fp_weights
    assert jload("correctness_results.json")["passed"]
    if os.path.exists(p("regression_results.json")) and os.path.exists(p("logs", "quantize_raw.json")):
        print("[quantize] done"); return
    os.makedirs(WDIR, exist_ok=True)
    model, _ = AV.load_partial()
    g = AV.jload("protocol.json")["gptq"]
    rows, K = AV.config_rows(CONFIG)
    hist = {"GPTQ": AV.weights_dir("GPTQ", CONFIG), "AGQ": AV.weights_dir("AGQ", CONFIG)}
    reg, raw, cost, files = {}, {}, {}, {}
    for L in LAYERS:
        cache = load_layer_cache(AV.ECACHE, L)
        exps = [e for _, e in experts_of(L)]
        Hq, Wq, info, aff = {}, {}, {}, None
        for q in QS:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            pr = (Profiler(model, L, K=K) if q == 0 else
                  Profiler(model, L, K=K, hessian_weighting="router_affinity", affinity_power=q,
                           collect_affinity=(q == 1)))
            pr.run(cache, rows, weights=[1.0] * len(rows))
            t_prof = pr.seconds
            Hq[q] = {(e, hk): pr.acc.H_sum((L, e, hk), list(range(K))).clone() for e in exps for hk in ("gu", "down")}
            tok = {e: pr.acc.tokens((L, e, "gu")) for e in exps}
            if q == 1:
                aff = pr.affinity
            pr.free(); del pr
            t0 = time.time()
            Wq[q], info[q] = {}, {}
            for e in exps:
                W = fp_weights(model, L, e)
                for proj, hk in PROJ:
                    inf = {}
                    Wq[q][(e, proj)] = GQ.gptq_quantize(W[proj], Hq[q][(e, hk)], g["bits"], g["group_size"],
                                                        g["percdamp"], info=inf)
                    info[q][(e, proj)] = inf
            torch.cuda.synchronize()
            cost[f"L{L}_q{q}"] = dict(profiler_s=t_prof, gptq_s=time.time() - t0, total_s=t_prof + time.time() - t0,
                                      peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_rss_gb=rss_gb())
            Hq[q] = {k: v.cpu() for k, v in Hq[q].items()}
            path = os.path.join(WDIR, f"L{L}_q{q}.pt")
            torch.save({f"{e}.{proj}": Wq[q][(e, proj)].to(torch.bfloat16).cpu() for e in exps for proj, _ in PROJ}, path)
            files[f"L{L}_q{q}"] = dict(path=path, sha256=sha256_file(path))
            print(f"[quantize] L{L} q={q}: profiler {t_prof:.1f} s, gptq {cost[f'L{L}_q{q}']['gptq_s']:.1f} s", flush=True)
        # regression: q0 == historical GPTQ, q1 == historical AGQ (BF16 installed weights), per expert/projection
        hf = {m: torch.load(os.path.join(hist[m], f"L{L}.pt")) for m in hist}
        for e in exps:
            for proj, _ in PROJ:
                reg[f"L{L}.{e}.{proj}"] = dict(
                    q0_equals_historical_gptq=bool(torch.equal(Wq[0][(e, proj)].to(torch.bfloat16).cpu(), hf["GPTQ"][f"{e}.{proj}"])),
                    q1_equals_historical_agq=bool(torch.equal(Wq[1][(e, proj)].to(torch.bfloat16).cpu(), hf["AGQ"][f"{e}.{proj}"])),
                    q2_finite=bool(torch.isfinite(Wq[2][(e, proj)]).all()),
                    q2_dead_cols=int(info[2][(e, proj)]["n_dead"]), q0_dead_cols=int(info[0][(e, proj)]["n_dead"]))
        # diagnostics
        for grp, e in experts_of(L):
            c = AV.expert_affinities(aff, L, e)
            W = fp_weights(model, L, e)
            rec = dict(layer=L, expert=e, exposure=grp, tokens=int(tok[e]), mean_c=float(c.mean()),
                       cv_c=float(c.std() / c.mean()), n=int(c.size),
                       n_eff={str(q): AG.effective_sample_size_q(c, q) for q in QS},
                       n_eff_frac={str(q): AG.effective_sample_size_q(c, q) / c.size for q in QS}, proj={})
            for proj, hk in PROJ:
                pr_ = {}
                for qa, qb in ((0, 1), (0, 2), (1, 2)):
                    hd = AG.hessian_diagnostics(Hq[qa][(e, hk)].to(MU.GPU), Hq[qb][(e, hk)].to(MU.GPU))
                    wd = AG.weight_diagnostics(Wq[qa][(e, proj)], Wq[qb][(e, proj)], W[proj].to(MU.GPU),
                                               info[qa][(e, proj)], info[qb][(e, proj)])
                    pr_[f"q{qa}_vs_q{qb}"] = dict(hessian=hd, weights=wd)
                pr_["quant_err_rel"] = {str(q): float(torch.linalg.norm(Wq[q][(e, proj)] - W[proj].to(MU.GPU)) /
                                                      torch.linalg.norm(W[proj])) for q in QS}
                rec["proj"][proj] = pr_
            raw[f"L{L}.{e}"] = rec
        del cache, Hq, Wq, info, hf
        torch.cuda.empty_cache()
    reg_sum = dict(per_projection=reg, n=len(reg),
                   q0_reproduces_historical_gptq=all(v["q0_equals_historical_gptq"] for v in reg.values()),
                   q1_reproduces_historical_agq=all(v["q1_equals_historical_agq"] for v in reg.values()),
                   q2_finite=all(v["q2_finite"] for v in reg.values()),
                   dead_columns_unchanged=all(v["q2_dead_cols"] == v["q0_dead_cols"] for v in reg.values()),
                   weight_files=files)
    reg_sum["passed"] = bool(reg_sum["q0_reproduces_historical_gptq"] and reg_sum["q1_reproduces_historical_agq"]
                             and reg_sum["q2_finite"] and reg_sum["dead_columns_unchanged"])
    jdump(reg_sum, p("regression_results.json"))
    jdump(dict(experts=raw), p("logs", "quantize_raw.json"))
    jdump(cost, p("logs", "cost_raw.json"))
    print(f"[quantize] regression passed={reg_sum['passed']} (q0 {reg_sum['q0_reproduces_historical_gptq']}, "
          f"q1 {reg_sum['q1_reproduces_historical_agq']})")
    if not reg_sum["passed"]:
        raise SystemExit("q0 / q1 regression FAILED - no quality comparison")


# ===================================================================== evaluate
@torch.no_grad()
def stage_evaluate(a):
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    assert jload("regression_results.json")["passed"]
    if os.path.exists(p("logs", "evaluate_raw.json")):
        print("[evaluate] done"); return
    model, _ = AV.load_partial()
    test_ids = list(range(*AV.jload("protocol.json")["evaluation"]["reconstruction_rows"]))
    k = MU.top_k(model)
    out = {}
    t_start = time.time()
    for L in LAYERS:
        blk = MU.moe_block(model, L)
        dig0 = PQ.layer_digest(model, L)
        exps = [e for _, e in experts_of(L)]
        orig = {e: {kk: w.detach().clone() for kk, w in MU.get_expert_weights(model, L, e).items()} for e in range(E)}
        Wfp = {e: {kk: v.to(MU.GPU, torch.float32) for kk, v in orig[e].items()} for e in range(E)}
        base = torch.load(os.path.join(AV.weights_dir("GPTQ", CONFIG), f"L{L}.pt"))
        pil = {q: torch.load(os.path.join(WDIR, f"L{L}_q{q}.pt")) for q in QS}
        Wq0 = {e: {kk: base[f"{e}.{kk}"].to(MU.GPU, torch.float32) for kk in AG.PROJ} for e in range(E)}
        Wsel = {q: {e: {kk: pil[q][f"{e}.{kk}"].to(MU.GPU, torch.float32) for kk in AG.PROJ} for e in exps} for q in QS}
        conds = {"A_q0": {}, "B_q1_subset": {e: Wsel[1][e] for e in exps}, "C_q2_subset": {e: Wsel[2][e] for e in exps}}
        # expert-level accumulators for the 12 experts x 3 q: [num_j, den_j] j = 0,1,2 for full expert & each projection
        ex = {q: {e: dict(full=np.zeros((3, 2)), **{pj: np.zeros((3, 2)) for pj, _ in PROJ}) for e in exps} for q in QS}
        lay = {cn: dict(total=0.0, diag=0.0, cross=0.0, den=0.0, per_seq=[]) for cn in conds}
        native = {cn: dict(num=0.0, den=0.0) for cn in conds}
        cache = load_layer_cache(AV.OCACHE, L)
        Hin, Y0 = [], []
        with MU.layer_on_gpu(model, L):
            sh = {kk: getattr(blk.shared_expert, kk).weight.float() for kk in AG.PROJ}
            sg = blk.shared_expert_gate.weight.float()
            for sid in test_ids:
                cap = {}
                hk = [blk.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach())),
                      blk.register_forward_hook(lambda m_, i, o: cap.__setitem__("y", o[0].detach())),
                      blk.gate.register_forward_hook(lambda m_, i, o: cap.__setitem__("lg", o.detach()))]
                try:
                    MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
                finally:
                    for h_ in hk:
                        h_.remove()
                h = cap["h"].reshape(-1, cap["h"].shape[-1])
                Hin.append(h); Y0.append(cap["y"].reshape(h.shape))
                hf = h.float()
                vals, sel = AG.routing_coefficients(cap["lg"], k)
                rows = AG.dispatch_rows(sel, vals, E)
                y = F.sigmoid(hf @ sg.T) * AG.expert_fp32(hf, sh)
                d0, dsel = {}, {q: {} for q in QS}
                for e, (tx, slot, c) in rows.items():
                    X = hf[tx]
                    xd = AG.down_input_fp32(X, Wfp[e])
                    o = xd @ Wfp[e]["down_proj"].T
                    y.index_add_(0, tx, o * c[:, None])
                    d0[e] = o - AG.expert_fp32(X, Wq0[e])
                    if e in exps:
                        cd = c.double()
                        cw = [torch.ones_like(cd), cd, cd ** 2]
                        on2 = (o.double() ** 2).sum(1)
                        inp = dict(gate_proj=X, up_proj=X, down_proj=xd)
                        for q in QS:
                            dq = o - AG.expert_fp32(X, Wsel[q][e])
                            dsel[q][e] = dq
                            dn2 = (dq.double() ** 2).sum(1)
                            A = ex[q][e]
                            for j in range(3):
                                A["full"][j] += [float((cw[j] * dn2).sum()), float((cw[j] * on2).sum())]
                            for pj, _ in PROJ:
                                r2 = ((inp[pj] @ (Wfp[e][pj] - Wsel[q][e][pj]).T).double() ** 2).sum(1)
                                t2 = ((inp[pj] @ Wfp[e][pj].T).double() ** 2).sum(1)
                                for j in range(3):
                                    A[pj][j] += [float((cw[j] * r2).sum()), float((cw[j] * t2).sum())]
                yn2 = float((y.double() ** 2).sum())
                for cn, repl in conds.items():
                    deltas = {e: (dsel[1][e] if cn == "B_q1_subset" else dsel[2][e]) if (e in repl) else d0[e]
                              for e in rows}
                    V = AG.slot_contributions(h.shape[0], h.shape[1], k, rows, deltas, MU.GPU)
                    dc = AG.decompose(V)
                    for kk in ("total", "diag", "cross"):
                        lay[cn][kk] += dc[kk]
                    lay[cn]["den"] += yn2
                    lay[cn]["per_seq"].append(dc["total"] / yn2)
            # native BF16 block forward under each condition
            for cn, repl in conds.items():
                for e in range(E):
                    src = pil[1 if cn == "B_q1_subset" else 2] if e in repl else base
                    MU.set_expert_weights(model, L, e, {kk: src[f"{e}.{kk}"] for kk in AG.PROJ})
                for h, y0 in zip(Hin, Y0):
                    yq, _ = blk(h.unsqueeze(0))
                    native[cn]["num"] += float(((yq.reshape(h.shape).double() - y0.double()) ** 2).sum())
                    native[cn]["den"] += float((y0.double() ** 2).sum())
            for e in range(E):
                MU.set_expert_weights(model, L, e, orig[e])
        assert PQ.layer_digest(model, L) == dig0, "pristine weights not restored"
        out[str(L)] = dict(
            layer={cn: dict(**{kk: v[kk] for kk in ("total", "diag", "cross", "den")},
                            rel_err=v["total"] / v["den"], diag_rel=v["diag"] / v["den"], cross_rel=v["cross"] / v["den"],
                            recon_check_abs=abs(v["total"] - v["diag"] - v["cross"]) / v["total"],
                            per_seq=v["per_seq"], native_rel_err=native[cn]["num"] / native[cn]["den"])
                   for cn, v in lay.items()},
            experts={str(e): {str(q): {kk: v.tolist() for kk, v in ex[q][e].items()} for q in QS} for e in exps})
        print(f"[evaluate] L{L}: " + ", ".join(f"{cn} {out[str(L)]['layer'][cn]['rel_err']:.6e}" for cn in conds), flush=True)
        del cache, Hin, Y0, Wfp, Wq0, Wsel
        torch.cuda.empty_cache()
    out["seconds"] = time.time() - t_start
    jdump(out, p("logs", "evaluate_raw.json"))


# ===================================================================== report
def rel(a, b):
    return (a - b) / b


def stage_report(a):
    prot = jload("protocol.json")
    rules = prot["decision_rule"]
    ev = jload("logs", "evaluate_raw.json")
    qr = jload("logs", "quantize_raw.json")["experts"]
    cr = jload("logs", "cost_raw.json")
    reg = jload("regression_results.json")
    corr = jload("correctness_results.json")
    rows, per_layer = [], {}
    pooled = {q: np.zeros((3, 2)) for q in QS}
    proj_pooled = {q: {pj: np.zeros((3, 2)) for pj, _ in PROJ} for q in QS}
    for L in LAYERS:
        for grp, e in experts_of(L):
            for q in QS:
                A = ev[str(L)]["experts"][str(e)][str(q)]
                f = np.array(A["full"])
                pooled[q] += f
                for pj, _ in PROJ:
                    proj_pooled[q][pj] += np.array(A[pj])
                rows.append(dict(layer=L, expert=e, exposure=grp, method=f"q={q}",
                                 E0=f[0, 0] / f[0, 1], E1=f[1, 0] / f[1, 1], E2=f[2, 0] / f[2, 1],
                                 **{f"P{j}_{pj}": np.array(A[pj])[j, 0] / np.array(A[pj])[j, 1] for pj, _ in PROJ
                                    for j in range(3)}))
    def agg(sel):
        out = {}
        for q in QS:
            rr = [r for r in rows if r["method"] == f"q={q}" and sel(r)]
            out[f"q{q}"] = {m: dict(mean=float(np.mean([r[m] for r in rr])), median=float(np.median([r[m] for r in rr])))
                            for m in ("E0", "E1", "E2")}
        for qa, qb in ((1, 0), (2, 0), (2, 1)):
            ra = [r for r in rows if r["method"] == f"q={qa}" and sel(r)]
            rb = [r for r in rows if r["method"] == f"q={qb}" and sel(r)]
            out[f"q{qa}_vs_q{qb}"] = {m: dict(rel_mean=rel(np.mean([r[m] for r in ra]), np.mean([r[m] for r in rb])),
                                              n_improved=int(sum(x[m] < y[m] for x, y in zip(ra, rb))), n=len(ra))
                                      for m in ("E0", "E1", "E2")}
        return out
    pooled_metrics = {f"q{q}": {f"E{j}": float(pooled[q][j, 0] / pooled[q][j, 1]) for j in range(3)} for q in QS}
    for qa, qb in ((1, 0), (2, 0), (2, 1)):
        pooled_metrics[f"q{qa}_vs_q{qb}"] = {f"E{j}": rel(pooled_metrics[f"q{qa}"][f"E{j}"], pooled_metrics[f"q{qb}"][f"E{j}"])
                                             for j in range(3)}
    proj_metrics = {f"q{q}": {pj: {f"P{j}": float(proj_pooled[q][pj][j, 0] / proj_pooled[q][pj][j, 1]) for j in range(3)}
                              for pj, _ in PROJ} for q in QS}
    for qa, qb in ((1, 0), (2, 0), (2, 1)):
        proj_metrics[f"q{qa}_vs_q{qb}"] = {pj: {f"P{j}": rel(proj_metrics[f"q{qa}"][pj][f"P{j}"], proj_metrics[f"q{qb}"][pj][f"P{j}"])
                                                for j in range(3)} for pj, _ in PROJ}
    exp_rec = dict(note="12 outcome-informed-layer experts; not population replicates", rows=rows,
                   mean_per_expert=dict(all=agg(lambda r: True),
                                        **{f"L{L}": agg(lambda r, L=L: r["layer"] == L) for L in LAYERS},
                                        **{g: agg(lambda r, g=g: r["exposure"] == g) for g in ("low", "median", "high")}),
                   pooled_tokens=pooled_metrics)
    jdump(exp_rec, p("expert_reconstruction.json"))
    jdump(dict(note="projection-level sum c^j ||(W-Wq)x||^2 / sum c^j ||Wx||^2, pooled over the 12 experts",
               pooled=proj_metrics), p("weighted_reconstruction.json"))
    sub, dec = {}, {}
    for L in LAYERS:
        Ly = ev[str(L)]["layer"]
        A_, B_, C_ = Ly["A_q0"], Ly["B_q1_subset"], Ly["C_q2_subset"]
        ps = {cn: np.array(Ly[cn]["per_seq"]) for cn in Ly}
        sub[str(L)] = dict(
            A_q0=A_["rel_err"], B_q1_subset=B_["rel_err"], C_q2_subset=C_["rel_err"],
            rel_q1_vs_q0=rel(B_["rel_err"], A_["rel_err"]), rel_q2_vs_q0=rel(C_["rel_err"], A_["rel_err"]),
            rel_q2_vs_q1=rel(C_["rel_err"], B_["rel_err"]),
            native=dict(A_q0=A_["native_rel_err"], B=B_["native_rel_err"], C=C_["native_rel_err"],
                        rel_q2_vs_q1=rel(C_["native_rel_err"], B_["native_rel_err"]),
                        rel_q1_vs_q0=rel(B_["native_rel_err"], A_["native_rel_err"]),
                        rel_q2_vs_q0=rel(C_["native_rel_err"], A_["native_rel_err"])),
            frac_seqs_C_better_than_B=float((ps["C_q2_subset"] < ps["B_q1_subset"]).mean()),
            frac_seqs_B_better_than_A=float((ps["B_q1_subset"] < ps["A_q0"]).mean()))
        dec[str(L)] = {cn: dict(total=v["total"], diag=v["diag"], cross=v["cross"], cross_sign="+" if v["cross"] >= 0 else "-",
                                cross_fraction=v["cross"] / v["total"], diag_rel=v["diag_rel"], cross_rel=v["cross_rel"],
                                reconstruction_rel_residual=v["recon_check_abs"],
                                delta_diag_vs_A=(v["diag"] - A_["diag"]) / A_["total"],
                                delta_cross_vs_A=(v["cross"] - A_["cross"]) / A_["total"],
                                delta_total_vs_A=(v["total"] - A_["total"]) / A_["total"])
                       for cn, v in Ly.items()}
    jdump(dict(note="subset replacement: 6 of 60 experts changed, 54 fixed at historical q=0; NOT a full c^2 layer", layers=sub),
          p("subset_layer_output.json"))
    jdump(dict(note="normalized by the same q=0 total; deltas are fractions of condition-A total error", layers=dec),
          p("error_decomposition.json"))
    # diagnostics
    hd, wd = {}, {}
    for key, r in qr.items():
        hd[key] = dict(layer=r["layer"], expert=r["expert"], exposure=r["exposure"], tokens=r["tokens"], mean_c=r["mean_c"],
                       cv_c=r["cv_c"], n_eff=r["n_eff"], n_eff_frac=r["n_eff_frac"],
                       proj={pj: {cmp: v["hessian"] for cmp, v in r["proj"][pj].items() if "_vs_" in cmp}
                             for pj, _ in PROJ})
        wd[key] = dict(layer=r["layer"], expert=r["expert"],
                       proj={pj: dict(quant_err_rel=r["proj"][pj]["quant_err_rel"],
                                      **{cmp: v["weights"] for cmp, v in r["proj"][pj].items() if "_vs_" in cmp})
                             for pj, _ in PROJ})
    def mean_over(d, f):
        return float(np.mean([f(v) for v in d.values()]))
    hsum = {cmp: {pj: dict(rel_frob_trace_norm=mean_over(hd, lambda v: v["proj"][pj][cmp]["rel_frob_trace_norm"]),
                           shape_residual=mean_over(hd, lambda v: v["proj"][pj][cmp]["shape_residual"]),
                           diag_corr=mean_over(hd, lambda v: v["proj"][pj][cmp]["diag_corr"]),
                           actorder_frac_changed=mean_over(hd, lambda v: v["proj"][pj][cmp]["actorder_frac_changed"]))
                  for pj, _ in PROJ} for cmp in ("q0_vs_q1", "q0_vs_q2", "q1_vs_q2")}
    neff = {str(q): dict(mean_frac=mean_over(hd, lambda v: v["n_eff_frac"][str(q)]),
                         min_frac=float(min(v["n_eff_frac"][str(q)] for v in hd.values()))) for q in QS}
    jdump(dict(summary=hsum, n_eff=neff, experts=hd), p("hessian_diagnostics.json"))
    wsum = {cmp: {pj: dict(rel_l2_diff=mean_over(wd, lambda v: v["proj"][pj][cmp]["rel_l2_diff"]),
                           frac_codes_changed=mean_over(wd, lambda v: v["proj"][pj][cmp]["frac_codes_changed"]))
                  for pj, _ in PROJ} for cmp in ("q0_vs_q1", "q0_vs_q2", "q1_vs_q2")}
    qerr = {str(q): mean_over(wd, lambda v: np.mean([v["proj"][pj]["quant_err_rel"][str(q)] for pj, _ in PROJ])) for q in QS}
    jdump(dict(summary=wsum, quant_err_rel_mean=qerr, experts=wd), p("weight_diagnostics.json"))
    # cost
    cost = {f"q{q}": dict(profiler_s=sum(cr[f"L{L}_q{q}"]["profiler_s"] for L in LAYERS),
                          gptq_s=sum(cr[f"L{L}_q{q}"]["gptq_s"] for L in LAYERS),
                          total_s=sum(cr[f"L{L}_q{q}"]["total_s"] for L in LAYERS),
                          peak_gpu_gb=max(cr[f"L{L}_q{q}"]["peak_gpu_gb"] for L in LAYERS)) for q in QS}
    for q in (1, 2):
        cost[f"q{q}_vs_q0"] = {kk: rel(cost[f"q{q}"][kk], cost["q0"][kk]) for kk in ("profiler_s", "gptq_s", "total_s", "peak_gpu_gb")}
    cost["peak_rss_gb_process"] = max(v["peak_rss_gb"] for v in cr.values())
    cost["per_layer"] = cr
    cost["notes"] = ("q = 2 adds one elementwise c.pow(2) per (expert, sequence) on the AGQ path; no extra forward pass; "
                     "timing protocol: per layer q = 0, 1, 2 run sequentially in one process (same Profiler replay + "
                     "36 solves); model load, cache I/O and evaluation excluded")
    cr_ = rules["cost"]
    cost_ok = (cost["q2_vs_q0"]["total_s"] <= cr_["max_time_increase_vs_q0"]
               and cost["q2_vs_q0"]["peak_gpu_gb"] <= cr_["max_peak_gpu_increase_vs_q0"])
    cost["modest"] = bool(cost_ok)
    jdump(cost, p("cost_results.json"))
    # decision
    unstable = not (reg["q2_finite"] and reg["dead_columns_unchanged"])
    correctness_ok = bool(corr["passed"] and reg["passed"] and all(
        max(dec[str(L)][cn]["reconstruction_rel_residual"] for cn in dec[str(L)]) < 1e-6 for L in LAYERS))
    d = AG.classify_c2(dict(surrogate_gain_rel=rules["surrogate_gain_rel"],
                            max_layer_regression_rel=rules["max_layer_regression_rel"]),
                       pooled_metrics["q2_vs_q1"]["E2"], {L: sub[str(L)]["rel_q2_vs_q1"] for L in LAYERS},
                       correctness_ok, cost_ok, unstable)
    decision = dict(primary=d, correctness_ok=correctness_ok, unstable=unstable,
                    effect_sizes=dict(pooled_E2_q2_vs_q1=pooled_metrics["q2_vs_q1"]["E2"],
                                      pooled_E0_q2_vs_q1=pooled_metrics["q2_vs_q1"]["E0"],
                                      layer_output_q2_vs_q1={str(L): sub[str(L)]["rel_q2_vs_q1"] for L in LAYERS},
                                      layer_output_q1_vs_q0={str(L): sub[str(L)]["rel_q1_vs_q0"] for L in LAYERS},
                                      layer_output_q2_vs_q0={str(L): sub[str(L)]["rel_q2_vs_q0"] for L in LAYERS}),
                    rules=rules)
    jdump(decision, p("decision.json"))
    summ = dict(verdict=d["verdict"], decision=decision, pooled_expert=pooled_metrics, subset_layer_output=sub,
                decomposition={L: {cn: dict(cross_fraction=v["cross_fraction"], delta_diag_vs_A=v["delta_diag_vs_A"],
                                            delta_cross_vs_A=v["delta_cross_vs_A"]) for cn, v in x.items()} for L, x in dec.items()},
                n_eff=neff, hessian=hsum, weights=wsum, quant_err_rel=qerr, cost={k: cost[k] for k in ("q0", "q1", "q2", "q1_vs_q0", "q2_vs_q0")})
    jdump(summ, p("summary.json"))
    figures(rows, sub, dec, pooled_metrics)
    write_md(summ, rows)
    print("[report]", d)


def figures(rows, sub, dec, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)
    def save(fig, nm):
        fig.tight_layout(); fig.savefig(p("figures", nm + ".png"), dpi=140); fig.savefig(p("figures", nm + ".pdf")); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    keys = sorted({(r["layer"], r["expert"], r["exposure"]) for r in rows})
    for i, m in enumerate(("E0", "E1", "E2")):
        for L, e, g in keys:
            v = {r["method"]: r[m] for r in rows if r["layer"] == L and r["expert"] == e}
            ax[i].plot([0, 1, 2], [(v[f"q={q}"] / v["q=1"] - 1) * 100 for q in QS], marker="o",
                       color="C0" if L == 8 else "C3", alpha=.6)
        ax[i].plot([0, 1, 2], [(pooled[f"q{q}"][m] / pooled["q1"][m] - 1) * 100 for q in QS], "k-", lw=2.5, label="pooled")
        ax[i].axhline(0, color="gray", lw=.8); ax[i].set_xticks([0, 1, 2], ["q=0", "q=1", "q=2"])
        ax[i].set(title=f"{m} (relative to q=1)", ylabel="% vs AGQ q=1")
    ax[0].legend(); fig.suptitle("Held-out expert reconstruction, 12 experts (blue L8, red L16)")
    save(fig, "fig1_expert_reconstruction")
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, (cn, lab) in enumerate((("A_q0", "A: all q=0"), ("B_q1_subset", "B: 6 experts q=1"), ("C_q2_subset", "C: 6 experts q=2"))):
        ax.bar(np.arange(2) + j * 0.27, [(sub[str(L)][cn] / sub[str(L)]["A_q0"] - 1) * 100 for L in LAYERS], 0.27, label=lab)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(np.arange(2) + 0.27, [f"L{L}" for L in LAYERS])
    ax.set(ylabel="layer output error vs condition A (%)", title="Subset-replacement MoE output (frozen routing)"); ax.legend()
    save(fig, "fig2_subset_layer_output")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, L in enumerate(LAYERS):
        cs = ["A_q0", "B_q1_subset", "C_q2_subset"]
        D = [dec[str(L)][c]["delta_diag_vs_A"] * 100 for c in cs]; C = [dec[str(L)][c]["delta_cross_vs_A"] * 100 for c in cs]
        T = [dec[str(L)][c]["delta_total_vs_A"] * 100 for c in cs]
        x = np.arange(3)
        ax[i].bar(x - .2, D, .2, label="Δ diagonal (c²‖ΔE‖²)"); ax[i].bar(x, C, .2, label="Δ cross"); ax[i].bar(x + .2, T, .2, label="Δ total")
        ax[i].axhline(0, color="k", lw=.8); ax[i].set_xticks(x, ["A", "B (q=1)", "C (q=2)"])
        ax[i].set(title=f"L{L}: change vs A, % of A total (cross frac A = {dec[str(L)]['A_q0']['cross_fraction']:+.3f})",
                  ylabel="%")
        ax[i].legend(fontsize=7)
    save(fig, "fig3_error_decomposition")


def write_md(s, rows):
    L = ["# c²-weighted Hessian pilot (EXPLORATORY; 12 experts, L8/L16, FR s0)", "",
         f"**Frozen screening verdict: `{s['verdict']}`**", "",
         "| Layer | Expert | Exposure | Method | E0 | E1 | E2 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['layer']} | {r['expert']} | {r['exposure']} | {r['method']} | {r['E0']:.5e} | {r['E1']:.5e} | {r['E2']:.5e} |")
    L += ["", "Pooled over the 12 experts' held-out tokens (relative differences):", ""]
    for cmp in ("q1_vs_q0", "q2_vs_q0", "q2_vs_q1"):
        L.append(f"- {cmp}: " + ", ".join(f"{m} {v * 100:+.3f} %" for m, v in s["pooled_expert"][cmp].items()))
    L += ["", "Subset-replacement layer output (6/60 experts changed):", ""]
    for Ly, v in s["subset_layer_output"].items():
        L.append(f"- L{Ly}: A {v['A_q0']:.6e}, B {v['B_q1_subset']:.6e}, C {v['C_q2_subset']:.6e}; q1 vs q0 "
                 f"{v['rel_q1_vs_q0'] * 100:+.3f} %, q2 vs q0 {v['rel_q2_vs_q0'] * 100:+.3f} %, q2 vs q1 {v['rel_q2_vs_q1'] * 100:+.3f} %")
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    dict(freeze=stage_freeze, correctness=stage_correctness, quantize=stage_quantize, evaluate=stage_evaluate,
         report=stage_report)[a.stage](a)


if __name__ == "__main__":
    main()
