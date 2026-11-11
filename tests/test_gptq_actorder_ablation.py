"""CPU tests for the GPTQ act-order ablation: solver override / read-only capture, integrity checks, metrics and the
frozen decision rule (tiny Qwen2-MoE from conftest + synthetic matrices)."""
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import actorder_ablation as AA
from moe_calib import frozen_routing as FRT
from moe_calib import frozen_routing_robustness as RB
from moe_calib import gptq as GQ
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
GS = 16
L0 = LAYERS[0]


def _problem(rows=24, cols=64, seed=0, dead=()):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(rows, cols, generator=g)
    X = torch.randn(4 * cols, cols, generator=g) * torch.linspace(0.2, 3, cols)
    H = X.T @ X
    for d in dead:
        H[d, :] = 0; H[:, d] = 0
    return W, H


def _original_gptq():
    """The solver exactly as committed before this study (git HEAD), for the default-path regression."""
    try:
        src = subprocess.run(["git", "show", "HEAD:moe_calib/gptq.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("git HEAD solver unavailable")
    ns = {}
    exec(compile(src, "gptq_head.py", "exec"), ns)
    return ns["gptq_quantize"]


# ---------------------------------------------------------------- 1-5 frozen inputs
def test_frozen_calibration_selections():
    assert RB.SELECTION_HASHES == {"FR": "ec355fb8d866e4ff", "FC": "0353229b6d3090db"}
    path = os.path.join(ROOT, "runs_frozen_routing", "protocol.json")
    if os.path.exists(path):
        pr = json.load(open(path))
        assert {c: v["ids_hash"] for c, v in pr["selections"].items()} == RB.SELECTION_HASHES and pr["n_union"] == 217


def test_grouping_manifests():
    import actorder_ablation as XA
    assert XA.G == ["G20261111", "G20261112", "G20261113", "G20261114", "G20261115"]
    d = os.path.join(ROOT, "runs_multigroup_robustness", "grouping_manifests")
    if os.path.isdir(d):
        assert [json.load(open(os.path.join(d, f"{g}.json")))["seed"] for g in XA.G] == list(RB.GROUPING_SEEDS)


def test_original_routing_reused_and_sequence_id_alignment():
    import actorder_ablation as XA
    src = inspect.getsource(XA.quant_pass)
    assert 'FRD.load_tables("O")' in src and "capture_routing" not in src
    assert "MRD.MG.remap(sel[c][\"chosen\"], union), dict(enumerate(union))" in src   # rows -> original sequence ids


def test_expert_token_counts_follow_forced_table(cache_dir):
    model, _ = tiny_model()
    cache = load_layer_cache(cache_dir, L0)
    rows = list(range(10))
    g = torch.Generator().manual_seed(2)
    table = {r: torch.stack([torch.randperm(E, generator=g)[:MU.top_k(model)] for _ in range(cache.shape[1])]) for r in rows}
    fp = FRT.ForcedProfiler(model, L0, table, {r: r for r in rows}, expect_natural=False, K=2).run(cache, rows)
    cnt = FRT.expert_token_counts(table, rows, E)
    assert all(fp.acc.tokens((L0, e, "gu")) == cnt[e] == fp.acc.tokens((L0, e, "down")) for e in range(E))


# ---------------------------------------------------------------- 6-13 solver
def test_actorder_capture_is_exact_solver_permutation():
    W, H = _problem(dead=(5, 9))
    info = {}
    GQ.gptq_quantize(W, H, 4, GS, 1.0, info=info)
    Hd = H.clone(); dd = torch.diag(Hd) == 0; Hd[dd, dd] = 1.0
    assert torch.equal(info["perm"], torch.argsort(torch.diag(Hd), descending=True))
    assert torch.equal(info["perm"], FRT.actorder_perm(H)) and info["n_dead"] == 2


def test_override_validation():
    W, H = _problem()
    for bad in (torch.arange(63), torch.zeros(64, dtype=torch.long), torch.arange(64).float(), torch.arange(64).reshape(8, 8)):
        with pytest.raises(ValueError):
            GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder_override=bad)
    with pytest.raises(ValueError):
        GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder=False, actorder_override=torch.arange(64))
    p = GQ.check_perm(torch.randperm(64).to(torch.int16), 64, "cpu")
    assert p.dtype == torch.int64


def test_inverse_permutation_and_frozen_order_equals_permuted_problem():
    """Override p == quantizing the explicitly permuted problem without act-order, then un-permuting (bitwise)."""
    W, H = _problem(seed=3, dead=(7,))
    p = torch.randperm(64, generator=torch.Generator().manual_seed(4))
    Q = GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder_override=p)
    Qp = GQ.gptq_quantize(W[:, p], H[p][:, p], 4, GS, 1.0, actorder=False)
    assert torch.equal(Q, Qp[:, torch.argsort(p)])


def test_group_membership_and_quantizer_grid():
    W, H = _problem(seed=5)
    p = torch.randperm(64, generator=torch.Generator().manual_seed(6))
    info = {}
    Q = GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder_override=p, info=info)
    grp = AA.group_of(p, GS)
    assert torch.equal(grp[p], torch.arange(64) // GS)
    assert info["scale"].shape == (64 // GS, 24) and info["zero"].shape == (64 // GS, 24)
    for j in range(64):                                # every column lies on its group's grid
        s, z = info["scale"][grp[j]], info["zero"][grp[j]]
        q = Q[:, j] / s + z
        assert torch.allclose(q, q.round(), atol=1e-3) and bool(((q.round() >= 0) & (q.round() <= 15)).all())
    pc = AA.perm_compare(p, torch.arange(64), GS)
    assert not pc["identical"] and 0 < pc["frac_columns_changed_group"] <= 1


def test_default_solver_unchanged_vs_git_head():
    orig = _original_gptq()
    for seed, dead in ((0, ()), (1, (3, 40))):
        W, H = _problem(seed=seed, dead=dead)
        assert torch.equal(GQ.gptq_quantize(W, H, 4, GS, 1.0), orig(W, H, 4, GS, 1.0))
        assert torch.equal(GQ.gptq_quantize(W, H, 4, GS, 1.0, info={}), orig(W, H, 4, GS, 1.0))   # capture is read-only
        assert torch.equal(GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder=False), orig(W, H, 4, GS, 1.0, actorder=False))


def test_natural_permutation_override_is_bitwise():
    for seed in range(3):
        W, H = _problem(seed=seed, dead=(2,) if seed else ())
        info = {}
        Q = GQ.gptq_quantize(W, H, 4, GS, 1.0, info=info)
        assert torch.equal(GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder_override=info["perm"]), Q)


def test_frozen_order_on_different_hessian_recomputes_quantizer():
    W, H1 = _problem(seed=7)
    _, H2 = _problem(seed=8)
    i1, i2, i2n = {}, {}, {}
    GQ.gptq_quantize(W, H1, 4, GS, 1.0, info=i1)
    Q2 = GQ.gptq_quantize(W, H2, 4, GS, 1.0, actorder_override=i1["perm"], info=i2)
    Q2n = GQ.gptq_quantize(W, H2, 4, GS, 1.0, info=i2n)
    assert torch.equal(i2["perm"], i1["perm"]) and not torch.equal(i2n["perm"], i1["perm"])
    assert not torch.equal(Q2, Q2n)
    # scales refitted (not copied): the first group's scale depends only on W[:, perm[:GS]] (identical), later groups
    # see the error feedback of H2
    assert torch.equal(i2["scale"][0], i1["scale"][0]) and not torch.equal(i2["scale"][1:], i1["scale"][1:])
    H2c = H2.clone()
    GQ.gptq_quantize(W, H2, 4, GS, 1.0, actorder_override=i1["perm"])
    assert torch.equal(H2, H2c)                                     # the supplied Hessian is never modified


def test_gate_up_share_hessian_and_down_separate():
    g = torch.Generator().manual_seed(9)
    Wd = {"gate_proj": torch.randn(32, 64, generator=g), "up_proj": torch.randn(32, 64, generator=g),
          "down_proj": torch.randn(64, 32, generator=g)}
    _, Hg = _problem(rows=32, cols=64, seed=10)
    _, Hd = _problem(rows=64, cols=32, seed=11)
    infos = {}
    Q = GQ.quantize_expert(Wd, Hg, Hd, 4, GS, 1.0, infos=infos)
    assert torch.equal(infos["gate_proj"]["perm"], infos["up_proj"]["perm"]) and infos["down_proj"]["perm"].numel() == 32
    Q0 = GQ.quantize_expert(Wd, Hg, Hd, 4, GS, 1.0)
    assert all(torch.equal(Q[k], Q0[k]) for k in Q0)
    Qn = GQ.quantize_expert(Wd, Hg, Hd, 4, GS, 1.0, perms={k: infos[k]["perm"] for k in AA.PROJ})
    assert all(torch.equal(Qn[k], Q0[k]) for k in Q0)


# ---------------------------------------------------------------- 14-21 E3 integration on the tiny model
def _e3(model, cache_dir, chosen, mode, perms=None, log=None):
    import e3_allocation as E3
    import actorder_ablation as XA
    test = split_ids(load_meta(cache_dir))["test"]
    orig = E3.quantize_from_acc_or_fallback
    seen, infos_all, hs = {}, {}, {}

    def qw(m, acc, L, e, bits, gs, pd):
        infos = {}
        if mode == "default":
            out = orig(m, acc, L, e, bits, gs, pd)
        else:
            out = orig(m, acc, L, e, bits, gs, pd, perms=perms[(L, e)], infos=infos)
        infos_all[(L, e)] = infos
        hs[(L, e)] = FRT.tensor_sha(acc.H_sum((L, e, "gu"), list(range(acc.K))))
        return out
    E3.quantize_from_acc_or_fallback = qw
    try:
        with XA.install_spy(log if log is not None else []):
            q = PQ.quantize_install(model, cache_dir, [L0], chosen, test, 2, 4, GS, 1.0)
    finally:
        E3.quantize_from_acc_or_fallback = orig
    return q, infos_all, hs


def test_e3_natural_override_identical_and_ao_aof_hessians_identical(cache_dir):
    chosen = list(range(10))
    m1, _ = tiny_model()
    q1, _, h1 = _e3(m1, cache_dir, chosen, "default")
    d1 = PQ.installed_digest(m1, [L0])
    # natural permutations captured from a capture pass
    m2, _ = tiny_model()
    q2, inf2, h2 = _e3(m2, cache_dir, chosen, "override", perms={(L0, e): {} for e in range(E)})
    live = [e for e in range(E) if inf2[(L0, e)]]                   # experts with calibration tokens (others: RTN)
    assert live
    nat = {(L0, e): ({k: inf2[(L0, e)][k]["perm"] for k in AA.PROJ} if e in live else {}) for e in range(E)}
    assert PQ.installed_digest(m2, [L0]) == d1                     # info capture alone changes nothing
    m3, _ = tiny_model()
    q3, inf3, h3 = _e3(m3, cache_dir, chosen, "override", perms=nat)
    assert PQ.installed_digest(m3, [L0]) == d1 and h3 == h1         # natural override bitwise; same Hessians
    assert q3["counts"]["eval_universe"] == q1["counts"]["eval_universe"]
    assert [r["mean"] for r in q3["results"]] == [r["mean"] for r in q1["results"]]
    # a foreign (frozen) permutation: Hessians identical, weights differ, applied perms == the frozen ones
    g = torch.Generator().manual_seed(12)
    frozen = {(L0, e): ({k: torch.randperm(nat[(L0, e)][k].numel(), generator=g) for k in AA.PROJ} if e in live else {})
              for e in range(E)}
    for e in live:
        frozen[(L0, e)]["up_proj"] = frozen[(L0, e)]["gate_proj"]
    m4, _ = tiny_model()
    q4, inf4, h4 = _e3(m4, cache_dir, chosen, "override", perms=frozen)
    assert h4 == h1 and PQ.installed_digest(m4, [L0]) != d1
    assert all(torch.equal(inf4[(L0, e)][k]["perm"], frozen[(L0, e)][k]) for e in live for k in AA.PROJ)
    assert q4["counts"]["eval_universe"] == q1["counts"]["eval_universe"]          # 21: same reconstruction universe


def test_router_untouched_no_inference_override_and_install_spy(cache_dir):
    model, _ = tiny_model()
    r0 = MU.moe_block(model, L0).gate.weight.clone()
    log = []
    _e3(model, cache_dir, list(range(10)), "default", log=log)
    assert torch.equal(MU.moe_block(model, L0).gate.weight, r0)
    FRT.assert_no_override(model)
    assert len(log) == 3 * E and all(x[1] and x[2] and x[4] for x in log)          # I1, I3, I4 at write-back


def test_no_hyperparameter_drift():
    sig = inspect.signature(GQ.gptq_quantize).parameters
    assert (sig["bits"].default, sig["group_size"].default, sig["percdamp"].default, sig["actorder"].default,
            sig["blocksize"].default, sig["actorder_override"].default, sig["info"].default) == (4, 128, 0.01, True, 128, None, None)
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    from moe_calib import frozen
    assert frozen._act_order_default() is True


# ---------------------------------------------------------------- 19-20 installation / BF16 integrity
def test_cast_rounding_check():
    W = torch.randn(64, 64) * 0.05
    ok, r = AA.cast_rounding_ok(W, W.to(torch.bfloat16))
    assert ok and r <= 1.0
    trunc = (W.contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    assert not torch.equal(trunc, W.to(torch.bfloat16))
    bad = W.to(torch.bfloat16).clone(); bad[0, 0] = bad[0, 0] * 1.5
    assert not AA.cast_rounding_ok(W, bad)[0]


def test_precision_decomposition_and_integrity_on_tiny_expert(cache_dir):
    model, _ = tiny_model()
    e = 1
    mod = MU.expert_module(model, L0, e)
    W0 = {k: w.detach().clone() for k, w in MU.get_expert_weights(model, L0, e).items()}
    g = torch.Generator().manual_seed(13)
    Wq32 = {k: (w.float() + 0.01 * torch.randn(w.shape, generator=g)) for k, w in W0.items()}
    X = torch.randn(40, W0["gate_proj"].shape[1], generator=g).to(torch.bfloat16).float()
    MU.set_expert_weights(model, L0, e, Wq32)
    dec = AA.precision_decomposition(mod, Wq32, W0, X, "cpu")
    assert AA.integrity_pass(dec) and dec["inputs_exactly_bf16"]
    ex = dec["expert"]
    assert ex["total_rel"] <= ex["weight_repr_rel"] + ex["arith_rel"] + 1e-9
    assert ex["old_check"]["tolerance"] == pytest.approx(ex["weight_repr_rel"] + 1.5 * ex["orig_weights_arith_rel"])
    # negative control: weights of another expert installed -> I1 / I2 fail
    MU.set_expert_weights(model, L0, e, {k: w.detach().clone() for k, w in MU.get_expert_weights(model, L0, 2).items()})
    bad = AA.precision_decomposition(mod, Wq32, W0, X, "cpu")
    assert not AA.integrity_pass(bad) and not bad["expert"]["module_bitwise_equal_independent_reference"]
    # inputs that are not exactly BF16 are flagged
    MU.set_expert_weights(model, L0, e, Wq32)
    assert not AA.precision_decomposition(mod, Wq32, W0, X + 1e-4, "cpu")["inputs_exactly_bf16"]


def test_frozen_ppl_evaluation_protocol():
    path = os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json")
    if not os.path.exists(path):
        pytest.skip("evaluation protocol not present")
    pr = json.load(open(path))
    assert pr["scored_targets_total"] == 65408
    import eval_partial_ppl as EP
    assert EP.BATCH == 8


# ---------------------------------------------------------------- 23-26 metrics and decision
def test_sd_range_and_distance():
    ao, aof = [9.266, 9.271, 9.276, 9.273, 9.268], [9.269, 9.270, 9.271, 9.270, 9.269]
    s = AA.ppl_spread(ao, aof, 9.2679)
    assert s["SD_AO"] == pytest.approx(np.std(ao, ddof=1)) and s["SD_ratio"] == pytest.approx(np.std(aof, ddof=1) / np.std(ao, ddof=1))
    assert s["Range_ratio"] == pytest.approx(0.002 / 0.010)
    assert s["n_AOF_closer_to_OO"] == sum(abs(a - 9.2679) < abs(b - 9.2679) for a, b in zip(aof, ao))
    assert AA.ppl_spread([1, 2, 3], [1, 1, 1], 1)["SD_ratio"] == 0.0
    assert AA.ppl_spread([1, 1, 1], [1, 2, 3], 1)["SD_ratio"] is None


def test_paired_deltas():
    d = AA.delta_stats([1.0, 1.0, 1.0, 1.0, 1.0], [1.01, 0.99, 1.0, 1.02, 1.005], 0.01)
    assert d["Delta"] == pytest.approx([0.01, -0.01, 0.0, 0.02, 0.005])
    assert d["n_positive"] == 3 and d["n_negative"] == 1 and d["n_preserved_vs_OO"] == 3 and d["n_reversals_vs_OO"] == 2


def test_weight_difference_and_reduction():
    import report_actorder_ablation as R
    W = {"a": torch.ones(4, 4, dtype=torch.bfloat16), "b": torch.ones(2, 2, dtype=torch.bfloat16)}
    V = {"a": W["a"].clone(), "b": W["b"].clone()}
    V["a"][0, 0] = 2
    pl = {"1": FRT.weight_diff(V, W)}
    t = R.wtotal(pl)
    assert t["frac_differ"] == pytest.approx(1 / 20) and t["rel_l2"] == pytest.approx((1 / 20) ** .5) and t["n_tensors_identical"] == 1
    r = AA.weight_reduction([0.05, 0.06], [0.07, 0.06])
    assert r["per_grouping"] == pytest.approx([1 - 5 / 7, 0.0]) and r["mean"] == pytest.approx((1 - 5 / 7) / 2)


def test_decision_classification():
    sp = lambda r1, r2: {"FR": {"SD_ratio": r1}, "FC": {"SD_ratio": r2}}
    w = lambda a, b: {"FR": {"mean": a}, "FC": {"mean": b}}
    dec = lambda W, S, D: AA.decide(AA.criteria(W, S, {"SD_Delta_ratio": D}))
    assert dec(w(.3, .4), sp(.5, .6), .7) == "ACTORDER-CONTRIBUTES-TO-WEIGHT-AND-PPL-SENSITIVITY"
    assert dec(w(.3, .4), sp(.5, .9), .5) == "ACTORDER-CONTRIBUTES-TO-WEIGHT-SENSITIVITY-ONLY"
    assert dec(w(.3, .4), sp(.5, .6), None) == "ACTORDER-CONTRIBUTES-TO-WEIGHT-SENSITIVITY-ONLY"
    assert dec(w(.1, .2), sp(.9, 1.2), .8) == "ACTORDER-NOT-PRIMARY"
    assert dec(w(.1, .2), sp(.5, 1.2), .8) == "ACTORDER-MIXED-EFFECT"
    assert dec(w(.3, .1), sp(.5, .6), .5) == "ACTORDER-MIXED-EFFECT"
    assert AA.decide({}, blocked=True) == "RUN-BLOCKED"
    assert AA.DECISION_RULE["criteria"]["W"].startswith("WeightReduction[FR] >= 0.25")


# ---------------------------------------------------------------- 27 backward compatibility
def test_backward_compatibility_of_call_chain():
    from moe_calib import quant_eval as QE
    for f in (QE.quantize_from_H, QE.quantize_from_acc, QE.quantize_from_acc_or_fallback):
        ps = inspect.signature(f).parameters
        assert ps["perms"].default is None and ps["infos"].default is None
    import e3_allocation as E3
    assert E3.quantize_from_acc_or_fallback is QE.quantize_from_acc_or_fallback
