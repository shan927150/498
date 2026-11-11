"""C4-prefix diversity-rescue study: prefix-conditioned generation (moe_calib.ebss_generator initial_token_ids),
region diversity helpers and frozen decisions (moe_calib.ebss_prefix), artifact integrity of runs_ebss_prefix_rescue/.
GPU-only behaviour (prefill vs teacher forcing, KV parent checks) is verified by the smoke stage; its stored results
are checked here when present."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import ebss_generator as G                                   # noqa: E402
from moe_calib import ebss_prefix as X                                      # noqa: E402

OUT = os.path.join(ROOT, "runs_ebss_prefix_rescue")
FID = os.path.join(ROOT, "runs_ebss_balance_fidelity")
STRONG = dict(balance_std_mode="algorithm1", usage_mode="per_token")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


NO = _load(os.path.join(FID, "normalization_oracle.py"), "normalization_oracle")
OLD = _load(os.path.join(OUT, "logs", "ebss_generator_pre_prefix.py"), "ebss_generator_pre_prefix_t")


def art(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not produced yet")
    return json.load(open(path))


def be(seed=0):
    return G.SyntheticBackend(V=30, L=3, E=8, k=2, seed=seed)


PRE = [[3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 5, 8], [9, 7, 9, 3, 2, 3]]


def gen(ub=True, pre=PRE, T=16, seed=0, audit=None, **kw):
    return G.generate(be(seed), None, T, 4, 1.3, ub, initial_token_ids=pre, audit=audit, **{**STRONG, **kw})


# 1-6 prefix extraction / manifest -------------------------------------------------------------------------------
def test_01_split_windows():
    w = [list(range(i, i + 512)) for i in range(3)]
    pre, cont = X.split_windows(w)
    assert all(len(x) == 64 for x in pre) and all(len(x) == 448 for x in cont)
    assert all(a + b == ww for a, b, ww in zip(pre, cont, w))


def test_02_prefix_checks_detect_problems():
    par = [list(range(i, i + 512)) for i in range(2)]
    pre, _ = X.split_windows(par)
    assert X.prefix_checks(pre, par)["valid"]
    assert not X.prefix_checks([pre[0], pre[0]], par)["valid"]                 # duplicate prefixes
    assert not X.prefix_checks([pre[0][:63], pre[1]], par)["valid"]            # wrong length


def test_03_frozen_manifest():
    m = art("prefix_manifest.json")
    assert m["checks"]["valid"] and m["checks"]["no_leakage"] and len(m["prefixes"]) == 16
    for e in m["prefixes"]:
        assert len(e["prefix_token_ids"]) == 64 and X.token_hash(e["prefix_token_ids"]) == e["prefix_sha256"]


def test_04_manifest_matches_c4_tokens():
    m = art("prefix_manifest.json")
    ids = os.path.expanduser("~/moe_cache/ids.pt")
    if not os.path.exists(ids):
        pytest.skip("C4 token cache unavailable")
    d = torch.load(ids)
    allt = torch.cat([d["pool"], d["holdout"]])
    for e in m["prefixes"]:
        row = allt[e["parent_row"]].tolist()
        assert X.token_hash(row) == e["parent_sha256"] and row[:64] == e["prefix_token_ids"]
        assert X.token_hash(row[64:]) == e["continuation_sha256"]


def test_05_no_leakage_rule():
    from moe_calib import ebss_pilot as P
    held = [list(range(1000, 1512))]
    par = [list(range(0, 512)), list(range(990, 1502))]          # second shares a >=32-gram with held-out
    r = X.prefix_checks([p[:64] for p in par], par, held, P.contamination)
    assert r["leakage"]["windows"] == [1] and not r["valid"]


# 7-15 generator ------------------------------------------------------------------------------------------------
def test_06_prefix_identity_and_length():
    r = gen()
    assert tuple(r.tokens.shape) == (3, 16) and r.tokens[:, :6].tolist() == PRE


def test_07_prefill_logp_equals_stepwise():
    b = be(); lg, sel, plp = b.prefill(torch.tensor(PRE))
    b2 = be(); ref = []
    l0, _ = b2.start([x[0] for x in PRE])
    for t in range(1, 6):
        lp = torch.log_softmax(l0.float(), -1); ref.append(lp[torch.arange(3), torch.tensor([x[t] for x in PRE])].double())
        l0, _ = b2.step([x[t] for x in PRE])
    assert torch.equal(plp, torch.stack(ref, 1)) and torch.equal(lg, l0) and tuple(sel.shape) == (3, 6, 3, 2)


def test_08_first_expansion_single_branch():
    tr = []
    gen(audit=tr.append)
    first = [d for d in tr if d["kind"] == "step"][0]
    assert first["n_branches"] == 1 and first["prefix_len"] == 6 and first["scores"].shape == (3, 1, 30)
    # kept tokens are the 4 distinct best expansions of the single parent
    tok = first["token"].view(3, 4)
    assert all(len(set(t.tolist())) == 4 for t in tok) and (first["parent"].view(3, 4) == torch.arange(3)[:, None]).all()


def test_09_initial_cumulative_logp():
    tr = []
    r = gen(audit=tr.append)
    b = be(); _, _, plp = b.prefill(torch.tensor(PRE))
    first = [d for d in tr if d["kind"] == "step"][0]
    assert torch.allclose(first["cum"], plp.sum(1)) and torch.equal(r.token_logp[:, :5], plp)
    assert torch.allclose(r.cum_logp, r.token_logp.sum(1), rtol=1e-12, atol=1e-12)


def test_10_prefix_expert_counts():
    tr = []
    gen(audit=tr.append)
    b = be(); _, sel, _ = b.prefill(torch.tensor(PRE))
    first = [d for d in tr if d["kind"] == "step"][0]
    for i in range(3):
        assert torch.equal(first["counts"][i], G.recount(sel[i], 8))           # all 6 prefix tokens


def test_11_balance_penalty_after_prefill():
    tr = []
    gen(audit=tr.append)
    for d in [x for x in tr if x["kind"] == "step"]:
        for r in range(d["counts"].shape[0]):
            o = NO.sigma(d["counts"][r].tolist(), d["prefix_len"], "algorithm1", "per_token", 2)
            assert abs(o - float(d["sigma"][r])) <= 1e-12 and o > 0
        sc = NO.scores(d["cum"].numpy(), d["logp"].numpy(), d["sigma"].numpy(), d["prefix_len"], 1.3, True)
        assert np.array_equal(sc, d["scores"].reshape(sc.shape).numpy())


def test_12_probability_only_control():
    a = gen(ub=False); b = G.generate(be(), None, 16, 4, 1.3, False, initial_token_ids=PRE)
    assert torch.equal(a.tokens, b.tokens)
    tr = []
    gen(ub=False, audit=tr.append)
    d = [x for x in tr if x["kind"] == "step"][3]
    sc = NO.scores(d["cum"].numpy(), d["logp"].numpy(), d["sigma"].numpy(), d["prefix_len"], 1.3, False)
    assert np.array_equal(sc, d["scores"].reshape(sc.shape).numpy())


def test_13_global_pruning_cross_parent():
    tr = []
    gen(audit=tr.append)
    for d in [x for x in tr if x["kind"] == "step"][1:]:
        B = d["n_branches"]; V = d["logp"].shape[1]
        for i in range(3):
            s = d["scores"][i].reshape(-1).numpy()
            top, _ = NO.rank(s, np.repeat(np.arange(B), V), np.tile(np.arange(V), B), 4)
            assert (top // V == d["parent"].view(3, 4)[i].numpy() - i * B).all()
            assert (top % V == d["token"].view(3, 4)[i].numpy()).all()


def test_14_parent_state_propagation():
    """Synthetic backend logits are a pure function of the full prefix: if the KV/branch state followed the wrong
    parent, the step logp would differ from a from-scratch recomputation."""
    tr = []
    gen(audit=tr.append, seed=3)
    for d in [x for x in tr if x["kind"] == "step"][1:6]:
        for r in range(d["tokens"].shape[0]):
            b = be(3); lg, _ = b.start([int(d["tokens"][r, 0])])
            for t in d["tokens"][r, 1:].tolist():
                lg, _ = b.step([t])
            assert torch.allclose(torch.log_softmax(lg[0].float(), -1), d["logp"][r])


def test_15_incremental_routing_equals_recount():
    r = gen(T=20)
    for i in range(3):
        assert torch.equal(r.counts[i], G.recount(r.hist_experts[i], 8))
        assert int(r.counts[i].sum()) == 20 * 3 * 2


def test_16_eos_is_ordinary_candidate():
    # no stop rule: output length is exact even if a "special" id is among the candidates
    r = gen(T=24)
    assert r.tokens.shape[1] == 24 and "eos" not in r.config


def test_17_deterministic():
    a, b = gen(), gen()
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.cum_logp, b.cum_logp) and torch.equal(a.final_score, b.final_score)


def test_18_historical_one_token_regression():
    for ub in (False, True):
        a = G.generate(be(), [1, 2, 3], 14, 4, 1.3, ub)
        b = OLD.generate(OLD.SyntheticBackend(V=30, L=3, E=8, k=2, seed=0), [1, 2, 3], 14, 4, 1.3, ub)
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.cum_logp, b.cum_logp) and torch.equal(a.final_score, b.final_score)
        c = G.generate(be(), None, 14, 4, 1.3, ub, initial_token_ids=[[1], [2], [3]])
        assert torch.equal(a.tokens, c.tokens) and torch.equal(a.counts, c.counts)


def test_19_balance_normalisation_regression():
    for v in G.BALANCE_VARIANTS.values():
        a = G.generate(be(1), [4, 5], 12, 4, 1.3, True, **v)
        b = OLD.generate(OLD.SyntheticBackend(V=30, L=3, E=8, k=2, seed=1), [4, 5], 12, 4, 1.3, True, **v)
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.final_score, b.final_score)


def test_20_score_sign_regression():
    tr = []
    gen(audit=tr.append)
    d = [x for x in tr if x["kind"] == "step"][2]
    lp = d["logp"].double(); cum = d["cum"]
    expect = -(cum[:, None] + lp) / (d["prefix_len"] + 1) + (d["sigma"] / 1.3)[:, None]
    assert torch.equal(expect.reshape(d["scores"].shape), d["scores"])        # positive penalty, minimised


def test_21_token_accounting():
    a = X.accounting(16, 64, 512, 4, 151646)
    assert a["emitted_continuation_tokens"] == 16 * 448 and a["final_calibration_tokens"] == 16 * 512
    assert a["prefix_tokens"] == 16 * 64 and a["forward_calls"] == 449
    assert a["scored_candidates"] == 16 * (151646 + 447 * 4 * 151646)


# 22-23 diversity --------------------------------------------------------------------------------------------------
def test_22_region_decomposition():
    w = list(range(64)) + [5, 6, 7, 8] * 112
    full = X.region_text([w], None, (0, 512))["per_sequence_mean"]
    cont = X.region_text([w], None, (64, 512))["per_sequence_mean"]
    assert abs(cont["distinct1"] - 4 / 448) < 1e-12 and full["distinct1"] > cont["distinct1"]
    bd = X.boundary_decomposition([w], 2)
    assert bd["prefix"] == 63 and bd["continuation"] == 4 and bd["full"] == 65
    rp = X.repetition_profile([w])
    assert rp["loop_onset_offset"] == [4] and rp["n_repeated_span_gt_64"] == 1


def test_23_natural_continuation_comparison():
    d = art("diversity_by_region.json")
    for s in ("C4R_s0", "PX_PO", "PX_EBSS"):
        assert d["sets"][s]["prefix"]["per_sequence_mean"] == d["sets"]["C4R_s0"]["prefix"]["per_sequence_mean"]
    assert d["sets"]["C4R_s0"]["continuation"]["n_tokens_per_seq"] == 448


def test_23b_diversity_label_rule():
    base = dict(distinct2=0.40, repetition_rate=0.75)
    good = dict(distinct2=0.46, repetition_rate=0.64)
    lab = X.diversity_label({"PO": good, "EBSS": good}, {"PO": base, "EBSS": base}, {"PO": good, "EBSS": good},
                            {"PO": base, "EBSS": base})
    assert lab["label"] == "CONTINUATION-DIVERSITY-IMPROVED"
    lab = X.diversity_label({"PO": base, "EBSS": base}, {"PO": base, "EBSS": base}, {"PO": good, "EBSS": base},
                            {"PO": base, "EBSS": base})
    assert lab["label"] == "PREFIX-CONTENT-ONLY" and not lab["mixed"]
    half = dict(distinct2=0.46, repetition_rate=0.74)
    lab = X.diversity_label({"PO": half, "EBSS": base}, {"PO": base, "EBSS": base}, {"PO": base, "EBSS": base},
                            {"PO": base, "EBSS": base})
    assert lab["label"] == "NO-DIVERSITY-IMPROVEMENT" and lab["mixed"]


# 24-28 protocol / decisions / artifacts ----------------------------------------------------------------------------
def test_24_reconstruction_protocol_integrity():
    m = art("logs/capture_manifest.json")
    assert m["passed"] and all(all(v.values()) for v in m["compatibility"]["layer_inputs"].values())
    for c in ("PX_PO", "PX_EBSS"):
        f = os.path.join(OUT, "results", f"{c}.json")
        if os.path.exists(f):
            r = json.load(open(f))
            assert r["valid"] and r["n_quantized_experts"] == 240 and r["n_projections"] == 720
            assert r["gptq"] == json.load(open(os.path.join(FID, "results", "ALG1_RATE.json")))["gptq"]


def test_25_ppl_protocol_integrity():
    for c in ("PX_PO", "PX_EBSS"):
        f = os.path.join(OUT, "results", f"{c}.json")
        if not os.path.exists(f):
            pytest.skip("not quantized yet")
        r = json.load(open(f))
        ref = json.load(open(os.path.join(FID, "results", "ALG1_RATE.json")))
        assert r["per_window_tokens"] == ref["per_window_tokens"] and r["repeat_eval_abs_diff_ppl"] == 0


def test_26_timing_accounting():
    for c in ("PX_PO", "PX_EBSS"):
        f = os.path.join(OUT, "generation_logs", f"generation_cost_{c}.json")
        if not os.path.exists(f):
            pytest.skip("not generated yet")
        g = json.load(open(f))
        assert abs(sum(g["breakdown_s"].values()) - g["timing"]["total_s"]) < 1e-6
        assert g["accounting"]["emitted_continuation_tokens"] == 16 * 448 and g["validation"]["valid"]


def test_27_decision_classification():
    base = dict(C4R_s0=1.0, C4R_s1=1.01, C4R_s2=0.99, RS_PO=1.13, RS_EBSS=1.14)
    assert X.primary_label({**base, "PX_PO": 1.2, "PX_EBSS": 0.98})["label"] == "PREFIX-RESCUE-BEATS-C4"
    assert X.primary_label({**base, "PX_PO": 1.12, "PX_EBSS": 1.2})["label"] == "PREFIX-RESCUE-PARTIAL"
    assert X.primary_label({**base, "PX_PO": 1.14, "PX_EBSS": 1.15})["label"] == "PREFIX-RESCUE-NO-GAIN"
    assert X.primary_label({**base, "PX_PO": 1.0, "PX_EBSS": None})["label"] == "PREFIX-RESCUE-RUN-BLOCKED"
    assert X.primary_label({**base, "PX_PO": 1.0, "PX_EBSS": 1.0}, valid=False)["label"] == "PREFIX-RESCUE-RUN-BLOCKED"
    assert abs(X.gap_recovery(1.13, 1.065, 1.0) - 0.5) < 1e-12


def test_28_backward_compatibility_and_smoke():
    r = G.generate(be(), [7, 8], 10, 4, 1.3, True)
    assert "initial_prefix_len" not in r.config and "prefill_wall_s" not in r.timing
    s = art("smoke_results.json")
    assert s["passed"] and all(s["checks"].values())
