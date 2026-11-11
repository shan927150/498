"""CPU tests for the model-matched EBSS generator and the EBSS generation pilot (no GPU, no model weights)."""
import json, os, sys, hashlib

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import ebss_generator as G          # noqa: E402
from moe_calib import ebss_pilot as P               # noqa: E402

SNAP = "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"


def _gen(use_balance, N=3, T=12, w=3, tau=0.05, seed=0, **kw):
    be = G.SyntheticBackend(V=40, L=3, E=8, k=2, seed=seed)
    return G.generate(be, [1, 2, 3][:N], seqlen=T, w=w, tau=tau, use_balance=use_balance, **kw), be


# 1-3 checkpoint, tokenizer, initialisation ------------------------------------------------------------
def test_exact_checkpoint_identification():
    from moe_calib.model_utils import _snapshot_dir
    d = _snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B")
    assert os.path.basename(d) == SNAP
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert cfg["num_experts"] == 60 and cfg["num_experts_per_tok"] == 4 and cfg["num_hidden_layers"] == 24
    assert "chat" not in d.lower()
    assert P.MODEL == "Qwen/Qwen1.5-MoE-A2.7B" and P.SNAPSHOT == SNAP


def test_tokenizer_compatibility():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
    assert len(t) == 151646 and t.all_special_ids == [151643, 151644, 151645]
    assert t.bos_token_id is None                    # no BOS is prepended: the random first token is position 0


def test_vocabulary_initialisation_frozen_and_shared():
    a = G.sample_first_tokens(16)
    assert a == G.sample_first_tokens(16) and len(set(a)) == 16
    assert all(0 <= x < 151643 for x in a)             # never a special token
    assert G.sample_first_tokens(4) == a[:4]           # smoke starts are the first 4 pilot starts


# 4-9 scoring ---------------------------------------------------------------------------------------
def test_cumulative_logprob_matches_token_logprobs():
    r, _ = _gen(True)
    assert torch.allclose(r.cum_logp, r.token_logp.sum(1), atol=0, rtol=1e-12)
    assert r.token_logp.shape[1] == r.tokens.shape[1] - 1


def test_beam_score_formula():
    cum = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    lp = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.1]]))
    sig = torch.tensor([0.2, 0.0], dtype=torch.float64)
    s = G.candidate_scores(cum, lp, sig, prefix_len=3, tau=2.0, use_balance=True)
    assert s[0, 0].item() == pytest.approx(-(-1.0 + np.log(0.5)) / 4 + 0.1)
    assert s[1, 1].item() == pytest.approx(-(-2.0 + np.log(0.1)) / 4 + 0.0)


def test_score_direction_is_minimisation():
    s = torch.tensor([[0.3, 0.1, 0.9], [0.2, 0.05, 0.4]], dtype=torch.float64)
    par, tok, best = G.select_best(s, 2)
    assert list(zip(par.tolist(), tok.tolist())) == [(1, 1), (0, 1)]
    assert G.final_scores(torch.tensor([-1.0, -5.0]), torch.tensor([0., 0.]), 4, 1.0, False).argmin() == 0


def test_balance_coefficient_divides_sigma_not_logits():
    cum = torch.zeros(1, dtype=torch.float64)
    lp = torch.log(torch.tensor([[0.7, 0.3]]))
    sig = torch.tensor([0.6], dtype=torch.float64)
    a = G.candidate_scores(cum, lp, sig, 1, 1.3, True)
    b = G.candidate_scores(cum, lp, sig, 1, 1.3, False)
    assert torch.allclose(a - b, torch.full_like(a, 0.6 / 1.3))     # additive sigma/tau, same for every token
    assert torch.equal(a.argsort(), b.argsort())                    # tau never rescales next-token logits


def test_probability_only_control_is_pure_beam_search():
    r, _ = _gen(False, tau=1e-9)                    # tau irrelevant when the balance term is off
    r2, _ = _gen(False, tau=10.0)
    assert torch.equal(r.tokens, r2.tokens) and r.intervention == {"not_applicable": True}


def test_prefix_based_balance_cannot_reorder_tokens_within_a_branch():
    cum = torch.zeros(2, dtype=torch.float64)
    lp = torch.log_softmax(torch.randn(2, 30), -1)
    sig = torch.tensor([0.0, 5.0], dtype=torch.float64)
    a = G.candidate_scores(cum, lp, sig, 4, 1.0, True)
    b = G.candidate_scores(cum, lp, sig, 4, 1.0, False)
    for i in range(2):
        assert torch.equal(a[i].argsort(), b[i].argsort())
    # ...but it can change which parent's tokens survive
    pa, _, _ = G.select_best(a, 3)
    assert set(pa.tolist()) == {0}


# 10-15 routing statistics and beam bookkeeping --------------------------------------------------------
def test_routing_membership_counts_each_token_once_per_selected_expert():
    hist = torch.tensor([[[0, 1]], [[1, 2]], [[1, 0]]], dtype=torch.int16)    # [T=3, L=1, k=2]
    c = G.recount(hist, 4)
    assert c.tolist() == [[2, 3, 1, 0]] and int(c.sum()) == 3 * 2


def test_per_layer_usage_statistics_sigma():
    counts = torch.tensor([[[4, 0, 0, 0], [1, 1, 1, 1]]])           # [1, L=2, E=4], 4 tokens, k = 1
    s = G.balance_sigma(counts, 4)
    u0 = np.array([1, 0, 0, 0.])
    assert s.item() == pytest.approx((u0.std(ddof=1) + 0.0) / 2)


def test_incremental_counts_equal_full_prefix_recount():
    r, _ = _gen(True, T=20)
    for i in range(r.tokens.shape[0]):
        assert torch.equal(r.counts[i], G.recount(r.hist_experts[i], 8))


def test_incremental_routing_equals_backend_replay():
    """The per-token routing stored for each chosen sentence equals a fresh replay of its exact prefix."""
    r, _ = _gen(True, T=15)
    be = G.SyntheticBackend(V=40, L=3, E=8, k=2, seed=0)
    for i in range(r.tokens.shape[0]):
        seq = r.tokens[i].tolist()
        _, sel = be.start([seq[0]])
        got = [sel[0]]
        for t in seq[1:]:
            _, sel = be.step([t]); got.append(sel[0])
        assert torch.equal(torch.stack(got).to(torch.int16), r.hist_experts[i])


def test_beam_parent_state_copy_not_aliased():
    r, _ = _gen(True, T=16, w=4)
    fb = r.final_beams
    for i in range(fb["tokens"].shape[0]):
        for b in range(fb["tokens"].shape[1]):
            # every surviving branch's counts are its own (sum = tokens * L * k) and sigma recomputes exactly
            assert int(fb["counts"][i, b].sum()) == 16 * 3 * 2
            assert torch.allclose(G.balance_sigma(fb["counts"][i, b], 16), fb["sigma"][i, b], rtol=0, atol=0)


def test_beam_pruning_keeps_w_distinct_branches():
    r, _ = _gen(True, T=10, w=3)
    for i in range(r.final_beams["tokens"].shape[0]):
        seqs = {tuple(x) for x in r.final_beams["tokens"][i].tolist()}
        assert len(seqs) == 3


def test_tie_handling_is_deterministic_by_parent_then_token():
    s = torch.zeros(3, 5, dtype=torch.float64)
    par, tok, _ = G.select_best(s, 4)
    assert list(zip(par.tolist(), tok.tolist())) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    fs = torch.tensor([[1.0, 1.0, 2.0]])
    assert int(torch.argmin(fs, dim=1)) == 0


# 16-21 cache, accounting, length, determinism, serialisation ------------------------------------------
def test_kv_reorder_follows_parents():
    """The backend state after reorder must belong to the kept parents (synthetic prefixes act as the KV)."""
    r, be = _gen(True, T=9, w=3)
    tails = [p[-1] for p in be.prefixes]
    assert tails == r.final_beams["tokens"][:, :, -1].reshape(-1).tolist()


def test_generated_token_accounting():
    r, be = _gen(False, N=2, T=10, w=3)
    assert be.calls == 10                              # 1 start + 9 steps (the last also yields final routing)
    assert r.tokens.shape == (2, 10)
    acc = P.token_accounting(n_sentences=2, seqlen=10, w=3, vocab=40)
    assert acc["emitted_tokens"] == 20 and acc["forward_rows"] == 2 * (1 + 9 * 3)
    assert acc["scored_candidates"] == 2 * (40 + 8 * 3 * 40)


def test_eos_does_not_stop_and_length_is_exact():
    be = G.SyntheticBackend(V=6, L=2, E=4, k=1, seed=3)
    r = G.generate(be, [5], seqlen=30, w=2, tau=1.0, use_balance=True)
    assert r.tokens.shape == (1, 30)                   # token 5 plays "EOS" and never ends the sequence


def test_deterministic_generation():
    a, _ = _gen(True); b, _ = _gen(True)
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.cum_logp, b.cum_logp)


def test_output_serialisation_roundtrip(tmp_path):
    r, _ = _gen(True)
    path = tmp_path / "x.jsonl"
    P.write_jsonl(r, str(path), decode=lambda ids: " ".join(map(str, ids)))
    rows = [json.loads(l) for l in open(path)]
    assert [x["token_ids"] for x in rows] == r.tokens.tolist()
    assert all(x["n_tokens"] == r.tokens.shape[1] and x["padding_tokens"] == 0 for x in rows)


def test_exact_sequence_length_validation():
    ok = P.validate_generated(torch.arange(16 * 512).view(16, 512), 16, 512, 151646)
    assert ok["valid"]
    bad = P.validate_generated(torch.zeros(15, 512, dtype=torch.long), 16, 512, 151646)
    assert not bad["valid"]
    dup = torch.arange(512).repeat(16, 1)
    assert not P.validate_generated(dup, 16, 512, 151646)["valid"]


# 22-27 diagnostics, accounting, gates, decision --------------------------------------------------------
def test_diversity_diagnostics():
    from moe_calib import ebss_official_data as ED
    assert ED.distinct_n([1, 1, 1, 1], 1) == 0.25 and ED.distinct_n([1, 2, 3, 4], 2) == 1.0
    d = P.distinct3([[1, 2, 3, 1, 2, 3]])
    assert d == pytest.approx(3 / 4)


def test_routing_representativeness_l1():
    from moe_calib import ebss_official_data as ED
    ref = np.full(4, 0.25)
    assert ED.routing_l1([1, 1, 1, 1], ref) == 0.0
    assert ED.routing_l1([4, 0, 0, 0], ref) == pytest.approx(1.5)


def test_generation_runtime_accounting():
    r, _ = _gen(True)
    for k in ("total_s", "stats_s", "score_s", "select_s", "reorder_gather_s"):
        assert r.timing[k] >= 0
    parts = r.timing["stats_s"] + r.timing["score_s"] + r.timing["select_s"] + r.timing["reorder_gather_s"]
    assert parts <= r.timing["total_s"] + 1e-6


def test_gpu_memory_accounting():
    m = P.memory_estimate(rows=64, seqlen=512, n_layers=24, kv_heads=16, head_dim=128, dtype_bytes=2)
    assert m["kv_cache_gb"] == pytest.approx(2 * 64 * 512 * 24 * 16 * 128 * 2 / 1e9)


def test_correctness_gate():
    good = dict(dispatch_mismatches=0, incremental_equals_recount=True, deterministic=True, logprob_identity=True,
                ranking_recomputed=True, po_ignores_balance=True, ebss_includes_balance=True, length_exact=True,
                token_ids_valid=True, no_chat_template=True, replay_logprob_close=True, parents_intact=True)
    assert P.smoke_gate(good)["passed"]
    assert not P.smoke_gate(dict(good, dispatch_mismatches=1))["passed"]
    assert not P.smoke_gate(dict(good, deterministic=False))["passed"]


def test_scientific_decision_classification():
    D = P.decide
    base = dict(pilot_complete=True, gates_ok=True)
    assert D(**base, d_sigma=-0.01, d_cv=-0.01, gptq=None) == "MODEL-MATCHED-EBSS-BALANCING-SUPPORTED"
    assert D(**base, d_sigma=0.01, d_cv=0.02, gptq=None) == "MODEL-MATCHED-EBSS-BALANCING-NOT-SUPPORTED"
    assert D(**base, d_sigma=-0.01, d_cv=0.01, gptq=None) == "MODEL-MATCHED-EBSS-MIXED"
    q = dict(valid=True, ebss=0.9, po=1.0, c4=[1.0, 1.01, 0.99])
    assert D(**base, d_sigma=0.01, d_cv=0.01, gptq=q) == "MODEL-MATCHED-EBSS-QUALITY-GAIN"
    q2 = dict(q, ebss=0.995)                            # beats PO, not the -0.5 % margin vs C4 mean / every seed
    assert D(**base, d_sigma=-0.01, d_cv=-0.01, gptq=q2) == "MODEL-MATCHED-EBSS-BALANCING-SUPPORTED"
    assert D(pilot_complete=False, gates_ok=True, d_sigma=-1, d_cv=-1, gptq=None) == "RUN-BLOCKED"
    assert D(pilot_complete=True, gates_ok=False, d_sigma=-1, d_cv=-1, gptq=None) == "RUN-BLOCKED"


# 28-30 no GPTQ change, no contamination, backward compatibility -----------------------------------------
GPTQ_SHA = "58e40d1c0e9f2d4fd39d22e4a12b83c9828763a37be25eea544f9db03d6fd39d"


def test_gptq_unmodified():
    h = hashlib.sha256(open(os.path.join(ROOT, "moe_calib", "gptq.py"), "rb").read()).hexdigest()
    assert h == GPTQ_SHA                               # same hash the official-EBSS study froze


def test_no_heldout_contamination_check():
    held = [list(range(100, 612))]
    gen_ok = [list(range(1000, 1512))]
    gen_bad = [list(range(1000, 1100)) + list(range(200, 240)) + list(range(2000, 2372))]
    assert P.contamination(gen_ok, held)["windows_with_32gram_match"] == 0
    assert P.contamination(gen_bad, held)["windows_with_32gram_match"] == 1


def test_backward_compatibility_previous_modules_import():
    from moe_calib import ebss_official_data as ED  # noqa: F401
    import official_ebss_validation as OV          # noqa: F401
    assert OV.CONFIGS[-1] == "EBSS" and ED.B == 128
    assert P.OUT == "runs_ebss_generation_pilot" and "official" not in P.CACHE
