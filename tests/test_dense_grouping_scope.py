"""CPU tests for the dense-MLP GPTQ adapter (tiny random Qwen2 dense model)."""
import os

os.environ.setdefault("MOE_DEVICE", "cpu")

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from moe_calib import dense_gptq as DG
from moe_calib import model_utils as MU
from moe_calib.gptq import gptq_quantize

D, I, T, N = 64, 160, 16, 10


def tiny_dense():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=D, intermediate_size=I, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    m.config.use_cache = False
    return m


def cache():
    g = torch.Generator().manual_seed(1)
    return torch.randn(N, T, D, generator=g).to(torch.bfloat16)


def test_layer_mapping():
    assert DG.map_layers([1, 8, 16, 23], 24, 28) == [1, 9, 19, 27]
    assert DG.map_layers([0, 23], 24, 24) == [0, 23]
    with pytest.raises(ValueError):
        DG.map_layers([1, 2], 24, 3)


def test_profiler_hessian_equals_manual_profiler_semantics():
    m, X = tiny_dense(), cache()
    MU._KW.clear()
    ids, K, L = [3, 0, 7, 5, 9, 1], 4, 2
    prof = DG.DenseProfiler(m, L, K).run(X, ids)
    got = {"gu": [], "down": []}
    mlp = DG.mlp(m, L)
    h1 = mlp.gate_proj.register_forward_pre_hook(lambda mod, a: got["gu"].append(a[0].reshape(-1, D).float()))
    h2 = mlp.down_proj.register_forward_pre_hook(lambda mod, a: got["down"].append(a[0].reshape(-1, I).float()))
    for s in ids:
        MU.run_layer(m, L, X[s:s + 1])
    h1.remove(); h2.remove()
    blocks = [(i * K) // len(ids) for i in range(len(ids))]
    for pj, d in (("gu", D), ("down", I)):
        Hb = torch.zeros(K, d, d)
        for b, x in zip(blocks, got[pj]):
            Hb[b] += x.T @ x
        assert torch.equal(prof.H(pj), Hb.sum(0))
        assert prof.tokens(pj) == len(ids) * T


def test_quantize_layer_is_unchanged_solver_and_info_path_bitwise():
    m, X = tiny_dense(), cache()
    MU._KW.clear()
    prof = DG.DenseProfiler(m, 1, 4).run(X, list(range(N)))
    Hg, Hd = prof.H("gu"), prof.H("down")
    Wq0, _ = DG.quantize_layer(m, 1, Hg, Hd, 4, 32, 1.0, with_info=False)
    Wq1, infos = DG.quantize_layer(m, 1, Hg, Hd, 4, 32, 1.0)
    W = {k: v.detach().float() for k, v in DG.get_weights(m, 1).items()}
    for k, H in (("gate_proj", Hg), ("up_proj", Hg), ("down_proj", Hd)):
        direct = gptq_quantize(W[k], H, 4, 32, 1.0)
        assert torch.equal(Wq0[k], direct) and torch.equal(Wq1[k], direct)
        q, gid, exact = DG.integer_codes(Wq1[k], infos[k], 4, 32)
        assert exact and q.dtype == torch.uint8 and int(q.max()) <= 15
        assert gid.shape[0] == W[k].shape[1] and int(torch.bincount(gid.long()).max()) == 32


def test_install_restore_and_digests():
    m, X = tiny_dense(), cache()
    MU._KW.clear()
    lay = [1, 3]
    snap, t0, u0 = DG.snapshot(m, lay), DG.target_digests(m, lay), DG.untouched_digests(m, lay)
    assert "lm_head.weight" in u0 and "model.layers.1.mlp.gate_proj.weight" not in u0
    prof = DG.DenseProfiler(m, 1, 2).run(X, list(range(N)))
    Wq, _ = DG.quantize_layer(m, 1, prof.H("gu"), prof.H("down"), 4, 32, 1.0, with_info=False)
    DG.set_weights(m, 1, Wq)
    assert all(torch.equal(w, Wq[k].to(torch.bfloat16)) for k, w in DG.get_weights(m, 1).items())
    assert DG.target_digests(m, lay) != t0 and DG.untouched_digests(m, lay) == u0
    DG.restore(m, snap)
    assert DG.matches(m, snap) == (6, 6) and DG.target_digests(m, lay) == t0
