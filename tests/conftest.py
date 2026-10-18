"""Tiny CPU fixtures: a random 3-layer Qwen2-MoE and a synthetic layer cache,
so the E1 -> E2 -> E3 code paths run in seconds without the 28 GB model."""
import json
import os
import sys

os.environ.setdefault("MOE_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from moe_calib import model_utils as MU

E, LAYERS, POOL, HOLDOUT, TUNE, T, D = 8, [0, 2], 40, 24, 12, 16, 64


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2MoeConfig(vocab_size=64, hidden_size=D, intermediate_size=64,
                         moe_intermediate_size=32, shared_expert_intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                         num_experts=E, num_experts_per_tok=2, max_position_embeddings=64,
                         decoder_sparse_step=1, mlp_only_layers=[])
    cfg._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(cfg).to(torch.bfloat16).eval()
    model.config.use_cache = False
    return model, None


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    monkeypatch.setattr(MU, "load_model", lambda *a, **k: tiny_model())
    MU._KW.clear()


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    g = torch.Generator().manual_seed(1)
    N = POOL + HOLDOUT
    torch.save({"pool": torch.zeros(POOL, T, dtype=torch.long),
                "holdout": torch.zeros(HOLDOUT, T, dtype=torch.long)}, d / "ids.pt")
    for L in LAYERS:
        torch.save(torch.randn(N, T, D, generator=g).to(torch.bfloat16), d / f"L{L}.pt")
    meta = dict(pool_size=POOL, holdout_size=HOLDOUT, tune_size=TUNE, seqlen=T,
                data_seed=0, layers=LAYERS, model="tiny", num_layers=3,
                cache_seconds=0.0, forward_s_per_seq=0.01)
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d)
