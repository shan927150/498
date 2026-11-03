"""Model loading + LAYER-WISE OFFLOAD helpers for Qwen1.5-MoE-A2.7B.

Why layer-wise: ICRN gives MIG slices (A100 20 GB / H200 16 GB VRAM), but the
bf16 model is ~28.6 GB. So the model lives in CPU RAM (use the A100 option:
72 GB RAM) and exactly ONE decoder layer is moved to the GPU at a time — the
same strategy the reference GPTQ implementation uses for large models.

Small always-needed modules (embedding, rotary, final norm, lm_head) are kept
on the GPU permanently (~1.3 GB).

Module layout (transformers Qwen2MoE, pinned in requirements.txt):
  model.model.layers[i].mlp                 -> Qwen2MoeSparseMoeBlock
  model.model.layers[i].mlp.gate            -> nn.Linear(hidden, num_experts) (router)
  model.model.layers[i].mlp.experts[j]      -> Qwen2MoeMLP {gate,up,down}_proj
  model.model.layers[i].mlp.shared_expert   -> NOT quantized in this study
gate_proj/up_proj share input x (d=2048) -> one Hessian H_gu;
down_proj input is act(gate x)*(up x) (d=1408) -> H_down.  [GPTQ: H = sum x x^T]
"""
from __future__ import annotations
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MOE_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")
GPU = os.environ.get("MOE_DEVICE", "cuda")   # override only for CPU tests
DTYPE = torch.bfloat16   # bf16 avoids fp16 overflow in MoE activations


def _partial_layers():
    """MOE_LOAD_LAYERS="1,8,16,23" -> load ONLY those decoder layers (+ embed,
    final norm, rotary). Every other layer stays on the meta device, so a
    process that only replays single layers needs ~5 GB of RAM instead of
    ~30 GB; this is what lets several experiments share one 72 GB ICRN
    session. Full-model work (cache building, PPL) must leave it unset."""
    v = os.environ.get("MOE_LOAD_LAYERS", "").strip()
    return sorted({int(x) for x in v.split(",") if x.strip()}) if v else None


def _snapshot_dir(name: str) -> str:
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"]
    try:
        return snapshot_download(name, allow_patterns=pats, local_files_only=True)
    except Exception:
        return snapshot_download(name, allow_patterns=pats)


def _load_partial(name: str, layers: list[int]):
    import json
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig
    path = _snapshot_dir(name)
    cfg = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=DTYPE,
                                                 attn_implementation="eager")
    bad = [L for L in layers if not 0 <= L < cfg.num_hidden_layers]
    if bad:
        raise ValueError(f"MOE_LOAD_LAYERS has layers {bad} outside 0..{cfg.num_hidden_layers - 1}")
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        wmap = json.load(open(idx))["weight_map"]
    else:
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    keep = tuple(f"model.layers.{L}." for L in layers) + ("model.embed_tokens.", "model.norm.")
    by_file: dict[str, list[str]] = {}
    for k, f in wmap.items():
        if k.startswith(keep):
            by_file.setdefault(f, []).append(k)
    n = 0
    for f, keys in by_file.items():
        with safe_open(os.path.join(path, f), framework="pt", device="cpu") as sf:
            for k in keys:
                set_module_tensor_to_device(model, k, "cpu", value=sf.get_tensor(k).to(DTYPE))
                n += 1
    for L in layers:                                  # every needed weight must be real
        for pn, p in model.model.layers[L].named_parameters():
            if p.is_meta:
                raise RuntimeError(f"layer {L} parameter {pn} missing from checkpoint")
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception:
        tok = None
    model.eval()
    model.config.use_cache = False
    model.partial_layers = list(layers)
    m = model.model
    m.embed_tokens.to(GPU)
    m.norm.to(GPU)
    if getattr(m, "rotary_emb", None) is not None:
        m.rotary_emb.to(GPU)
    print(f"[model] partial load: layers {layers} ({n} tensors); lm_head and other "
          "layers left on meta", flush=True)
    return model, tok


def load_model(name: str = MODEL_NAME):
    layers = _partial_layers()
    if layers is not None:
        return _load_partial(name, layers)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=DTYPE, attn_implementation="eager",
        low_cpu_mem_usage=True)                      # stays on CPU
    model.eval()
    model.config.use_cache = False
    m = model.model
    m.embed_tokens.to(GPU)
    m.norm.to(GPU)
    if getattr(m, "rotary_emb", None) is not None:
        m.rotary_emb.to(GPU)
    model.lm_head.to(GPU)
    return model, tok


# ---------------------------------------------------------------- structure

def num_layers(model) -> int:
    return len(model.model.layers)


def moe_block(model, layer_idx: int):
    return model.model.layers[layer_idx].mlp


def num_experts(model) -> int:
    return model.config.num_experts            # 60


def top_k(model) -> int:
    return model.config.num_experts_per_tok    # 4


def expert_dims(model):
    return model.config.hidden_size, model.config.moe_intermediate_size  # 2048, 1408


def expert_module(model, layer_idx: int, expert_idx: int):
    return moe_block(model, layer_idx).experts[expert_idx]


def get_expert_weights(model, layer_idx: int, expert_idx: int):
    e = expert_module(model, layer_idx, expert_idx)
    return {"gate_proj": e.gate_proj.weight,
            "up_proj": e.up_proj.weight,
            "down_proj": e.down_proj.weight}


@torch.no_grad()
def set_expert_weights(model, layer_idx: int, expert_idx: int, wq: dict):
    e = expert_module(model, layer_idx, expert_idx)
    for name, w in wq.items():
        p = getattr(e, name).weight
        p.copy_(w.to(p.device, p.dtype))


# ------------------------------------------------------- layer-wise forward

class _Caught(Exception):
    pass


class _Catcher(nn.Module):
    """Replaces layer 0 for one call to capture the kwargs (causal mask,
    position ids / rotary embeddings, ...) the model passes to every decoder
    layer. Version-robust: we never construct these ourselves."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.args, self.kw = None, None
        if hasattr(inner, "attention_type"):
            self.attention_type = inner.attention_type

    def forward(self, hs, *args, **kwargs):
        self.args, self.kw = args, kwargs
        raise _Caught


_KW: dict[int, tuple] = {}


@torch.no_grad()
def layer_kwargs(model, seqlen: int):
    if seqlen in _KW:
        return _KW[seqlen]
    layers = model.model.layers
    cat = _Catcher(layers[0])
    layers[0] = cat
    try:
        model.model(input_ids=torch.zeros(1, seqlen, dtype=torch.long, device=GPU),
                    use_cache=False)
    except _Caught:
        pass
    finally:
        layers[0] = cat.inner
    kw = dict(cat.kw)
    kw["use_cache"] = False
    kw.pop("past_key_value", None)
    kw.pop("past_key_values", None)
    _KW[seqlen] = (cat.args, kw)
    return _KW[seqlen]


@contextmanager
def layer_on_gpu(model, layer_idx: int):
    layer = model.model.layers[layer_idx]
    layer.to(GPU)
    try:
        yield layer
    finally:
        layer.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def run_layer(model, layer_idx: int, hs: torch.Tensor) -> torch.Tensor:
    """hs: [B, T, d] on GPU. Layer must already be on GPU."""
    args, kw = layer_kwargs(model, hs.shape[1])
    out = model.model.layers[layer_idx](hs, *args, **kw)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def embed(model, ids: torch.Tensor) -> torch.Tensor:
    """ids: [B, T] -> hidden [B, T, d] on GPU."""
    return model.model.embed_tokens(ids.to(GPU))
