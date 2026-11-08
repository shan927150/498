"""Partial-model GPTQ installation for model-level (PPL) evaluation.

Scope (identical to E3): every routed expert's gate_proj / up_proj / down_proj
in the profiled layers is quantized with the frozen GPTQ solver from the
NON-SEQUENTIAL Hessians of cached BF16 layer inputs; everything else
(attention, routers, shared experts, embeddings, norms, lm_head, all other
layers) keeps its original BF16 weights.

Weights are FAKE-QUANTIZED: the solver returns the dequantized FP32 matrix
scale * (q - zero) (act-order permutation already undone, group mapping
applied); it is cast to the model dtype (BF16) and copied into the expert's
nn.Linear in its native [out, in] orientation (Qwen2MoeMLP projections have
no bias). This is simulated weight-only GPTQ, not packed 4-bit inference: no
memory or throughput claim follows from it.

The quantization itself is E3's own `quantize_and_score(..., write_back=True)`,
unchanged; this module only adds pristine-weight snapshots / restoration,
weight capture for verification, checksums of untouched parameters, and an
expert-level output check.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager

import torch

from . import model_utils as MU

PROJ = ("gate_proj", "up_proj", "down_proj")


def target_names(model, layers):
    E = MU.num_experts(model)
    return {f"model.layers.{L}.mlp.experts.{e}.{p}.weight" for L in layers for e in range(E) for p in PROJ}


# ---------------------------------------------------------------- pristine state
@torch.no_grad()
def snapshot_experts(model, layers):
    """CPU clones of every target expert weight (the pristine BF16 state)."""
    E = MU.num_experts(model)
    return {(L, e): {k: w.detach().to("cpu", copy=True) for k, w in MU.get_expert_weights(model, L, e).items()}
            for L in layers for e in range(E)}


@torch.no_grad()
def restore_experts(model, snap):
    for (L, e), W in snap.items():
        MU.set_expert_weights(model, L, e, W)


@torch.no_grad()
def experts_match(model, snap):
    """(n_tensors_equal, n_tensors_total) against a snapshot, bitwise."""
    eq = tot = 0
    for (L, e), W in snap.items():
        cur = MU.get_expert_weights(model, L, e)
        for k in W:
            tot += 1
            eq += bool(torch.equal(cur[k].detach().cpu(), W[k]))
    return eq, tot


@torch.no_grad()
def _digest(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha1(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16
                        else t.numpy().tobytes()).hexdigest()


@torch.no_grad()
def untouched_digests(model, layers):
    """sha1 of every parameter / buffer that must NOT change (all but the targets)."""
    tgt = target_names(model, layers)
    out = {}
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if n not in tgt and not p.is_meta:
            out[n] = _digest(p.view(torch.int16) if p.dtype == torch.bfloat16 else p)
    return out


@torch.no_grad()
def installed_digest(model, layers):
    """One sha256 over all target expert weights in a fixed order (run identity)."""
    h = hashlib.sha256()
    E = MU.num_experts(model)
    for L in layers:
        for e in range(E):
            for k, w in MU.get_expert_weights(model, L, e).items():
                h.update(w.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- quantize + install
@contextmanager
def capture_installs(keys):
    """Record the FP32 W_q that E3 writes back for the (L, e) in `keys`."""
    keys, cap, orig = set(keys), {}, MU.set_expert_weights

    def spy(model, L, e, wq):
        if (L, e) in keys:
            cap[(L, e)] = {k: v.detach().float().cpu().clone() for k, v in wq.items()}
        return orig(model, L, e, wq)

    MU.set_expert_weights = spy
    try:
        yield cap
    finally:
        MU.set_expert_weights = orig


def quantize_install(model, cache_dir, layers, chosen, eval_ids, K, bits, group_size, percdamp, capture=(),
                     eval_cache_dir=None):
    """E3's quantize_and_score with write_back=True (all 60 x len(layers) experts
    quantized and installed). Returns E3's per-expert scores and counts plus
    the captured FP32 W_q for `capture` experts. `eval_cache_dir` (E3's own option): reconstruction
    activations from a different cache than calibration (eval_ids index ITS rows)."""
    import e3_allocation as E3
    with capture_installs(capture) as cap:
        results, counts, orig, t_h, t_q = E3.quantize_and_score(
            model, cache_dir, layers, chosen, {}, eval_ids, K, bits, group_size,
            write_back=True, percdamp=percdamp, eval_cache_dir=eval_cache_dir)
    assert counts["n_experts_quantized_total"] == len(layers) * MU.num_experts(model)
    return dict(results=results, counts=counts, hessian_s=t_h, gptq_s=t_q, captured=cap)


# ---------------------------------------------------------------- expert-level check
@torch.no_grad()
def expert_inputs(model, L, e, layer_inputs, max_tokens=2048):
    """Routed inputs of expert (L, e) for the given cached layer inputs [N, T, d]
    (forward pre-hook on the expert during a normal layer forward)."""
    got = []
    mod = MU.expert_module(model, L, e)
    h = mod.register_forward_pre_hook(lambda m, a: got.append(a[0].detach().float().cpu()))
    try:
        with MU.layer_on_gpu(model, L):
            for i in range(layer_inputs.shape[0]):
                MU.run_layer(model, L, layer_inputs[i:i + 1].to(MU.GPU))
    finally:
        h.remove()
    X = torch.cat(got) if got else torch.zeros(0, model.config.hidden_size)
    return X[:max_tokens]


def _mlp32(X, W, act):
    return (act(X @ W["gate_proj"].T) * (X @ W["up_proj"].T)) @ W["down_proj"].T


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


@torch.no_grad()
def expert_output_check(model, L, e, Wq32, W_orig, X):
    """Compare, on identical inputs X [n, d]:
      y_e3   = FP32 MLP with E3's FP32 W_q (the weights E3's evaluator scores)
      y_inst = the installed BF16 module, run as the model runs it
    and report the BF16 installation / compute deviation relative to the size of
    the quantization perturbation itself (y_e3 vs FP32 MLP with original weights)."""
    mod = MU.expert_module(model, L, e)
    act = mod.act_fn
    dev = MU.GPU
    Xf = X.to(dev, torch.float32)
    Wq = {k: v.to(dev, torch.float32) for k, v in Wq32.items()}
    Wo = {k: v.to(dev, torch.float32) for k, v in W_orig.items()}
    Wi = {k: getattr(mod, k).weight.detach().to(dev, torch.float32) for k in PROJ}
    y_fp = _mlp32(Xf, Wo, act)
    y_e3 = _mlp32(Xf, Wq, act)
    y_inst32 = _mlp32(Xf, Wi, act)
    m_dev = next(mod.parameters()).device
    mod.to(dev)
    try:
        y_inst = mod(X.to(dev, MU.DTYPE)).float()
        # BF16-compute noise reference: the same module with the ORIGINAL weights
        saved = {k: getattr(mod, k).weight.detach().clone() for k in PROJ}
        for k in PROJ:
            getattr(mod, k).weight.copy_(W_orig[k].to(dev, MU.DTYPE))
        y_orig_bf16 = mod(X.to(dev, MU.DTYPE)).float()
        for k in PROJ:
            getattr(mod, k).weight.copy_(saved[k])
    finally:
        mod.to(m_dev)
    w_exact = all(torch.equal(getattr(mod, k).weight.detach().cpu(), Wq32[k].to(MU.DTYPE)) for k in PROJ)
    w_changed = all(not torch.equal(getattr(mod, k).weight.detach().cpu(), W_orig[k].cpu()) for k in PROJ)
    q = _rel(y_e3, y_fp)
    out = dict(layer=L, expert=e, n_tokens=int(X.shape[0]),
               installed_equals_bf16_of_e3_wq=w_exact, installed_differs_from_bf16_original=w_changed,
               quant_effect_rel=q,                                   # ||y_e3 - y_fp|| / ||y_fp||
               installed_vs_e3_rel=_rel(y_inst, y_e3),               # what the model computes vs E3
               weight_rounding_only_rel=_rel(y_inst32, y_e3),        # BF16 cast of W_q, FP32 compute
               bf16_compute_noise_rel=_rel(y_orig_bf16, y_fp))       # same deviation for untouched weights
    out["ratio_to_quant_effect"] = out["installed_vs_e3_rel"] / max(q, 1e-30)
    return out


# ---------------------------------------------------------------- full-depth helpers
def resolve_layers(spec, model):
    """'all' -> every decoder layer that has a routed-expert MoE block; else a list of ints.
    Raises if a requested layer has no routed experts (would silently skip experts)."""
    n = MU.num_layers(model)
    moe = [L for L in range(n) if hasattr(MU.moe_block(model, L), "experts")]
    if spec in ("all", ["all"]):
        return moe
    layers = sorted({int(x) for x in (spec if isinstance(spec, (list, tuple)) else str(spec).split(","))})
    bad = [L for L in layers if L not in moe]
    if bad:
        raise ValueError(f"layers {bad} have no routed experts")
    return layers


@torch.no_grad()
def target_tensor_digests(model, layers):
    """{'L.e.proj': sha1} for every target expert projection (pristine / installed identity)."""
    E = MU.num_experts(model)
    return {f"{L}.{e}.{k}": _digest(w.view(torch.int16))
            for L in layers for e in range(E) for k, w in MU.get_expert_weights(model, L, e).items()}


@torch.no_grad()
def nonfinite_targets(model, layers):
    E = MU.num_experts(model)
    return [f"{L}.{e}.{k}" for L in layers for e in range(E)
            for k, w in MU.get_expert_weights(model, L, e).items() if not torch.isfinite(w).all()]


@torch.no_grad()
def layer_digest(model, L):
    h = hashlib.sha256()
    for e in range(MU.num_experts(model)):
        for k, w in MU.get_expert_weights(model, L, e).items():
            h.update(w.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def save_layer(model, L, path):
    """Installed (BF16 effective GPTQ) weights of one layer; written atomically."""
    import os
    d = {f"{e}.{k}": w.detach().cpu().clone() for e in range(MU.num_experts(model))
         for k, w in MU.get_expert_weights(model, L, e).items()}
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def install_layer_file(model, L, path):
    d = torch.load(path)
    E = MU.num_experts(model)
    assert len(d) == E * len(PROJ), f"{path}: {len(d)} tensors"
    for e in range(E):
        MU.set_expert_weights(model, L, e, {k: d[f"{e}.{k}"] for k in PROJ})
