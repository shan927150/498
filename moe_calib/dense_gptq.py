"""Minimal dense-MLP adapter around the EXISTING Hessian accumulator and GPTQ solver (dense-model control).

Nothing here re-implements GPTQ or Hessian accumulation:
  * Hessians: moe_calib.hessian.BlockHessianAccumulator with ONE pseudo-expert (index 0) per layer, keys
    (L, 0, "gu") / (L, 0, "down"); `add` is called from forward pre-hooks exactly as moe_calib.profiler.Profiler
    does for routed experts (gate_proj input -> "gu", shared by gate_proj and up_proj; down_proj input -> "down"),
    one sequence at a time on cached layer inputs, the K blocks assigned by Profiler.run's default contiguous
    K-way split, weight 1.0, then summed over all K blocks (quant_eval.quantize_from_acc).
  * Solver: moe_calib.gptq.quantize_expert (gate/up share H_gu, down uses H_down), unchanged.
  * Install: exactly model_utils.set_expert_weights' cast  p.copy_(W_q.to(p.device, p.dtype)).
The MoE code path is not modified; for a dense Qwen2 decoder layer the MLP is model.model.layers[L].mlp with
gate_proj / up_proj / down_proj (no bias).
"""
from __future__ import annotations

import hashlib
import time

import torch

from . import model_utils as MU
from .gptq import quantize_expert
from .hessian import BlockHessianAccumulator

PROJ = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- layer mapping
def map_layers(src_layers, n_src, n_dst):
    """Relative-depth mapping of decoder-layer indices: L -> round(L * (n_dst - 1) / (n_src - 1)).
    Endpoints map to endpoints (0 -> 0, n_src-1 -> n_dst-1). Raises on collisions."""
    out = [int(round(L * (n_dst - 1) / (n_src - 1))) for L in src_layers]
    if len(set(out)) != len(out) or not all(0 <= x < n_dst for x in out):
        raise ValueError(f"layer mapping {src_layers} -> {out} invalid for depth {n_dst}")
    return out


def mlp(model, L):
    return model.model.layers[L].mlp


def dims(model):
    return model.config.hidden_size, model.config.intermediate_size


def get_weights(model, L):
    m = mlp(model, L)
    return {k: getattr(m, k).weight for k in PROJ}


@torch.no_grad()
def set_weights(model, L, wq):
    """Same cast as model_utils.set_expert_weights."""
    m = mlp(model, L)
    for k, w in wq.items():
        p = getattr(m, k).weight
        p.copy_(w.to(p.device, p.dtype))


def target_names(layers):
    return {f"model.layers.{L}.mlp.{k}.weight" for L in layers for k in PROJ}


# ---------------------------------------------------------------- digests / snapshots
def _bytes(t):
    t = t.detach().contiguous().cpu()
    return (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()


@torch.no_grad()
def target_digests(model, layers):
    return {f"{L}.{k}": hashlib.sha1(_bytes(w)).hexdigest() for L in layers for k, w in get_weights(model, L).items()}


@torch.no_grad()
def untouched_digests(model, layers):
    """sha1 of every parameter / buffer except the target MLP projections (tied lm_head == embed is one tensor)."""
    tgt = target_names(layers)
    seen, out = set(), {}
    for n, p in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers()):
        if n in tgt or p.is_meta:
            continue
        out[n] = hashlib.sha1(_bytes(p)).hexdigest()
    return out


@torch.no_grad()
def installed_digest(model, layers):
    h = hashlib.sha256()
    for L in layers:
        for k, w in get_weights(model, L).items():
            h.update(_bytes(w))
    return h.hexdigest()


@torch.no_grad()
def snapshot(model, layers):
    return {L: {k: w.detach().to("cpu", copy=True) for k, w in get_weights(model, L).items()} for L in layers}


@torch.no_grad()
def restore(model, snap):
    for L, W in snap.items():
        set_weights(model, L, W)


@torch.no_grad()
def matches(model, snap):
    eq = tot = 0
    for L, W in snap.items():
        cur = get_weights(model, L)
        for k in W:
            tot += 1
            eq += bool(torch.equal(cur[k].detach().cpu(), W[k]))
    return eq, tot


# ---------------------------------------------------------------- Hessians (Profiler semantics)
class DenseProfiler:
    """Profiler.run for a dense MLP: replay ONE decoder layer on cached inputs one sequence at a time; forward
    pre-hooks on mlp.gate_proj ("gu") and mlp.down_proj ("down") feed BlockHessianAccumulator.add."""

    def __init__(self, model, L, K):
        self.model, self.L, self.K = model, L, K
        d_gu, d_down = dims(model)
        self.acc = BlockHessianAccumulator([L], 1, d_gu, d_down, K, device=MU.GPU, store_device=MU.GPU)
        self.cur_block, self.cur_weight = 0, 1.0
        self.seconds = 0.0

    def _hook(self, proj):
        key = (self.L, 0, proj)

        def fn(module, args):
            X = args[0].reshape(-1, args[0].shape[-1])
            self.acc.add(key, X, self.cur_block, self.cur_weight)
        return fn

    @torch.no_grad()
    def run(self, cache, seq_ids):
        n = len(seq_ids)
        blocks = [(i * self.K) // max(n, 1) for i in range(n)]            # Profiler.run default
        m = mlp(self.model, self.L)
        hs = [m.gate_proj.register_forward_pre_hook(self._hook("gu")),
              m.down_proj.register_forward_pre_hook(self._hook("down"))]
        t0 = time.time()
        try:
            with MU.layer_on_gpu(self.model, self.L):
                for i, sid in enumerate(seq_ids):
                    self.cur_block = blocks[i]
                    MU.run_layer(self.model, self.L, cache[sid:sid + 1].to(MU.GPU, non_blocking=True))
        finally:
            for h in hs:
                h.remove()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds += time.time() - t0
        return self

    def H(self, proj):
        """quant_eval.quantize_from_acc: sum over all K blocks."""
        return self.acc.H_sum((self.L, 0, proj), list(range(self.K)))

    def tokens(self, proj):
        return self.acc.tokens((self.L, 0, proj))


# ---------------------------------------------------------------- GPTQ + integer codes
@torch.no_grad()
def quantize_layer(model, L, H_gu, H_down, bits, group_size, percdamp, with_info=True):
    """quantize_expert on the dense MLP's FP32 weights (quant_eval.fp_weights semantics: .detach().float()).
    with_info: the solver's read-only `info` capture (perm, per-group scale/zero) -> integer codes."""
    W = {k: v.detach().float() for k, v in get_weights(model, L).items()}
    if not with_info:
        return quantize_expert(W, H_gu, H_down, bits, group_size, percdamp), None
    infos = {}
    Wq = quantize_expert(W, H_gu, H_down, bits, group_size, percdamp, perms={}, infos=infos)
    return Wq, infos


@torch.no_grad()
def integer_codes(Wq, info, bits, group_size):
    """Recover the solver's integer codes q (uint8, ORIGINAL column order) from the FP32 dequantized W_q
    = scale * (q - zero) and the captured per-group (scale, zero) of the PERMUTED column order. Returns
    (codes, group_id [cols] in original order, exact) where exact = scale*(codes-zero) reproduces W_q bitwise."""
    perm = info["perm"].long()
    sc, zr = info["scale"].float(), info["zero"].float()                   # [n_groups, rows]
    Qp = Wq.detach().float().cpu()[:, perm]
    cols = Qp.shape[1]
    g = torch.arange(cols) // group_size
    S, Z = sc[g].T, zr[g].T                                                # [rows, cols] permuted order
    q = torch.clamp(torch.round(Qp / S + Z), 0, 2 ** bits - 1)
    exact = bool(torch.equal(S * (q - Z), Qp))
    inv = torch.argsort(perm)
    gid = torch.empty(cols, dtype=torch.int32)
    gid[perm] = g.int()
    return q[:, inv].to(torch.uint8), gid, exact
