"""Opt-in, diagnostic-only inference-time routing recorder / top-k override (transformers 4.51.3 Qwen2MoE).

Two modes of a RoutingSession, attached to EVERY MoE block of the model for the duration of a `with attach(...)`:

  natural  the class forward of Qwen2MoeSparseMoeBlock runs unmodified. A forward hook on the router linear
           (`mlp.gate`) receives the router logits and RE-COMPUTES softmax(logits, fp32) + topk exactly as the block
           does (same ops, same input tensor -> bitwise the same indices/values) and stores them. The hook returns
           None, so the block output is untouched.
  force    additionally `blk.forward` is replaced by moe_calib.frozen_routing.make_forced_forward, the verbatim
           copy of the class forward in which only `selected_experts` is substituted (here: by the stored
           reference indices of the same window / token, in the stored slot order). Router logits and softmax are
           still computed from the CURRENT hidden state; the routing weights of the forced experts are the
           current softmax probabilities gathered at the forced indices (norm_topk_prob is honoured, it is False
           for Qwen1.5-MoE-A2.7B). Expert dispatch (expert loop 0..E-1, index_add_), the shared expert and the
           BF16 casts are the original code. The gate hook still records the NATURAL top-k the model would have
           chosen, plus the current probabilities of the forced experts.

Nothing is written to any parameter; `detach()` removes every hook and override.
Token mapping: the evaluator runs batches of `batch` complete windows; the MoE block sees [B*T, d] rows, row r of
batch bi is window bi*batch + r // T, position r % T.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from . import model_utils as MU
from . import frozen_routing as FRT


class RoutingSession:
    def __init__(self, n_layers, n_win, T, k, n_experts, batch, mode="natural", ref_idx=None, record=True):
        assert mode in ("natural", "force")
        if mode == "force":
            assert ref_idx is not None and tuple(ref_idx.shape) == (n_layers, n_win, T, k), \
                (None if ref_idx is None else ref_idx.shape)
            assert ref_idx.dtype in (torch.int16, torch.int32, torch.int64)
        self.n_layers, self.n_win, self.T, self.k, self.E, self.batch = n_layers, n_win, T, k, n_experts, batch
        self.mode, self.ref, self.record = mode, ref_idx, record
        self.bi = None                                   # current batch index, set by the evaluator
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt)
        if record:
            self.idx = z(n_layers, n_win, T, k, dt=torch.int16)          # natural ordered top-k
            self.pk = z(n_layers, n_win, T, k)                           # natural top-k probabilities (slot order)
            self.p_next = z(n_layers, n_win, T)                          # (k+1)-th largest probability
            self.p_forced = z(n_layers, n_win, T, k) if mode == "force" else None   # current probs of executed experts
        self.dispatch = torch.zeros(n_layers, n_experts, dtype=torch.int64)       # rows each expert actually received
        self.n_gate_calls = torch.zeros(n_layers, dtype=torch.int64)
        self.n_forced_calls = torch.zeros(n_layers, dtype=torch.int64)
        self._handles, self._overridden = [], []

    # ------------------------------------------------------------ positions
    def _rows(self, n_rows):
        assert self.bi is not None, "evaluator must set session.bi"
        B = n_rows // self.T
        assert B * self.T == n_rows
        w0 = self.bi * self.batch
        assert w0 + B <= self.n_win
        return w0, B

    def ref_slice(self, L, n_rows):
        w0, B = self._rows(n_rows)
        return self.ref[L, w0:w0 + B].reshape(B * self.T, self.k)

    # ------------------------------------------------------------ hooks
    def _gate_hook(self, L):
        def hook(mod, inp, out):
            self.n_gate_calls[L] += 1
            if not self.record:
                return None
            with torch.no_grad():
                probs = F.softmax(out, dim=1, dtype=torch.float)
                w, i = torch.topk(probs, self.k, dim=-1)
                nxt = torch.topk(probs, self.k + 1, dim=-1).values[:, self.k]
                w0, B = self._rows(out.shape[0])
                sl = slice(w0, w0 + B)
                self.idx[L, sl] = i.view(B, self.T, self.k).to("cpu", torch.int16)
                self.pk[L, sl] = w.view(B, self.T, self.k).cpu()
                self.p_next[L, sl] = nxt.view(B, self.T).cpu()
                if self.mode == "force":
                    f = self.ref_slice(L, out.shape[0]).to(probs.device, torch.long)
                    self.p_forced[L, sl] = probs.gather(1, f).view(B, self.T, self.k).cpu()
            return None
        return hook

    def _count_hook(self, L, e):
        def hook(mod, inp):
            self.dispatch[L, e] += inp[0].shape[0]
            return None
        return hook

    def _provider(self, L):
        def provider(selected):
            self.n_forced_calls[L] += 1
            return self.ref_slice(L, selected.shape[0]).to(selected.device, selected.dtype)
        return provider

    def attach(self, model):
        assert not self._handles and not self._overridden, "session already attached"
        for L in range(MU.num_layers(model)):
            blk = MU.moe_block(model, L)
            assert "forward" not in blk.__dict__, f"layer {L}: MoE block already overridden"
            self._handles.append(blk.gate.register_forward_hook(self._gate_hook(L)))
            for e in range(len(blk.experts)):
                self._handles.append(blk.experts[e].register_forward_pre_hook(self._count_hook(L, e)))
            if self.mode == "force":
                blk.forward = FRT.make_forced_forward(blk, self._provider(L))
                self._overridden.append(blk)

    def detach(self):
        for h in self._handles:
            h.remove()
        for blk in self._overridden:
            del blk.forward
        self._handles, self._overridden = [], []


@contextmanager
def attach(model, session):
    session.attach(model)
    try:
        yield session
    finally:
        session.detach()
        FRT.assert_no_override(model)


def new_session(model, windows, batch, mode="natural", ref_idx=None, record=True):
    N, T = windows.shape
    return RoutingSession(MU.num_layers(model), N, T, MU.top_k(model), MU.num_experts(model), batch, mode=mode,
                          ref_idx=ref_idx, record=record)


# ---------------------------------------------------------------- loss / evaluator
def nll_tokens_from_logits(logits, ids):
    """Op-for-op diagnostics.eval_partial_ppl.nll_from_logits (mask=None) that also returns the per-target NLL.
    Returns (per-row sum NLL float64 [B], n scored [B], per-target NLL float64 [B, T-1])."""
    B, T, V = logits.shape
    lp = logits[:, :-1].float()
    tgt = ids[:, 1:].to(lp.device)
    nll = F.cross_entropy(lp.reshape(-1, V), tgt.reshape(-1), reduction="none").view(B, T - 1).double()
    valid = torch.ones_like(nll, dtype=torch.bool)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return nll.sum(1), valid.sum(1), nll


def _sha_hs(h, hasher):
    hasher.update(h.detach().contiguous().view(torch.int16).cpu().numpy().tobytes())


@torch.no_grad()
def layerwise_eval(model, windows, batch, session=None, layer_hashes=False):
    """The historical EP.layerwise_nll forward (same calls in the same order) with an optional RoutingSession
    (session.bi is set before every MoE-bearing layer call) and per-token NLL.
    Returns dict(nll [N] f64, cnt [N] i64, token_nll [N, T-1] f64, final_sha, layer_sha list|None)."""
    N, T = windows.shape
    hs = [MU.embed(model, windows[i:i + batch]) for i in range(0, N, batch)]
    lsha = [] if layer_hashes else None
    for L in range(MU.num_layers(model)):
        with MU.layer_on_gpu(model, L):
            out = []
            for bi, h in enumerate(hs):
                if session is not None:
                    session.bi = bi
                out.append(MU.run_layer(model, L, h))
            hs = out
        if layer_hashes:
            hh = hashlib.sha256()
            for h in hs:
                _sha_hs(h, hh)
            lsha.append(hh.hexdigest())
    if session is not None:
        session.bi = None
    fh = hashlib.sha256()
    nll, cnt, tok = [], [], []
    for bi, h in enumerate(hs):
        _sha_hs(h, fh)
        for j in range(h.shape[0]):
            logits = model.lm_head(model.model.norm(h[j:j + 1]))
            s, n, t = nll_tokens_from_logits(logits, windows[bi * batch + j:bi * batch + j + 1])
            nll.append(float(s[0])); cnt.append(int(n[0])); tok.append(t[0].cpu())
        hs[bi] = None
    return dict(nll=np.array(nll, dtype=np.float64), cnt=np.array(cnt, dtype=np.int64),
                token_nll=torch.stack(tok).numpy(), final_sha=fh.hexdigest(), layer_sha=lsha)


# ---------------------------------------------------------------- routing comparisons
def set_disagreement(a, b):
    """a, b: int [..., k] ordered top-k. Returns (set_changed bool [...], order_only bool [...]):
    set_changed = membership differs; order_only = same membership, different slot order."""
    sa, _ = torch.sort(a.long(), dim=-1)
    sb, _ = torch.sort(b.long(), dim=-1)
    set_changed = (sa != sb).any(-1)
    any_slot = (a.long() != b.long()).any(-1)
    return set_changed, any_slot & ~set_changed


def n_membership_changes(a, b):
    """Number of experts of `a` not in `b` per token (0..k)."""
    a, b = a.long(), b.long()
    inb = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)
    return (~inb).sum(-1)


def validate_indices(idx, n_experts):
    """(all in range, no repeats within a token)."""
    i = idx.long()
    in_range = bool(((i >= 0) & (i < n_experts)).all())
    s, _ = torch.sort(i, dim=-1)
    no_rep = bool((s[..., 1:] != s[..., :-1]).all())
    return in_range, no_rep


def expert_counts(idx, n_experts):
    """[L, E] token-slot counts per expert from int [L, ..., k] indices."""
    Lc = idx.shape[0]
    flat = idx.reshape(Lc, -1).long()
    return torch.stack([torch.bincount(flat[L], minlength=n_experts) for L in range(Lc)])
