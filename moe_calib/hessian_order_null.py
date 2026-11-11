"""FP32 Hessian accumulation-order null experiment for MoE GPTQ (helpers).

Actual accumulation mechanism (moe_calib/profiler.py + moe_calib/hessian.py, inspected, unchanged):
  * The E3 Profiler replays ONE decoder layer per calibration sequence (batch 1) in the order of `chosen`
    (the allocation's selection order, NOT sorted). For each expert, a forward pre-hook on gate_proj / down_proj
    receives that sequence's routed rows X (BF16; rows of all top-k slots, slot-major from torch.where on the
    expert mask; a token appears at most once per expert since its top-k experts are distinct).
  * BlockHessianAccumulator.add: Xf = X.float(); outer = Xf.T @ Xf (one FP32 cuBLAS GEMM per sequence and
    expert); weight == 1.0 for FR / FC so no scaling; H[key][block] += outer (FP32 elementwise in-place add).
  * block = (i * K) // n of the sequence's replay POSITION i (K = 4, n = 128): 4 contiguous 32-sequence blocks.
  * GPTQ receives H_sum = H[key][[0,1,2,3]].sum(dim=0) (unnormalized; GPTQ's relative damping is scale-free).
So the replay order enters the Hessian through the sequence of FP32 additions inside each block (and, under the
default positional block rule, through block membership). Nothing sorts or canonicalizes the contributions.

Intervention: the replay order is permuted with a prespecified index permutation; every sequence KEEPS ITS
ORIGINAL BLOCK (block_of_sid from the original position), so each (expert, block) receives exactly the same set
of per-sequence contributions and only the FP32 addition order inside each block changes. The cross-block sum
(dim-0 sum over blocks 0..3) is unchanged. Per-sequence contributions are computed by the unchanged path and are
verified (fingerprints) to be bitwise identical across orders.
"""
from __future__ import annotations
import hashlib
import json
from contextlib import contextmanager

import numpy as np
import torch

from .frozen_routing import ForcedProfiler, tensor_sha

ORDER_SEEDS = (20261121, 20261122, 20261123, 20261124, 20261125)
FP64_SEEDS = (20261121, 20261122)
BF16_PPL = 9.237089824789335
TAU_PPL = 1e-4 * BF16_PPL                     # 0.000924: operational materiality threshold (frozen)
PROJ = ("gate_proj", "up_proj", "down_proj")
PROJ_H = {"gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}


# ---------------------------------------------------------------- orders and blocks
def positional_blocks(n, K):
    """The Profiler's default block rule (Profiler.run with blocks=None)."""
    return [(i * K) // max(n, 1) for i in range(n)]


def index_permutation(seed, n=128):
    return [int(x) for x in np.random.default_rng(seed).permutation(n)]


def order_spec(chosen, seed, K):
    """(replay order, blocks) for one accumulation order. seed None = the ORIGINAL replay order (identity).
    Every sequence keeps the block of its ORIGINAL replay position."""
    chosen = [int(s) for s in chosen]
    n = len(chosen)
    blk = dict(zip(chosen, positional_blocks(n, K)))
    perm = list(range(n)) if seed is None else index_permutation(seed, n)
    order = [chosen[j] for j in perm]
    return order, [blk[s] for s in order], perm


def within_block_orders(order, blocks, K):
    return {k: [s for s, b in zip(order, blocks) if b == k] for k in range(K)}


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def selection_sha(chosen):
    return hashlib.sha256(json.dumps([int(s) for s in chosen]).encode()).hexdigest()


# ---------------------------------------------------------------- ordered Hessian Profiler
class OrderedForcedProfiler(ForcedProfiler):
    """ForcedProfiler (frozen ORIGINAL routing; expect_natural=True asserts it equals the natural top-k for every
    token) whose run() replays `order` with `blocks` instead of the caller's `chosen` with positional blocks.
    The caller (E3) must pass exactly the original `chosen`, positional blocks and unit weights.
    record: optional dict; if given, records per (key, block) the contributing sequence ids for every key and,
    for `fp_keys`, sha256 of each sequence's routed input X and of its contribution Xf.T @ Xf (recomputed with the
    accumulator's own expression; the accumulation itself uses the unchanged BlockHessianAccumulator.add)."""
    def __init__(self, model, layer, table, row_to_sid, expect_natural, chosen, order, blocks, record=None,
                 fp_keys=(), **kw):
        super().__init__(model, layer, table, row_to_sid, expect_natural, **kw)
        self.chosen, self.order, self.blocks = [int(s) for s in chosen], [int(s) for s in order], list(blocks)
        assert sorted(self.order) == sorted(self.chosen) and len(set(self.order)) == len(self.order)
        self.record, self.fp_keys = record, set(fp_keys)
        if record is not None and self.acc is not None:
            acc_cls = type(self.acc)        # unbound: no acc -> closure -> bound method -> acc cycle (GPU memory)

            def add(key, X, block, weight=1.0):
                if X.numel() > 0:
                    record.setdefault("members", {}).setdefault((key, block), []).append(int(self.cur_seq))
                    if key in self.fp_keys:
                        Xf = X.to(torch.float32)
                        record.setdefault("contrib", {})[(key, int(self.cur_seq))] = (tensor_sha(X), tensor_sha(Xf.T @ Xf))
                return acc_cls.add(self.acc, key, X, block, weight)
            self.acc.add = add

    def run(self, cache, seq_ids, blocks=None, weights=None, routing_only=False):
        assert [int(s) for s in seq_ids] == self.chosen, "caller did not pass the original replay order"
        assert blocks is None, "caller passed explicit blocks"
        assert weights is None or all(w == 1.0 for w in weights), "non-unit weights"
        return super().run(cache, self.order, blocks=self.blocks, weights=[1.0] * len(self.order),
                           routing_only=routing_only)


@contextmanager
def ordered_forced_e3(E3, tables, row_to_sid, chosen, order, blocks, on_hessian=None):
    """As frozen_routing.forced_e3 (expect_natural=True), with the Hessian-building Profiler replaced by an
    OrderedForcedProfiler; the held-out reconstruction Profiler stays the plain original."""
    orig_prof, orig_q = E3.Profiler, E3.quantize_from_acc_or_fallback
    stats = {}

    def factory(model, layer, K=4, collect_hessian=True, collect_holdout=False, **kw):
        if collect_hessian and not collect_holdout:
            fp = OrderedForcedProfiler(model, layer, tables[layer], row_to_sid, True, chosen, order, blocks, K=K,
                                       collect_hessian=True, collect_holdout=False, **kw)
            stats[layer] = fp
            return fp
        return orig_prof(model, layer, K=K, collect_hessian=collect_hessian, collect_holdout=collect_holdout, **kw)

    def qwrap(model, acc, L, e, *a, **k):
        if on_hessian is not None:
            on_hessian(L, e, acc)
        return orig_q(model, acc, L, e, *a, **k)

    E3.Profiler, E3.quantize_from_acc_or_fallback = factory, qwrap
    try:
        yield stats
    finally:
        E3.Profiler, E3.quantize_from_acc_or_fallback = orig_prof, orig_q


# ---------------------------------------------------------------- FP64 accumulation control
class Fp64Accumulator:
    """Diagnostic accumulator with the Profiler's `acc` interface (add / tokens) for a subset of keys: each
    sequence contribution is computed with the accumulator's own FP32 expression (outer = Xf.T @ Xf) and
    accumulated per block both in FP32 (H32, the original arithmetic) and in FP64 (H64)."""
    def __init__(self, keys, dims, K, device):
        self.K, self.keys = K, set(keys)
        self.H32 = {k: torch.zeros(K, dims[k[2]], dims[k[2]], dtype=torch.float32, device=device) for k in keys}
        self.H64 = {k: torch.zeros(K, dims[k[2]], dims[k[2]], dtype=torch.float64, device=device) for k in keys}
        self.raw_n = {k: torch.zeros(K, dtype=torch.float64) for k in keys}

    @torch.no_grad()
    def add(self, key, X, block, weight=1.0):
        if key not in self.keys or X.numel() == 0:
            return
        assert weight == 1.0
        Xf = X.to(torch.float32)
        outer = Xf.T @ Xf
        self.H32[key][block] += outer
        self.H64[key][block] += outer.to(torch.float64)
        self.raw_n[key][block] += X.shape[0]

    def sums(self, key):
        return self.H32[key].sum(dim=0), self.H64[key].sum(dim=0)


# ---------------------------------------------------------------- Hessian difference metrics
def ordered_int(a):
    """Monotone integer image of FP32 values (adjacent floats differ by 1): ULP distance = |ordered_int diff|."""
    i = a.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(i < 0, -(i & 0x7FFFFFFF), i)


@torch.no_grad()
def hdiff(H, H0):
    """Elementwise comparison of an order-variant FP32 Hessian H with the baseline H0 (same shape)."""
    D = H.double() - H0.double()
    A = H0.double()
    neq = H != H0
    n = H0.numel()
    nd = int(neq.sum())
    dg = torch.diagonal(D)
    off = D.clone()
    off.fill_diagonal_(0)
    tr0 = float(torch.trace(A))
    ulp = (ordered_int(H.float()) - ordered_int(H0.float())).abs()
    return dict(n=n, n_diff=nd, frac_diff=nd / n, bitwise_identical=nd == 0,
                max_abs=float(D.abs().max()), mean_abs=float(D.abs().mean()),
                rel_fro=float(D.norm() / A.norm()) if float(A.norm()) > 0 else 0.0,
                rel_diag=float(dg.norm() / torch.diagonal(A).norm()) if float(torch.diagonal(A).norm()) > 0 else 0.0,
                trace_diff=float(torch.trace(D)), trace_rel_diff=float(torch.trace(D)) / tr0 if tr0 else 0.0,
                max_offdiag_abs=float(off.abs().max()), max_ulp=int(ulp.max()),
                median_ulp_changed=float(ulp[neq].double().median()) if nd else 0.0,
                nonfinite=int((~torch.isfinite(H)).sum()))


def solver_preprocess(H, perm, percdamp):
    """Exactly the solver's steps before the column loop (moe_calib/gptq.py): dead columns, permutation, damping,
    upper Cholesky of H^-1. Returns (damped permuted H, Hinv factor). Diagnostic copy; the solver is unchanged."""
    H = H.to(torch.float32).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    perm = torch.as_tensor(perm).long().to(H.device)
    H = H[perm][:, perm]
    H += torch.eye(H.shape[0], device=H.device) * (percdamp * torch.mean(torch.diag(H)))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    return H, Hinv


@torch.no_grad()
def factorization_sensitivity(H, H0, perm, percdamp):
    Hd, U = solver_preprocess(H, perm, percdamp)
    Hd0, U0 = solver_preprocess(H0, perm, percdamp)
    ev = torch.linalg.eigvalsh(Hd0.double())
    kappa = float(ev.max() / ev.min())
    rel_damped = float((Hd.double() - Hd0.double()).norm() / Hd0.double().norm())
    dU = torch.diagonal(U).double() / torch.diagonal(U0).double() - 1
    return dict(rel_damped_perturbation=rel_damped, cond_damped=kappa, first_order_bound=kappa * rel_damped,
                rel_hinv_factor=float((U.double() - U0.double()).norm() / U0.double().norm()),
                max_rel_hinv_diag=float(dU.abs().max()), damped_bitwise_identical=bool(torch.equal(Hd, Hd0)),
                hinv_bitwise_identical=bool(torch.equal(U, U0)))


def natural_perm(H):
    """argsort(diag H, descending) after dead-column handling (the solver's act-order if not overridden)."""
    d = torch.diag(H).clone().float()
    d[d == 0] = 1.0
    return torch.argsort(d, descending=True)


# ---------------------------------------------------------------- quantizer codes (read-only)
@torch.no_grad()
def codes_from(Wq, perm, scale, zero, group_size=128):
    """Integer codes of the solver output: Wq = scale_g * (q - zero_g) in permuted column order. Returns (codes in
    ORIGINAL column order as uint8, exact) where exact = the recomputed dequantization equals Wq bitwise."""
    Wq = Wq.detach().float().cpu()
    perm = torch.as_tensor(perm).long().cpu()
    cols = Wq.shape[1]
    idx = torch.arange(cols) // group_size
    S, Z = scale.float()[idx].T, zero.float()[idx].T
    Wp = Wq[:, perm]
    q = torch.round(Wp / S + Z)
    exact = bool(torch.equal(S * (q - Z), Wp))
    return q[:, torch.argsort(perm)].to(torch.uint8), exact


# ---------------------------------------------------------------- statistics
def sd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if x.size > 1 else float("nan")


def paired(fr, fc, fr_oo, fc_oo):
    fr, fc = np.asarray(fr, float), np.asarray(fc, float)
    d = fc - fr
    d_oo = fc_oo - fr_oo
    return dict(delta=d.tolist(), delta_oo=d_oo, sd_fr=sd(fr), sd_fc=sd(fc), sd_delta=sd(d), mean_fr=float(fr.mean()),
                mean_fc=float(fc.mean()), mean_delta=float(d.mean()), min_delta=float(d.min()), max_delta=float(d.max()),
                sign_reversals=int((np.sign(d) != np.sign(d_oo)).sum()),
                mean_abs_dev_fr=float(np.abs(fr - fr_oo).mean()), max_abs_dev_fr=float(np.abs(fr - fr_oo).max()),
                mean_abs_dev_fc=float(np.abs(fc - fc_oo).mean()), max_abs_dev_fc=float(np.abs(fc - fc_oo).max()))


# ---------------------------------------------------------------- frozen decision rule
DECISION_RULE = {
    "tau_PPL": "1e-4 x BF16 PPL = %.9f (operational materiality threshold, not a significance test)" % TAU_PPL,
    "precedence": [
        "RUN-BLOCKED: any correctness requirement fails - baseline (original-order) regression (Hessians 480/480 == "
        "frozen OO, natural act-order == P_O, weights / reconstruction / PPL / per-window NLL == frozen OO), routing "
        "== R_O for every token, identical per-(expert, block) contributing sequences and token counts across orders, "
        "bitwise-identical per-sequence contributions of the diagnostic experts across orders, non-finite Hessians, "
        "any required quantized configuration missing or invalid (240 experts, 720 tensors, 0 RTN fallbacks, "
        "permutation == P_O 240/240, Hessians == preflight 480/480, integrity I1-I4, repeat evaluation identical), "
        "or the deterministic repeatability control differs",
        "NULL-ORDER-INVARIANT: all 10 shuffled-order Hessian sets (480 Hessians each) are bitwise identical to the "
        "original-order Hessians (no GPTQ is run)",
        "ORDER-SENSITIVE-HESSIAN-ONLY: some Hessian differs, but the installed effective weights of all 10 "
        "configurations are bitwise identical to OO",
        "ORDER-SENSITIVE-WEIGHTS-ONLY: effective weights differ in at least one configuration, and "
        "max over the 10 configurations of |PPL - PPL_OO| <= tau_PPL",
        "ORDER-SENSITIVE-PPL: effective weights differ and at least one |PPL - PPL_OO| > tau_PPL",
    ],
    "reported_not_gating": "strategies affected, paired Delta change (|Delta_g - Delta_OO| > tau_PPL), FR/FC sign "
                           "reversals, SDs vs the earlier AA / AO / AOF spreads (context only, never pooled)",
    "repeatability_config": "the first configuration in the prespecified order (FR then FC; seeds ascending) whose "
                            "Hessians differ from the original-order Hessians is quantized twice in fresh processes",
    "frozen": "fixed before any shuffled-order GPTQ / PPL; never adjusted after outcomes",
}


def decide(blocked, any_hessian_differs, weights_all_equal_oo, ppl_devs, tau=TAU_PPL):
    """ppl_devs: |PPL - PPL_OO| of the 10 configurations."""
    if blocked:
        return "RUN-BLOCKED"
    if not any_hessian_differs:
        return "NULL-ORDER-INVARIANT"
    if weights_all_equal_oo:
        return "ORDER-SENSITIVE-HESSIAN-ONLY"
    if max(ppl_devs) <= tau:
        return "ORDER-SENSITIVE-WEIGHTS-ONLY"
    return "ORDER-SENSITIVE-PPL"


def repeat_config(hess_differs):
    """hess_differs: {(strategy, seed): bool}. Prespecified repeatability configuration."""
    for c in ("FR", "FC"):
        for s in ORDER_SEEDS:
            if hess_differs.get((c, s)):
                return c, s
    return None
