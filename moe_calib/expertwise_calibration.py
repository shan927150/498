"""Expert-wise activation calibration MVP: covariance-sketch selection, token matching, aligned capture.

Question: does letting each MoE expert pick its own calibration sequences (instead of all experts sharing one
sequence set) improve isolated-expert GPTQ reconstruction once per-expert routed-token counts are matched?

Four conditions (B sequences each):
  SR shared random              one random set S, every expert uses its natural routed tokens from S
  SC shared covariance-aware    one S minimizing mean_e D_e(S)
  EC expert-wise covariance     per expert S_e minimizing D_e(S_e)
  ER expert-wise random         per expert an independent random S_e

Selection surrogate (sketch space, fixed Gaussian projections R_p in R^{d_p x k}, z = R_p^T x):
  H_{e,p}(S) = sum_{s in S} M_{s,e,p} / sum_{s in S} n_{s,e},       M_{s,e,p} = sum_{t routed to e in s} z_t z_t^T
  D_e(S)     = 1/2 sum_p ||H_{e,p}(S) - H_{e,p}^pool||_F / (||H_{e,p}^pool||_F + eps)
Empty-token rule: if sum_{s in S} n_{s,e} == 0, H_{e,p}(S) := 0, i.e. that term equals
||H^pool|| / (||H^pool|| + eps) (~1, the mismatch of "no information").

Storage: a symmetric k x k matrix is stored as its upper triangle with off-diagonal entries scaled by sqrt(2)
("svech"), so the Euclidean norm of the vector equals the Frobenius norm of the matrix and all sums stay linear.

Everything here except `ExpertCapture` is pure CPU tensor/numpy code (unit-tested on synthetic data). The GPTQ
solver, the Profiler, routing and the evaluation metric are NOT touched by this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

PROJS = ("gu", "down")
CONDITIONS = ("SR", "SC", "EC", "ER")
COND_CODE = {c: i for i, c in enumerate(CONDITIONS)}
EPS = 1e-12


# ===================================================================== expert choice
def select_experts(counts, n: int = 8) -> list[int]:
    """Deterministic frequency-stratified choice: sort experts by pool routed-token count ascending (ties -> lower
    id), take ranks round(i * (E - 1) / (n - 1)), i = 0..n-1 (lowest, ..., median, ..., highest)."""
    counts = np.asarray(counts)
    E = counts.shape[0]
    if not 2 <= n <= E:
        raise ValueError(f"need 2 <= n <= E, got n={n}, E={E}")
    order = np.lexsort((np.arange(E), counts))            # primary: count, secondary: id
    ranks = [int(round(i * (E - 1) / (n - 1))) for i in range(n)]
    if len(set(ranks)) != n:
        raise ValueError("rank collision")
    return [int(order[r]) for r in ranks]


# ===================================================================== sketch algebra
def projection_matrices(dims: dict, k: int = 64, seed: int = 20261116) -> dict:
    """{proj: [d_p, k] fp32}, entries N(0, 1/k), generated on CPU from a fixed seed (identical on every run)."""
    g = torch.Generator().manual_seed(seed)
    return {p: (torch.randn(dims[p], k, generator=g, dtype=torch.float64) / math.sqrt(k)).float() for p in PROJS}


def svech_index(k: int, device="cpu"):
    iu = torch.triu_indices(k, k, device=device)
    w = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).to(device=device, dtype=torch.float64)
    return iu, w


def svech(S: torch.Tensor, iu=None, w=None) -> torch.Tensor:
    """[..., k, k] symmetric -> [..., k(k+1)/2] with ||svech(S)||_2 == ||S||_F."""
    k = S.shape[-1]
    if iu is None:
        iu, w = svech_index(k, S.device)
    return S[..., iu[0], iu[1]].to(torch.float64) * w.to(S.device)


def sketch_outer(X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """sum_t z_t z_t^T for the rows of X, z = R^T x (projection in fp32, outer products in fp64)."""
    z = X.to(R.device, torch.float32) @ R
    z = z.double()
    return z.T @ z


def rel_frob(H: torch.Tensor, n: float, Href: torch.Tensor, nref: float, eps: float = EPS) -> float:
    """||H/n - Href/nref||_F / (||Href/nref||_F + eps) for raw (un-normalized) sums; n == 0 -> H/n := 0."""
    ref = Href.double() / nref
    cur = torch.zeros_like(ref) if n == 0 else H.double() / n
    return float(torch.linalg.norm(cur - ref) / (torch.linalg.norm(ref) + eps))


def mismatch_from_sums(A: dict, n: float, ref: dict, eps: float = EPS) -> float:
    """D_e from summed svech sketches A[p] (raw), token count n and normalized references ref[p]."""
    out = 0.0
    for p in PROJS:
        R = ref[p]
        cur = torch.zeros_like(R) if n == 0 else A[p] / n
        out += 0.5 * float(torch.linalg.norm(cur - R) / (torch.linalg.norm(R) + eps))
    return out


# ===================================================================== objective
@dataclass
class Term:
    M: torch.Tensor          # [N, V] fp64 per-sequence svech sketch sums
    n: torch.Tensor          # [N] fp64 per-sequence routed-token counts
    weight: float
    R: torch.Tensor = field(init=False)    # [V] normalized pool reference
    Rn: float = field(init=False)
    MM: torch.Tensor = field(init=False)   # [N] ||M_c||^2
    MR: torch.Tensor = field(init=False)   # [N] <M_c, R>

    def __post_init__(self):
        tot = float(self.n.sum())
        self.R = self.M.sum(0) / tot if tot > 0 else torch.zeros(self.M.shape[1], dtype=torch.float64)
        self.Rn = float(torch.linalg.norm(self.R))
        self.MM = (self.M * self.M).sum(1)
        self.MR = self.M @ self.R


class Objective:
    """J(S) = sum_t w_t ||A_t(S)/n_t(S) - R_t|| / (||R_t|| + eps), with A_t(S) = sum_{s in S} M_t[s]."""

    def __init__(self, terms: list[Term], eps: float = EPS):
        if not terms:
            raise ValueError("no terms")
        self.terms, self.eps = terms, eps
        self.N = terms[0].M.shape[0]
        if any(t.M.shape[0] != self.N or t.n.shape[0] != self.N for t in terms):
            raise ValueError("terms disagree on the number of sequences")

    # ---- exact evaluation (used for acceptance and reporting)
    def exact(self, sel) -> float:
        idx = torch.as_tensor(list(sel), dtype=torch.long)
        v = 0.0
        for t in self.terms:
            n = float(t.n[idx].sum()) if idx.numel() else 0.0
            cur = torch.zeros_like(t.R) if n == 0 else t.M[idx].sum(0) / n
            v += t.weight * float(torch.linalg.norm(cur - t.R)) / (t.Rn + self.eps)
        return v

    def per_term(self, sel) -> list[float]:
        idx = torch.as_tensor(list(sel), dtype=torch.long)
        out = []
        for t in self.terms:
            n = float(t.n[idx].sum()) if idx.numel() else 0.0
            cur = torch.zeros_like(t.R) if n == 0 else t.M[idx].sum(0) / n
            out.append(float(torch.linalg.norm(cur - t.R)) / (t.Rn + self.eps))
        return out

    # ---- incremental state: A_t, n_t, MA_t = M_t @ A_t
    def state(self, sel) -> list[dict]:
        idx = torch.as_tensor(list(sel), dtype=torch.long)
        st = []
        for t in self.terms:
            A = t.M[idx].sum(0) if idx.numel() else torch.zeros_like(t.R)
            st.append(dict(A=A, n=float(t.n[idx].sum()) if idx.numel() else 0.0, MA=t.M @ A))
        return st

    def _scores(self, st, A_shift=None, n_shift=None, MA_shift=None) -> torch.Tensor:
        """Objective of (S [- i]) + c for EVERY candidate c (closed form, no materialization).
        *_shift: per-term corrections for the removal of one member i (A - M_i, n - n_i, MA - M M_i)."""
        tot = torch.zeros(self.N, dtype=torch.float64)
        for j, (t, s) in enumerate(zip(self.terms, st)):
            A, n, MA = s["A"], s["n"], s["MA"]
            if A_shift is not None:
                A, n, MA = A - A_shift[j], n - n_shift[j], MA - MA_shift[j]
            AA, AR = float(A @ A), float(A @ t.R)
            den = n + t.n
            num2 = AA + 2.0 * MA + t.MM
            numR = AR + t.MR
            safe = torch.where(den > 0, den, torch.ones_like(den))
            q = num2 / safe ** 2 - 2.0 * numR / safe + t.Rn ** 2
            val = torch.sqrt(q.clamp_min(0.0))
            val = torch.where(den > 0, val, torch.full_like(val, t.Rn))
            tot += t.weight * val / (t.Rn + self.eps)
        return tot

    def add_scores(self, st) -> torch.Tensor:
        return self._scores(st)

    def swap_scores(self, st, i: int) -> torch.Tensor:
        A_s = [t.M[i] for t in self.terms]
        n_s = [float(t.n[i]) for t in self.terms]
        MA_s = [t.M @ t.M[i] for t in self.terms]
        return self._scores(st, A_s, n_s, MA_s)

    def apply_add(self, st, c: int):
        for t, s in zip(self.terms, st):
            s["A"] = s["A"] + t.M[c]
            s["n"] += float(t.n[c])
            s["MA"] = s["MA"] + t.M @ t.M[c]

    def apply_remove(self, st, i: int):
        for t, s in zip(self.terms, st):
            s["A"] = s["A"] - t.M[i]
            s["n"] -= float(t.n[i])
            s["MA"] = s["MA"] - t.M @ t.M[i]


# ===================================================================== selection
def _rng(seed: int, *tags: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, tags)]))


def random_select(N: int, B: int, rng: np.random.Generator) -> list[int]:
    if B > N:
        raise ValueError("B > N")
    return sorted(int(x) for x in rng.choice(N, size=B, replace=False))


def greedy_select(obj: Objective, B: int, rng: np.random.Generator, sample_frac: float = 0.1) -> list[int]:
    """Seeded stochastic greedy: at each step evaluate a uniformly sampled ceil(sample_frac * N) subset of the
    remaining candidates (all of them if sample_frac >= 1) and add the one with the lowest objective
    (ties -> lowest index)."""
    N = obj.N
    sel, in_sel = [], torch.zeros(N, dtype=torch.bool)
    st = obj.state([])
    m = N if sample_frac >= 1 else max(1, int(math.ceil(sample_frac * N)))
    for _ in range(B):
        rem = np.flatnonzero(~in_sel.numpy())
        cand = rem if m >= rem.size else np.sort(rng.choice(rem, size=m, replace=False))
        sc = obj.add_scores(st)[torch.as_tensor(cand)]
        c = int(cand[int(torch.argmin(sc))])
        sel.append(c)
        in_sel[c] = True
        obj.apply_add(st, c)
    return sel


def swap_refine(obj: Objective, sel: list[int], rng: np.random.Generator, max_passes: int = 5,
                rel_tol: float = 1e-9) -> tuple[list[int], list[float], dict]:
    """Best-improvement swap refinement. Each pass visits the B slots in a seeded random order; for slot i it
    scores every out-of-set candidate c for S - i + c in closed form and proposes the best one. The swap is
    ACCEPTED only if the exact objective strictly decreases by more than rel_tol (relative) — so the accepted
    trajectory is monotone non-increasing by construction (checked). Stops after a pass with no accepted swap."""
    sel = list(sel)
    in_sel = torch.zeros(obj.N, dtype=torch.bool)
    in_sel[sel] = True
    st = obj.state(sel)
    cur = obj.exact(sel)
    trace = [cur]
    n_acc = n_prop = passes = 0
    for _ in range(max_passes):
        passes += 1
        improved = False
        for slot in rng.permutation(len(sel)):
            i = sel[slot]
            sc = obj.swap_scores(st, i)
            sc[in_sel] = float("inf")
            c = int(torch.argmin(sc))
            if not math.isfinite(float(sc[c])) or float(sc[c]) >= cur * (1 - rel_tol):
                continue
            n_prop += 1
            trial = sel.copy()
            trial[slot] = c
            new = obj.exact(trial)
            if new < cur * (1 - rel_tol):
                obj.apply_remove(st, i)
                obj.apply_add(st, c)
                in_sel[i], in_sel[c] = False, True
                sel, cur = trial, new
                trace.append(cur)
                n_acc += 1
                improved = True
        if not improved:
            break
    if any(b > a for a, b in zip(trace, trace[1:])):
        raise AssertionError("objective increased during accepted swaps")
    return sorted(sel), trace, dict(passes=passes, accepted=n_acc, proposed=n_prop)


def covariance_select(obj: Objective, B: int, rng: np.random.Generator, sample_frac: float = 0.1,
                      max_passes: int = 5) -> tuple[list[int], dict]:
    g = greedy_select(obj, B, rng, sample_frac)
    j_greedy = obj.exact(g)
    sel, trace, info = swap_refine(obj, g, rng, max_passes)
    info.update(greedy_objective=j_greedy, final_objective=trace[-1], trace=trace)
    return sel, info


def build_terms(sketch_M: dict, sketch_n: dict, experts: list[int], shared: bool) -> Objective | dict:
    """sketch_M[(e, p)]: [N, V] fp64; sketch_n[e]: [N] (gu == down counts). Shared -> one objective with weight
    1/(2|E|) per term (= mean_e D_e); expert-wise -> {e: objective with weight 1/2}."""
    def terms(es, w):
        return [Term(sketch_M[(e, p)], sketch_n[e].double(), w) for e in es for p in PROJS]
    if shared:
        return Objective(terms(experts, 1.0 / (2 * len(experts))))
    return {e: Objective(terms([e], 0.5)) for e in experts}


def select_all(sketch_M, sketch_n, experts, B, seed, sample_frac=0.1, max_passes=5) -> dict:
    """All four conditions for one allocation seed. Returns {cond: {e: sorted ids}} plus optimizer info.
    RNG streams: SR (seed, 0), SC (seed, 1), EC (seed, 2, e), ER (seed, 3, e) — independent of each other."""
    N = next(iter(sketch_n.values())).shape[0]
    out, info = {}, {}
    sr = random_select(N, B, _rng(seed, COND_CODE["SR"]))
    out["SR"] = {e: sr for e in experts}
    sc_obj = build_terms(sketch_M, sketch_n, experts, shared=True)
    sc, info["SC"] = covariance_select(sc_obj, B, _rng(seed, COND_CODE["SC"]), sample_frac, max_passes)
    out["SC"] = {e: sc for e in experts}
    ec_obj = build_terms(sketch_M, sketch_n, experts, shared=False)
    out["EC"], info["EC"] = {}, {}
    for e in experts:
        out["EC"][e], info["EC"][e] = covariance_select(ec_obj[e], B, _rng(seed, COND_CODE["EC"], e),
                                                        sample_frac, max_passes)
    out["ER"] = {e: random_select(N, B, _rng(seed, COND_CODE["ER"], e)) for e in experts}
    return dict(selection=out, optimizer=info)


def expert_mismatch(sketch_M, sketch_n, e, sel) -> float:
    """D_e(sel) in sketch space (exact)."""
    return build_terms(sketch_M, sketch_n, [e], shared=False)[e].exact(sel)


# ===================================================================== token matching
def match_tokens(seq_counts: dict, k: int, rng: np.random.Generator) -> dict:
    """seq_counts {seq: n routed rows}. Uniformly choose exactly k of the sum(n) (seq, row) pairs without
    replacement; returns {seq: sorted LongTensor of row indices} for sequences with >= 1 chosen row. Row indices
    address the expert's dispatch rows of that sequence, which are IDENTICAL for gate/up and down inputs."""
    seqs = sorted(seq_counts)
    n = np.array([int(seq_counts[s]) for s in seqs], dtype=np.int64)
    tot = int(n.sum())
    if not 0 <= k <= tot:
        raise ValueError(f"cannot match k={k} from {tot} tokens")
    pick = np.sort(rng.choice(tot, size=k, replace=False)) if k else np.zeros(0, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(n)])
    which = np.searchsorted(off, pick, side="right") - 1
    out = {}
    for j in np.unique(which):
        rows = pick[which == j] - off[j]
        out[seqs[j]] = torch.as_tensor(rows, dtype=torch.long)
    return out


def matching_plan(selection: dict, seq_expert_counts, experts, seed) -> dict:
    """selection {cond: {e: ids}}; seq_expert_counts(e) -> [N] counts. k_e = min_c n_e^c.
    Returns {cond: {e: {seq: rows}}}, counts {cond: {e: n}}, quota {e: k_e}. RNG (seed, 7919, cond, e)."""
    counts = {c: {e: int(sum(int(seq_expert_counts(e)[s]) for s in selection[c][e])) for e in experts}
              for c in CONDITIONS}
    quota = {e: min(counts[c][e] for c in CONDITIONS) for e in experts}
    plan = {c: {e: match_tokens({s: int(seq_expert_counts(e)[s]) for s in selection[c][e]}, quota[e],
                                _rng(seed, 7919, COND_CODE[c], e)) for e in experts} for c in CONDITIONS}
    return dict(plan=plan, counts=counts, quota=quota)


# ===================================================================== diversity / overlap
def source_diversity(rows_per_seq: dict) -> dict:
    m = np.array([len(v) if hasattr(v, "__len__") else int(v) for v in rows_per_seq.values()], dtype=float)
    m = m[m > 0]
    if m.size == 0:
        return dict(distinct=0, inv_simpson=0.0, exp_entropy=0.0, max_share=0.0)
    p = m / m.sum()
    return dict(distinct=int(m.size), inv_simpson=float(1.0 / (p ** 2).sum()),
                exp_entropy=float(np.exp(-(p * np.log(p)).sum())), max_share=float(p.max()))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def overlap_stats(selection: dict, experts) -> dict:
    out = {}
    for c in CONDITIONS:
        sets = [set(selection[c][e]) for e in experts]
        union = set().union(*sets)
        pair = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
        out[c] = dict(union=len(union), mean_pairwise_jaccard=float(np.mean(pair)) if pair else 1.0,
                      mean_jaccard_vs_SC=float(np.mean([jaccard(selection[c][e], selection["SC"][e])
                                                        for e in experts])),
                      mean_jaccard_vs_SR=float(np.mean([jaccard(selection[c][e], selection["SR"][e])
                                                        for e in experts])))
    return out


def validate_selection(selection: dict, experts, B: int, N: int):
    """Hard checks: exactly the predetermined experts, B distinct in-pool ids per set, shared conditions share."""
    for c in CONDITIONS:
        if sorted(selection[c]) != sorted(experts):
            raise AssertionError(f"{c}: expert set mismatch")
        for e in experts:
            ids = selection[c][e]
            if len(ids) != B or len(set(ids)) != B:
                raise AssertionError(f"{c}/{e}: {len(ids)} ids, {len(set(ids))} distinct (need {B})")
            if min(ids) < 0 or max(ids) >= N:
                raise AssertionError(f"{c}/{e}: id outside candidate pool")
    for c in ("SR", "SC"):
        if len({tuple(selection[c][e]) for e in experts}) != 1:
            raise AssertionError(f"{c}: shared condition has differing sets")


# ===================================================================== capture (model hooks)
class ExpertCapture:
    """Hooks on one Qwen2-MoE layer: the router (natural top-k, the model's own softmax/topk), and gate_proj / down_proj
    pre-hooks of the TARGET experts. For every replayed sequence it delivers each target expert's dispatch rows to
    `callback(seq, e, X_gu, X_down)` after checking
      * gu rows == router input at the router-selected token positions, in the model's slot-major dispatch order
        (bitwise) -> natural routing;
      * down rows == act(gate_proj(X_gu)) * up_proj(X_gu) (bitwise) -> gate/up and down rows are the SAME tokens.
    Read-only: never modifies activations or weights."""

    def __init__(self, block, experts: list[int], top_k: int, callback):
        self.blk, self.experts, self.k, self.cb = block, list(experts), top_k, callback
        self.E = len(block.experts)
        self.cur_seq = -1
        self._rows, self._router_in, self._gu = {}, None, {}
        self._hooks = []
        self.n_checks = 0

    def _gate_hook(self, m, inputs, out):
        probs = torch.softmax(out, dim=1, dtype=torch.float)
        sel = torch.topk(probs, self.k, dim=-1).indices
        mask = F.one_hot(sel, num_classes=self.E).permute(2, 1, 0)
        self._router_in = inputs[0].reshape(-1, inputs[0].shape[-1])
        self._rows = {e: torch.where(mask[e])[1] for e in self.experts}

    def _gu_hook(self, e):
        def fn(m, args):
            X = args[0].reshape(-1, args[0].shape[-1])
            top_x = self._rows[e]
            if X.shape[0] != top_x.numel() or not torch.equal(X, self._router_in[top_x].to(X.dtype)):
                raise RuntimeError(f"expert {e}: gate_proj rows != router-selected inputs (seq {self.cur_seq})")
            self._gu[e] = X
        return fn

    def _down_hook(self, e):
        ex = self.blk.experts[e]

        def fn(m, args):
            Xd = args[0].reshape(-1, args[0].shape[-1])
            Xg = self._gu.pop(e)
            ref = ex.act_fn(F.linear(Xg, ex.gate_proj.weight, ex.gate_proj.bias)) * \
                F.linear(Xg, ex.up_proj.weight, ex.up_proj.bias)
            if Xd.shape[0] != Xg.shape[0] or not torch.equal(Xd, ref.to(Xd.dtype)):
                raise RuntimeError(f"expert {e}: down_proj rows not aligned with gate/up rows (seq {self.cur_seq})")
            self.n_checks += 1
            self.cb(self.cur_seq, e, Xg, Xd)
        return fn

    def attach(self):
        self._hooks.append(self.blk.gate.register_forward_hook(self._gate_hook))
        for e in self.experts:
            ex = self.blk.experts[e]
            self._hooks.append(ex.gate_proj.register_forward_pre_hook(self._gu_hook(e)))
            self._hooks.append(ex.down_proj.register_forward_pre_hook(self._down_hook(e)))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def finish_sequence(self):
        """Experts with zero routed rows in a sequence still get the (empty) model call; any expert whose rows
        were never delivered must have had zero routed tokens."""
        for e in self.experts:
            if e in self._gu:
                raise RuntimeError(f"expert {e}: gate/up rows without matching down rows")
        self._rows = {}


@torch.no_grad()
def replay(model, L: int, cache, seq_ids, experts, callback, run_layer=None, device=None) -> dict:
    """Replay layer L (must already be on `device`) over cached inputs, one sequence at a time, delivering target
    expert rows to callback(seq, e, X_gu, X_down)."""
    from . import model_utils as MU
    run_layer = run_layer or MU.run_layer
    device = device or MU.GPU
    cap = ExpertCapture(MU.moe_block(model, L), experts, MU.top_k(model), callback).attach()
    try:
        for sid in seq_ids:
            cap.cur_seq = int(sid)
            run_layer(model, L, cache[sid:sid + 1].to(device))
            cap.finish_sequence()
    finally:
        cap.detach()
    return dict(n_alignment_checks=cap.n_checks)
