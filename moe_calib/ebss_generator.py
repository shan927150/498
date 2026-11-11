"""Model-matched Expert-Balanced Self-Sampling (EBSS, MoEQuant ICML 2025, Sec. 4.2 / Algorithm 1) and a
matched Probability-Only control.

Paper semantics implemented (sources in runs_ebss_generation_pilot/source_review.md):
  * Eq. 13  score(S||v) = -(R_S + log P(v|S)) / (l + 1) + sigma(M, S) / tau      (l = |S|, LOWER is better)
  * Eq. 11  R_S = sum_t log P(S_t | S_<t)  (cumulative log-probability of the branch)
  * Eq. 14  keep the w best of the w x |V| expansions ("argtopk"); with Eq. 9-10 (argmin objective) and the
            leading minus of Eq. 13, "best" = smallest score  -> resolved as MINIMISATION.
  * Eq. 3/4 sigma = (1/L) sum_l sigma_l,  sigma_l = sqrt( 1/(E-1) sum_e (u_l^e - mean_l)^2 ), all L MoE layers.
  * "deferred expert balance": the balance term of a candidate S||v uses the PREFIX S only (the routing of v
    is unknown until v is fed), so it ranks branches with different prefixes and never tokens within a branch.
  * Algorithm 1: first token Random(V); Seqlen - 1 expansion steps; final sentence = best Score(c)
    = -R/|c| + sigma(c)/tau over the w surviving branches (minimum, see above).
Implementation choices for undocumented details are labelled in GENERATOR_CHOICES.

Probability-Only = identical procedure with the balance term removed (use_balance=False): same candidate
universe, same length normalisation, same scheduling, same initial tokens.

The beam logic is backend-agnostic (`Backend.start / step / reorder`) so it can be unit-tested with a
synthetic backend; `HFOffloadBackend` drives the BF16 Qwen1.5-MoE-A2.7B checkpoint with all non-expert
weights resident on the GPU and each layer's 60 routed experts streamed from pinned host memory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

W_REF = 4            # search width w (paper Sec. 5.2 / Table 7; rebuttal R3: 49.59 at w=4)
TAU_REF = 1.3        # expert-balance temperature tau (task protocol; rebuttal R2 Qwen: 1.2 -> 49.60, 1.3 -> 49.61)
SEQLEN = 512
INIT_SEED = 20261110

GENERATOR_CHOICES = {
    "usage_frequency": "u_l^e = (# tokens of the branch routed to expert e at layer l, top-k membership, each token "
                       "once per selected expert) / (# tokens of the branch). Paper: 'usage frequency', normalisation "
                       "undocumented. Per-token rate chosen because it reproduces the scale of the authors' reported "
                       "Expert Balance STD (rebuttal: wikitext2 0.0427 / EBSS 0.0052 on 64-expert DeepSeek).",
    "routed_counts_not_affinities": "counts of selected (top-k) experts; routing probabilities are not used",
    "shared_expert": "excluded (always active; not part of the 60 routed experts)",
    "layers_in_sigma": "all 24 MoE layers (Eq. 3: L = number of MoE layers); layers 1/8/16/23 reported separately",
    "balance_scope": "per branch (prefix S of the candidate); no state is shared between sentences",
    "first_token": "uniform Random(V) over the non-special tokenizer ids 0..151642 with numpy default_rng(INIT_SEED); "
                   "sampled once and shared by both methods",
    "first_token_logprob": "the unconditioned first token contributes 0 to R; it counts toward |S| and the 512 "
                           "tokens and its routing counts toward sigma",
    "vocabulary_V": "tokenizer ids 0..151645 (all real tokens incl. the 3 specials); embedding padding rows "
                    "151646..151935 (never trained) are excluded",
    "eos": "no stop rule (Algorithm 1 iterates exactly Seqlen-1 steps over all of V); an EOS/special token is "
           "an ordinary candidate and generation continues; every one of the 512 tokens is a real model choice",
    "length": "Seqlen = 512 counts the first token: 1 random + 511 generated; no padding",
    "final_selection": "argmin Score(c) over the w surviving branches (Alg. 1 line 20 says argmax; resolved to the "
                       "minimisation objective of Eq. 9-10/13)",
    "ties": "candidates ordered by (score, parent beam index, token id): stable sort of the flattened [w, V] "
            "score array; final selection ties -> lowest beam index",
    "sentence_batching": "all sentences decoded in lock-step in one batch (rebuttal R3/R6: sequences stitched "
                         "along the batch dimension); KV cache reordered by parent index after every pruning",
    "precision": "BF16 weights/activations, eager attention; log_softmax in fp32; R and scores in fp64",
}


# ---------------------------------------------------------------- pure scoring functions
BALANCE_STD_MODES = ("eq4", "algorithm1")
USAGE_MODES = ("per_token", "assignment_share")
BALANCE_DEFAULT = dict(balance_std_mode="eq4", usage_mode="per_token")     # historical EQ4_RATE
BALANCE_VARIANTS = {                                                         # balance-strength fidelity study
    "EQ4_RATE": dict(balance_std_mode="eq4", usage_mode="per_token"),
    "EQ4_SHARE": dict(balance_std_mode="eq4", usage_mode="assignment_share"),
    "ALG1_SHARE": dict(balance_std_mode="algorithm1", usage_mode="assignment_share"),
    "ALG1_RATE": dict(balance_std_mode="algorithm1", usage_mode="per_token"),
}


def balance_sigma(counts: torch.Tensor, n_tokens: int | torch.Tensor, balance_std_mode: str = "eq4",
                  usage_mode: str = "per_token", top_k: int | None = None) -> torch.Tensor:
    """counts [..., L, E] routed-token counts of a branch -> sigma [...] (mean over the L MoE layers).

    usage_mode   per_token        u_e = c_e / n            (historical default)
                 assignment_share u_e = c_e / (k n)        (share of the k*n top-k assignments; needs top_k)
    balance_std  eq4              sigma_l = sqrt( 1/(E-1) sum_e (u_e - mean u)^2 )   (paper Eq. 4)
                 algorithm1       sigma_l = sqrt( E/(E-1) sum_e (u_e - mean u)^2 )   (Algorithm 1 line 30)
    E is read from the last axis of `counts` (never hard-coded). The default (eq4, per_token) executes exactly the
    historical operations, so it is bitwise identical to the pre-study implementation."""
    if balance_std_mode not in BALANCE_STD_MODES or usage_mode not in USAGE_MODES:
        raise ValueError(f"unknown balance normalisation {balance_std_mode!r} / {usage_mode!r}")
    n = torch.as_tensor(n_tokens, dtype=torch.float64, device=counts.device)
    while n.dim() < counts.dim():
        n = n.unsqueeze(-1)
    u = counts.to(torch.float64) / n
    if usage_mode == "assignment_share":
        if not top_k:
            raise ValueError("assignment_share usage needs top_k")
        u = u / top_k
    if balance_std_mode == "eq4":
        return u.std(dim=-1, unbiased=True).mean(dim=-1)
    E = u.shape[-1]
    d = u - u.mean(dim=-1, keepdim=True)
    return torch.sqrt(E / (E - 1) * (d * d).sum(dim=-1)).mean(dim=-1)


def candidate_scores(cum_logp: torch.Tensor, logp: torch.Tensor, sigma: torch.Tensor, prefix_len: int,
                     tau: float, use_balance: bool) -> torch.Tensor:
    """Eq. 13 for every expansion. cum_logp [B], logp [B, V], sigma [B]; prefix_len = l = |S|. Lower = better."""
    s = -(cum_logp[:, None] + logp.to(torch.float64)) / (prefix_len + 1)
    if use_balance:
        s = s + (sigma / tau)[:, None]
    return s


def select_best(scores: torch.Tensor, w: int):
    """scores [B, V] for ONE sentence -> (parent, token, score) of the w smallest, ties by (parent, token)."""
    flat = scores.reshape(-1)
    order = torch.sort(flat, stable=True).indices[:w]
    V = scores.shape[1]
    return order // V, order % V, flat[order]


def final_scores(cum_logp, sigma, length, tau, use_balance):
    """Alg. 1 Score(c) = -R/|c| + sigma/tau (lower is better)."""
    s = -cum_logp / length
    return s + sigma / tau if use_balance else s


def recount(hist_experts: torch.Tensor, E: int) -> torch.Tensor:
    """Reference recount from the stored per-token expert ids [T, L, k] -> counts [L, E]."""
    T, L, k = hist_experts.shape
    out = torch.zeros(L, E, dtype=torch.long)
    for l in range(L):
        out[l] = torch.bincount(hist_experts[:, l, :].reshape(-1).long().cpu(), minlength=E)
    return out


def sample_first_tokens(n: int, seed: int = INIT_SEED, n_regular: int = 151643) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, n_regular, size=n)]


# ---------------------------------------------------------------- backends
class Backend:
    """start(first_tokens [R]) -> (logits [R, V], sel [R, L, k]) for the first tokens;
    step(tokens [R]) -> same for the next token of every row; reorder(rows [R']) re-indexes the KV state."""
    vocab_limit: int
    n_layers: int
    n_experts: int
    top_k: int

    def start(self, tokens): raise NotImplementedError
    def step(self, tokens): raise NotImplementedError
    def reorder(self, rows): raise NotImplementedError

    def prefill(self, ids):
        """ids [N, P] (one shared-length prefix per sentence) -> (logits [N, V] at the last prefix position,
        sel [N, P, L, k] routing of every prefix token, prefix_logp [N, P-1] teacher-forced log P(x_t | x_<t),
        t = 1..P-1). Generic fallback: start + P-1 single-token steps (exact for any causal backend)."""
        ids = torch.as_tensor(ids, dtype=torch.long)
        N, Pl = ids.shape
        logits, sel = self.start(ids[:, 0].tolist())
        sels, lps = [sel], []
        for t in range(1, Pl):
            lp = torch.log_softmax(logits.float(), dim=-1)
            lps.append(lp[torch.arange(N), ids[:, t].to(lp.device)].double())
            logits, sel = self.step(ids[:, t].tolist())
            sels.append(sel)
        plp = torch.stack(lps, 1) if lps else torch.zeros(N, 0, dtype=torch.float64, device=logits.device)
        return logits, torch.stack(sels, 1), plp


class SyntheticBackend(Backend):
    """Deterministic CPU backend for tests: logits and routing are pure functions of the full prefix."""

    def __init__(self, V=50, L=3, E=8, k=2, seed=0, logit_scale=3.0):
        self.vocab_limit, self.n_layers, self.n_experts, self.top_k = V, L, E, k
        self.seed, self.scale = seed, logit_scale
        self.prefixes: list[tuple] = []
        self.calls = 0

    def _out(self):
        lg, sel = [], []
        for pfx in self.prefixes:
            g = torch.Generator().manual_seed(hash((self.seed,) + pfx) % (2 ** 31))
            lg.append(torch.randn(self.vocab_limit, generator=g) * self.scale)
            r = torch.randn(self.n_layers, self.n_experts, generator=g)
            sel.append(torch.topk(r, self.top_k, dim=-1).indices)
        self.calls += 1
        return torch.stack(lg), torch.stack(sel)

    def start(self, tokens):
        self.prefixes = [(int(t),) for t in tokens]
        return self._out()

    def step(self, tokens):
        self.prefixes = [p + (int(t),) for p, t in zip(self.prefixes, tokens)]
        return self._out()

    def reorder(self, rows):
        self.prefixes = [self.prefixes[int(r)] for r in rows]


class HFOffloadBackend(Backend):
    """BF16 Qwen1.5-MoE-A2.7B on a 20 GB MIG slice: everything except the routed experts lives on the GPU;
    before each MoE block runs, that layer's 60 x 3 expert matrices are copied from pinned host memory into
    one reusable GPU buffer (weights are bit-identical; only their location changes). Routing is read from a
    hook on each router (softmax(fp32) -> topk(k): the block's own op on the same tensor) and cross-checked
    against the experts' actual dispatch when `check_dispatch` is on."""

    def __init__(self, model, vocab_limit=151646, check_dispatch=False, device="cuda"):
        from transformers import DynamicCache
        self.DynamicCache = DynamicCache
        self.model, self.dev = model, device
        cfg = model.config
        self.vocab_limit, self.n_layers = vocab_limit, cfg.num_hidden_layers
        self.n_experts, self.top_k = cfg.num_experts, cfg.num_experts_per_tok
        self.timers = dict(forward_s=0.0, expert_transfer_s=0.0, routing_hook_s=0.0)
        self.n_forward, self.rows_forwarded = 0, 0
        self.bytes_transferred = 0
        self._sel = [None] * self.n_layers
        self._disp = None
        self.check_dispatch = check_dispatch
        self.dispatch_mismatches = 0
        self._place()
        self._hooks()
        self.cache = None

    # placement ------------------------------------------------------------
    def _place(self):
        m = self.model
        t0 = time.time()
        m.model.embed_tokens.to(self.dev); m.model.norm.to(self.dev); m.lm_head.to(self.dev)
        if getattr(m.model, "rotary_emb", None) is not None:
            m.model.rotary_emb.to(self.dev)
        self.expert_params = []
        n_per_layer = None
        for L, layer in enumerate(m.model.layers):
            layer.self_attn.to(self.dev); layer.input_layernorm.to(self.dev)
            layer.post_attention_layernorm.to(self.dev)
            mlp = layer.mlp
            mlp.gate.to(self.dev); mlp.shared_expert.to(self.dev); mlp.shared_expert_gate.to(self.dev)
            ps = []
            for ex in mlp.experts:
                for proj in (ex.gate_proj, ex.up_proj, ex.down_proj):
                    w = proj.weight
                    w.data = w.data.pin_memory()
                    ps.append(w)
            self.expert_params.append(ps)
            n = sum(p.numel() for p in ps)
            n_per_layer = n if n_per_layer is None else n_per_layer
            assert n == n_per_layer
        self.buf = torch.empty(n_per_layer, dtype=torch.bfloat16, device=self.dev)
        self.layer_bytes = n_per_layer * 2
        self.place_s = time.time() - t0

    def _hooks(self):
        m = self.model
        for L, layer in enumerate(m.model.layers):
            mlp = layer.mlp
            mlp.register_forward_pre_hook(self._pre(L))
            mlp.register_forward_hook(self._post(L))
            mlp.gate.register_forward_hook(self._gate(L))
            for e, ex in enumerate(mlp.experts):
                ex.register_forward_pre_hook(self._disp_hook(L, e))

    def _pre(self, L):
        def fn(mod, args):
            torch.cuda.synchronize(); t0 = time.time()
            off = 0
            for w in self.expert_params[L]:
                n = w.numel()
                dst = self.buf[off:off + n].view(w.shape)
                dst.copy_(w.data, non_blocking=True)
                w._cpu = w.data
                w.data = dst
                off += n
            torch.cuda.synchronize()
            self.timers["expert_transfer_s"] += time.time() - t0
            self.bytes_transferred += self.layer_bytes
        return fn

    def _post(self, L):
        def fn(mod, args, out):
            for w in self.expert_params[L]:
                w.data = w._cpu
                del w._cpu
        return fn

    def _gate(self, L):
        def fn(mod, args, out):
            t0 = time.time()
            probs = torch.softmax(out, dim=1, dtype=torch.float)
            self._sel[L] = torch.topk(probs, self.top_k, dim=-1).indices
            self.timers["routing_hook_s"] += time.time() - t0
        return fn

    def _disp_hook(self, L, e):
        def fn(mod, args):
            if self._disp is not None:
                self._disp[L, e] += args[0].reshape(-1, args[0].shape[-1]).shape[0]
        return fn

    # stepping -------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, tokens):
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.dev).view(-1, 1)
        R = ids.shape[0]
        if self.check_dispatch:
            self._disp = torch.zeros(self.n_layers, self.n_experts, dtype=torch.long)
        torch.cuda.synchronize(); t0 = time.time()
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        logits = out.logits[:, -1, :self.vocab_limit].float()
        self.cache = out.past_key_values
        sel = torch.stack([s.view(R, self.top_k) for s in self._sel], dim=1)        # [R, L, k]
        torch.cuda.synchronize()
        self.timers["forward_s"] += time.time() - t0
        self.n_forward += 1; self.rows_forwarded += R
        if self.check_dispatch:
            ref = torch.stack([torch.bincount(sel[:, l].reshape(-1).cpu(), minlength=self.n_experts)
                               for l in range(self.n_layers)])
            self.dispatch_mismatches += int((ref != self._disp).sum())
            self._disp = None
        return logits, sel

    def start(self, tokens):
        self.cache = self.DynamicCache()
        return self._forward(tokens)

    @torch.no_grad()
    def prefill(self, ids):
        """One batched causal forward of the N prefixes ([N, P], no padding: all prefixes have length P) into a
        fresh KV cache. Returns the last-position logits, the routing of all P tokens and the teacher-forced
        log-probabilities of prefix tokens 1..P-1 (fp32 log_softmax, fp64 values)."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        N, Pl = ids.shape
        x0 = self.timers["expert_transfer_s"]
        self.cache = self.DynamicCache()
        torch.cuda.synchronize(); t0 = time.time()
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        lg = out.logits[:, :, :self.vocab_limit]
        last = lg[:, -1].float()
        plp = torch.empty(N, Pl - 1, dtype=torch.float64, device=self.dev)
        for i in range(N):                                   # row-wise fp32 log_softmax (bounded memory)
            lp = torch.log_softmax(lg[i, :-1].float(), dim=-1)
            plp[i] = lp.gather(1, ids[i, 1:, None])[:, 0].double()
        sel = torch.stack([s.view(N, Pl, self.top_k) for s in self._sel], dim=2)    # [N, P, L, k]
        torch.cuda.synchronize()
        dt = time.time() - t0
        self.timers["prefill_s"] = self.timers.get("prefill_s", 0.0) + dt
        self.timers["prefill_expert_transfer_s"] = (self.timers.get("prefill_expert_transfer_s", 0.0)
                                                    + self.timers["expert_transfer_s"] - x0)
        self.n_forward += 1; self.rows_forwarded += N
        self.prefill_tokens = getattr(self, "prefill_tokens", 0) + N * Pl
        return last, sel, plp

    def step(self, tokens):
        return self._forward(tokens)

    def reorder(self, rows):
        self.cache.reorder_cache(torch.as_tensor(rows, dtype=torch.long, device=self.dev))

    def kv_cache_bytes(self):
        if self.cache is None:
            return 0
        return sum(t.numel() * t.element_size() for t in self.cache.key_cache + self.cache.value_cache)

    @torch.no_grad()
    def replay(self, tokens):
        """Teacher-forced full-sequence forward at batch 1 without KV cache (reference for log-probs/routing)."""
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.dev).view(1, -1)
        out = self.model(input_ids=ids, use_cache=False)
        lp = torch.log_softmax(out.logits[0, :, :self.vocab_limit].float(), dim=-1)
        T = ids.shape[1]
        sel = torch.stack([s.view(T, self.top_k) for s in self._sel], dim=1)        # [T, L, k]
        tok_lp = lp[:-1].gather(1, ids[0, 1:, None])[:, 0].double()
        return tok_lp.cpu(), sel.cpu()


# ---------------------------------------------------------------- generator
@dataclass
class GenResult:
    method: str
    tokens: torch.Tensor           # [N, T] chosen sentence per start token
    cum_logp: torch.Tensor         # [N]
    token_logp: torch.Tensor       # [N, T-1]
    counts: torch.Tensor           # [N, L, E] incremental counts of the chosen sentence
    hist_experts: torch.Tensor     # [N, T, L, k] per-token routed experts (generation-time)
    sigma: torch.Tensor            # [N]
    final_score: torch.Tensor      # [N]
    final_beams: dict              # all w surviving branches: tokens, cum, sigma, score
    step_log: list                 # recorded beam-score detail for selected steps
    intervention: dict             # how often the balance term changed the kept set vs probability ranking
    timing: dict
    config: dict = field(default_factory=dict)


@torch.no_grad()
def generate(backend: Backend, first_tokens: list[int], seqlen: int = SEQLEN, w: int = W_REF,
             tau: float = TAU_REF, use_balance: bool = True, record_steps=(), record_top: int = 8,
             progress_every: int = 0, audit=None, balance_std_mode: str = "eq4",
             usage_mode: str = "per_token", initial_token_ids=None) -> GenResult:
    """Algorithm 1 for all start tokens in lock-step. use_balance=False is the Probability-Only control.
    audit: optional callable (score audit, opt-in); receives CPU copies of the exact pre-pruning state, the
    production scores and the production pruning decision of every step, and the final-selection state.
    Read-only: it never alters any value used by the search.
    balance_std_mode / usage_mode: normalisation of the balance term (see balance_sigma); only sigma changes.
    initial_token_ids: optional [N][P] natural-text prefixes (same P for every sentence; P >= 1). The prefix is
    prefilled in one forward; its routing initialises the expert counts (c_total = c_prefix + c_generated), its
    teacher-forced transitions 1..P-1 initialise R (token 0 contributes 0, as the historical random first token),
    |S| counts the prefix, and expansion starts from ONE branch per sentence. seqlen counts the prefix.
    With initial_token_ids=None the historical one-token path is executed unchanged."""
    if initial_token_ids is not None:
        pre = torch.as_tensor(initial_token_ids, dtype=torch.long)
        assert pre.dim() == 2 and pre.shape[1] >= 1, "initial_token_ids must be [N, P]"
        assert first_tokens is None or list(map(int, first_tokens)) == pre[:, 0].tolist()
        assert pre.shape[1] < seqlen
        first_tokens = pre[:, 0].tolist()
    N = len(first_tokens)
    L, E, k = backend.n_layers, backend.n_experts, backend.top_k
    V = backend.vocab_limit
    t_score = t_select = t_stats = t_reorder = 0.0
    t_all = time.time()
    timers0 = dict(getattr(backend, "timers", {}))       # backend timers are cumulative across calls
    tok0 = torch.as_tensor(first_tokens, dtype=torch.long)
    if initial_token_ids is None:
        P0 = 1
        logits, sel = backend.start(tok0.tolist())
        dev = logits.device
        tokens = tok0.view(N, 1).to(dev)                 # [N*B, len]
        cum = torch.zeros(N, dtype=torch.float64, device=dev)
        tlp = torch.zeros(N, 0, dtype=torch.float64, device=dev)
        counts = torch.zeros(N, L, E, dtype=torch.long, device=dev)
        hist = torch.zeros(N, 0, L, k, dtype=torch.int16, device=dev)
    else:
        P0 = pre.shape[1]
        t_pf = time.time()
        logits, sel_pre, plp = backend.prefill(pre)
        dev = logits.device
        tokens = pre.to(dev)
        tlp = plp.to(dev, torch.float64)
        cum = tlp.sum(dim=1) if P0 > 1 else torch.zeros(N, dtype=torch.float64, device=dev)
        sel_pre = sel_pre.to(dev)
        hist = sel_pre[:, :P0 - 1].to(torch.int16)       # routing of prefix tokens 0..P-2 (token P-1: loop below)
        counts = torch.zeros(N, L, E, dtype=torch.long, device=dev)
        if P0 > 1:
            counts.scatter_add_(2, hist.long().permute(0, 2, 1, 3).reshape(N, L, -1),
                                torch.ones(N, L, (P0 - 1) * k, dtype=torch.long, device=dev))
        sel = sel_pre[:, P0 - 1]
        t_prefill = time.time() - t_pf
    B = 1                                                # live branches per sentence (1 before the first pruning)
    step_log, n_diff, n_steps, parents_diff = [], 0, 0, 0
    for j in range(P0, seqlen):
        l = j                                            # prefix length |S| (tokens incl. the first)
        t0 = time.time()
        sel = sel.to(dev)
        counts.scatter_add_(2, sel.long(), torch.ones_like(sel, dtype=torch.long))
        hist = torch.cat([hist, sel.to(torch.int16)[:, None]], dim=1)
        sigma = balance_sigma(counts, l, balance_std_mode, usage_mode, k)
        t_stats += time.time() - t0
        t0 = time.time()
        logp = torch.log_softmax(logits.float(), dim=-1)
        sc = candidate_scores(cum, logp, sigma, l, tau, use_balance).view(N, B, V)
        t_score += time.time() - t0
        t0 = time.time()
        par, tok, best = [], [], []
        for i in range(N):
            pp, tt, ss = select_best(sc[i], w)
            par.append(pp + i * B); tok.append(tt); best.append(ss)
            if use_balance and B > 1:                    # what pure probability ranking would have kept
                po = candidate_scores(cum[i * B:(i + 1) * B], logp[i * B:(i + 1) * B], sigma[i * B:(i + 1) * B],
                                      l, tau, False)
                qp, qt, _ = select_best(po, w)
                same = set(zip((pp).tolist(), tt.tolist())) == set(zip(qp.tolist(), qt.tolist()))
                n_diff += int(not same); n_steps += 1
                parents_diff += int(sorted(pp.tolist()) != sorted(qp.tolist()))
            if j in record_steps:
                flat = sc[i].reshape(-1)
                order = torch.sort(flat, stable=True).indices[:record_top]
                pi, ti = order // V, order % V
                gi = pi + i * B
                step_log.append(dict(
                    step=j, sentence=i, prefix_len=l, n_branches=B, w=w, tau=tau, use_balance=use_balance,
                    candidates=[dict(rank=r, parent=int(pi[r]), token=int(ti[r]),
                                     R_parent=float(cum[gi[r]]), logp_v=float(logp[gi[r], ti[r]]),
                                     prob_term=float(-(cum[gi[r]] + logp[gi[r], ti[r]].double()) / (l + 1)),
                                     sigma_parent=float(sigma[gi[r]]),
                                     balance_term=float(sigma[gi[r]] / tau) if use_balance else 0.0,
                                     score=float(flat[order[r]]), kept=r < w)
                                for r in range(len(order))],
                    branch_sigma=[float(x) for x in sigma[i * B:(i + 1) * B]],
                    branch_R=[float(x) for x in cum[i * B:(i + 1) * B]]))
        par = torch.cat(par); tok = torch.cat(tok).to(dev)
        t_select += time.time() - t0
        if audit is not None:
            audit(dict(kind="step", step=j, prefix_len=l, n_branches=B, w=w, tau=tau, use_balance=use_balance,
                       logits=logits.float().cpu().clone(), logp=logp.cpu().clone(), cum=cum.cpu().clone(),
                       token_logp=tlp.cpu().clone(), sigma=sigma.cpu().clone(), counts=counts.cpu().clone(),
                       hist=hist.cpu().clone(), tokens=tokens.cpu().clone(), scores=sc.cpu().clone(),
                       parent=par.cpu().clone(), token=tok.cpu().clone()))
        t0 = time.time()
        newlp = logp[par, tok].double()
        tokens = torch.cat([tokens[par], tok[:, None]], dim=1)
        cum = cum[par] + newlp
        tlp = torch.cat([tlp[par], newlp[:, None]], dim=1)
        counts = counts[par].clone()
        hist = hist[par].clone()
        backend.reorder(par.tolist() if not torch.is_tensor(par) else par)
        B = w
        t_reorder += time.time() - t0
        logits, sel = backend.step(tok.tolist())
        if progress_every and j % progress_every == 0:
            print(f"[gen {'EBSS' if use_balance else 'PO'}] step {j}/{seqlen - 1} "
                  f"{time.time() - t_all:.0f} s", flush=True)
    # routing of the last token -> complete statistics of every surviving branch (needed by Score(c))
    t0 = time.time()
    sel = sel.to(dev)
    counts.scatter_add_(2, sel.long(), torch.ones_like(sel, dtype=torch.long))
    hist = torch.cat([hist, sel.to(torch.int16)[:, None]], dim=1)
    sigma = balance_sigma(counts, seqlen, balance_std_mode, usage_mode, k)
    t_stats += time.time() - t0
    fs = final_scores(cum, sigma, seqlen, tau, use_balance).view(N, B)
    pick = torch.argmin(fs, dim=1)                       # first minimum -> lowest beam index on ties
    if audit is not None:
        audit(dict(kind="final", length=seqlen, tau=tau, use_balance=use_balance, cum=cum.cpu().clone(),
                   token_logp=tlp.cpu().clone(), sigma=sigma.cpu().clone(), counts=counts.cpu().clone(),
                   hist=hist.cpu().clone(), tokens=tokens.cpu().clone(), final_scores=fs.cpu().clone(),
                   pick=pick.cpu().clone()))
    rows = torch.arange(N, device=dev) * B + pick
    timing = dict(total_s=time.time() - t_all, stats_s=t_stats, score_s=t_score, select_s=t_select,
                  reorder_gather_s=t_reorder)
    if initial_token_ids is not None:
        timing["prefill_wall_s"] = t_prefill
    for kk, v in getattr(backend, "timers", {}).items():
        timing[kk] = v - timers0.get(kk, 0.0)
    intervention = dict(steps_with_choice=n_steps, steps_kept_set_differs_from_probability_ranking=n_diff,
                        frac=(n_diff / n_steps) if n_steps else None,
                        steps_kept_parents_differ=parents_diff) if use_balance else dict(not_applicable=True)
    return GenResult(
        method="EBSS" if use_balance else "PROBABILITY_ONLY", tokens=tokens[rows].cpu(), cum_logp=cum[rows].cpu(),
        token_logp=tlp[rows].cpu(), counts=counts[rows].cpu(), hist_experts=hist[rows].cpu(),
        sigma=sigma[rows].cpu(), final_score=fs[torch.arange(N), pick].cpu(),
        final_beams=dict(tokens=tokens.view(N, B, -1).cpu(), cum_logp=cum.view(N, B).cpu(),
                         sigma=sigma.view(N, B).cpu(), score=fs.cpu(), pick=pick.cpu(),
                         counts=counts.view(N, B, L, E).cpu()),
        step_log=step_log, intervention=intervention, timing=timing,
        config=dict(seqlen=seqlen, w=w, tau=tau, use_balance=use_balance, first_tokens=list(map(int, first_tokens)),
                    n_sentences=N, balance_std_mode=balance_std_mode, usage_mode=usage_mode, top_k=k,
                    **({} if initial_token_ids is None else dict(initial_prefix_len=int(P0),
                                                                  prefix_logp_in_R="teacher-forced tokens 1..P-1"))))


class EBSSGenerator:
    def __init__(self, backend, w=W_REF, tau=TAU_REF, seqlen=SEQLEN):
        self.backend, self.w, self.tau, self.seqlen = backend, w, tau, seqlen

    def generate(self, first_tokens, **kw):
        return generate(self.backend, first_tokens, self.seqlen, self.w, self.tau, use_balance=True, **kw)


class ProbabilityOnlyGenerator(EBSSGenerator):
    """Identical to EBSSGenerator except that the expert-balance term is removed from every score."""

    def generate(self, first_tokens, **kw):
        return generate(self.backend, first_tokens, self.seqlen, self.w, self.tau, use_balance=False, **kw)


# ---------------------------------------------------------------- set-level exposure statistics
def set_exposure(counts_sum: np.ndarray) -> dict:
    """counts_sum [L, E] routed tokens of a whole generated set -> per-layer + aggregate balance metrics."""
    from .ebss_official_data import exposure_stats
    c = np.asarray(counts_sum, float)
    n_tok = c[0].sum() / 4.0
    per = [exposure_stats(c[l]) for l in range(c.shape[0])]
    u = c / n_tok
    sig_l = u.std(axis=1, ddof=1)
    return dict(paper_sigma=float(sig_l.mean()), paper_sigma_per_layer=sig_l.tolist(), per_layer=per,
                mean_over_layers={kk: float(np.mean([p[kk] for p in per]))
                                  for kk in ("sd", "cv", "gini", "min", "p5", "p10", "max", "n_zero")})
