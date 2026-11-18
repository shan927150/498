"""Integer-code alignment, code/grid decomposition and spatial statistics of GPTQ code flips (pure functions).

Code record (as saved by moe_calib.dense_gptq.integer_codes + the solver's read-only info capture):
  q [R, C] uint8 integer codes in ORIGINAL column order; group [C] execution group of each original column
  (= position in the act-order permutation // group_size); perm [C] act-order permutation (execution position ->
  original column); scale / zero [n_groups, R] fitted at each group start (moe_calib.gptq._qparams).
The solver's dequantization is dq = scale * (q - zero) in FP32 (moe_calib.gptq.gptq_quantize), followed by the
installation cast .to(bfloat16).
"""
from __future__ import annotations

import torch

SEG = 128                                        # group_size == lazy-batch blocksize == segment length
LAGS = (1, 2, 4, 8, 16, 32, 64)                  # within-segment lags
WIN = (2, 3, 4, 8)                               # all-flip windows of length L (within segment)
EDGE = 8                                         # first / last positions of a segment


# ---------------------------------------------------------------- alignment
def element_grid(rec):
    """Per-element scale / zero in ORIGINAL coordinates: S[r, j] = scale[group[j], r]."""
    g = rec["group"].long()
    return rec["scale"].float()[g].T.contiguous(), rec["zero"].float()[g].T.contiguous()


def dequant(rec):
    """The solver's FP32 dequantized weights, original coordinates: scale * (q - zero)."""
    S, Z = element_grid(rec)
    return S * (rec["q"].float() - Z)


def validate_record(rec, W_bf16, group_size=SEG):
    perm = rec["perm"].long()
    C = rec["q"].shape[1]
    ok_perm = torch.equal(torch.sort(perm).values, torch.arange(C))
    pos = torch.empty(C, dtype=torch.long)
    pos[perm] = torch.arange(C)
    W32 = dequant(rec)
    return dict(perm_is_permutation=bool(ok_perm),
                group_equals_position_div_gs=bool(torch.equal(rec["group"].long(), pos // group_size)),
                n_groups_matches=rec["scale"].shape[0] == -(-C // group_size),
                codes_in_range=bool(int(rec["q"].max()) <= 15),
                bf16_reconstruction_bitwise=bool(torch.equal(W32.to(torch.bfloat16), W_bf16)),
                fp32_finite=bool(torch.isfinite(W32).all()))


# ---------------------------------------------------------------- code / grid decomposition
@torch.no_grad()
def decompose(ra, rb, Wa, Wb):
    """Element-level categories between variant a and reference b (original coordinates).
    A same code & same BF16, B same code & diff BF16, C diff code & same BF16, D diff code & diff BF16."""
    qa, qb = ra["q"], rb["q"]
    Sa, Za = element_grid(ra)
    Sb, Zb = element_grid(rb)
    cd, wd = qa != qb, Wa != Wb
    sd, zd = Sa != Sb, Za != Zb
    grid_d = sd | zd
    gd = (ra["group"] != rb["group"])[None, :].expand_as(cd)
    lvl_d = (qa.float() - Za) != (qb.float() - Zb)                  # offset from the zero point
    n = cd.numel()
    A, B_, C_, D = (~cd & ~wd), (~cd & wd), (cd & ~wd), (cd & wd)
    # counterfactuals: a's codes on b's grid (and reverse), cast like the installation path
    cf_ab = (Sb * (qa.float() - Zb)).to(torch.bfloat16)               # == Wb  -> difference explained by the grid
    cf_ba = (Sa * (qb.float() - Za)).to(torch.bfloat16)               # == Wa
    grid_only_ab = wd & (cf_ab == Wb)
    grid_only_ba = wd & (cf_ba == Wa)
    nw = max(int(wd.sum()), 1)
    f = lambda m: int(m.sum()) / n
    rs = ((Sa - Sb).abs() / Sb.abs().clamp_min(1e-30))
    return dict(
        n=n, frac_code_changed=f(cd), frac_bf16_changed=f(wd), frac_level_changed=f(lvl_d),
        cat_A=f(A), cat_B=f(B_), cat_C=f(C_), cat_D=f(D),
        frac_bf16_changed_same_code=int(B_.sum()) / nw, frac_code_changed_same_bf16=int(C_.sum()) / max(int(cd.sum()), 1),
        frac_scale_changed=f(sd), frac_zero_changed=f(zd), frac_grid_changed=f(grid_d), frac_group_changed=f(gd),
        median_rel_scale_diff_changed=float(rs[sd].median()) if bool(sd.any()) else 0.0,
        B_with_grid_change=int((B_ & grid_d).sum()) / max(int(B_.sum()), 1),
        C_same_level=int((C_ & ~lvl_d).sum()) / max(int(C_.sum()), 1),
        D_same_grid=int((D & ~grid_d).sum()) / max(int(D.sum()), 1),            # fixed-grid index changes
        code_changed_same_grid=int((cd & ~grid_d).sum()) / n,
        code_changed_changed_grid=int((cd & grid_d).sum()) / n,
        code_changed_same_group=int((cd & ~gd).sum()) / n,
        code_changed_changed_group=int((cd & gd).sum()) / n,
        bf16_changed_grid_only_a_on_b=int(grid_only_ab.sum()) / nw,
        bf16_changed_grid_only_b_on_a=int(grid_only_ba.sum()) / nw,
        note="counterfactual shares are order-dependent and not an additive decomposition")


# ---------------------------------------------------------------- spatial statistics
def to_exec(F, perm):
    """Reorder columns of an original-coordinate mask into an execution order (position p -> column perm[p])."""
    return F[:, perm.long().to(F.device)]


@torch.no_grad()
def stats(M, seg=SEG, full=True):
    """M [R, C] bool (C multiple of seg). Within-segment lag-pair counts, lag-seg (cross-segment) pairs,
    all-flip window counts, segment-edge counts. full=True adds runs / distances / position profile."""
    R, C = M.shape
    ns = C // seg
    S = M.view(R, ns, seg)
    out = dict(n=int(M.numel()), flips=int(M.sum()))
    for d in LAGS:
        a, b = S[..., :-d], S[..., d:]
        out[f"pairs_{d}"] = int((a & b).sum())
        out[f"first_{d}"] = int(a.sum())
        out[f"second_{d}"] = int(b.sum())
        out[f"slots_{d}"] = R * ns * (seg - d)
    out["pairs_cross"] = int((M[:, :-seg] & M[:, seg:]).sum())
    for L in WIN:
        out[f"win_{L}"] = int(S.unfold(-1, L, 1).all(-1).sum())
    out["edge_first"] = int(S[..., :EDGE].sum())
    out["edge_last"] = int(S[..., -EDGE:].sum())
    if full:
        prof = S.sum((0, 1)).float()
        out["position_profile"] = prof.cpu().tolist()
        Ln, mx = 1, 0
        while True:
            w = int(S.unfold(-1, Ln, 1).all(-1).sum()) if Ln <= seg else 0
            if w == 0:
                break
            mx, Ln = Ln, Ln + 1
        out["max_run"] = mx
        out["runs"] = out["flips"] - out["pairs_1"]
        rc = M.sum(1).float()
        out["rows_with_flip"] = int((rc > 0).sum())
        p = out["flips"] / max(out["n"], 1)
        mu = float(rc.mean())
        out["row_dispersion"] = float(rc.var() / (mu * (1 - p))) if mu > 0 and p < 1 else float("nan")
        # inter-flip distances along the row (whole row, execution order)
        idx = M.nonzero()
        if idx.shape[0] > 1:
            same = idx[1:, 0] == idx[:-1, 0]
            dist = (idx[1:, 1] - idx[:-1, 1])[same]
            edges = torch.tensor([1, 2, 3, 5, 9, 17, 33, 65, 129, 10 ** 9], device=dist.device)
            h = torch.bucketize(dist, edges, right=True)
            out["interflip_hist"] = {lab: int((h == i + 1).sum()) for i, lab in
                                     enumerate(["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"])}
    return out


def enrichment(s, d):
    """E_d = P(F_{j+d}=1 | F_j=1) / P(F_{j+d}=1), within segments."""
    if s[f"first_{d}"] == 0 or s[f"second_{d}"] == 0:
        return float("nan")
    return (s[f"pairs_{d}"] / s[f"first_{d}"]) / (s[f"second_{d}"] / s[f"slots_{d}"])


def analytic_pairs(M, seg=SEG):
    """Exact expectation of within-segment lag-d pair counts when each (row, segment)'s k flips are placed uniformly
    at random in its seg positions: k(k-1)(seg-d)/(seg(seg-1)); and the same for a row-only null (whole row)."""
    R, C = M.shape
    k = M.view(R, C // seg, seg).sum(-1).double()
    kr = M.sum(1).double()
    out = {}
    for d in LAGS:
        out[f"segnull_pairs_{d}"] = float((k * (k - 1)).sum() * (seg - d) / (seg * (seg - 1)))
        # row-only null: pairs at lag d inside the same segment, rows' flips uniform over C columns
        n_slots = (C // seg) * (seg - d)
        out[f"rownull_pairs_{d}"] = float((kr * (kr - 1)).sum() * n_slots / (C * (C - 1)))
    return out


@torch.no_grad()
def null_masks(M, gen, seg=SEG):
    """One permutation of M preserving the flip count of every (row, segment): uniform positions within segment."""
    R, C = M.shape
    S = M.view(R, C // seg, seg)
    k = S.sum(-1, keepdim=True)
    keys = torch.rand(S.shape, generator=gen, device=M.device)
    idx = keys.argsort(-1)
    take = torch.arange(seg, device=M.device).view(1, 1, seg) < k
    out = torch.zeros_like(S)
    out.scatter_(-1, idx, take)
    return out.view(R, C)


@torch.no_grad()
def permutation_null(M, n_perm, seed, seg=SEG):
    """n_perm conditioned permutations; returns per-statistic arrays (E_d, cross pairs, windows, edges)."""
    gen = torch.Generator(device=M.device)
    gen.manual_seed(int(seed))
    keys = [f"E_{d}" for d in LAGS] + ["pairs_cross", "edge_first", "edge_last", "runs"] + [f"win_{L}" for L in WIN]
    acc = {k: [] for k in keys}
    for _ in range(n_perm):
        s = stats(null_masks(M, gen, seg), seg, full=False)
        for d in LAGS:
            acc[f"E_{d}"].append(enrichment(s, d))
        acc["pairs_cross"].append(s["pairs_cross"])
        acc["edge_first"].append(s["edge_first"]); acc["edge_last"].append(s["edge_last"])
        acc["runs"].append(s["flips"] - s["pairs_1"])
        for L in WIN:
            acc[f"win_{L}"].append(s[f"win_{L}"])
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in acc.items()}


def summarize_null(obs, null):
    """Observed value vs null distribution: mean, 0.5/2.5/97.5/99.5 % quantiles, upper-tail p (with +1 correction)."""
    v = null[~torch.isnan(null)]
    if v.numel() == 0:
        return dict(obs=obs, null_mean=float("nan"))
    q = torch.quantile(v, torch.tensor([0.005, 0.025, 0.5, 0.975, 0.995], dtype=torch.float64))
    return dict(obs=float(obs), null_mean=float(v.mean()), null_sd=float(v.std()), q005=float(q[0]), q025=float(q[1]),
                q50=float(q[2]), q975=float(q[3]), q995=float(q[4]),
                p_upper=float(((v >= obs).sum() + 1) / (v.numel() + 1)), p_lower=float(((v <= obs).sum() + 1) / (v.numel() + 1)),
                ratio_to_null_mean=float(obs / float(v.mean())) if float(v.mean()) != 0 else float("nan"))
