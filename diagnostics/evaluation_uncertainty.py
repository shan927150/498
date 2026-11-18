"""Decomposing GPTQ execution variability and evaluation-sampling uncertainty (CPU only).

Reuses the stored per-window NLL of runs_heldout_bitwidth_validation/results (WikiText-2 = WT2, C4-validation = C4HO).
No model forward, no GPU, no GPTQ solve.

Stages (each resumable; outputs in runs_evaluation_uncertainty/):
  audit     artifact_inventory.json, alignment_validation.json, window_structure_audit.json
            (WT2 article mapping uses the tokenizer only: CPU, no model)
  protocol  protocol.json (frozen + sha256) BEFORE any variance / reliability result is computed
  synth     statistical_validation.json (synthetic tests A-F)
  analyze   joint covariance, variance components (+ nested bootstrap), split-half, null, cross-dataset, MoE CIs,
            dense / dose files, decision.json
  figures   figures/fig1..fig7
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, sys, time, bisect

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "runs_heldout_bitwidth_validation", "results")
OUT = os.path.join(ROOT, "runs_evaluation_uncertainty")
TGT = 511
NW = 128
GROUPS = ["G20261111", "G20261112", "G20261113", "G20261114", "G20261115"]
SEEDS = [0, 1, 2]
DOSES = [1, 4, 16, 64]
DS = ["WT2", "C4HO"]
DSKEY = {"WT2": "wikitext", "C4HO": "heldout"}

# ---- frozen numerical settings (copied into protocol.json) ----
B_COV = 2000          # joint bootstrap draws for the covariance matrix
B_OUTER = 2000        # nested bootstrap, outer
B_INNER = 400         # nested bootstrap, inner (covariance re-estimated inside every outer draw)
K_SPLIT = 1000        # split-half partitions
R_NULL = 500          # null datasets for the split-half null
K_NULL = 40           # partitions per null dataset
B_CROSS = 2000        # cross-dataset correlation bootstrap
SEED = 20261118
WT2_SCHEMES = ["article", "block4", "block8", "block16", "window"]   # first = primary
C4_SCHEMES = ["window"]                                              # one window per distinct document


def p(*a):
    return os.path.join(OUT, *a)


def jdump(o, f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(o, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, f)


def jload(f):
    with open(f) as fh:
        return json.load(fh)


def sha_file(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    os.makedirs(p("logs"), exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(p("logs", "run.log"), "a") as fh:
        fh.write(line + "\n")


# =====================================================================================================================
# data
# =====================================================================================================================
def rec(path, policy=None):
    d = jload(path)
    if policy:
        d = d["policies"][policy]
    return d


def window_means(r, ds):
    if DSKEY[ds] not in r and ds == "WT2" and "historical_wikitext_per_window_nll" in r:
        # dose k = 0 was not re-evaluated on WT2 by the heldout run; its historical WT2 array is embedded
        h = np.asarray(r["historical_wikitext_per_window_nll"], float)
        r[DSKEY[ds]] = dict(per_window_nll=h.tolist(), per_window_tokens=[TGT] * NW, total_nll=float(h.sum()),
                            total_tokens=TGT * NW, mean_nll=float(h.sum() / (TGT * NW)), ppl=r["historical_wikitext_ppl"],
                            source="historical_wikitext_per_window_nll (embedded)")
    e = r[DSKEY[ds]]
    nll = np.asarray(e["per_window_nll"], float)
    tok = np.asarray(e["per_window_tokens"], float)
    return nll / tok, nll, tok, e


def config_paths():
    """(set, id) -> (file, policy). Every configuration that the analysis may use."""
    c = {}
    for st in ("FR", "FC"):
        for s in SEEDS:
            for g in GROUPS:
                c[("moe", f"{st}_s{s}_{g}")] = (os.path.join(SRC, "moe", f"{st}_s{s}_{g}.json"), None)
        c[("moe", f"{st}_ORIGINAL")] = (os.path.join(SRC, "moe", f"{st}_ORIGINAL.json"), None)
    c[("moe", "BF16")] = (os.path.join(SRC, "moe", "BF16.json"), None)
    for g in GROUPS + ["ORIGINAL"]:
        c[("dense4", g)] = (os.path.join(SRC, "dense", f"dense4_{g}.json"), None)
        c[("dense3", g)] = (os.path.join(SRC, "dense3", f"dense3_{g}.json"), None)
    c[("dense", "BF16")] = (os.path.join(SRC, "dense", "BF16.json"), None)
    for pol in ("natural", "fixed"):
        c[(f"dose_{pol}", "k000")] = (os.path.join(SRC, "dense", "dose_b1_k000.json"), pol)
        for k in DOSES:
            for r in range(8):
                c[(f"dose_{pol}", f"k{k:03d}_r{r}")] = (os.path.join(SRC, "dense", f"dose_b1_k{k:03d}_r{r}.json"), pol)
    return c


def load_matrix():
    """L[set][id][ds] = per-window mean NLL (128,)"""
    L = {}
    for (st, cid), (f, pol) in config_paths().items():
        r = rec(f, pol)
        for ds in DS:
            L.setdefault(st, {}).setdefault(cid, {})[ds] = window_means(r, ds)[0]
    return L


def analysis_sets(L):
    """Return {name: dict(cols, X{ds}: (128,G), blocks, ref{ds}, contrast{ds}(128,C), contrast_ids, kind)}

    X columns are the per-window quantities whose column means are the variant estimates theta.
    Centering blocks: list of column-index lists (MoE: one per seed; others: one block).
    contrasts: the prespecified non-null sign contrasts (MoE: raw FC-FR; dense: g - ORIGINAL for 4 non-null groupings;
    dose: replicate - k0)."""
    S = {}
    cols = [f"s{s}_{g}" for s in SEEDS for g in GROUPS]
    X = {ds: np.stack([L["moe"][f"FC_s{s}_{g}"][ds] - L["moe"][f"FR_s{s}_{g}"][ds] for s in SEEDS for g in GROUPS], 1)
         for ds in DS}
    S["moe"] = dict(kind="moe", cols=cols, X=X, blocks=[list(range(5 * i, 5 * i + 5)) for i in range(3)],
                    C=X, contrast_ids=cols, primary=True,
                    FR={ds: np.stack([L["moe"][f"FR_s{s}_{g}"][ds] for s in SEEDS for g in GROUPS], 1) for ds in DS},
                    FC={ds: np.stack([L["moe"][f"FC_s{s}_{g}"][ds] for s in SEEDS for g in GROUPS], 1) for ds in DS})
    for b in ("dense4", "dense3"):
        X = {ds: np.stack([L[b][g][ds] for g in GROUPS], 1) for ds in DS}
        nn = GROUPS[:4]
        C = {ds: np.stack([L[b][g][ds] - L[b]["ORIGINAL"][ds] for g in nn], 1) for ds in DS}
        S[b] = dict(kind="dense", cols=GROUPS, X=X, blocks=[list(range(5))], C=C, contrast_ids=nn,
                    primary=(b == "dense4"), ref={ds: L[b]["ORIGINAL"][ds] for ds in DS})
    for pol in ("natural", "fixed"):
        for k in DOSES:
            ids = [f"k{k:03d}_r{r}" for r in range(8)]
            X = {ds: np.stack([L[f"dose_{pol}"][i][ds] for i in ids], 1) for ds in DS}
            C = {ds: X[ds] - L[f"dose_{pol}"]["k000"][ds][:, None] for ds in DS}
            S[f"dose_{pol}_k{k}"] = dict(kind="dose", cols=ids, X=X, blocks=[list(range(8))], C=C, contrast_ids=ids,
                                         primary=False)
    return S


# =====================================================================================================================
# stage: audit
# =====================================================================================================================
def wt2_articles():
    """Map each WT2 evaluation window to its WikiText-2 test article (CPU tokenizer only)."""
    sys.path.insert(0, ROOT)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from moe_calib import model_utils as MU
    import torch
    prot = jload(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json"))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    assert hashlib.sha256(text.encode()).hexdigest() == prot["dataset"]["text_sha256"]
    tok = AutoTokenizer.from_pretrained(MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B"))
    enc = tok(text, return_offsets_mapping=True)
    ids, off = enc.input_ids, enc.offset_mapping
    assert len(ids) == prot["total_corpus_tokens"]
    heads, titles, pos = [], [], 0
    for line in ds["text"]:
        if re.fullmatch(r" = [^=].*[^=] = \n", line):
            heads.append(pos)
            titles.append(line.strip(" =\n"))
        pos += len(line) + 2
    win = torch.load(os.path.join(ROOT, "runs_partial_ppl", "windows.pt"))
    starts = [o[0] for o in off]
    rows = []
    for i, w in enumerate(prot["window_indices"]):
        assert torch.equal(torch.tensor(ids[w * 512:(w + 1) * 512]), win[i])
        art = [max(bisect.bisect_right(heads, starts[t]) - 1, 0) for t in range(w * 512, w * 512 + 512)]
        vals, cnt = np.unique(art[1:], return_counts=True)          # scored targets = tokens 1..511
        major = int(vals[np.argmax(cnt)])
        rows.append(dict(window=i, stream_window_index=w, token_offset=w * 512, article=major, article_title=titles[major],
                         articles_spanned=[int(v) for v in vals], frac_targets_in_major=float(cnt.max() / 511),
                         contiguous_with_previous=bool(i > 0 and w == prot["window_indices"][i - 1] + 1)))
    return dict(n_articles_in_test=len(heads), rows=rows,
                rule="article = WikiText-2 test article (top-level ' = Title = ' header) holding the majority of the "
                     "window's 511 scored targets; tokens before the first header are assigned to article 0",
                windows_verified_equal_runs_partial_ppl=True)


def stage_audit(_):
    os.makedirs(p("logs"), exist_ok=True)
    paths = config_paths()
    inv = dict(source_dir=os.path.relpath(SRC, ROOT), configs={}, missing=[], format=(
        "per-config JSON; <dataset>.per_window_nll = per-window SUM of token NLL over 511 scored targets; "
        "per_window_tokens = 511; 128 windows; mean NLL per window = sum / tokens"))
    for (st, cid), (f, pol) in paths.items():
        if not os.path.exists(f):
            inv["missing"].append(dict(set=st, id=cid, file=f))
            continue
        inv["configs"][f"{st}/{cid}"] = dict(file=os.path.relpath(f, ROOT), policy=pol, sha256=sha_file(f))
    inv["n_configs"] = len(inv["configs"])
    inv["analysis_sets"] = dict(
        moe="30 MoE Family D configs -> 15 paired FC-FR columns (3 seeds x 5 groupings); FR/FC_ORIGINAL descriptive",
        dense4="5 historical groupings (G20261115 = ORIGINAL bitwise); ORIGINAL = descriptive reference",
        dense3="same 5 groupings at 3 bits",
        dose_natural="batch-1 natural act-order, k = 1/4/16/64 x 8 replicates, k0 = reference",
        dose_fixed="batch-1 fixed act-order (ORIGINAL permutation), same replicate IDs, secondary")
    inv["not_used"] = "reference matrix (ref_*); dose strong-path and k = 32/96 families were never evaluated on C4HO"
    jdump(inv, p("artifact_inventory.json"))
    log(f"[audit] inventory {inv['n_configs']} configs, missing {len(inv['missing'])}")

    # ---- alignment
    al = dict(checks={}, per_config={})
    hist_moe = {}
    for s in (1, 2):
        for st in ("FR", "FC"):
            for g in GROUPS:
                f = os.path.join(ROOT, "runs_cross_seed_grouping", "results", f"{st}_s{s}_{g}.json")
                if os.path.exists(f):
                    hist_moe[f"{st}_s{s}_{g}"] = (f, np.asarray(jload(f)["per_window_nll"], float))
    natural = jload(os.path.join(ROOT, "runs_grouping_routing_amplification", "natural_ppl_results.json"))["per_config"]
    dscope = jload(os.path.join(ROOT, "runs_dense_grouping_scope", "grouping_ppl_results.json"))["configs"]
    ok_all = True
    for (st, cid), (f, pol) in paths.items():
        r = rec(f, pol)
        c = {}
        for ds in DS:
            m, nll, tok, e = window_means(r, ds)
            c[ds] = dict(shape_ok=nll.shape == (NW,), finite=bool(np.isfinite(nll).all()),
                         tokens_all_511=bool((tok == TGT).all()), total_tokens_65408=int(tok.sum()) == 65408 == e["total_tokens"],
                         sum_reproduces_total_nll=abs(nll.sum() - e["total_nll"]) <= 1e-9 * e["total_nll"],
                         mean_reproduces_mean_nll=abs(nll.sum() / tok.sum() - e["mean_nll"]) <= 1e-12,
                         exp_reproduces_ppl=abs(math.exp(nll.sum() / tok.sum()) - e["ppl"]) <= 1e-9 * e["ppl"],
                         equal_weight_window_mean_equals_token_weighted=abs(m.mean() - e["mean_nll"]) <= 1e-12)
        hw = r.get("historical_wikitext_per_window_nll")
        if hw is not None:
            c["WT2"]["per_window_equals_historical_embedded"] = bool(np.array_equal(np.asarray(hw, float), window_means(r, "WT2")[1]))
        if st == "moe" and cid in hist_moe:
            c["WT2"]["per_window_equals_runs_cross_seed_grouping"] = bool(np.array_equal(hist_moe[cid][1], window_means(r, "WT2")[1]))
        if st == "moe" and cid in natural:
            c["WT2"]["ppl_equals_runs_grouping_routing_amplification"] = natural[cid]["ppl"] == r["wikitext"]["ppl"]
        if st == "dense4" and cid in dscope:
            c["WT2"]["ppl_equals_runs_dense_grouping_scope"] = dscope[cid]["ppl"] == r["wikitext"]["ppl"]
        c["record_valid_flag"] = bool(r.get("valid", True))
        flat = [v for ds in DS for v in c[ds].values()] + [c["record_valid_flag"]]
        c["all_ok"] = all(flat)
        ok_all &= c["all_ok"]
        al["per_config"][f"{st}/{cid}"] = c
    # exact null
    L = load_matrix()
    al["checks"]["dense4_G20261115_equals_ORIGINAL_bitwise"] = {ds: bool(np.array_equal(L["dense4"]["G20261115"][ds], L["dense4"]["ORIGINAL"][ds])) for ds in DS}
    al["checks"]["dense3_G20261115_equals_ORIGINAL_bitwise"] = {ds: bool(np.array_equal(L["dense3"]["G20261115"][ds], L["dense3"]["ORIGINAL"][ds])) for ds in DS}
    al["checks"]["dose_k0_natural_equals_dense4_ORIGINAL_bitwise"] = {ds: bool(np.array_equal(L["dose_natural"]["k000"][ds], L["dense4"]["ORIGINAL"][ds])) for ds in DS}
    al["checks"]["dose_k0_WT2_source"] = "historical_wikitext_per_window_nll embedded in dose_b1_k000.json (k = 0 not re-evaluated on WT2)"
    hv = jload(os.path.join(ROOT, "runs_heldout_bitwidth_validation", "heldout_evaluator_validation.json"))
    al["checks"]["window_identity"] = dict(
        WT2="all configs scored on runs_partial_ppl/windows.pt (sha e646cf7c...), the same 128 windows in sorted stream "
            "order; the heldout run verified WT2 per-window NLL bitwise against history for every parity config",
        C4HO="all configs scored on runs_heldout_bitwidth_validation/heldout_windows.pt (sha e105f974...), fixed order "
             "by shard row; same file for MoE and dense (dense tokenizer re-tokenizes identically)",
        heldout_evaluator_validation_keys=list(hv)[:20])
    al["checks"]["weighting"] = ("every window has 511 scored targets, so the token-weighted mean NLL equals the equal-"
                                 "weight mean of window mean NLL; analyses use window mean NLL = sum / 511")
    al["all_configs_ok"] = ok_all
    al["n_configs"] = len(al["per_config"])
    jdump(al, p("alignment_validation.json"))
    log(f"[audit] alignment all ok = {ok_all}; nulls {al['checks']['dense4_G20261115_equals_ORIGINAL_bitwise']}")

    # ---- window structure
    man = jload(os.path.join(ROOT, "runs_heldout_bitwidth_validation", "evaluation_manifest.json"))
    pw = man["per_window"]
    c4 = dict(n_windows=len(pw), distinct_rows=len({w["c4_validation_row"] for w in pw}),
              distinct_urls=len({w["url"] for w in pw}), distinct_text_sha=len({w["text_sha256"] for w in pw}),
              one_window_per_document=man["sequence_construction"]["one_window_per_document"],
              window_order="ascending C4 validation shard row")
    c4["resampling_unit"] = "document (= window)" if c4["distinct_rows"] == c4["distinct_text_sha"] == NW else "UNRESOLVED"
    wt = wt2_articles()
    arts = [r["article"] for r in wt["rows"]]
    ua, cnt = np.unique(arts, return_counts=True)
    contig = sum(r["contiguous_with_previous"] for r in wt["rows"])
    runs, cur = [], 1
    for r in wt["rows"][1:]:
        if r["contiguous_with_previous"]:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    wt2 = dict(construction=jload(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json"))["packing"],
               n_windows=NW, n_articles_covered=len(ua), n_articles_in_test=wt["n_articles_in_test"],
               windows_per_article=dict(min=int(cnt.min()), median=float(np.median(cnt)), max=int(cnt.max()),
                                        hist={int(k): int(v) for k, v in zip(*np.unique(cnt, return_counts=True))}),
               windows_spanning_two_articles=sum(len(r["articles_spanned"]) > 1 for r in wt["rows"]),
               windows_contiguous_with_previous=contig, contiguous_run_lengths=dict(
                   max=max(runs), n_runs=len(runs), hist={int(k): int(v) for k, v in zip(*np.unique(runs, return_counts=True))}),
               articles_monotone_in_window_order=bool(all(a <= b for a, b in zip(arts, arts[1:]))),
               rule=wt["rule"], rows=wt["rows"])
    wt2["resampling_unit_primary"] = "article (cluster of windows)"
    wt2["block_sensitivity"] = {f"block{b}": dict(block_len=b, n_blocks=NW // b, complete=NW % b == 0) for b in (4, 8, 16)}
    wt2["blocks_note"] = ("blocks are non-overlapping runs of b consecutive windows in the sorted stream order; 128 is "
                          "divisible by 4/8/16, so no window is dropped and there is no incomplete block; b = 16 leaves "
                          "only 8 resampling units (unstable)")
    # lag-1 autocorrelation of BF16 window NLL (description of serial dependence)
    def ac1(x):
        x = x - x.mean()
        return float((x[1:] * x[:-1]).sum() / (x * x).sum())
    wt2["lag1_autocorr_bf16_moe_window_nll"] = ac1(L["moe"]["BF16"]["WT2"])
    wt2["lag1_autocorr_moe_paired_effect_s0_G20261111"] = ac1(L["moe"]["FC_s0_G20261111"]["WT2"] - L["moe"]["FR_s0_G20261111"]["WT2"])
    c4["lag1_autocorr_bf16_moe_window_nll"] = ac1(L["moe"]["BF16"]["C4HO"])
    jdump(dict(C4HO=c4, WT2=wt2), p("window_structure_audit.json"))
    log(f"[audit] C4 unique docs {c4['distinct_rows']}; WT2 articles {len(ua)}, contiguous pairs {contig}")


# =====================================================================================================================
# resampling machinery
# =====================================================================================================================
def units_for(scheme, ds):
    """unit id per window (len 128)."""
    if scheme == "window":
        return np.arange(NW)
    if scheme.startswith("block"):
        b = int(scheme[5:])
        return np.arange(NW) // b
    if scheme == "article":
        assert ds == "WT2"
        return np.asarray([r["article"] for r in jload(p("window_structure_audit.json"))["WT2"]["rows"]])
    raise ValueError(scheme)


class Units:
    """Aggregate windows into resampling units: S (m, G) sums, n (m,) window counts."""
    def __init__(self, X, uid):
        self.ids, inv = np.unique(uid, return_inverse=True)
        self.m = len(self.ids)
        self.n = np.bincount(inv, minlength=self.m).astype(float)
        self.S = np.zeros((self.m, X.shape[1]))
        np.add.at(self.S, inv, X)
        self.inv = inv

    def theta(self, C):
        """C: (B, m) unit counts -> (B, G)"""
        return (C @ self.S) / (C @ self.n)[:, None]


def draw_counts(rng, m, B):
    return rng.multinomial(m, np.full(m, 1.0 / m), size=B).astype(float)


def center(theta, blocks):
    out = np.empty_like(theta)
    for b in blocks:
        out[..., b] = theta[..., b] - theta[..., b].mean(-1, keepdims=True)
    return out


def df_of(blocks):
    return sum(len(b) - 1 for b in blocks)


def vstats_from(theta_hat, draws, blocks):
    """V_obs, V_samp (= tr(P Sigma)/df) from bootstrap draws (B, G)."""
    df = df_of(blocks)
    v_obs = float((center(theta_hat, blocks) ** 2).sum() / df)
    cd = center(draws - draws.mean(0), blocks)
    v_samp = float((cd ** 2).sum() / (draws.shape[0] - 1) / df)
    return v_obs, v_samp


def nested(U, blocks, rng, n_out, n_in, null=False, theta_hat=None):
    """Outer bootstrap of units; inside every outer draw the sampling covariance is re-estimated by an inner bootstrap.
    Returns arrays V_obs*, V_samp*, V_corr* (n_out,). If null: the variant effects are removed first (common-effect
    world that keeps the window-level covariance)."""
    S = U.S
    if null:
        th = center(theta_hat, blocks)
        S = U.S - U.n[:, None] * th[None, :]
    df = df_of(blocks)
    vo, vs = np.empty(n_out), np.empty(n_out)
    m = U.m
    for o in range(n_out):
        idx = rng.integers(0, m, m)
        So, no = S[idx], U.n[idx]
        t = So.sum(0) / no.sum()
        vo[o] = (center(t, blocks) ** 2).sum() / df
        C = draw_counts(rng, m, n_in)
        ti = (C @ So) / (C @ no)[:, None]
        cd = center(ti - ti.mean(0), blocks)
        vs[o] = (cd ** 2).sum() / (n_in - 1) / df
    return vo, vs, vo - vs


# ---- split-half ----------------------------------------------------------------------------------------------------
def greedy_split(rng, uid_unique, sizes):
    """Shuffle units, assign each to the half with fewer windows (ties -> A). Returns boolean mask over units (A)."""
    order = rng.permutation(len(uid_unique))
    a = np.zeros(len(uid_unique), bool)
    na = nb = 0.0
    for i in order:
        if na <= nb:
            a[i] = True; na += sizes[i]
        else:
            nb += sizes[i]
    return a


def rankdata(x):
    """average ranks along last axis (ties averaged), x (..., G)"""
    o = np.argsort(x, axis=-1, kind="mergesort")
    r = np.empty_like(x, dtype=float)
    G = x.shape[-1]
    np.put_along_axis(r, o, np.broadcast_to(np.arange(1, G + 1, dtype=float), x.shape), axis=-1)
    # tie averaging (rare; exact duplicates only)
    xs = np.take_along_axis(x, o, -1)
    if (np.diff(xs, axis=-1) == 0).any():
        from scipy.stats import rankdata as rd
        r = np.apply_along_axis(rd, -1, x)
    return r


def pearson(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(-1) / den


def pair_list(blocks):
    return [(i, j) for b in blocks for ii, i in enumerate(b) for j in b[ii + 1:]]


def reliability_stats(zA, zB, cA, cB, blocks, se_full, thr_mask):
    """zA,zB (K,G) centred effects; cA,cB (K,C) raw contrasts. Returns dict of (K,) arrays."""
    out = {}
    out["pearson"] = pearson(zA, zB)
    if len(blocks) > 1:
        rA = np.concatenate([rankdata(zA[:, b]) for b in blocks], 1)
        rB = np.concatenate([rankdata(zB[:, b]) for b in blocks], 1)
        out["spearman_within_seed_ranks"] = pearson(rA, rB)
        out["per_seed_pearson"] = np.stack([pearson(zA[:, b], zB[:, b]) for b in blocks], 1)
    else:
        out["spearman"] = pearson(rankdata(zA), rankdata(zB))
    pl = pair_list(blocks)
    I = np.array([i for i, _ in pl]); J = np.array([j for _, j in pl])
    sa = np.sign(zA[:, I] - zA[:, J]); sb = np.sign(zB[:, I] - zB[:, J])
    out["kendall_tau_a"] = (sa * sb).mean(1)
    out["pair_order_agreement"] = (sa * sb > 0).mean(1)
    top = []; bot = []
    for b in blocks:
        top.append(np.argmax(zA[:, b], 1) == np.argmax(zB[:, b], 1))
        bot.append(np.argmin(zA[:, b], 1) == np.argmin(zB[:, b], 1))
    out["top_agreement"] = np.mean(top, 0)
    out["bottom_agreement"] = np.mean(bot, 0)
    out["sign_agreement_all"] = (np.sign(cA) * np.sign(cB) > 0).mean(1)
    if thr_mask.any():
        out["sign_agreement_thresholded"] = (np.sign(cA[:, thr_mask]) * np.sign(cB[:, thr_mask]) > 0).mean(1)
    return out


def summarize(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return dict(median=float(np.median(x)), mean=float(x.mean()), p05=float(np.quantile(x, .05)),
                p95=float(np.quantile(x, .95)), frac_positive=float((x > 0).mean()), frac_gt_0p5=float((x > .5).mean()),
                frac_negative=float((x < 0).mean()), n=int(x.size))


def split_masks(rng, scheme, ds, K, uid=None):
    """K boolean (128,) masks for half A. WT2 'chronological' handled separately."""
    uid = units_for(scheme, ds) if uid is None else uid
    uu, inv = np.unique(uid, return_inverse=True)
    sizes = np.bincount(inv).astype(float)
    M = np.empty((K, len(uid)), bool)
    for k in range(K):
        a = greedy_split(rng, uu, sizes)
        M[k] = a[inv]
    return M


def half_thetas(X, M):
    W = M.astype(float)
    tA = (W @ X) / W.sum(1, keepdims=True)
    tB = ((1 - W) @ X) / (1 - W).sum(1, keepdims=True)
    return tA, tB


# =====================================================================================================================
# stage: protocol
# =====================================================================================================================
def stage_protocol(_):
    f = p("protocol.json")
    if os.path.exists(f):
        log("[protocol] already frozen; not rewritten")
        return
    for need in ("artifact_inventory.json", "alignment_validation.json", "window_structure_audit.json"):
        assert os.path.exists(p(need)), need
    assert not os.path.exists(p("variance_components.json")), "results already exist; protocol must precede them"
    prot = dict(
        title="Decomposing GPTQ execution variability and evaluation-sampling uncertainty",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        cpu_only=True,
        data=dict(source="runs_heldout_bitwidth_validation/results (per-window NLL sums / 511 targets)",
                  datasets=dict(WT2="runs_partial_ppl/windows.pt, 128 x 512 WikiText-2 test windows",
                                C4HO="runs_heldout_bitwidth_validation/heldout_windows.pt, 128 x 512, one per C4-val document"),
                  inventory_sha256=sha_file(p("artifact_inventory.json")),
                  window_structure_sha256=sha_file(p("window_structure_audit.json"))),
        analysis_sets=dict(
            primary=dict(moe="15 paired columns D = L_FC - L_FR (per-window mean NLL), 3 seeds x 5 groupings; "
                             "within-seed block centering P_within = diag(P5,P5,P5), df = 12",
                         dense4="5 grouping columns L_g (G20261111..15); P5, df = 4; ORIGINAL = reference for contrasts only"),
            secondary=dict(dense3="as dense4 at 3 bits",
                           dose_natural="k = 1/4/16/64 analysed separately; 8 replicate columns; P8, df = 7; contrasts vs k = 0"),
            exploratory=dict(dose_fixed="fixed act-order dose arrays, same treatment",
                             disattenuated_cross_correlation="only if both datasets' corrected-variance CI lower bound > 0")),
        estimand=("between-variant variance of dataset-average mean NLL (finite set of executed variants), corrected for "
                  "evaluation-text sampling: V_corr = V_obs - tr(P Sigma)/df, Sigma = joint bootstrap covariance of the "
                  "variant mean-NLL vector with identical resampled indices for all variants"),
        resampling=dict(C4HO="window = document (verified one window per distinct document); iid bootstrap of 128 units",
                        WT2_primary="article cluster bootstrap (window -> majority WikiText-2 test article)",
                        WT2_sensitivity=["block4 (32 units)", "block8 (16)", "block16 (8, unstable)",
                                         "window (naive iid, for comparison only)"],
                        cluster_estimator="ratio estimator: sum of resampled unit sums / resampled window count"),
        bootstrap=dict(B_cov=B_COV, nested_outer=B_OUTER, nested_inner=B_INNER, cross_dataset=B_CROSS,
                       variance_ci=("nested bootstrap: Sigma re-estimated in every outer draw; pivot interval for V_corr: "
                                    "[V_corr - q975(V_corr* - V_obs), V_corr - q025(V_corr* - V_obs)] (V_obs = bootstrap-"
                                    "world true finite-set variance); V_obs and V_samp reported with percentile intervals"),
                       null_test=("common-effect null: remove the centred variant effects from every unit sum (keeps the "
                                  "window-level joint covariance), rerun the nested bootstrap; p = P(V_corr*_null >= V_corr)"),
                       sampling_fraction="V_samp / V_obs; interval = 1 - (V_corr pivot interval) / V_obs"),
        split_half=dict(K=K_SPLIT, C4HO="random 64/64 documents", WT2_primary="balanced random article split "
                        "(shuffle articles, assign to the half with fewer windows)",
                        WT2_sensitivity=["block4", "block8", "block16", "chronological first 64 vs last 64 "
                                         "(single deterministic split, confounded by topic/position)"],
                        statistics=["Pearson of centred effects (MoE: within-seed centred, 15-dim) [primary]",
                                    "Spearman (MoE: within-seed ranks)", "Kendall tau-a (MoE: within-seed pairs only)",
                                    "pairwise order agreement", "top / bottom agreement (MoE: per seed, averaged)",
                                    "sign agreement of prespecified non-null contrasts",
                                    "sign agreement restricted to contrasts with |full-data estimate| > 1 x its own "
                                    "primary-scheme bootstrap SE (rule fixed before results; unthresholded = primary)"],
                        contrasts=dict(moe="raw FC - FR paired effect (15)", dense="g - ORIGINAL, 4 non-null groupings "
                                       "(G20261115 = ORIGINAL excluded)", dose="replicate - k0 (8)"),
                        summary="median, mean, 5th/95th percentiles, frac > 0, frac > 0.5, frac < 0 over partitions; "
                                "partitions reuse the same windows and are NOT independent replicates"),
        split_half_null=dict(R=R_NULL, K_per_dataset=K_NULL, construction=(
            "common-effect world (variant effects removed from unit sums); draw a bootstrap dataset of units (same "
            "scheme), split its DRAWS with the same balanced rule (each draw is a unit, so the halves are independent iid "
            "samples from the effect-free population, as complementary halves of an iid sample are), compute "
            "reliability; statistic compared = median Pearson over partitions; p = P(null median >= observed median)")),
        cross_dataset=dict(statistics=["Pearson / Spearman of full-data centred effects", "sign agreement of contrasts",
                                       "pairwise order agreement"],
                           ci="independent primary-scheme bootstraps of the two datasets, percentile",
                           comparability=("split-half medians are stepped up to full length with Spearman-Brown "
                                          "2r/(1+r) before comparison with the full-vs-full cross correlation; "
                                          "expected cross correlation under identical true effects = sqrt(lam_WT2 lam_C4), "
                                          "lam = V_corr / V_obs (only if both > 0)")),
        moe_ci=dict(statistic="per pair mean of D (nats/token) with percentile CI under every scheme; PPL difference "
                              "exp(l_FC) - exp(l_FR) reported separately with its own percentile CI", level=0.95),
        decision_rules=dict(
            resolved_variance="V_corr pivot 95% CI lower bound > 0 AND common-effect null p < 0.05 (primary scheme)",
            reliable_within="median split-half Pearson (primary scheme) >= 0.5 AND exceeds the null median "
                            "distribution at p < 0.05",
            setting_verdict=dict(
                A="both datasets resolved AND reliable, AND cross-dataset Pearson upper 95% bound < sqrt(lam_WT2 lam_C4)",
                B="neither dataset resolved AND neither reliable",
                C="exactly one dataset resolved-and-reliable, the other neither (or settings within a family differ)",
                D="any other pattern (e.g. resolved but unreliable, both reliable but cross-dataset agreement not "
                  "distinguishable from the attenuated expectation)"),
            note="dose levels are reported per k with the same rules; a family-level dose summary is descriptive"),
        seeds=dict(master=SEED, derivation="np.random.default_rng([SEED, stage_code, set_index, dataset_index, scheme_index])"),
        code_sha256=sha_file(os.path.abspath(__file__)),
        constraints=["no model forward / GPU / GPTQ", "no new data", "historical artifacts read-only",
                     "negative V_corr reported raw; zero only as a separately labelled constrained estimate"])
    jdump(prot, f)
    h = sha_file(f)
    with open(p("logs", "protocol_sha256.txt"), "w") as fh:
        fh.write(h + "\n")
    os.chmod(f, 0o444)
    log(f"[protocol] frozen sha256 {h}")


def check_protocol():
    prot = jload(p("protocol.json"))
    h = open(p("logs", "protocol_sha256.txt")).read().strip()
    assert sha_file(p("protocol.json")) == h, "protocol changed"
    return prot


def rng_for(*key):
    return np.random.default_rng([SEED] + [int(k) for k in key])


# =====================================================================================================================
# stage: synthetic validation
# =====================================================================================================================
def synth_data(rng, n, G, delta, ar=0.0, shared_sd=0.35, idio_sd=0.004, common_sd=0.003):
    """window-level X (n,G): base difficulty + variant-common noise + idiosyncratic noise + delta."""
    def series(sd, k):
        e = rng.normal(0, sd, (n, k))
        if ar > 0:
            for t in range(1, n):
                e[t] = ar * e[t - 1] + math.sqrt(1 - ar * ar) * e[t]
        return e
    base = series(shared_sd, 1)
    common = series(common_sd, 1)
    idio = series(idio_sd, G)
    return base + common + idio + delta[None, :]


def stage_synth(_):
    if os.path.exists(p("statistical_validation.json")):
        log("[synth] exists; skip")
        return
    check_protocol()
    t0 = time.time()
    out = {}
    G = 5
    blocks = [list(range(G))]
    # ---- A: no true differences
    rng = rng_for(1, 1)
    vc, vs, cover, reps = [], [], [], 200
    for r in range(reps):
        X = synth_data(rng, 128, G, np.zeros(G))
        U = Units(X, np.arange(128))
        th = X.mean(0)
        dr = U.theta(draw_counts(rng, 128, 1000))
        vo, vsm = vstats_from(th, dr, blocks)
        vc.append(vo - vsm); vs.append(vsm)
        if r < 60:
            o, s_, c_ = nested(U, blocks, rng, 300, 100)
            d = c_ - vo
            lo, hi = (vo - vsm) - np.quantile(d, .975), (vo - vsm) - np.quantile(d, .025)
            cover.append(lo <= 0 <= hi)
    vc, vs = np.array(vc), np.array(vs)
    out["A_no_true_differences"] = dict(reps=reps, mean_V_corr=float(vc.mean()), se_mean=float(vc.std(ddof=1) / math.sqrt(reps)),
                                        mean_V_samp=float(vs.mean()), bias_relative_to_V_samp=float(vc.mean() / vs.mean()),
                                        frac_negative=float((vc < 0).mean()), pivot_ci_coverage_of_zero=float(np.mean(cover)),
                                        n_coverage_reps=len(cover),
                                        passed=bool(abs(vc.mean()) < 3 * vc.std(ddof=1) / math.sqrt(reps)))
    # ---- B: stable differences
    rng = rng_for(1, 2)
    tau = 0.0006
    delta = rng.normal(0, tau, G); delta -= delta.mean()
    true_v = float((delta ** 2).sum() / (G - 1))
    res = {}
    for n in (128, 1024):
        vcs, rs = [], []
        for r in range(100):
            X = synth_data(rng, n, G, delta)
            U = Units(X, np.arange(n))
            dr = U.theta(draw_counts(rng, n, 500))
            vo, vsm = vstats_from(X.mean(0), dr, blocks)
            vcs.append(vo - vsm)
            M = np.zeros((20, n), bool)
            for k in range(20):
                M[k, rng.permutation(n)[: n // 2]] = True
            tA, tB = half_thetas(X, M)
            rs.append(np.median(pearson(center(tA, blocks), center(tB, blocks))))
        res[str(n)] = dict(mean_V_corr=float(np.mean(vcs)), true_finite_set_V=true_v,
                           ratio=float(np.mean(vcs) / true_v), median_split_half_r=float(np.median(rs)))
    out["B_stable_differences"] = dict(per_n=res, passed=bool(abs(res["1024"]["ratio"] - 1) < 0.25 and
                                                                res["1024"]["median_split_half_r"] > res["128"]["median_split_half_r"]))
    # ---- C: domain-dependent
    rng = rng_for(1, 3)
    d1 = rng.normal(0, tau, G); d2 = rng.normal(0, tau, G)
    res = {}
    for n in (128, 1024, 4096):
        win1, win2, cross = [], [], []
        for r in range(50):
            X1 = synth_data(rng, n, G, d1); X2 = synth_data(rng, n, G, d2)
            M = np.zeros((10, n), bool)
            for k in range(10):
                M[k, rng.permutation(n)[: n // 2]] = True
            a, b = half_thetas(X1, M); win1.append(np.median(pearson(a, b)))
            a, b = half_thetas(X2, M); win2.append(np.median(pearson(a, b)))
            cross.append(float(pearson(X1.mean(0), X2.mean(0))))
        res[str(n)] = dict(within_domain1=float(np.median(win1)), within_domain2=float(np.median(win2)),
                           cross_domain=float(np.median(cross)))
    out["C_domain_dependent"] = dict(true_cross_corr_of_effects=float(pearson(d1 - d1.mean(), d2 - d2.mean())), per_n=res,
                                     passed=bool(res["4096"]["within_domain1"] > res["128"]["within_domain1"] and
                                                 res["4096"]["within_domain1"] > 0.8))
    # ---- D: serial correlation
    rng = rng_for(1, 4)
    true_means, se_naive, se_block = [], [], []
    for r in range(300):
        X = synth_data(rng, 128, 2, np.zeros(2), ar=0.8, idio_sd=0.01)
        dcol = X[:, 1:] - X[:, :1]
        true_means.append(dcol.mean())
        if r < 100:
            for sch, store in (("window", se_naive), ("block16", se_block)):
                U = Units(dcol, units_for(sch, None) if sch != "window" else np.arange(128))
                store.append(U.theta(draw_counts(rng, U.m, 500)).std(ddof=1))
    sd_true = float(np.std(true_means, ddof=1))
    out["D_serial_correlation"] = dict(ar1=0.8, true_sd_of_mean=sd_true, mean_se_naive_window=float(np.mean(se_naive)),
                                       mean_se_block16=float(np.mean(se_block)),
                                       naive_over_true=float(np.mean(se_naive) / sd_true),
                                       block_over_true=float(np.mean(se_block) / sd_true),
                                       passed=bool(abs(np.mean(se_block) / sd_true - 1) < abs(np.mean(se_naive) / sd_true - 1)))
    # ---- E: nested MoE
    rng = rng_for(1, 5)
    bl = [list(range(5 * i, 5 * i + 5)) for i in range(3)]
    delta = rng.normal(0, tau, 15)
    offs = np.repeat([0.004, -0.003, 0.001], 5)
    dW = center(delta, bl)
    true_within = float((dW ** 2).sum() / 12)
    vw, vflat = [], []
    for r in range(100):
        X = synth_data(rng, 128, 15, delta + offs)
        U = Units(X, np.arange(128))
        dr = U.theta(draw_counts(rng, 128, 500))
        vo, vsm = vstats_from(X.mean(0), dr, bl); vw.append(vo - vsm)
        vo, vsm = vstats_from(X.mean(0), dr, [list(range(15))]); vflat.append(vo - vsm)
    out["E_nested_moe"] = dict(true_within_seed_V=true_within, mean_V_corr_within=float(np.mean(vw)),
                               mean_V_corr_flattened=float(np.mean(vflat)),
                               ratio_within=float(np.mean(vw) / true_within), ratio_flat=float(np.mean(vflat) / true_within),
                               passed=bool(abs(np.mean(vw) / true_within - 1) < 0.3 and np.mean(vflat) > 5 * true_within))
    # ---- F: exact null duplicate
    rng = rng_for(1, 6)
    X = synth_data(rng, 128, 5, np.zeros(5))
    X[:, 4] = X[:, 0]          # duplicate reference (variant 4 == reference 0 bitwise)
    ref = X[:, 0]
    C = X[:, 1:] - ref[:, None]
    U = Units(C, np.arange(128))
    dr = U.theta(draw_counts(rng, 128, 1000))
    M = split_masks(rng, "window", "C4HO", 200)
    cA, cB = half_thetas(C, M)
    sa_incl = (np.sign(cA) * np.sign(cB) > 0).mean(1)
    sa_excl = (np.sign(cA[:, :3]) * np.sign(cB[:, :3]) > 0).mean(1)
    out["F_exact_null"] = dict(null_contrast_exactly_zero_all_draws=bool((dr[:, 3] == 0).all()),
                               null_contrast_sign_in_halves=float(np.abs(np.sign(cA[:, 3])).max()),
                               sign_agreement_including_null_counts_it_as=("disagreement (sign 0)"),
                               mean_sign_agreement_incl=float(sa_incl.mean()), mean_sign_agreement_excl=float(sa_excl.mean()),
                               rule="null contrasts are excluded from sign denominators; correlations use the 5 grouping "
                                    "values (the duplicate is one realized grouping, not two)",
                               passed=bool((dr[:, 3] == 0).all()))
    out["all_passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = time.time() - t0
    jdump(out, p("statistical_validation.json"))
    log(f"[synth] all passed {out['all_passed']}: " + ", ".join(f"{k}={v['passed']}" for k, v in out.items() if isinstance(v, dict)))


# =====================================================================================================================
# stage: analyze
# =====================================================================================================================
def schemes(ds):
    return WT2_SCHEMES if ds == "WT2" else C4_SCHEMES


def primary_scheme(ds):
    return schemes(ds)[0]


def var_block(S, si, ds, di, sch, sj, nested_on):
    X = S["X"][ds]
    blocks = S["blocks"]
    U = Units(X, units_for(sch, ds))
    th = X.mean(0)
    rng = rng_for(2, si, di, sj)
    draws = U.theta(draw_counts(rng, U.m, B_COV))
    cov = np.cov(draws, rowvar=False)
    vo, vs = vstats_from(th, draws, blocks)
    vc = vo - vs
    r = dict(scheme=sch, n_units=int(U.m), V_obs=vo, V_samp=vs, V_corr=vc, V_corr_constrained=max(vc, 0.0),
             SD_obs=math.sqrt(vo), SD_corr=math.sqrt(vc) if vc > 0 else None,
             sampling_fraction=vs / vo if vo > 0 else None, df=df_of(blocks))
    ccov = dict(scheme=sch, n_units=int(U.m), cols=S["cols"], theta=th, cov=cov,
                se=np.sqrt(np.diag(cov)), corr=cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))))
    if nested_on:
        rng = rng_for(3, si, di, sj)
        o, s_, c_ = nested(U, blocks, rng, B_OUTER, B_INNER)
        d = c_ - vo                       # V_corr* - V_obs (bootstrap-world truth = V_obs)
        lo, hi = vc - np.quantile(d, .975), vc - np.quantile(d, .025)
        rngn = rng_for(4, si, di, sj)
        on, sn, cn = nested(U, blocks, rngn, B_OUTER, B_INNER, null=True, theta_hat=th)
        r.update(V_obs_ci=[float(np.quantile(o, .025)), float(np.quantile(o, .975))],
                 V_samp_ci=[float(np.quantile(s_, .025)), float(np.quantile(s_, .975))],
                 V_corr_pivot_ci=[float(lo), float(hi)],
                 V_corr_percentile_of_bootstrap=[float(np.quantile(c_, .025)), float(np.quantile(c_, .975))],
                 SD_corr_ci_upper=math.sqrt(hi) if hi > 0 else 0.0,
                 SD_corr_ci_lower=math.sqrt(lo) if lo > 0 else 0.0,
                 sampling_fraction_ci=[float(1 - hi / vo), float(1 - lo / vo)],
                 null_p=float((cn >= vc).mean()),
                 null_V_corr_q95=float(np.quantile(cn, .95)),
                 null_V_corr_draws_summary=dict(mean=float(cn.mean()), sd=float(cn.std())),
                 resolved=bool(lo > 0 and (cn >= vc).mean() < 0.05))
        r["_draws"] = dict(V_corr_star=c_, null=cn)
    return r, ccov


def split_block(S, si, ds, di, sch, sj, se_full, null_on):
    X, C, blocks = S["X"][ds], S["C"][ds], S["blocks"]
    rng = rng_for(5, si, di, sj)
    thr = np.abs(C.mean(0)) > se_full
    if sch == "chronological":
        M = np.zeros((1, NW), bool); M[0, :64] = True
    else:
        M = split_masks(rng, sch, ds, K_SPLIT)
    tA, tB = half_thetas(X, M)
    cA, cB = half_thetas(C, M)
    st = reliability_stats(center(tA, blocks), center(tB, blocks), cA, cB, blocks, se_full, thr)
    out = dict(scheme=sch, n_partitions=int(M.shape[0]), n_threshold_contrasts=int(thr.sum()),
               threshold_rule="|full-data contrast| > its primary-scheme bootstrap SE",
               half_sizes=[int(M[0].sum()), int((~M[0]).sum())])
    for k, v in st.items():
        if k == "per_seed_pearson":
            out[k] = {f"s{i}": summarize(v[:, i]) for i in range(v.shape[1])}
        else:
            out[k] = summarize(v)
    out["_pearson"] = st["pearson"]
    if null_on and sch != "chronological":
        th = X.mean(0)
        uid = units_for(sch, ds)
        U = Units(X, uid)
        S0 = U.S - U.n[:, None] * center(th, blocks)[None, :]
        rngn = rng_for(6, si, di, sj)
        med, allr = [], []
        for r in range(R_NULL):
            idx = rngn.integers(0, U.m, U.m)
            # expand to a window-level null dataset: each drawn unit contributes its windows' rows
            rows, orig = [], []
            for c, u in enumerate(idx):
                w = np.nonzero(U.inv == u)[0]
                rows.append(w); orig.append(np.full(len(w), c))   # each draw = its own split unit -> independent halves
            rows = np.concatenate(rows); orig = np.concatenate(orig)
            Xn = X[rows] - center(th, blocks)[None, :]
            Mn = split_masks(rngn, None, ds, K_NULL, uid=orig)
            a, b = half_thetas(Xn, Mn)
            rr = pearson(center(a, blocks), center(b, blocks))
            med.append(np.nanmedian(rr)); allr.append(rr)
        med = np.array(med)
        obs = out["pearson"]["median"]
        out["null"] = dict(R=R_NULL, K=K_NULL, null_median_pearson=summarize(med),
                           null_pooled_pearson=summarize(np.concatenate(allr)), observed_median=obs,
                           p_value=float((med >= obs).mean()))
        out["_null_pooled"] = np.concatenate(allr)
        out["_null_med"] = med
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def stage_analyze(a):
    prot = check_protocol()
    assert os.path.exists(p("statistical_validation.json")), "run synth first"
    t0 = time.time()
    L = load_matrix()
    S = analysis_sets(L)
    names = list(S)
    if a.sets:
        names = [n for n in names if n in a.sets.split(",")]
    cache_f = p("logs", "analyze_cache.npz")
    covj, varj, splj, nullj = {}, {}, {}, {}
    raw = {}
    for si, name in enumerate(S):
        if name not in names:
            continue
        St = S[name]
        partf = p("logs", "parts", f"{name}.json")
        if os.path.exists(partf) and not a.force:
            d = jload(partf)
            covj[name], varj[name], splj[name] = d["cov"], d["var"], d["split"]
            rawf = p("logs", "parts", f"{name}.npz")
            if os.path.exists(rawf):
                raw[name] = dict(np.load(rawf))
            log(f"[analyze] {name}: cached")
            continue
        covj[name], varj[name], splj[name] = {}, {}, {}
        rw = {}
        for di, ds in enumerate(DS):
            covj[name][ds], varj[name][ds], splj[name][ds] = {}, {}, {}
            for sj, sch in enumerate(schemes(ds)):
                r, c = var_block(St, si, ds, di, sch, sj, nested_on=True)
                d_ = r.pop("_draws", None)
                if d_ is not None:
                    rw[f"{ds}_{sch}_Vcorr_star"] = d_["V_corr_star"]; rw[f"{ds}_{sch}_Vcorr_null"] = d_["null"]
                varj[name][ds][sch] = r
                covj[name][ds][sch] = c
                log(f"[analyze] {name} {ds} {sch}: Vobs {r['V_obs']:.3e} Vsamp {r['V_samp']:.3e} Vcorr {r['V_corr']:.3e} "
                    f"CI {r.get('V_corr_pivot_ci')} p {r.get('null_p')}")
            # contrast SEs (primary scheme) for threshold rule
            Uc = Units(St["C"][ds], units_for(primary_scheme(ds), ds))
            se_c = Uc.theta(draw_counts(rng_for(7, si, di), Uc.m, B_COV)).std(0, ddof=1)
            for sj, sch in enumerate(schemes(ds) + (["chronological"] if ds == "WT2" else [])):
                sp = split_block(St, si, ds, di, sch, sj, se_c, null_on=(sch == primary_scheme(ds)))
                if "_null_pooled" in sp:
                    rw[f"{ds}_{sch}_null_pooled"] = sp.pop("_null_pooled"); rw[f"{ds}_{sch}_null_med"] = sp.pop("_null_med")
                rw[f"{ds}_{sch}_pearson"] = sp.pop("_pearson")
                splj[name][ds][sch] = sp
                log(f"[analyze] {name} {ds} split {sch}: median r {sp['pearson']['median'] if sp['pearson'] else None} "
                    f"null p {sp.get('null', {}).get('p_value')}")
        os.makedirs(p("logs", "parts"), exist_ok=True)
        jdump(dict(cov=covj[name], var=varj[name], split=splj[name]), partf)
        np.savez(p("logs", "parts", f"{name}.npz"), **rw)
        raw[name] = rw
    if a.sets:
        log("[analyze] partial run (--sets); final files not written")
        return

    # ---- cross-dataset
    cross = {}
    for si, name in enumerate(S):
        St = S[name]
        bl = St["blocks"]
        th = {ds: St["X"][ds].mean(0) for ds in DS}
        z = {ds: center(th[ds], bl) for ds in DS}
        cvec = {ds: St["C"][ds].mean(0) for ds in DS}
        r_obs = float(pearson(z["WT2"], z["C4HO"]))
        if len(bl) > 1:
            rk = {ds: np.concatenate([rankdata(z[ds][b]) for b in bl]) for ds in DS}
        else:
            rk = {ds: rankdata(z[ds]) for ds in DS}
        sp_obs = float(pearson(rk["WT2"], rk["C4HO"]))
        pl = pair_list(bl)
        po = float(np.mean([np.sign(z["WT2"][i] - z["WT2"][j]) * np.sign(z["C4HO"][i] - z["C4HO"][j]) > 0 for i, j in pl]))
        sg = float(np.mean(np.sign(cvec["WT2"]) * np.sign(cvec["C4HO"]) > 0))
        # bootstrap CI (independent datasets, primary schemes)
        rng = rng_for(8, si)
        dr = {}
        for ds in DS:
            U = Units(St["X"][ds], units_for(primary_scheme(ds), ds))
            dr[ds] = center(U.theta(draw_counts(rng, U.m, B_CROSS)), bl)
        rb = pearson(dr["WT2"], dr["C4HO"])
        lam = {}
        for ds in DS:
            v = varj[name][ds][primary_scheme(ds)]
            lam[ds] = v["V_corr"] / v["V_obs"]
        both_pos = all(varj[name][ds][primary_scheme(ds)].get("V_corr_pivot_ci", [0])[0] > 0 for ds in DS)
        exp_r = math.sqrt(lam["WT2"] * lam["C4HO"]) if lam["WT2"] > 0 and lam["C4HO"] > 0 else None
        sh = {ds: splj[name][ds][primary_scheme(ds)]["pearson"]["median"] for ds in DS}
        sb = {ds: (2 * sh[ds] / (1 + sh[ds]) if sh[ds] is not None and sh[ds] > -1 else None) for ds in DS}
        cross[name] = dict(
            pearson_full_WT2_vs_C4=r_obs, pearson_bootstrap_ci=[float(np.nanquantile(rb, .025)), float(np.nanquantile(rb, .975))],
            spearman_full=sp_obs, pair_order_agreement=po, n_pairs=len(pl),
            sign_agreement_contrasts=sg, n_contrasts=int(len(cvec["WT2"])),
            sign_disagreements=[St["contrast_ids"][i] for i in range(len(cvec["WT2"])) if np.sign(cvec["WT2"][i]) != np.sign(cvec["C4HO"][i])],
            split_half_median_pearson=sh, spearman_brown_full_length_reliability=sb,
            reliability_lambda=lam, expected_cross_r_if_identical_true_effects=exp_r,
            disattenuated_cross_r=(r_obs / exp_r if (exp_r and both_pos) else None),
            disattenuation_reported=bool(exp_r and both_pos),
            note=("full-vs-full correlation of centred effects (MoE: within-seed centred); bootstrap resamples both datasets "
                  "independently with the primary schemes; the 15 MoE / 5 dense values are not independent evidence"))
        raw.setdefault(name, {})["cross_boot"] = rb

    # ---- MoE window CIs
    moe = S["moe"]
    mci = dict(pairs={}, counts={})
    for di, ds in enumerate(DS):
        for sj, sch in enumerate(schemes(ds)):
            uid = units_for(sch, ds)
            rng = rng_for(9, di, sj)
            U = Units(moe["X"][ds], uid)
            Cn = draw_counts(rng, U.m, B_COV)
            dD = U.theta(Cn)
            UR = Units(moe["FR"][ds], uid); UC = Units(moe["FC"][ds], uid)
            dP = np.exp(UC.theta(Cn)) - np.exp(UR.theta(Cn))
            th = moe["X"][ds].mean(0)
            pth = np.exp(moe["FC"][ds].mean(0)) - np.exp(moe["FR"][ds].mean(0))
            for g, c in enumerate(moe["cols"]):
                e = mci["pairs"].setdefault(c, {}).setdefault(ds, {})
                lo, hi = np.quantile(dD[:, g], [.025, .975])
                plo, phi = np.quantile(dP[:, g], [.025, .975])
                e[sch] = dict(mean_nll_diff=float(th[g]), ci=[float(lo), float(hi)], se=float(dD[:, g].std(ddof=1)),
                              excludes_zero=bool(lo > 0 or hi < 0), sign=int(np.sign(th[g])),
                              ppl_diff=float(pth[g]), ppl_diff_ci=[float(plo), float(phi)],
                              fr_better=bool(th[g] > 0))
            mci["counts"].setdefault(ds, {})[sch] = dict(
                n_excluding_zero=int(sum(mci["pairs"][c][ds][sch]["excludes_zero"] for c in moe["cols"])),
                median_ci_width=float(np.median([np.diff(mci["pairs"][c][ds][sch]["ci"])[0] for c in moe["cols"]])),
                median_se=float(np.median([mci["pairs"][c][ds][sch]["se"] for c in moe["cols"]])))
            # execution-averaged comparison: grand mean of the 15 paired effects (seed means averaged)
            gm = dD.mean(1)
            mci["counts"][ds][sch]["grand_mean_nll_diff"] = float(th.mean())
            mci["counts"][ds][sch]["grand_mean_eval_only_ci"] = [float(np.quantile(gm, .025)), float(np.quantile(gm, .975))]
            raw.setdefault("moe", {})[f"{ds}_{sch}_pair_se"] = dD.std(0, ddof=1)
    for c in moe["cols"]:
        w, h = mci["pairs"][c]["WT2"]["article"], mci["pairs"][c]["C4HO"]["window"]
        mci["pairs"][c]["point_signs_agree"] = w["sign"] == h["sign"]
        mci["pairs"][c]["opposite_significant"] = bool(w["excludes_zero"] and h["excludes_zero"] and w["sign"] != h["sign"])
    mci["summary"] = dict(
        WT2_excluding_zero={s: mci["counts"]["WT2"][s]["n_excluding_zero"] for s in WT2_SCHEMES},
        C4HO_excluding_zero=mci["counts"]["C4HO"]["window"]["n_excluding_zero"],
        point_sign_disagreements=int(sum(not mci["pairs"][c]["point_signs_agree"] for c in moe["cols"])),
        opposite_significant_directions=int(sum(mci["pairs"][c]["opposite_significant"] for c in moe["cols"])),
        WT2_ci_width_ratio_vs_naive_window={s: mci["counts"]["WT2"][s]["median_ci_width"] / mci["counts"]["WT2"]["window"]["median_ci_width"]
                                            for s in WT2_SCHEMES},
        execution_dispersion={ds: dict(within_seed_SD_obs=varj["moe"][ds][primary_scheme(ds)]["SD_obs"],
                                       corrected_SD=varj["moe"][ds][primary_scheme(ds)]["SD_corr"],
                                       corrected_V=varj["moe"][ds][primary_scheme(ds)]["V_corr"],
                                       median_window_SE_of_one_pair=mci["counts"][ds][primary_scheme(ds)]["median_se"],
                                       ratio_SDobs_over_SE=varj["moe"][ds][primary_scheme(ds)]["SD_obs"] /
                                       mci["counts"][ds][primary_scheme(ds)]["median_se"]) for ds in DS},
        primary_schemes=dict(WT2="article", C4HO="window"),
        distinctions=("(i) a pair CI = uncertainty of ONE fixed-weight FR/FC comparison over evaluation text; (ii) the "
                      "within-seed SD of the 15 paired effects = variability across calibration executions, which still "
                      "contains evaluation noise (V_samp); (iii) the uncertainty of the execution-averaged comparison "
                      "needs both: eval-only CI of the grand mean plus between-execution variance / 15"))
    for ds in DS:
        vcor = max(varj["moe"][ds][primary_scheme(ds)]["V_corr"], 0.0)
        ci = mci["counts"][ds][primary_scheme(ds)]["grand_mean_eval_only_ci"]
        se_eval = (ci[1] - ci[0]) / 3.92
        mci["summary"].setdefault("execution_averaged", {})[ds] = dict(
            grand_mean=mci["counts"][ds][primary_scheme(ds)]["grand_mean_nll_diff"], eval_only_se=se_eval,
            between_execution_se_from_Vcorr_constrained=math.sqrt(vcor / 15),
            combined_se=math.sqrt(se_eval ** 2 + vcor / 15),
            note="constrained (>= 0) V_corr used for the combined SE; seed-level variance is not separately identified")

    # ---- write files
    jdump(strip(covj), p("joint_bootstrap_covariance.json"))
    vc_out = {n: {ds: {sch: {k: v for k, v in r.items() if not k.endswith("_ci") and k not in
                             ("null_p", "null_V_corr_q95", "null_V_corr_draws_summary", "resolved", "SD_corr_ci_upper",
                              "SD_corr_ci_lower", "V_corr_percentile_of_bootstrap")}
                       for sch, r in varj[n][ds].items()} for ds in DS} for n in varj}
    jdump(dict(definition=prot["estimand"], units="nats/token squared (mean-NLL scale)", sets=vc_out,
               primary_schemes=dict(WT2="article", C4HO="window")), p("variance_components.json"))
    unc = {n: {ds: {sch: {k: r.get(k) for k in ("V_obs", "V_obs_ci", "V_samp", "V_samp_ci", "V_corr", "V_corr_pivot_ci",
                                                "V_corr_percentile_of_bootstrap", "V_corr_constrained", "SD_corr",
                                                "SD_corr_ci_lower", "SD_corr_ci_upper", "sampling_fraction",
                                                "sampling_fraction_ci", "null_p", "null_V_corr_q95", "resolved", "n_units")}
                    for sch, r in varj[n][ds].items()} for ds in DS} for n in varj}
    jdump(dict(method=prot["bootstrap"], sets=unc,
               caveat=("the pivot interval treats V_corr* - V_obs as approximately pivotal; with 47 WT2 articles / 128 "
                       "C4 documents and df = 4-12 it is approximate; b = 16 has 8 units and is unstable")),
          p("variance_component_uncertainty.json"))
    jdump(dict(sets=strip(splj), note=prot["split_half"]["summary"]), p("split_half_reliability.json"))
    nulls = {n: {ds: dict(scheme=primary_scheme(ds), observed_median_pearson=splj[n][ds][primary_scheme(ds)]["pearson"]["median"],
                          **splj[n][ds][primary_scheme(ds)].get("null", {}),
                          variance_resolved=varj[n][ds][primary_scheme(ds)]["resolved"],
                          V_corr=varj[n][ds][primary_scheme(ds)]["V_corr"],
                          V_corr_pivot_ci=varj[n][ds][primary_scheme(ds)]["V_corr_pivot_ci"])
                 for ds in DS} for n in splj}
    jdump(dict(construction=prot["split_half_null"]["construction"], sets=nulls), p("split_half_null_comparison.json"))
    jdump(dict(sets=cross, method=prot["cross_dataset"]), p("cross_dataset_reliability.json"))
    jdump(mci, p("moe_window_ci_analysis.json"))

    # dense / dose convenience files
    def setsummary(n):
        St = S[n]
        o = dict(columns=St["cols"], per_dataset={})
        for ds in DS:
            th = St["X"][ds].mean(0)
            o["per_dataset"][ds] = dict(
                mean_nll=dict(zip(St["cols"], th.tolist())), centred=dict(zip(St["cols"], center(th, St["blocks"]).tolist())),
                contrasts=dict(zip(St["contrast_ids"], St["C"][ds].mean(0).tolist())),
                variance={sch: unc[n][ds][sch] for sch in unc[n][ds]},
                split_half_primary={k: v for k, v in strip(splj[n][ds][primary_scheme(ds)]).items()},
                split_half_sensitivity={sch: strip(splj[n][ds][sch])["pearson"] for sch in splj[n][ds]})
        o["cross_dataset"] = cross[n]
        return o
    dense = {}
    for b in ("dense4", "dense3"):
        o = setsummary(b)
        for ds in DS:
            o["per_dataset"][ds]["ref_ORIGINAL_mean_nll"] = float(S[b]["ref"][ds].mean())
            o["per_dataset"][ds]["G20261115_minus_ORIGINAL_exact"] = float((S[b]["X"][ds][:, 4] - S[b]["ref"][ds]).mean())
        dense[b] = o
    dense["note"] = ("theta_g - theta_ORIGINAL = contrasts; across-grouping variance uses the 5 groupings centred by their own "
                     "mean (ORIGINAL not double counted); sign agreement uses the 4 non-null groupings")
    jdump(dense, p("dense_grouping_uncertainty.json"))
    dose = {n: setsummary(n) for n in S if n.startswith("dose_")}
    dose["note"] = "each k analysed separately (8 replicate columns); fixed act-order = exploratory"
    jdump(dose, p("dense_dose_uncertainty.json"))

    # ---- decision
    dec = dict(rules=prot["decision_rules"], settings={})
    for n in S:
        per = {}
        for ds in DS:
            v = varj[n][ds][primary_scheme(ds)]
            sp = splj[n][ds][primary_scheme(ds)]
            rel = bool(sp["pearson"]["median"] >= 0.5 and sp["null"]["p_value"] < 0.05)
            per[ds] = dict(resolved_variance=v["resolved"], V_corr=v["V_corr"], V_corr_ci=v["V_corr_pivot_ci"],
                           null_p_variance=v["null_p"], median_split_half_r=sp["pearson"]["median"],
                           split_half_null_p=sp["null"]["p_value"], reliable_within=rel,
                           sampling_fraction=v["sampling_fraction"])
        good = {ds: per[ds]["resolved_variance"] and per[ds]["reliable_within"] for ds in DS}
        none = {ds: (not per[ds]["resolved_variance"]) and (not per[ds]["reliable_within"]) for ds in DS}
        cr = cross[n]
        if all(good.values()):
            exp_r = cr["expected_cross_r_if_identical_true_effects"]
            verdict = ("A-DISTRIBUTION-DEPENDENT-EFFECTS-SUPPORTED" if exp_r and cr["pearson_bootstrap_ci"][1] < exp_r
                       else "D-INCONCLUSIVE")
        elif all(none.values()):
            verdict = "B-SAMPLING-UNCERTAINTY-DOMINATES"
        elif (good["WT2"] and none["C4HO"]) or (good["C4HO"] and none["WT2"]):
            verdict = "C-MIXED"
        else:
            verdict = "D-INCONCLUSIVE"
        dec["settings"][n] = dict(per_dataset=per, verdict=verdict,
                                  cross_r=cr["pearson_full_WT2_vs_C4"], cross_r_ci=cr["pearson_bootstrap_ci"])
    dec["verdicts"] = dict(moe=dec["settings"]["moe"]["verdict"], dense4=dec["settings"]["dense4"]["verdict"],
                           dense3=dec["settings"]["dense3"]["verdict"],
                           dose_natural={f"k{k}": dec["settings"][f"dose_natural_k{k}"]["verdict"] for k in DOSES},
                           dose_fixed_exploratory={f"k{k}": dec["settings"][f"dose_fixed_k{k}"]["verdict"] for k in DOSES})
    jdump(dec, p("decision.json"))
    np.savez(p("logs", "figure_data.npz"), **{f"{n}__{k}": v for n, d in raw.items() for k, v in d.items()})
    log(f"[analyze] done in {time.time() - t0:.0f}s; verdicts {dec['verdicts']}")


# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "protocol", "synth", "analyze", "all"])
    ap.add_argument("--sets", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(p("logs"), exist_ok=True)
    st = ["audit", "protocol", "synth", "analyze"] if a.stage == "all" else [a.stage]
    for s in st:
        dict(audit=stage_audit, protocol=stage_protocol, synth=stage_synth, analyze=stage_analyze)[s](a)


if __name__ == "__main__":
    main()
