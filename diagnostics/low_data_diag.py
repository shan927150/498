"""Diagnostics for the E1 16->32 anomaly (held-out error gets worse for ~90% of
experts when calibration goes from 16 to 32 sequences). DIAGNOSTIC ONLY:
nothing here feeds E1/E2/E3, nothing is written under runs/, and the final
test split is never profiled (its ids are only used to check disjointness).

Subcommands (run in this order; each is resumable or cheap to rerun):
  invariants  A: partial-sum equivalence, Hessian scale invariance, draw integrity
  nested      B: strictly nested prefixes C16 < C32 < ... < C256, absolute tuning error
  calibtune   C: q16/q32/q64 error on calibration sets, the new chunks, unions, tuning
  damp        D: 16->32 (and 32->64) under percdamp x actorder (production unchanged)
  report      print tables from runs_diag/*

Error definitions
  tune_e1  : E1's metric -- mean over tuning sequences of
             sum_proj ||X_s (W-Wq)^T||^2 / sum_proj ||X_s W^T||^2
             (first 128 routed tokens per sequence, Profiler default).
  *_h      : the same relative error pooled over ALL tokens of a set S,
             computed exactly from S's Hessian:
             sum_proj tr(D H_S D^T) / sum_proj tr(W H_S W^T), D = W - Wq.
             Identical definition for calibration sets, new chunks and tuning.
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e1_label_sufficiency import label_expert                       # noqa: E402
from moe_calib import model_utils as MU                             # noqa: E402
from moe_calib.data_utils import load_meta, load_layer_cache, split_ids  # noqa: E402
from moe_calib.gptq import gptq_quantize                            # noqa: E402
from moe_calib.profiler import Profiler                             # noqa: E402
from moe_calib.quant_eval import fp_weights, quantize_from_H, per_seq_rel_errors  # noqa: E402

PROJ = ("gate_proj", "up_proj", "down_proj")
HKEY = {"gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}
PREFIXES = [16, 32, 64, 128, 256]
ORDER_SALT = 7_000_003          # nested ordering: Random(ORDER_SALT + seed).sample(pool, 256)
OUT = "runs_diag"


# ---------------------------------------------------------------- helpers

def splits(cache_dir):
    meta = load_meta(cache_dir)
    sp = split_ids(meta)
    return meta, sp


def nested_order(pool, seed):
    return random.Random(ORDER_SALT + seed).sample(pool, PREFIXES[-1])


def hessians(model, L, cache, ids, K=1):
    """Per expert {"gu": H, "down": H} summed over K blocks (production
    Profiler + BlockHessianAccumulator), and routed-token counts."""
    P = Profiler(model, L, K=K).run(cache, ids)
    E, ks = MU.num_experts(model), list(range(K))
    H = {e: {p: P.acc.H_sum((L, e, p), ks).clone() for p in ("gu", "down")} for e in range(E)}
    tok = {e: P.acc.tokens((L, e, "gu")) for e in range(E)}
    P.free()
    del P
    return H, tok


def tune_holdout(model, L, cache, tune):
    return Profiler(model, L, collect_hessian=False, collect_holdout=True).run(cache, tune).holdout


def gpu_weights(model, L, e):
    return {k: v.to(MU.GPU) for k, v in fp_weights(model, L, e).items()}


def quant(W, H, percdamp=0.01, actorder=True, bits=4, gs=128):
    return {n: gptq_quantize(W[n], H[HKEY[n]], bits, gs, percdamp, actorder) for n in PROJ}


def rtn(W, bits=4, gs=128):
    """Production zero-token fallback: H = I, act-order off == per-group RTN."""
    return {n: gptq_quantize(W[n], torch.eye(W[n].shape[1], device=MU.GPU), bits, gs,
                             actorder=False) for n in PROJ}


@torch.no_grad()
def hrel(W, Q, H):
    num = den = 0.0
    for n in PROJ:
        Hp = H[HKEY[n]].to(MU.GPU)
        D = W[n] - Q[n].to(MU.GPU)
        num += float(((D @ Hp) * D).sum())
        den += float(((W[n] @ Hp) * W[n]).sum())
    return num / den if den > 0 else float("nan")


def tune_err(model, L, e, Q, HO):
    d = per_seq_rel_errors(model, L, e, Q, HO)
    return float(np.mean(list(d.values()))) if d else float("nan"), d


def relF(A, B):
    return float(torch.linalg.norm(A - B) / torch.linalg.norm(B))


def qdiff(Qa, Qb, W):
    """fraction of differing quantized entries, max |diff| / max |W|."""
    n = tot = 0
    mx = 0.0
    for k in PROJ:
        a, b = Qa[k].to(MU.GPU), Qb[k].to(MU.GPU)
        ne = (a != b)
        n += int(ne.sum()); tot += a.numel()
        mx = max(mx, float((a - b).abs().max() / W[k].abs().max()))
    return n / tot, mx


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def e1_dir(seed, bits=4):
    d = glob.glob(f"runs/e1_b{bits}_s{seed}_*")
    assert len(d) == 1, d
    return d[0]


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()))


# ------------------------------------------------------------- A: invariants

def draw_integrity(meta, sp):
    pool, tune, test = set(sp["pool"]), set(sp["tune"]), set(sp["test"])
    z = np.load("runs/routing_profile.npz")
    C, layers, E = z["C"], z["layers"].tolist(), int(z["E"])
    row = {s: i for i, s in enumerate(z["pool_ids"].tolist())}
    out = []
    for seed in (0, 1, 2):
        d = e1_dir(seed)
        draws = {int(k): v for k, v in json.load(open(os.path.join(d, "draws.json"))).items()}
        raw = [r for r in read_jsonl(os.path.join(d, "raw.jsonl")) if not r.get("marker")]
        tok_raw = {(r["layer"], r["expert"], r["n0"], r["rep"]): r["tokens_n0"] for r in raw}
        for n0, dr in draws.items():
            base = dr["base"]
            res = dict(char_seed=seed, n0=n0, base_size=len(base), base_unique=len(set(base)) == len(base),
                       base_in_pool=set(base) <= pool, base_tune_overlap=len(set(base) & tune),
                       base_test_overlap=len(set(base) & test))
            nb = C[[row[s] for s in base]].sum(0)
            res["ext"] = []
            for r, ext in enumerate(dr["ext"]):
                nx = C[[row[s] for s in ext]].sum(0)
                mism = 0
                for li, L in enumerate(layers):
                    for e in range(E):
                        if (L, e, n0, r) in tok_raw and tok_raw[(L, e, n0, r)] != nb[li * E + e]:
                            mism += 1
                res["ext"].append(dict(
                    rep=r, size=len(ext), unique=len(set(ext)) == len(ext),
                    base_overlap=len(set(base) & set(ext)), in_pool=set(ext) <= pool,
                    tune_overlap=len(set(ext) & tune), test_overlap=len(set(ext) & test),
                    other_ext_overlap=sum(len(set(ext) & set(o)) for j, o in enumerate(dr["ext"]) if j != r),
                    experts_not_strictly_increasing=int((nx == 0).sum()),
                    min_ext_tokens=int(nx.min()),
                    raw_tokens_n0_mismatch_vs_routing=mism))
            out.append(res)
    return out


def cmd_invariants(a):
    meta, sp = splits(a.cache_dir)
    res = dict(draw_integrity=draw_integrity(meta, sp))
    L, n0, rep, seed = a.layer, 16, 0, 0
    d = e1_dir(seed)
    draws = {int(k): v for k, v in json.load(open(os.path.join(d, "draws.json"))).items()}
    base, ext = draws[n0]["base"], draws[n0]["ext"][rep]
    raw = {(r["layer"], r["expert"]): r for r in read_jsonl(os.path.join(d, "raw.jsonl"))
           if not r.get("marker") and r["n0"] == n0 and r["rep"] == rep}

    model, _ = MU.load_model()
    E = MU.num_experts(model)
    cache = load_layer_cache(a.cache_dir, L)
    HO = tune_holdout(model, L, cache, sp["tune"])
    HA, tA = hessians(model, L, cache, base, K=4)          # exactly E1's accumulators
    HB, tB = hessians(model, L, cache, ext, K=4)
    HD, tD = hessians(model, L, cache, base + ext, K=1)    # direct profile of the union
    per = []
    for e in range(E):
        W = gpu_weights(model, L, e)
        HS = {p: HA[e][p] + HB[e][p] for p in ("gu", "down")}
        r = dict(expert=e, tokens_base=tA[e], tokens_ext=tB[e], tokens_direct=tD[e],
                 relF_gu=relF(HS["gu"], HD[e]["gu"]), relF_down=relF(HS["down"], HD[e]["down"]),
                 dead_cols_base_gu=int((torch.diag(HA[e]["gu"]) == 0).sum()),
                 dead_cols_base_down=int((torch.diag(HA[e]["down"]) == 0).sum()))
        q0 = quantize_from_H(model, L, e, HA[e]["gu"], HA[e]["down"])   # production path
        qS = quantize_from_H(model, L, e, HS["gu"], HS["down"])
        qD = quantize_from_H(model, L, e, HD[e]["gu"], HD[e]["down"])
        r["helper_equals_production"] = qdiff(quant(W, HA[e]), q0, W)[0] == 0.0
        r["q_sum_vs_direct_frac_diff"], r["q_sum_vs_direct_maxdiff"] = qdiff(qS, qD, W)
        e0, d0 = tune_err(model, L, e, q0, HO)
        eS, dS = tune_err(model, L, e, qS, HO)
        eD, dD = tune_err(model, L, e, qD, HO)
        r.update(tune_err_n0=e0, tune_err_2n0_sum=eS, tune_err_2n0_direct=eD,
                 same_tune_seqs=sorted(d0) == sorted(dS) == sorted(dD))
        lab, _ = label_expert(d0, dS, tA[e], 1, 8)
        r["rel_delta_recomputed"] = lab["rel_delta"] if lab else None
        r["rel_delta_stored_e1"] = raw.get((L, e), {}).get("rel_delta")
        r["tokens_stored_e1"] = raw.get((L, e), {}).get("tokens_n0")
        # Hessian scale invariance (relative damping) at low data and at 2n0
        for tag, H in (("n0", HA[e]), ("2n0", HS)):
            q1 = quant(W, H)
            e1v, _ = tune_err(model, L, e, q1, HO)
            for s in (2, 10):
                qs = quant(W, {p: s * H[p] for p in H})
                es, _ = tune_err(model, L, e, qs, HO)
                fd, md = qdiff(q1, qs, W)
                r[f"scale{s}_{tag}_frac_diff"] = fd
                r[f"scale{s}_{tag}_maxdiff"] = md
                r[f"scale{s}_{tag}_tune_err_reldiff"] = (es - e1v) / e1v
        per.append(r)
        print(f"[inv] L{L} e{e}: relF gu={r['relF_gu']:.2e} down={r['relF_down']:.2e} "
              f"qdiff={r['q_sum_vs_direct_frac_diff']:.2e} scale10 n0 frac={r['scale10_n0_frac_diff']:.2e} "
              f"rd stored={r['rel_delta_stored_e1']} recomputed={r['rel_delta_recomputed']}", flush=True)
    res.update(layer=L, n0=n0, rep=rep, char_seed=seed, per_expert=per)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/invariants.json", "w"), indent=1)


# --------------------------------------------------------------- B: nested

def cmd_nested(a):
    meta, sp = splits(a.cache_dir)
    path = f"{OUT}/nested.jsonl"
    os.makedirs(OUT, exist_ok=True)
    done = {(r["seed"], r["layer"]) for r in read_jsonl(path) if r.get("marker")}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    for L in a.layers:
        todo = [s for s in a.seeds if (s, L) not in done]
        if not todo:
            continue
        cache = load_layer_cache(a.cache_dir, L)
        HO = tune_holdout(model, L, cache, sp["tune"])
        HT, _ = hessians(model, L, cache, sp["tune"], K=1)
        Ws = {e: gpu_weights(model, L, e) for e in range(E)}
        R = {e: rtn(Ws[e]) for e in range(E)}
        rtn_tune = [tune_err(model, L, e, R[e], HO)[0] for e in range(E)]
        rtn_tune_h = [hrel(Ws[e], R[e], HT[e]) for e in range(E)]
        for seed in todo:
            order = nested_order(sp["pool"], seed)
            run, tok, lo, recs = None, None, 0, []
            for n in PREFIXES:
                Hc, tc = hessians(model, L, cache, order[lo:n], K=1)
                if run is None:
                    run, tok = Hc, tc
                else:
                    for e in range(E):
                        for p in ("gu", "down"):
                            run[e][p] += Hc[e][p]
                        tok[e] += tc[e]
                    del Hc
                rec = dict(seed=seed, layer=L, n=n, tokens=[tok[e] for e in range(E)],
                           tune_err=[], tune_err_h=[], calib_err_h=[], rtn_calib_err_h=[],
                           rtn_tune_err=rtn_tune, rtn_tune_err_h=rtn_tune_h)
                for e in range(E):
                    q = quant(Ws[e], run[e])
                    rec["tune_err"].append(tune_err(model, L, e, q, HO)[0])
                    rec["tune_err_h"].append(hrel(Ws[e], q, HT[e]))
                    rec["calib_err_h"].append(hrel(Ws[e], q, run[e]))
                    rec["rtn_calib_err_h"].append(hrel(Ws[e], R[e], run[e]))
                recs.append(rec)
                print(f"[nested] seed={seed} L={L} n={n}: mean tune_err={np.mean(rec['tune_err']):.5f} "
                      f"calib_h={np.mean(rec['calib_err_h']):.5f} tokens med={np.median(rec['tokens']):.0f}",
                      flush=True)
                lo = n
            del run
            with open(path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.write(json.dumps(dict(seed=seed, layer=L, marker=True)) + "\n")
        del cache, HO, HT, Ws, R
        torch.cuda.empty_cache()


# ----------------------------------------------------- C: calibration vs tune

def cmd_calibtune(a):
    meta, sp = splits(a.cache_dir)
    path = f"{OUT}/calibtune.jsonl"
    os.makedirs(OUT, exist_ok=True)
    done = {(r["seed"], r["layer"]) for r in read_jsonl(path)}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    for L in a.layers:
        for seed in a.seeds:
            if (seed, L) in done:
                continue
            cache = load_layer_cache(a.cache_dir, L)
            order = nested_order(sp["pool"], seed)
            HO = tune_holdout(model, L, cache, sp["tune"])
            HT, _ = hessians(model, L, cache, sp["tune"], K=1)
            Ha, ta = hessians(model, L, cache, order[0:16], K=1)
            Hb, tb = hessians(model, L, cache, order[16:32], K=1)
            Hc, tc = hessians(model, L, cache, order[32:64], K=1)
            sets = ["C16", "S17_32", "C32", "S33_64", "C64", "tune"]
            out = {q: {s: [] for s in sets + ["tune_e1"]} for q in ("q16", "q32", "q64", "rtn")}
            for e in range(E):
                W = gpu_weights(model, L, e)
                H = dict(C16=Ha[e], S17_32=Hb[e], S33_64=Hc[e], tune=HT[e])
                H["C32"] = {p: Ha[e][p] + Hb[e][p] for p in ("gu", "down")}
                H["C64"] = {p: H["C32"][p] + Hc[e][p] for p in ("gu", "down")}
                Q = dict(q16=quant(W, H["C16"]), q32=quant(W, H["C32"]), q64=quant(W, H["C64"]),
                         rtn=rtn(W))
                for qn, q in Q.items():
                    for s in sets:
                        out[qn][s].append(hrel(W, q, H[s]))
                    out[qn]["tune_e1"].append(tune_err(model, L, e, q, HO)[0])
            rec = dict(seed=seed, layer=L, tokens=dict(C16=[ta[e] for e in range(E)],
                                                      S17_32=[tb[e] for e in range(E)],
                                                      S33_64=[tc[e] for e in range(E)]), err=out)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[calibtune] seed={seed} L={L} done", flush=True)
            del cache, HO, HT, Ha, Hb, Hc
            torch.cuda.empty_cache()


# ---------------------------------------------------- D: damping / act-order

def cmd_damp(a):
    meta, sp = splits(a.cache_dir)
    path = f"{OUT}/damp.jsonl"
    os.makedirs(OUT, exist_ok=True)
    done = {(r["seed"], r["layer"], r["percdamp"], r["actorder"]) for r in read_jsonl(path)}
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    L, seed = a.layer, a.seed
    cache = load_layer_cache(a.cache_dir, L)
    order = nested_order(sp["pool"], seed)
    HO = tune_holdout(model, L, cache, sp["tune"])
    HT, _ = hessians(model, L, cache, sp["tune"], K=1)
    H16, t16 = hessians(model, L, cache, order[:16], K=1)
    Hb, _ = hessians(model, L, cache, order[16:32], K=1)
    Hc, _ = hessians(model, L, cache, order[32:64], K=1)
    H32 = {e: {p: H16[e][p] + Hb[e][p] for p in ("gu", "down")} for e in range(E)}
    H64 = {e: {p: H32[e][p] + Hc[e][p] for p in ("gu", "down")} for e in range(E)}
    del Hb, Hc
    for pd in a.percdamp:
        for ao in (True, False):
            if (seed, L, pd, ao) in done:
                continue
            rec = dict(seed=seed, layer=L, percdamp=pd, actorder=ao,
                       tokens16=[t16[e] for e in range(E)],
                       err16=[], err32=[], err64=[], rel_delta_16_32=[], rel_delta_32_64=[],
                       tune_h16=[], tune_h32=[], tune_h64=[],
                       calib_h16=[], calib_h32=[], calib_h64=[])
            for e in range(E):
                W = gpu_weights(model, L, e)
                ds = {}
                for n, H in ((16, H16[e]), (32, H32[e]), (64, H64[e])):
                    q = quant(W, H, percdamp=pd, actorder=ao)
                    m, ds[n] = tune_err(model, L, e, q, HO)
                    rec[f"err{n}"].append(m)
                    rec[f"tune_h{n}"].append(hrel(W, q, HT[e]))
                    rec[f"calib_h{n}"].append(hrel(W, q, H))
                for a0, a1 in ((16, 32), (32, 64)):
                    lab, _ = label_expert(ds[a0], ds[a1], 1, 1, 8)
                    rec[f"rel_delta_{a0}_{a1}"].append(lab["rel_delta"] if lab else None)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            rd = np.array(rec["rel_delta_16_32"], float)
            print(f"[damp] percdamp={pd} actorder={ao}: 16->32 mean rel_delta={np.nanmean(rd):+.4f} "
                  f"worse={np.mean(rd < 0):.2f}", flush=True)


# ------------------------------------------------------------------ report

def tr_stats(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                frac_improved=float((d > 0).mean()), frac_worsened=float((d < 0).mean()), n=int(d.size))


def cmd_report(a):
    rep = {}
    nested = [r for r in read_jsonl(f"{OUT}/nested.jsonl") if not r.get("marker")]
    if nested:
        key = lambda r: (r["seed"], r["layer"])
        units = sorted({key(r) for r in nested})
        by = {(key(r), r["n"]): r for r in nested}
        trans = list(zip(PREFIXES[:-1], PREFIXES[1:]))
        B = dict(units=units, abs_err={}, transitions={})
        for n in PREFIXES:
            rows = [by[(u, n)] for u in units if (u, n) in by]
            B["abs_err"][n] = dict(
                tune_err=stats(sum((r["tune_err"] for r in rows), [])),
                tune_err_h=stats(sum((r["tune_err_h"] for r in rows), [])),
                calib_err_h=stats(sum((r["calib_err_h"] for r in rows), [])),
                rtn_tune_err=stats(sum((r["rtn_tune_err"] for r in rows), [])),
                tokens=stats(sum((r["tokens"] for r in rows), [])))
        for n0, n1 in trans:
            def diffs(sel, f="tune_err"):
                out = []
                for u in units:
                    if sel(u) and (u, n0) in by and (u, n1) in by:
                        out += list(np.array(by[(u, n0)][f]) - np.array(by[(u, n1)][f]))
                return out
            t = dict(overall=tr_stats(diffs(lambda u: True)),
                     overall_tune_h=tr_stats(diffs(lambda u: True, "tune_err_h")),
                     overall_calib_h_own_set=tr_stats(diffs(lambda u: True, "calib_err_h")),
                     by_layer={L: tr_stats(diffs(lambda u, L=L: u[1] == L))
                               for L in sorted({u[1] for u in units})},
                     by_seed={s: tr_stats(diffs(lambda u, s=s: u[0] == s))
                              for s in sorted({u[0] for u in units})})
            B["transitions"][f"{n0}->{n1}"] = t
        rep["nested"] = B
    ct = read_jsonl(f"{OUT}/calibtune.jsonl")
    if ct:
        rep["calibtune"] = [dict(seed=r["seed"], layer=r["layer"],
                                 tokens={k: stats(v) for k, v in r["tokens"].items()},
                                 mean_err={q: {s: float(np.nanmean(v)) for s, v in d.items()}
                                           for q, d in r["err"].items()},
                                 frac_q32_worse_than_q16={
                                     s: float(np.mean(np.array(r["err"]["q32"][s]) > np.array(r["err"]["q16"][s])))
                                     for s in r["err"]["q16"]})
                            for r in ct]
    dp = read_jsonl(f"{OUT}/damp.jsonl")
    if dp:
        rep["damp"] = []
        for r in dp:
            x = dict(percdamp=r["percdamp"], actorder=r["actorder"], seed=r["seed"], layer=r["layer"])
            for tr in ("16_32", "32_64"):
                rd = np.array(r[f"rel_delta_{tr}"], float)
                x[tr] = dict(mean_rel_delta=float(np.nanmean(rd)), frac_improved=float(np.mean(rd > 0)),
                             frac_worsened=float(np.mean(rd < 0)))
            for n in (16, 32, 64):
                x[f"mean_err{n}"] = float(np.mean(r[f"err{n}"]))
                x[f"mean_calib_h{n}"] = float(np.mean(r[f"calib_h{n}"]))
            tk = np.array(r["tokens16"], float)
            x["tokens16"] = stats(tk)
            x["tokens32_approx_median"] = None
            rep["damp"].append(x)
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["invariants", "nested", "calibtune", "damp", "report"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--percdamp", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    a = ap.parse_args(argv)
    dict(invariants=cmd_invariants, nested=cmd_nested, calibtune=cmd_calibtune,
         damp=cmd_damp, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
