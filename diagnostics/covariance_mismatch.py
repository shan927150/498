"""Read-only activation second-moment mismatch diagnostic (Part E, post hoc).

Runs AFTER the primary coverage_lp_repr result is recorded; it never feeds back
into selection, tau or any threshold.

For every validated run (FR, FC, TR, TC, HR, HC from runs_typicality_matched/,
RC from runs_repr_constrained/; seeds 0/1/2) and every routed expert of layers
1/8/16/23, using the gate/up-projection INPUT (the 'gu' key, the matrix GPTQ
uses for gate_proj / up_proj):

    H_cal,e = X_cal,e^T X_cal,e / N_cal,e     (the run's 128 calibration sequences)
    H_ref,e = X_ref,e^T X_ref,e / N_ref,e     (reference split, all tokens)

Reference source: the TUNING split (cache rows 2000-2127). It is disjoint from
the candidate pool and from the test split that scores reconstruction, and was
not used by this experiment's selection or tau. It WAS used earlier (E1 labels,
P0 damping selection), so it is an independent *activation source*, not a
pristine confirmation split.

Metrics per expert: normalized Frobenius distance ||H_cal - H_ref||_F / ||H_ref||_F,
relative diagonal difference ||diag diff||_2 / ||diag H_ref||_2, effective rank
exp(entropy of normalized eigenvalues) of both, and top-16 eigenspace alignment
||U_cal^T U_ref||_F^2 / 16. Correlations with reconstruction error are
descriptive (Spearman), never causal.

Writes <out>/covariance/cov_metrics.json and cov_summary.md.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import report_repr_constrained as RR                 # noqa: E402
from moe_calib import model_utils as MU              # noqa: E402
from moe_calib.data_utils import load_meta, load_layer_cache, split_ids   # noqa: E402
from moe_calib.frozen import load_frozen             # noqa: E402
from moe_calib.profiler import Profiler              # noqa: E402

TOPK = 16


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def spectral(H):
    w, U = torch.linalg.eigh(H.double())
    w = w.clamp_min(0)
    p = w / w.sum().clamp_min(1e-30)
    er = float(torch.exp(-(p[p > 0] * torch.log(p[p > 0])).sum()))
    return er, U[:, -TOPK:].float()


def second_moments(model, cache, L, ids, E):
    prof = Profiler(model, L, K=1).run(cache, ids)
    out = {}
    for e in range(E):
        n = prof.acc.tokens((L, e, "gu"))
        if n > 0:
            out[e] = (prof.acc.H_sum((L, e, "gu"), [0]) / n, n)
    return prof, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_repr_constrained")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    a = ap.parse_args(argv)
    cfg = load_frozen(a.frozen)
    meta = load_meta(a.cache_dir)
    sp = split_ids(meta)
    assert sp["definition"] == cfg["split"], "split differs from the frozen config"
    ref_ids = sp["tune"]
    runs = {}
    for sd in RR.SEEDS:
        rs, _ = RR.load_seed(a.outdir, a.matched_dir, cfg, sd)
        for k, r in rs.items():
            runs[f"{k}_s{sd}"] = r
    layers = list(cfg["layers"])
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    res = dict(reference="tuning split rows %s (disjoint from pool and test)" % sp["definition"]["tune"],
               projection="gu input (gate/up)", topk=TOPK, per_run={k: {} for k in runs})
    t0 = time.time()
    for L in layers:
        cache = load_layer_cache(a.cache_dir, L)
        pref, Href = second_moments(model, cache, L, ref_ids, E)
        ref_spec = {e: spectral(H) for e, (H, _) in Href.items()}
        for name, r in runs.items():
            prof, Hc = second_moments(model, cache, L, r["chosen"], E)
            errs = {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert"]}
            for e, (Hr, nr) in Href.items():
                if e not in Hc:
                    continue
                H, nc = Hc[e]
                er_c, Uc = spectral(H)
                er_r, Ur = ref_spec[e]
                res["per_run"][name][f"{L}:{e}"] = dict(
                    fro=float(torch.linalg.norm(H - Hr) / torch.linalg.norm(Hr)),
                    diag=float(torch.linalg.norm(H.diagonal() - Hr.diagonal()) / torch.linalg.norm(Hr.diagonal())),
                    eff_rank_cal=er_c, eff_rank_ref=er_r,
                    align=float((Uc.T @ Ur).pow(2).sum() / TOPK),
                    n_cal=nc, n_ref=nr, recon=errs.get((L, e)))
            prof.free()
            del Hc
        pref.free()
        del cache, Href, ref_spec
        torch.cuda.empty_cache()
        print(f"[cov] layer {L} done ({time.time() - t0:.0f}s)", flush=True)
    # summaries
    cond = {}
    for name, d in res["per_run"].items():
        k = name.split("_s")[0]
        v = list(d.values())
        cond.setdefault(k, []).append(dict(
            fro=np.mean([x["fro"] for x in v]), diag=np.mean([x["diag"] for x in v]),
            eff_rank_ratio=np.mean([x["eff_rank_cal"] / x["eff_rank_ref"] for x in v]),
            align=np.mean([x["align"] for x in v]), recon=runs[name]["summary"]["recon_mean"]))
    res["by_condition"] = {k: {m: float(np.mean([x[m] for x in v])) for m in v[0]} for k, v in cond.items()}
    names = list(res["per_run"])
    run_fro = [np.mean([x["fro"] for x in res["per_run"][n].values()]) for n in names]
    run_rec = [runs[n]["summary"]["recon_mean"] for n in names]
    # within-seed (paired) correlation across the 7 conditions, then mean over seeds
    ws = []
    for sd in RR.SEEDS:
        idx = [i for i, n in enumerate(names) if n.endswith(f"_s{sd}")]
        ws.append(spearman([run_fro[i] for i in idx], [run_rec[i] for i in idx]))
    # within-expert correlation across the 21 runs (controls expert heterogeneity)
    keys = list(res["per_run"][names[0]])
    we = [spearman([res["per_run"][n].get(k, {}).get("fro", np.nan) for n in names],
                   [res["per_run"][n].get(k, {}).get("recon", np.nan) or np.nan for n in names]) for k in keys]
    we = [x for x in we if x is not None]
    res["correlation"] = dict(
        run_level_spearman_fro_vs_recon=spearman(run_fro, run_rec),
        within_seed_spearman_fro_vs_recon=ws,
        within_expert_spearman_fro_vs_recon_median=float(np.median(we)),
        within_expert_frac_positive=float(np.mean(np.array(we) > 0)),
        n_experts=len(we))
    os.makedirs(os.path.join(a.outdir, "covariance"), exist_ok=True)
    json.dump(res, open(os.path.join(a.outdir, "covariance", "cov_metrics.json"), "w"))
    L_ = ["# Activation second-moment mismatch (post-hoc, descriptive)\n",
          f"Reference: {res['reference']}; projection: {res['projection']}. Mean over 240 experts and 3 seeds.\n",
          "| condition | rel. Frobenius | rel. diagonal | eff-rank ratio cal/ref | top-16 alignment | recon |",
          "|---|---|---|---|---|---|"]
    for k in ("FR", "FC", "TR", "TC", "HR", "HC", "RC"):
        if k in res["by_condition"]:
            g = res["by_condition"][k]
            L_.append(f"| {RR.NAMES[k]} | {g['fro']:.4f} | {g['diag']:.4f} | {g['eff_rank_ratio']:.4f} | "
                      f"{g['align']:.4f} | {g['recon']:.6f} |")
    c = res["correlation"]
    L_.append(f"\nSpearman(mean Frobenius mismatch, recon) over 21 runs: {c['run_level_spearman_fro_vs_recon']:.3f}; "
              f"within seed (7 conditions): {', '.join(f'{x:.3f}' for x in c['within_seed_spearman_fro_vs_recon'])}; "
              f"within expert across runs: median {c['within_expert_spearman_fro_vs_recon_median']:.3f} "
              f"({100 * c['within_expert_frac_positive']:.0f} % of {c['n_experts']} experts positive). "
              "Descriptive only; correlation is not causation.")
    open(os.path.join(a.outdir, "covariance", "cov_summary.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
