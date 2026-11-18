"""POST-HOC, EXPLORATORY (not in protocol.json; written after the primary results were seen).

1. Is the WT2 corrected between-variant variance larger than the C4HO one?  Pivot interval for the difference
   V_corr(WT2) - V_corr(C4HO) from the two independent nested bootstraps already saved by the primary analysis.
2. Article influence on WT2: leave-one-article-out V_corr / sampling fraction / null-free split-half median, to check
   whether the WT2 signal is carried by one article.
3. Sign concentration: per-window share of the centred between-variant sum of squares (WT2 vs C4HO).
"""
from __future__ import annotations
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation_uncertainty as E   # noqa: E402

SETS = ["moe", "dense4", "dense3", "dose_natural_k4", "dose_natural_k16"]


def main():
    E.check_protocol()
    L = E.load_matrix()
    S = E.analysis_sets(L)
    out = dict(label="POST-HOC EXPLORATORY: not prespecified; does not change any primary verdict", difference={},
               leave_one_article_out={}, concentration={})
    for n in SETS:
        z = np.load(E.p("logs", "parts", f"{n}.npz"))
        vj = E.jload(E.p("logs", "parts", f"{n}.json"))["var"]
        w, c = vj["WT2"]["article"], vj["C4HO"]["window"]
        dW = z["WT2_article_Vcorr_star"] - w["V_obs"]
        dC = z["C4HO_window_Vcorr_star"] - c["V_obs"]
        diff = w["V_corr"] - c["V_corr"]
        dd = dW - dC                                   # independent bootstraps, paired by draw index (independent)
        lo, hi = diff - np.quantile(dd, .975), diff - np.quantile(dd, .025)
        out["difference"][n] = dict(V_corr_WT2=w["V_corr"], V_corr_C4HO=c["V_corr"], difference=diff,
                                    pivot_ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0),
                                    ratio_V_obs_WT2_over_C4HO=w["V_obs"] / c["V_obs"],
                                    ratio_V_samp_WT2_over_C4HO=w["V_samp"] / c["V_samp"])
        # leave-one-article-out on WT2
        St = S[n]
        X, bl = St["X"]["WT2"], St["blocks"]
        art = E.units_for("article", "WT2")
        rows = []
        for i, a in enumerate(np.unique(art)):
            keep = art != a
            Xk = X[keep]
            U = E.Units(Xk, art[keep])
            rng = E.rng_for(11, SETS.index(n), i)
            dr = U.theta(E.draw_counts(rng, U.m, 1000))
            vo, vs = E.vstats_from(Xk.mean(0), dr, bl)
            M = E.split_masks(rng, None, "WT2", 200, uid=art[keep])
            tA, tB = E.half_thetas(Xk, M)
            r = np.nanmedian(E.pearson(E.center(tA, bl), E.center(tB, bl)))
            rows.append(dict(article=int(a), n_windows=int((~keep).sum()), V_corr=vo - vs, sampling_fraction=vs / vo,
                             split_half_median_r=float(r)))
        vc = np.array([r["V_corr"] for r in rows]); fr = np.array([r["sampling_fraction"] for r in rows])
        sr = np.array([r["split_half_median_r"] for r in rows])
        worst = rows[int(np.argmin(vc))]
        out["leave_one_article_out"][n] = dict(V_corr_full=w["V_corr"], V_corr_min=float(vc.min()), V_corr_max=float(vc.max()),
                                               sampling_fraction_max=float(fr.max()), split_half_r_min=float(sr.min()),
                                               split_half_r_median=float(np.median(sr)), most_influential=worst,
                                               n_articles=len(rows))
        # concentration of the centred between-variant signal over windows
        conc = {}
        for ds in E.DS:
            Xd = St["X"][ds]
            th = E.center(Xd.mean(0), bl)
            contrib = (E.center(Xd, bl) * th[None, :]).sum(1)      # window contribution to theta' P theta * W
            tot = contrib.sum()
            srt = np.sort(contrib)[::-1]
            conc[ds] = dict(top5_window_share=float(srt[:5].sum() / tot), top10_window_share=float(srt[:10].sum() / tot),
                            frac_windows_positive=float((contrib > 0).mean()))
        out["concentration"][n] = conc
        print(n, out["difference"][n]["pivot_ci"], out["leave_one_article_out"][n]["V_corr_min"],
              out["leave_one_article_out"][n]["split_half_r_min"], flush=True)
    E.jdump(out, E.p("posthoc_exploratory.json"))


if __name__ == "__main__":
    main()
