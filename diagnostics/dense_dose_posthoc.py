"""POST-HOC (not prespecified; does not enter any protocol criterion): is R_func driven by the measured Hessian
perturbation D_H or by the number of perturbed calibration rows k? Joint OLS of log R_func on log k and log D_H (batch-1
family, k >= 1, replicate configs + k=128 + historical singletons), per policy; within-dose Spearman(D_H, R_func)."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_dose_response as S
import report_dense_dose_response as R

def main():
    pr, cfgs, rows, fun, sq_quant, sq_bf16 = R.load_all()
    out = dict(note=__doc__)
    for pol in S.POLICIES:
        rs = [r for r in rows if r["family"] == "b1" and r["policy"] == pol and r["k"] >= 1 and "R_func" in r]
        y = np.log([r["R_func"] for r in rs]); lk = np.log([r["k"] for r in rs]); ld = np.log([r["D_H"] for r in rs])
        def fit(X):
            A = np.column_stack(X + [np.ones_like(y)]); c = np.linalg.lstsq(A, y, rcond=None)[0]
            res = y - A @ c; return c.tolist(), float(1 - res.var() / y.var())
        ck, r2k = fit([lk]); cd, r2d = fit([ld]); cb, r2b = fit([lk, ld])
        within = {k: float(spearmanr([r["D_H"] for r in rs if r["k"] == k], [r["R_func"] for r in rs if r["k"] == k]).statistic)
                  for k in sorted({r["k"] for r in rs if r["rep"] is not None})}
        hs = [r for r in rs if r["historical"]]
        out[pol] = dict(n=len(rs), log_k_only=dict(coef=ck, r2=r2k), log_DH_only=dict(coef=cd, r2=r2d),
                        joint=dict(coef_log_k=cb[0], coef_log_DH=cb[1], r2=r2b), within_dose_spearman_DH_Rfunc=within,
                        historical_singletons=dict(D_H_range=[min(r["D_H"] for r in hs), max(r["D_H"] for r in hs)],
                                                   R_func_range=[min(r["R_func"] for r in hs), max(r["R_func"] for r in hs)]))
    S.jdump(out, S.p("logs", "posthoc_k_vs_DH.json"))
    print(json.dumps(out, indent=1)[:3000])

if __name__ == "__main__":
    main()
