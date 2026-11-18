"""Analysis, figures and machine-readable outputs for runs_heldout_bitwidth_validation (definitions frozen in
protocol.json; this module only reads results)."""
from __future__ import annotations
import itertools, json, math, os, sys, time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import heldout_bitwidth_validation as H                       # noqa: E402
from moe_calib import cross_seed_grouping as CS                # noqa: E402

p, jload, jdump = H.p, H.jload, H.jdump
NT = H.N_WIN * (H.SEQLEN - 1)
TPW = H.SEQLEN - 1
DS = ("wikitext", "heldout")
DSN = {"wikitext": "WikiText-2", "heldout": "C4-val held-out"}
G = H.GROUPS
BOOT_SEED, N_WBOOT, N_GBOOT = 20261117, 2000, 20000
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- basic stats
def ppl_of(nll):
    return math.exp(float(np.sum(nll)) / NT)


def lppl_of(nll):
    return float(np.sum(nll)) / NT


def sd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def chi2_ci(s, n, a=0.05):
    d = n - 1
    return [float(s * math.sqrt(d / stats.chi2.ppf(1 - a / 2, d))), float(s * math.sqrt(d / stats.chi2.ppf(a / 2, d)))]


def f_band(r, d1, d2, a=0.05):
    """95% CI for sigma1/sigma2 from observed SD ratio r (independent-sample F; descriptive when samples share weights)."""
    return [float(r / math.sqrt(stats.f.ppf(1 - a / 2, d1, d2))), float(r / math.sqrt(stats.f.ppf(a / 2, d1, d2)))]


def ratio(a, b):
    return float(a / b) if b not in (0, 0.0) and b == b else float("nan")


def penalty_ratio(S, Q, Q_se):
    if Q <= 0:
        return dict(value=None, status="UNDEFINED (Q <= 0)")
    if Q < 2 * Q_se:
        return dict(value=float(S / Q), status="UNSTABLE (Q < 2 x window-bootstrap SE)")
    return dict(value=float(S / Q), status="ok")


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def pearson(a, b):
    return float(stats.pearsonr(a, b)[0])


# ---------------------------------------------------------------- loading
def R(*x):
    return p("results", *x)


def load_all():
    d = dict(dense_bf16=jload(R("dense", "BF16.json")), moe_bf16=jload(R("moe", "BF16.json")))
    d["dense4"] = {g: jload(R("dense", f"dense4_{g}.json")) for g in H.DENSE_CONFIGS}
    d["dense3"] = {g: jload(R("dense3", f"dense3_{g}.json")) for g in H.DENSE_CONFIGS}
    d["dose"] = {n: jload(R("dense", f"dose_{n}.json")) for n in H.dose_names()}
    d["ref"] = {(i, j): jload(R("dense", f"ref_{i}__{j}.json")) for i in G for j in G}
    d["moe"] = {n: jload(R("moe", f"{n}.json")) for n in H.MOE_CONFIGS + H.MOE_REFS}
    return d


def nll_dense(rec, ds):
    return np.array(rec[ds]["per_window_nll"], float)


def nll_moe(rec, ds):
    return np.array(rec["historical_wikitext_per_window_nll"] if ds == "wikitext" else rec["heldout"]["per_window_nll"], float)


# ---------------------------------------------------------------- window bootstrap (conditional on weights)
def wboot_idx(n_boot=N_WBOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, H.N_WIN, size=(n_boot, H.N_WIN))


def boot_ppl(nll, idx):
    return np.exp(nll[idx].sum(1) / (idx.shape[1] * TPW))


def group_stats(nlls, bf, idx=None):
    """nlls: [5, 128] per-window NLL of the 5 groupings; bf [128]. Point + optional window bootstrap of Q, S, R."""
    ppl = np.array([ppl_of(x) for x in nlls]); lp = np.array([lppl_of(x) for x in nlls])
    pb, lb = ppl_of(bf), lppl_of(bf)
    S, Sl, Q, Ql = sd(ppl), sd(lp), float(ppl.mean() - pb), float(lp.mean() - lb)
    out = dict(ppl=ppl.tolist(), mean_ppl=float(ppl.mean()), bf16_ppl=pb, S=S, S_log=Sl, Q=Q, Q_log=Ql,
               S_ci95_chi2=chi2_ci(S, 5), S_log_ci95_chi2=chi2_ci(Sl, 5),
               max_pairwise=float(ppl.max() - ppl.min()), cv=float(S / ppl.mean()))
    if idx is not None:
        P = np.stack([boot_ppl(x, idx) for x in nlls], 1)
        Pb = boot_ppl(bf, idx)
        Sb, Qb = P.std(1, ddof=1), P.mean(1) - Pb
        Lb = np.log(P); Lbf = np.log(Pb)
        Rb = Sb / Qb; Rlb = Lb.std(1, ddof=1) / (Lb.mean(1) - Lbf)
        out["window_boot"] = dict(Q_se=float(Qb.std()), Q_ci95=np.quantile(Qb, [.025, .975]).tolist(),
                                  S_se=float(Sb.std()), S_ci95=np.quantile(Sb, [.025, .975]).tolist(),
                                  R_ci95=np.quantile(Rb, [.025, .975]).tolist(), R_log_ci95=np.quantile(Rlb, [.025, .975]).tolist(),
                                  n_boot=int(idx.shape[0]), seed=BOOT_SEED,
                                  note="joint window resampling, weights fixed: evaluation-text uncertainty only")
        out["_boot"] = dict(S=Sb, Q=Qb, R=Rb, Rlog=Rlb)
        Qse = out["window_boot"]["Q_se"]
    else:
        Qse = 0.0
    out["R"] = penalty_ratio(S, Q, Qse)
    out["R_log"] = penalty_ratio(Sl, Ql, Qse / pb if pb else 0)
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o


def per_window_pairs(nlls):
    """10 grouping pairs: per-window mean-NLL differences (nats/token)."""
    out = []
    for i, j in itertools.combinations(range(len(nlls)), 2):
        d = (nlls[i] - nlls[j]) / TPW
        out.append(dict(pair=[G[i], G[j]], mean=float(d.mean()), sd_over_windows=float(d.std(ddof=1)),
                        frac_windows_differ=float((d != 0).mean()), max_abs=float(np.abs(d).max())))
    return out


# ---------------------------------------------------------------- MoE
def moe_analysis(d):
    res = {}
    idx = wboot_idx()
    from full_depth_pilot import paired_bootstrap
    for ds in DS:
        ppl, lpl, nl = {}, {}, {}
        for s in H.MOE_SEEDS:
            for g in G:
                for st in H.MOE_STRATS:
                    r = d["moe"][f"{st}_s{s}_{g}"]
                    x = nll_moe(r, ds)
                    nl[(st, s, g)] = x
                    ppl[(st, s, g)] = ppl_of(x); lpl[(st, s, g)] = lppl_of(x)
        dl = CS.paired_deltas(ppl)
        dn = CS.paired_deltas(lpl)                      # mean-NLL (log PPL) differences
        v = CS.variance_summary(dl)
        vn = CS.variance_summary(dn)
        hb = CS.hierarchical_bootstrap(dl)
        dec, det = CS.decide(dl)
        M = v["overall_mean_delta"]
        flat = [(s, g, dl[s][g]) for s in sorted(dl) for g in sorted(dl[s])]
        r1 = [f"s{s}_{g}" for s, g, x in flat if np.sign(x) == -np.sign(M)]
        bf = np.array(d["moe_bf16"][ds]["per_window_nll"], float)
        pb = ppl_of(bf)
        # seed-0 ORIGINAL reference on the same dataset
        o = {st: nll_moe(d["moe"][f"{st}_ORIGINAL"], ds) for st in H.MOE_STRATS}
        dO = ppl_of(o["FC"]) - ppl_of(o["FR"])
        r2 = [f"s0_{g}" for g in G if np.sign(dl[0][g]) != np.sign(dO)]
        s0 = np.array([dl[0][g] for g in G])
        rank = {st: int(1 + sum(ppl[(st, 0, g)] < ppl_of(o[st]) for g in G)) for st in H.MOE_STRATS}
        # FR / FC individual PPL SD (pooled within seed) and penalty
        indiv = {}
        for st in H.MOE_STRATS:
            sds = [sd([ppl[(st, s, g)] for g in G]) for s in H.MOE_SEEDS]
            allp = [ppl[(st, s, g)] for s in H.MOE_SEEDS for g in G]
            pooled = float(np.sqrt(np.mean(np.square(sds))))
            Qst = float(np.mean(allp) - pb)
            # window bootstrap of the pooled SD / Q (weights fixed)
            P = {k: boot_ppl(nl[k], idx) for k in nl if k[0] == st}
            Pb = boot_ppl(bf, idx)
            sdb = np.sqrt(np.mean([np.stack([P[(st, s, g)] for g in G], 1).std(1, ddof=1) ** 2 for s in H.MOE_SEEDS], 0))
            Qb = np.mean(np.stack(list(P.values()), 1), 1) - Pb
            sd0 = sds[0]
            indiv[st] = dict(per_seed_sd=sds, pooled_within_seed_sd=pooled, pooled_sd_ci95_chi2=chi2_ci(pooled, 13),
                             seed0_sd=sd0, mean_ppl=float(np.mean(allp)), seed0_mean_ppl=float(np.mean([ppl[(st, 0, g)] for g in G])),
                             Q=Qst, Q_seed0=float(np.mean([ppl[(st, 0, g)] for g in G]) - pb),
                             Q_window_boot_se=float(Qb.std()),
                             R=penalty_ratio(pooled, Qst, float(Qb.std())),
                             R_seed0=penalty_ratio(sd0, float(np.mean([ppl[(st, 0, g)] for g in G]) - pb), float(Qb.std())),
                             R_window_boot_ci95=np.quantile(sdb / Qb, [.025, .975]).tolist(),
                             pooled_sd_log=float(np.sqrt(np.mean([sd([lpl[(st, s, g)] for g in G]) ** 2 for s in H.MOE_SEEDS]))),
                             original_ppl=ppl_of(o[st]), original_rank_among_seed0=rank[st],
                             note="pooled within-seed SD has 3 x 4 = 12 df (chi-square CI uses n = 13)")
        pairs = {}
        nci = 0
        for s, g, x in flat:
            fr, fc = nl[("FR", s, g)], nl[("FC", s, g)]
            ci = paired_bootstrap(fc, fr, np.full(H.N_WIN, TPW))
            ex = ci["ci95"][0] > 0 or ci["ci95"][1] < 0
            nci += ex
            dw = (fc - fr) / TPW
            pairs[f"s{s}_{g}"] = dict(FR=ppl[("FR", s, g)], FC=ppl[("FC", s, g)], delta=x, delta_mean_nll=dn[s][g],
                                      window_ci95_mean_nll=ci["ci95"], window_ci_excludes_0=bool(ex),
                                      per_window_diff_sd=float(dw.std(ddof=1)), frac_windows_fc_worse=float((dw > 0).mean()))
        res[ds] = dict(
            dataset=DSN[ds], bf16_ppl=pb, pairs=pairs, variance=v, variance_mean_nll=vn, hierarchical_bootstrap=hb,
            decision=dec, decision_details=det, pooled_sd_delta=v["pooled_within_sd"],
            pooled_sd_delta_ci95_chi2=chi2_ci(v["pooled_within_sd"], 13),
            pooled_sd_delta_mean_nll=vn["pooled_within_sd"], seed_sd_delta=v["sd_grouping"],
            relative_pooled_sd=v["pooled_within_sd"] / float(np.mean(list(ppl.values()))),
            n_positive=v["n_fc_worse_total"], n_negative=v["n_fc_better_total"],
            R1=dict(count=len(r1), of=15, configs=r1, M=M),
            R2_seed0=dict(count=len(r2), of=5, configs=r2, original_delta=dO, original_FR=ppl_of(o["FR"]), original_FC=ppl_of(o["FC"]),
                          original_delta_z_vs_seed0_groupings=float((dO - s0.mean()) / s0.std(ddof=1)),
                          original_delta_rank_among_seed0=int(1 + (s0 < dO).sum())),
            individual=indiv, n_pairs_window_ci_excluding_0=int(nci))
    w, h = res["wikitext"], res["heldout"]
    k = sorted(w["pairs"])
    a, b = np.array([w["pairs"][x]["delta"] for x in k]), np.array([h["pairs"][x]["delta"] for x in k])
    an, bn = np.array([w["pairs"][x]["delta_mean_nll"] for x in k]), np.array([h["pairs"][x]["delta_mean_nll"] for x in k])
    Rd = h["pooled_sd_delta"] / w["pooled_sd_delta"]
    Rn = h["pooled_sd_delta_mean_nll"] / w["pooled_sd_delta_mean_nll"]
    cross = dict(
        R_dataset=Rd, R_dataset_F_band_95=f_band(Rd, 12, 12),
        R_dataset_mean_nll=Rn, R_dataset_mean_nll_F_band_95=f_band(Rn, 12, 12),
        R_dataset_relative=h["relative_pooled_sd"] / w["relative_pooled_sd"],
        historical_reference_sd=0.006872, wikitext_recomputed_sd=w["pooled_sd_delta"],
        wikitext_reproduces_historical=abs(w["pooled_sd_delta"] - 0.006872) < 5e-7,
        sign_agreement=int((np.sign(a) == np.sign(b)).sum()), sign_disagreement=int((np.sign(a) != np.sign(b)).sum()),
        sign_disagreement_configs=[x for x, u, v_ in zip(k, a, b) if np.sign(u) != np.sign(v_)],
        pearson_delta=pearson(a, b), spearman_delta=spearman(a, b), pearson_delta_mean_nll=pearson(an, bn),
        within_seed_centered_pearson=pearson(np.concatenate([a[i * 5:(i + 1) * 5] - a[i * 5:(i + 1) * 5].mean() for i in range(3)]),
                                             np.concatenate([b[i * 5:(i + 1) * 5] - b[i * 5:(i + 1) * 5].mean() for i in range(3)])),
        order=k,
        note="F band treats the two SDs as independent 12-df estimates; the same weights are evaluated on both datasets, so "
             "the band is descriptive; correlations are descriptive (15 nested pairs, 3 seeds)")
    return res, cross


# ---------------------------------------------------------------- dense 4 / 3
def dense_analysis(d, key):
    out = {}
    idx = wboot_idx()
    for ds in DS:
        if key == "dense4":
            recs = d["dense4"]
        else:
            recs = d["dense3"]
        nlls = np.stack([nll_dense(recs[g], ds) for g in G])
        bf = nll_dense(d["dense_bf16"], ds)
        st = group_stats(nlls, bf, idx)
        orig = ppl_of(nll_dense(recs["ORIGINAL"], ds))
        st.update(original_ppl=orig, Q0_original=orig - ppl_of(bf),
                  per_config={g: ppl_of(nll_dense(recs[g], ds)) for g in H.DENSE_CONFIGS},
                  deviation_from_mean={g: ppl_of(nll_dense(recs[g], ds)) - st["mean_ppl"] for g in G},
                  original_rank_among_5=int(1 + sum(ppl_of(x) < orig for x in nlls)),
                  null_G20261115_equals_ORIGINAL=recs["G20261115"][ds]["per_window_nll"] == recs["ORIGINAL"][ds]["per_window_nll"],
                  per_window_pairs=per_window_pairs(nlls),
                  mean_per_window_pair_sd=float(np.mean([x["sd_over_windows"] for x in per_window_pairs(nlls)])))
        out[ds] = st
    return out


def dense_cross(a4):
    w, h = a4["wikitext"], a4["heldout"]
    rS = h["S"] / w["S"]
    rR = (h["R"]["value"] or float("nan")) / (w["R"]["value"] or float("nan"))
    dv = [w["deviation_from_mean"][g] for g in G], [h["deviation_from_mean"][g] for g in G]
    return dict(S_ratio=rS, S_ratio_F_band_95=f_band(rS, 4, 4), S_log_ratio=h["S_log"] / w["S_log"],
                S_log_ratio_F_band_95=f_band(h["S_log"] / w["S_log"], 4, 4),
                Q_ratio=h["Q"] / w["Q"], R_ratio=rR, R_ratio_F_band_95=f_band(rR, 4, 4) if rR == rR else None,
                deviation_pearson=pearson(*dv), deviation_spearman=spearman(*dv),
                deviation_sign_agreement=int(sum(np.sign(x) == np.sign(y) for x, y in zip(*dv))),
                ranking_wikitext=sorted(G, key=lambda g: w["per_config"][g]), ranking_heldout=sorted(G, key=lambda g: h["per_config"][g]),
                note="n = 5 groupings (4 single-row perturbations + exact null); F(4,4) bands are wide (~3.1x factor)")


def bitwidth(a4, a3):
    out = {}
    rng = np.random.default_rng(BOOT_SEED)
    gi = rng.integers(0, 5, size=(N_GBOOT, 5))
    for ds in DS:
        w4, w3 = a4[ds], a3[ds]
        p4, p3 = np.array(w4["ppl"]), np.array(w3["ppl"])
        bfp = w4["bf16_ppl"]
        S4, S3 = p4[gi].std(1, ddof=1), p3[gi].std(1, ddof=1)
        Q4, Q3 = p4[gi].mean(1) - bfp, p3[gi].mean(1) - bfp
        ok = (S4 > 0)
        RR = (S3[ok] / Q3[ok]) / (S4[ok] / Q4[ok])
        SS = S3[ok] / S4[ok]
        R4, R3 = w4["R"]["value"], w3["R"]["value"]
        rrel = R3 / R4 if (R4 and R3) else float("nan")
        # window bootstrap of R3/R4 (weights fixed)
        wb = w3["_boot"]["R"] / w4["_boot"]["R"]
        lab_int = np.quantile(RR, [.025, .975]).tolist()
        label = "B1" if lab_int[0] <= 1 <= lab_int[1] else ("B2" if lab_int[1] < 1 else "B3")
        out[ds] = dict(
            dataset=DSN[ds], Q4=w4["Q"], Q3=w3["Q"], S4=w4["S"], S3=w3["S"], S4_log=w4["S_log"], S3_log=w3["S_log"],
            R4=R4, R3=R3, R4_status=w4["R"]["status"], R3_status=w3["R"]["status"],
            R4_log=w4["R_log"]["value"], R3_log=w3["R_log"]["value"],
            R_relative=rrel, R_log_relative=(w3["R_log"]["value"] / w4["R_log"]["value"]),
            S_ratio=w3["S"] / w4["S"], Q_ratio=w3["Q"] / w4["Q"], S_log_ratio=w3["S_log"] / w4["S_log"],
            S_ratio_F_band_95_independent=f_band(w3["S"] / w4["S"], 4, 4),
            grouping_paired_bootstrap=dict(R_relative_ci95=lab_int, S_ratio_ci95=np.quantile(SS, [.025, .975]).tolist(),
                                           n_draws=int(N_GBOOT), n_valid=int(ok.sum()), seed=BOOT_SEED,
                                           note="resample the 5 groupings jointly across bit-widths (pairing kept); n = 5 "
                                                "makes this a crude sensitivity interval, not inference"),
            window_bootstrap_R_relative_ci95=np.quantile(wb, [.025, .975]).tolist(),
            classification=label,
            paired_table={g: dict(ppl4=w4["per_config"][g], ppl3=w3["per_config"][g],
                                  diff=w3["per_config"][g] - w4["per_config"][g],
                                  penalty4=w4["per_config"][g] - bfp, penalty3=w3["per_config"][g] - bfp,
                                  dev4=w4["deviation_from_mean"][g], dev3=w3["deviation_from_mean"][g]) for g in G},
            deviation_pearson=pearson([w4["deviation_from_mean"][g] for g in G], [w3["deviation_from_mean"][g] for g in G]),
            deviation_spearman=spearman([w4["deviation_from_mean"][g] for g in G], [w3["deviation_from_mean"][g] for g in G]),
            deviation_sign_agreement=int(sum(np.sign(w4["deviation_from_mean"][g]) == np.sign(w3["deviation_from_mean"][g]) for g in G)))
    labs = {out[ds]["classification"] for ds in DS}
    out["overall_classification"] = labs.pop() if len(labs) == 1 else "B4"
    out["rule"] = "protocol: B1 interval of R3/R4 contains 1; B2 upper < 1; B3 lower > 1; B4 datasets disagree"
    return out


# ---------------------------------------------------------------- dose
def dose_analysis(d, dense_bf):
    out = dict(doses={}, Q0={})
    for ds in DS:
        bf = ppl_of(nll_dense(dense_bf, ds))
        k0 = d["dose"]["b1_k000"]["policies"]["natural"]
        k0n = np.array(k0["historical_wikitext_per_window_nll"] if ds == "wikitext" else k0["heldout"]["per_window_nll"])
        out["Q0"][ds] = ppl_of(k0n) - bf
        out.setdefault("bf16", {})[ds] = bf
        out.setdefault("ppl_k0", {})[ds] = ppl_of(k0n)
    for pol in ("natural", "fixed"):
        for k in H.DOSE_K:
            names = [f"b1_k{k:03d}_r{r}" for r in range(H.DOSE_REPS)]
            e = {}
            for ds in DS:
                pp = []
                for n in names:
                    rp = d["dose"][n]["policies"][pol]
                    x = np.array(rp["historical_wikitext_per_window_nll"] if ds == "wikitext" else rp["heldout"]["per_window_nll"])
                    pp.append(ppl_of(x))
                pp = np.array(pp)
                s = sd(pp)
                e[ds] = dict(ppl=pp.tolist(), mean=float(pp.mean()), s_k=s, s_k_ci95_chi2=chi2_ci(s, 8),
                             s_k_log=sd(np.log(pp)), R_PPL=s / out["Q0"][ds], R_PPL_ci95=[x / out["Q0"][ds] for x in chi2_ci(s, 8)],
                             mean_shift_vs_k0=float(pp.mean() - out["ppl_k0"][ds]))
            dev = [np.array(e[ds]["ppl"]) - e[ds]["mean"] for ds in DS]
            e["cross"] = dict(s_ratio=e["heldout"]["s_k"] / e["wikitext"]["s_k"],
                              s_ratio_F_band_95=f_band(e["heldout"]["s_k"] / e["wikitext"]["s_k"], 7, 7),
                              R_PPL_ratio=e["heldout"]["R_PPL"] / e["wikitext"]["R_PPL"],
                              replicate_deviation_pearson=pearson(*dev), replicate_deviation_spearman=spearman(*dev),
                              replicate_ids=names)
            out["doses"][f"{pol}|k{k}"] = e
    trend = {}
    for pol in ("natural", "fixed"):
        for ds in DS:
            ks = np.array(H.DOSE_K, float)
            s = np.array([out["doses"][f"{pol}|k{k}"][ds]["s_k"] for k in H.DOSE_K])
            sl, ic = np.polyfit(np.log(ks), np.log(s), 1)
            trend[f"{pol}|{ds}"] = dict(s_k=s.tolist(), loglog_slope_s_vs_k=float(sl), s64_over_s1=float(s[-1] / s[0]),
                                        max_over_min=float(s.max() / s.min()),
                                        pooled_s_over_doses=float(np.sqrt(np.mean(s ** 2))),
                                        R_PPL_pooled=float(np.sqrt(np.mean(s ** 2)) / out["Q0"][ds]),
                                        k1_share_of_k64=float(s[0] / s[-1]))
    out["trend"] = trend
    for ds in DS:
        out.setdefault("fixed_over_natural_sd", {})[ds] = {f"k{k}": out["doses"][f"fixed|k{k}"][ds]["s_k"] / out["doses"][f"natural|k{k}"][ds]["s_k"]
                                                           for k in H.DOSE_K}
    out["note"] = ("s_k over 8 different perturbation subsets per dose (deterministic reruns are not replicates); "
                   "Q0 = PPL(k=0 = ORIGINAL) - BF16 on the same dataset; WikiText values from stored historical per-window NLL "
                   "(parity re-verified bitwise on the prespecified subset)")
    return out


def refmatrix_analysis(d):
    if any(v is None for v in d["ref"].values()):
        return dict(available=False)
    out = {}
    for ds in DS:
        M = np.array([[ppl_of(nll_dense(d["ref"][(i, j)], ds)) for j in G] for i in G])   # rows: reference pi_i
        means = M.mean(1)
        # two-way additive ANOVA (reference x Hessian, no replication): reference effect vs interaction residual
        gm = M.mean(); rr = M.mean(1) - gm; cc = M.mean(0) - gm
        res = M - gm - rr[:, None] - cc[None, :]
        ms_ref = 5 * (rr ** 2).sum() / 4; ms_h = 5 * (cc ** 2).sum() / 4; ms_e = (res ** 2).sum() / 16
        F_ref = ms_ref / ms_e
        out[ds] = dict(matrix=M.tolist(), rows="reference permutation pi_i", cols="Hessian H_j",
                       mean_by_reference={g: float(m) for g, m in zip(G, means)},
                       sd_within_reference={g: sd(M[i]) for i, g in enumerate(G)},
                       mean_within_reference_sd=float(np.mean([sd(M[i]) for i in range(5)])),
                       between_reference_range=float(means.max() - means.min()),
                       best_reference=G[int(means.argmin())], worst_reference=G[int(means.argmax())],
                       original_reference_rank=int(1 + (means < means[G.index("G20261115")]).sum()),
                       anova_posthoc=dict(F_reference=float(F_ref), p_reference=float(1 - stats.f.cdf(F_ref, 4, 16)),
                                          F_hessian=float(ms_h / ms_e), p_hessian=float(1 - stats.f.cdf(ms_h / ms_e, 4, 16)),
                                          note="post-hoc, not prespecified: additive two-way ANOVA, interaction as error"))
    w, h = out["wikitext"], out["heldout"]
    out["cross"] = dict(reference_mean_spearman=spearman(list(w["mean_by_reference"].values()), list(h["mean_by_reference"].values())),
                        original_worst_wikitext=w["worst_reference"] == "G20261115",
                        original_worst_heldout=h["worst_reference"] == "G20261115",
                        best_wikitext=w["best_reference"], best_heldout=h["best_reference"],
                        note="the held-out set is used only to re-evaluate; no reference is selected on it")
    out["available"] = True
    return out


# ---------------------------------------------------------------- power
def power(moe, mcross, a4, a3):
    deltas = (0.0025, 0.005, 0.01)
    n = lambda s, dl: int(math.ceil((Z * s / dl) ** 2))
    rows = []
    for ds in DS:
        s = moe[ds]["pooled_sd_delta"]
        rows.append(dict(design="paired FC-FR, MoE Family D (empirical paired SD)", dataset=DSN[ds], s_used=s,
                         s_definition="pooled within-seed SD of Delta = PPL(FC)-PPL(FR) over groupings",
                         n_by_delta={str(dl): n(s, dl) for dl in deltas}, n_meaning="paired execution replicates"))
    for lab, a in (("dense 4-bit", a4), ("dense 3-bit", a3)):
        for ds in DS:
            s = a[ds]["S"]
            rows.append(dict(design=f"independent arms, {lab} single-method SD (ASSUMES independent arm executions)",
                             dataset=DSN[ds], s_used=math.sqrt(2) * s, single_method_sd=s,
                             s_definition="sqrt(2) x SD(PPL) over the 5 groupings (1 calibration seed)",
                             n_by_delta={str(dl): n(math.sqrt(2) * s, dl) for dl in deltas}, n_meaning="replicates per arm"))
    for st in H.MOE_STRATS:
        for ds in DS:
            s = moe[ds]["individual"][st]["pooled_within_seed_sd"]
            rows.append(dict(design=f"independent arms, MoE {st} single-method pooled SD (ASSUMES independence)", dataset=DSN[ds],
                             s_used=math.sqrt(2) * s, single_method_sd=s, s_definition="sqrt(2) x pooled within-seed SD(PPL)",
                             n_by_delta={str(dl): n(math.sqrt(2) * s, dl) for dl in deltas}, n_meaning="replicates per arm"))
    return dict(formula="n = ((z_0.975 + z_0.80) s / delta)^2, alpha 0.05 two-sided, power 0.8 (normal approximation; "
                        "ignores t-correction, so small n are optimistic)", z_sum=Z, deltas=deltas, rows=rows,
                caveat="planning estimates conditional on this numerical stack, these models and the assumed variance; not "
                       "universal requirements. The 3-bit dense study has one calibration strategy, so it cannot give a "
                       "paired FR/FC SD.")


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 2})
    return plt


def save(fig, nm):
    os.makedirs(p("figures"), exist_ok=True)
    fig.savefig(p("figures", nm + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(p("figures", nm + ".pdf"), bbox_inches="tight")


def figures(moe, mcross, a4, a3, bw, dose, refm, table):
    plt = _plt()
    # Fig 1: MoE paired Delta across datasets
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    k = mcross["order"]
    x = np.arange(15)
    for j, ds in enumerate(DS):
        v = [moe[ds]["pairs"][q]["delta"] for q in k]
        ax.scatter(x + (j - .5) * .25, v, s=40, color=COL[j], label=DSN[ds], zorder=3, edgecolor=SURF, linewidth=1)
    for i, q in enumerate(k):
        if q in mcross["sign_disagreement_configs"]:
            ax.axvspan(i - .45, i + .45, color=GRID, alpha=.6, zorder=0)
    for s in (5, 10):
        ax.axvline(s - .5, color=INK2, lw=.8)
    ax.axhline(0, color=INK2, lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([q.replace("_G2026111", " G") for q in k], rotation=60, fontsize=7)
    ax.set_ylabel("Δ = PPL(FC) − PPL(FR)")
    ax.set_title(f"Fig. 1  MoE Family D paired FC−FR differences (shaded = sign differs across datasets; "
                 f"{mcross['sign_disagreement']}/15)", fontsize=9, loc="left")
    ax.legend(loc="upper right")
    for i, s in enumerate(H.MOE_SEEDS):
        ax.text(i * 5 + 2, 0.98, f"seed {s}", ha="center", va="top", color=INK2, fontsize=8,
                transform=ax.get_xaxis_transform())
    save(fig, "fig1_moe_paired_deltas"); plt.close(fig)
    # Fig 2: dense 4-bit grouping across datasets (deviation from 5-grouping mean, both datasets)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for j, ds in enumerate(DS):
        ax = axs[j]
        v = [a4[ds]["per_config"][g] for g in G]
        ax.scatter(range(5), np.array(v) - a4[ds]["bf16_ppl"], color=COL[0], s=45, zorder=3, label="grouping variant")
        ax.axhline(a4[ds]["Q0_original"], color=COL[1], lw=1.5, ls="--", label="ORIGINAL (= G20261115)")
        ax.set_xticks(range(5)); ax.set_xticklabels([g[-2:] for g in G])
        ax.set_xlabel("grouping G202611xx"); ax.set_ylabel("PPL − BF16 PPL")
        lo = min(v) - a4[ds]["bf16_ppl"]; hi = max(v) - a4[ds]["bf16_ppl"]
        ax.set_ylim(lo - (hi - lo) * .6, hi + (hi - lo) * .6)
        ax.set_title(f"{DSN[ds]}: SD {a4[ds]['S']:.5f}, Q {a4[ds]['Q']:.4f}", fontsize=9, loc="left")
        if j == 0:
            ax.legend(loc="lower left", fontsize=7)
    fig.suptitle("Fig. 2  Dense 4-bit grouping variants on both evaluation sets", x=0.01, ha="left", fontsize=10)
    save(fig, "fig2_dense4_datasets"); plt.close(fig)
    # Fig 3: 3-bit vs 4-bit grouping-level PPL deviations and SD
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for j, ds in enumerate(DS):
        ax = axs[j]
        for b, (a, c_) in enumerate(((a4, COL[0]), (a3, COL[1]))):
            dv = [a[ds]["deviation_from_mean"][g] for g in G]
            ax.plot(range(5), dv, marker="o", ms=6, color=c_, label=f"{4 - b}-bit (SD {a[ds]['S']:.4f})")
        ax.axhline(0, color=INK2, lw=.8)
        ax.set_xticks(range(5)); ax.set_xticklabels([g[-2:] for g in G])
        ax.set_xlabel("grouping G202611xx"); ax.set_ylabel("PPL − bit-width mean")
        ax.set_title(DSN[ds], fontsize=9, loc="left"); ax.legend(fontsize=7)
    fig.suptitle("Fig. 3  Dense grouping deviations: 3-bit vs 4-bit (same Hessians)", x=0.01, ha="left", fontsize=10)
    save(fig, "fig3_dense_3v4_groupings"); plt.close(fig)
    # Fig 4: relative variability SD/Q with window-boot CI (dense) and MoE reference
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    items = []
    for ds in DS:
        items.append((f"dense 4b\n{DSN[ds]}", a4[ds]["R"]["value"], a4[ds]["window_boot"]["R_ci95"], COL[0]))
        items.append((f"dense 3b\n{DSN[ds]}", a3[ds]["R"]["value"], a3[ds]["window_boot"]["R_ci95"], COL[1]))
    for ds in DS:
        r = moe[ds]["individual"]["FR"]
        items.append((f"MoE FR 4b\n{DSN[ds]}\n(pooled 3 seeds)", r["R"]["value"], r["R_window_boot_ci95"], COL[2]))
    for i, (lab, v, ci, c_) in enumerate(items):
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=c_, ms=7, capsize=3)
    for ds_i, ds in enumerate(DS):
        for a, off in ((a4, 0), (a3, 1)):
            v = a[ds]["R"]["value"]; lo, hi = [x / a[ds]["Q"] for x in a[ds]["S_ci95_chi2"]]
            ax.plot([ds_i * 2 + off + .18] * 2, [lo, hi], color=INK2, lw=1, alpha=.6)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([x[0] for x in items], fontsize=7)
    ax.set_ylabel("execution SD / quantization penalty")
    ax.set_title("Fig. 4  Relative execution variability (colored bars: window bootstrap; grey: chi-square over groupings)",
                 fontsize=9, loc="left")
    save(fig, "fig4_relative_variability"); plt.close(fig)
    # Fig 5: SD vs Q
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    mk = {"wikitext": "o", "heldout": "s"}
    for ds in DS:
        for a, c_, lab in ((a4, COL[0], "dense 4b"), (a3, COL[1], "dense 3b")):
            ax.errorbar(a[ds]["Q"], a[ds]["S"], yerr=[[a[ds]["S"] - a[ds]["S_ci95_chi2"][0]], [a[ds]["S_ci95_chi2"][1] - a[ds]["S"]]],
                        fmt=mk[ds], color=c_, ms=7, capsize=3, label=f"{lab}, {DSN[ds]}")
        r = moe[ds]["individual"]["FR"]
        ax.errorbar(r["Q"], r["pooled_within_seed_sd"], yerr=[[r["pooled_within_seed_sd"] - r["pooled_sd_ci95_chi2"][0]],
                    [r["pooled_sd_ci95_chi2"][1] - r["pooled_within_seed_sd"]]], fmt=mk[ds], color=COL[2], ms=7, capsize=3,
                    label=f"MoE FR 4b (pooled), {DSN[ds]}")
    qq = np.linspace(0.005, max(a3[ds]["Q"] for ds in DS) * 1.1, 50)
    for rr in (0.02, 0.05, 0.1):
        ax.plot(qq, rr * qq, color=GRID, lw=1)
        ax.text(qq[-1], rr * qq[-1], f" {rr:g}×Q", color=INK2, fontsize=7, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("quantization penalty Q = mean PPL − BF16"); ax.set_ylabel("execution SD(PPL)")
    ax.set_title("Fig. 5  Raw execution SD vs quantization penalty (chi-square 95% CI)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "fig5_sd_vs_penalty"); plt.close(fig)
    # Fig 6: dose-response SD on both datasets
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for j, pol in enumerate(("natural", "fixed")):
        ax = axs[j]
        for i, ds in enumerate(DS):
            s = [dose["doses"][f"{pol}|k{k}"][ds]["R_PPL"] for k in H.DOSE_K]
            ci = [dose["doses"][f"{pol}|k{k}"][ds]["R_PPL_ci95"] for k in H.DOSE_K]
            xs = np.arange(4) + (i - .5) * .15
            ax.errorbar(xs, s, yerr=[[a - b[0] for a, b in zip(s, ci)], [b[1] - a for a, b in zip(s, ci)]], fmt="o-",
                        color=COL[i], ms=6, capsize=3, label=DSN[ds])
        ax.set_xticks(range(4)); ax.set_xticklabels([f"k={k}" for k in H.DOSE_K])
        ax.set_ylabel("s_k / Q0"); ax.set_title(f"{pol} act-order", fontsize=9, loc="left"); ax.legend(fontsize=7)
    fig.suptitle("Fig. 6  Dense batch-1 dose response: replicate PPL SD / Q0 (chi-square 95% CI, 8 replicates)", x=0.01,
                 ha="left", fontsize=10)
    save(fig, "fig6_dose_heldout"); plt.close(fig)
    # Fig 7: replicate planning
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    dl = np.linspace(0.002, 0.012, 100)
    pw = table["power"]["rows"]
    sel = [(r, c_) for r, c_ in zip([x for x in pw if x["design"].startswith("paired")], COL[:2])]
    for r, c_ in sel:
        ax.plot(dl, (Z * r["s_used"] / dl) ** 2, color=c_, label=f"paired MoE FC−FR, {r['dataset']} (s={r['s_used']:.4f})")
    ind = [x for x in pw if x["design"].startswith("independent arms, dense 4")]
    for r, ls in zip(ind, ("--", ":")):
        ax.plot(dl, (Z * r["s_used"] / dl) ** 2, color=COL[2], ls=ls, label=f"indep. arms dense 4b, {r['dataset']} (√2·{r['single_method_sd']:.4f})")
    ind3 = [x for x in pw if x["design"].startswith("independent arms, dense 3")]
    for r, ls in zip(ind3, ("--", ":")):
        ax.plot(dl, (Z * r["s_used"] / dl) ** 2, color=COL[3], ls=ls, label=f"indep. arms dense 3b, {r['dataset']} (√2·{r['single_method_sd']:.4f})")
    ax.set_yscale("log"); ax.set_xlabel("detectable PPL difference δ"); ax.set_ylabel("required execution replicates")
    for d_ in (0.0025, 0.005, 0.01):
        ax.axvline(d_, color=GRID, lw=1)
    ax.set_title("Fig. 7  Illustrative replicate counts (α 0.05, power 0.8, normal approx.)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5)
    save(fig, "fig7_replicate_planning"); plt.close(fig)


# ---------------------------------------------------------------- outputs
def analyze():
    d = load_all()
    missing = [k for k, v in list(d["dense4"].items()) + list(d["dense3"].items()) + list(d["moe"].items()) if v is None]
    assert not missing, f"missing results: {missing}"
    pr = H.check_protocol(require_code=False)
    # -- validation summary
    val = dict(protocol_sha256=pr["protocol_sha256"])
    val["dense_bf16"] = d["dense_bf16"]["checks"]
    val["moe_bf16"] = d["moe_bf16"]["checks"]
    val["dense4_wikitext_parity"] = {g: d["dense4"][g]["checks"]["wikitext_parity_bitwise"] for g in H.DENSE_CONFIGS}
    val["dose_wikitext_parity"] = {f"{n}|{pol}": d["dose"][n]["policies"][pol]["checks"]["wikitext_parity_bitwise"]
                                   for n in H.DOSE_PARITY for pol in ("natural", "fixed")}
    val["refmatrix_wikitext_parity"] = {f"{i}|{j}": (d["ref"][(i, j)] or {}).get("checks", {}).get("wikitext_parity_bitwise")
                                        for i in G for j in G}
    val["moe_wikitext_parity"] = {n: d["moe"][n]["checks"]["wikitext_parity_bitwise"] for n in H.MOE_PARITY}
    val["repeat_eval_bitwise"] = dict(
        dense4=all(d["dense4"][g]["checks"]["heldout_repeat_bitwise"] for g in H.DENSE_CONFIGS),
        dense3=all(d["dense3"][g]["checks"]["heldout_repeat_bitwise"] and d["dense3"][g]["checks"]["wikitext_repeat_bitwise"] for g in H.DENSE_CONFIGS),
        moe=all(d["moe"][n]["checks"]["heldout_repeat_bitwise"] for n in d["moe"]),
        dose_subset=all(d["dose"][n]["policies"][pol]["checks"].get("heldout_repeat_bitwise", True) for n in d["dose"] for pol in ("natural", "fixed")))
    val["all_configs_valid"] = dict(dense4=all(v["valid"] for v in d["dense4"].values()), dense3=all(v["valid"] for v in d["dense3"].values()),
                                    dose=all(v["valid"] for v in d["dose"].values()),
                                    refmatrix=all((v or {}).get("valid", False) for v in d["ref"].values()),
                                    moe=all(v["valid"] for v in d["moe"].values()))
    val["end_checks"] = dict(dense=jload(p("logs", "dense_end_checks.json")), moe=jload(p("logs", "moe_full_untouched_checks.json")),
                             dense3=(jload(p("logs", "dense3_checks_raw.json")) or {}).get("end"))
    flat = [v for v in list(val["dense4_wikitext_parity"].values()) + list(val["dose_wikitext_parity"].values()) +
            list(val["moe_wikitext_parity"].values()) + list(val["repeat_eval_bitwise"].values()) + list(val["all_configs_valid"].values())]
    val["passed"] = all(flat) and all(v for v in val["dense_bf16"].values() if isinstance(v, bool)) and \
        all(v for v in val["moe_bf16"].values() if isinstance(v, bool))
    jdump(val, p("heldout_evaluator_validation.json"))
    # -- MoE
    moe, mcross = moe_analysis(d)
    jdump(dict(per_config={n: dict(heldout_ppl=r["heldout"]["ppl"], heldout_per_window_nll=r["heldout"]["per_window_nll"],
                                   wikitext_ppl_historical=r["historical_wikitext_ppl"],
                                   wikitext_per_window_nll_historical=r["historical_wikitext_per_window_nll"],
                                   installed_weights_sha256=r["installed_weights_sha256"], checks=r["checks"])
                           for n, r in d["moe"].items()},
               bf16=dict(wikitext_ppl=d["moe_bf16"]["wikitext"]["ppl"], heldout_ppl=d["moe_bf16"]["heldout"]["ppl"],
                         heldout_per_window_nll=d["moe_bf16"]["heldout"]["per_window_nll"]),
               note="WikiText values are the stored historical natural results (parity re-verified bitwise on "
                    f"{H.MOE_PARITY}); held-out values are new"), p("moe_heldout_ppl_results.json"))
    jdump(dict(datasets=moe, cross_dataset=mcross), p("moe_heldout_pairing_analysis.json"))
    # -- dense 4
    a4 = dense_analysis(d, "dense4")
    c4 = dense_cross(a4)
    jdump(dict(datasets=strip(a4), cross_dataset=c4,
               historical_reference=dict(S=0.00503, S_log=3.99e-4, Q=0.1007, R=0.05,
                                         note="historical summary values (WikiText); recomputed here from the same windows"),
               per_config_per_window={g: {ds: d["dense4"][g][ds]["per_window_nll"] for ds in DS} for g in H.DENSE_CONFIGS},
               bf16=dict({ds: d["dense_bf16"][ds]["ppl"] for ds in DS})), p("dense4_heldout_ppl_results.json"))
    # -- dose + refmatrix
    dose = dose_analysis(d, d["dense_bf16"])
    refm = refmatrix_analysis(d)
    jdump(dict(dose=dose, reference_matrix=refm,
               per_config_heldout={n: {pol: d["dose"][n]["policies"][pol]["heldout"]["ppl"] for pol in ("natural", "fixed")}
                                   for n in d["dose"]}), p("dense4_dose_heldout_analysis.json"))
    # -- dense 3
    a3 = dense_analysis(d, "dense3")
    c3 = dense_cross(a3)
    man = {g: dict(per_layer={L: dict(file=os.path.join(H.CACHE, "dense3", g, f"L{L}.pt"), file_sha256=v["file_sha256"],
                                      hessian_sha256=v["hessian_sha256"], weight_sha256=v["weight_sha256"],
                                      code_range={k: [m["code_min"], m["code_max"]] for k, m in v["meta"].items()},
                                      levels_used={k: m["n_levels_used"] for k, m in v["meta"].items()},
                                      frac_bf16_equal_to_4bit={k: x for k, x in v["frac_bf16_equal_to_4bit_historical"].items()})
                              for L, v in d["dense3"][g]["per_layer"].items()},
                   installed_digest=d["dense3"][g]["checks"]["installed_digest"], gptq=d["dense3"][g]["gptq"], gptq_s=d["dense3"][g]["gptq_s"])
           for g in H.DENSE_CONFIGS}
    jdump(dict(configs=man, solver="moe_calib.dense_gptq.quantize_layer -> gptq.quantize_expert (unchanged), bits=3, maxq=7",
               storage=os.path.join(H.CACHE, "dense3")), p("dense3_quantization_manifest.json"))
    raw = jload(p("logs", "dense3_checks_raw.json")) or {}
    corr = dict(per_config={g: {k: v for k, v in d["dense3"][g]["checks"].items() if k != "installed_digest"} for g in H.DENSE_CONFIGS},
                null_control=raw.get("null_G20261115_vs_ORIGINAL"), fresh_process_repeat=jload(p("logs", "dense3_fresh_process_repeat.json")),
                bits4_parity={g: {L: v["bits4_parity"] for L, v in d["dense3"][g]["per_layer"].items()} for g in H.DENSE_CONFIGS},
                end=raw.get("end"))
    corr["passed"] = all(d["dense3"][g]["valid"] for g in H.DENSE_CONFIGS) and bool((corr["null_control"] or {}).get("passed")) \
        and bool((corr["fresh_process_repeat"] or {}).get("passed"))
    jdump(corr, p("dense3_correctness.json"))
    jdump(dict(datasets=strip(a3), cross_dataset=c3,
               per_config_per_window={g: {ds: d["dense3"][g][ds]["per_window_nll"] for ds in DS} for g in H.DENSE_CONFIGS}),
          p("dense3_ppl_results.json"))
    bw = bitwidth(a4, a3)
    jdump(bw, p("bitwidth_variability_comparison.json"))
    # -- joint table
    rows = []
    for b, a in ((4, a4), (3, a3)):
        for ds in DS:
            rows.append(dict(model="Dense", bits=b, dataset=DSN[ds], mean_ppl=a[ds]["mean_ppl"], bf16_ppl=a[ds]["bf16_ppl"],
                             Q=a[ds]["Q"], S=a[ds]["S"], R=a[ds]["R"]["value"], R_status=a[ds]["R"]["status"],
                             S_log=a[ds]["S_log"], R_log=a[ds]["R_log"]["value"],
                             sampling="1 calibration seed, 5 groupings (4 single-row perturbations + exact null), ddof 1",
                             source="historical weights re-evaluated" if b == 4 else "new 3-bit solves"))
    for st in H.MOE_STRATS:
        for ds in DS:
            r = moe[ds]["individual"][st]
            rows.append(dict(model=f"MoE {st}", bits=4, dataset=DSN[ds], mean_ppl=r["mean_ppl"], bf16_ppl=moe[ds]["bf16_ppl"],
                             Q=r["Q"], S=r["pooled_within_seed_sd"], R=r["R"]["value"], R_status=r["R"]["status"],
                             S_log=r["pooled_sd_log"], seed0_S=r["seed0_sd"], seed0_R=r["R_seed0"]["value"],
                             sampling="3 allocation seeds x 5 groupings; pooled within-seed SD (12 df) - NOT comparable 1:1 "
                                      "with the dense single-seed SD; seed-0-only SD given as seed0_S"))
    A_lab = classify_A(moe, mcross, a4, c4)
    table = dict(primary_table=rows,
                 moe_paired=[dict(dataset=DSN[ds], mean_delta=moe[ds]["variance"]["overall_mean_delta"],
                                  pooled_sd_delta=moe[ds]["pooled_sd_delta"], pooled_sd_delta_ci95=moe[ds]["pooled_sd_delta_ci95_chi2"],
                                  pooled_sd_mean_nll=moe[ds]["pooled_sd_delta_mean_nll"], fr_better=moe[ds]["n_positive"],
                                  fc_better=moe[ds]["n_negative"], R1=moe[ds]["R1"]["count"], R2_seed0=moe[ds]["R2_seed0"]["count"],
                                  hier_boot_ci95=moe[ds]["hierarchical_bootstrap"]["ci95"], decision=moe[ds]["decision"]) for ds in DS],
                 classification_A=A_lab, classification_B=bw["overall_classification"],
                 classification_B_by_dataset={ds: bw[ds]["classification"] for ds in DS})
    table["power"] = power(moe, mcross, a4, a3)
    jdump(table, p("dataset_bitwidth_comparison.json"))
    jdump(table["power"], p("replicate_power_analysis.json"))
    cost(d)
    figures(moe, mcross, a4, a3, bw, dose, refm, table)
    print(json.dumps(dict(A=A_lab, B=bw["overall_classification"], validation=val["passed"]), indent=1))


def classify_A(moe, mcross, a4, c4):
    w, h = moe["wikitext"], moe["heldout"]
    a3_ = h["pooled_sd_delta_ci95_chi2"][1] < 0.5 * w["pooled_sd_delta"] and a4["heldout"]["S_ci95_chi2"][1] < 0.5 * a4["wikitext"]["S"]
    inside = lambda band: band is not None and band[0] <= 1 <= band[1]
    a1 = inside(mcross["R_dataset_F_band_95"]) and inside(c4["S_ratio_F_band_95"]) and inside(c4["R_ratio_F_band_95"])
    lab = "A3" if a3_ else ("A1" if a1 else "A2")
    return dict(label=lab, moe_R_dataset=mcross["R_dataset"], moe_band=mcross["R_dataset_F_band_95"],
                dense_S_ratio=c4["S_ratio"], dense_S_band=c4["S_ratio_F_band_95"], dense_R_ratio=c4["R_ratio"],
                dense_R_band=c4["R_ratio_F_band_95"], rule="protocol decision_criteria.A")


def cost(d):
    raw = jload(p("logs", "cost_raw.json")) or {}
    def tot(stage):
        v = raw.get(stage, {})
        s = 0.0
        for x in v.values():
            s += x if isinstance(x, (int, float)) else x.get("total_s", 0.0)
        return s
    out = dict(stage_seconds={k: tot(k) for k in raw},
               dense3_gptq_s={g: d["dense3"][g]["gptq_s"] for g in H.DENSE_CONFIGS},
               moe_install_s_median=float(np.median([r["install_s"] for r in d["moe"].values()])),
               moe_config_s_median=float(np.median([r["seconds"] for r in d["moe"].values()])),
               moe_peak_cpu_gb=max(r["peak_cpu_gb"] for r in d["moe"].values()),
               dense_eval_s_median=float(np.median([d["dense4"][g]["heldout"]["eval_s"] for g in H.DENSE_CONFIGS])),
               moe_eval_s_median=float(np.median([r["heldout"]["eval_s"] for r in d["moe"].values()])),
               end_checks=dict(dense=jload(p("logs", "dense_end_checks.json")), moe=jload(p("logs", "moe_full_untouched_checks.json"))),
               gptq_solves=dict(new_3bit=6 + 4, bits4_parity_resolves=6, historical_4bit_reruns_used_as_results=0),
               storage_gb=sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(H.CACHE) for f in fs) / 1e9)
    jdump(out, p("cost_results.json"))


if __name__ == "__main__":
    analyze()
