"""Analysis for runs_wt2_fresh_generalization (CPU). Reuses diagnostics/evaluation_uncertainty.py machinery.

stage_synth   cluster-structured synthetic validation / power with the fresh_article article sizes
stage_analyze all analysis JSONs + decision + cost; figures via report_wt2_fresh_generalization.py
"""
from __future__ import annotations
import math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation_uncertainty as E          # noqa: E402
import wt2_fresh_generalization as W        # noqa: E402

p, jdump, jload, log = W.p, W.jdump, W.jload, W.log
GROUPS, SEEDS = W.GROUPS, W.SEEDS
HBV = os.path.join(W.HBV_DIR, "results")
NW_ORIG = 128
COH = ["original", "fresh_article", "fresh_window", "C4HO"]
COH_X = ["fresh_article_noearly", "fresh_article_window", "fresh_window_window"]   # sensitivity variants
B_COV, B_OUT, B_IN, K_SPLIT, R_NULL, K_NULL, B_CROSS = 2000, 2000, 400, 1000, 500, 40, 2000


# =====================================================================================================================
# data
# =====================================================================================================================
def per_window(rec):
    return np.asarray(rec["per_window_nll"], float) / np.asarray(rec["per_window_tokens"], float)


def orig_pw(h):
    """original-WT2 per-window mean NLL: re-evaluated array if present, else the verified historical array
    (heldout run evaluated WT2 only for the parity subset; alignment verified in runs_evaluation_uncertainty)."""
    if "wikitext" in h:
        return per_window(h["wikitext"])
    return np.asarray(h["historical_wikitext_per_window_nll"], float) / 511.0


def load_all():
    """L[cohort][model][config] = (n,) per-window mean NLL; U[cohort] = unit ids (n,)."""
    L = {c: {"moe": {}, "dense4": {}, "dense3": {}} for c in COH}
    for n in W.MOE_CONFIGS + W.MOE_REFS + ["BF16"]:
        h = jload(os.path.join(HBV, "moe", f"{n}.json"))
        L["original"]["moe"][n] = orig_pw(h)
        L["C4HO"]["moe"][n] = per_window(h["heldout"])
        f = jload(p("results", "moe", f"{n}.json"))
        for c in ("fresh_article", "fresh_window"):
            L[c]["moe"][n] = per_window(f[c])
    for bits in (4, 3):
        for g in W.DENSE_CONFIGS:
            h = jload(os.path.join(HBV, "dense" if bits == 4 else "dense3", f"dense{bits}_{g}.json"))
            f = jload(p("results", "dense", f"dense{bits}_{g}.json"))
            L["original"][f"dense{bits}"][g] = orig_pw(h)
            L["C4HO"][f"dense{bits}"][g] = per_window(h["heldout"])
            for c in ("fresh_article", "fresh_window"):
                L[c][f"dense{bits}"][g] = per_window(f[c])
    hb = jload(os.path.join(HBV, "dense", "BF16.json")); fb = jload(p("results", "dense", "BF16.json"))
    for c, src in (("original", hb["wikitext"]), ("C4HO", hb["heldout"]), ("fresh_article", fb["fresh_article"]),
                   ("fresh_window", fb["fresh_window"])):
        L[c]["dense4"]["BF16"] = L[c]["dense3"]["BF16"] = per_window(src)
    U = dict(original=np.asarray([r["article"] for r in E.jload(os.path.join(E.OUT, "window_structure_audit.json"))["WT2"]["rows"]]),
             C4HO=np.arange(128),
             fresh_article=np.asarray([r["article"] for r in jload(p("fresh_article_manifest.json"))["windows"]]),
             fresh_window=np.asarray([r["article"] for r in jload(p("fresh_window_manifest.json"))["windows"]]))
    return L, U


def sets_for(L, c):
    """MoE / dense4 / dense3 analysis sets for one cohort."""
    out = {}
    m = L[c]["moe"]
    cols = [f"s{s}_{g}" for s in SEEDS for g in GROUPS]
    out["moe"] = dict(cols=cols, blocks=[list(range(5 * i, 5 * i + 5)) for i in range(3)],
                      X=np.stack([m[f"FC_s{s}_{g}"] - m[f"FR_s{s}_{g}"] for s in SEEDS for g in GROUPS], 1),
                      FR=np.stack([m[f"FR_s{s}_{g}"] for s in SEEDS for g in GROUPS], 1),
                      FC=np.stack([m[f"FC_s{s}_{g}"] for s in SEEDS for g in GROUPS], 1),
                      ref=m["FC_ORIGINAL"] - m["FR_ORIGINAL"], bf16=m["BF16"])
    out["moe"]["C"], out["moe"]["contrast_ids"] = out["moe"]["X"], cols
    for b in ("dense4", "dense3"):
        d = L[c][b]
        X = np.stack([d[g] for g in GROUPS], 1)
        out[b] = dict(cols=GROUPS, blocks=[list(range(5))], X=X, ref=d["ORIGINAL"], bf16=d["BF16"],
                      C=X[:, :4] - d["ORIGINAL"][:, None], contrast_ids=GROUPS[:4],
                      null_exact=bool(np.array_equal(d["G20261115"], d["ORIGINAL"])))
    return out


def cohort_views(L, U):
    """dict name -> (sets, uid, scheme label)."""
    V = {c: (sets_for(L, c), U[c], "article" if c != "C4HO" else "document") for c in COH}
    early = set(jload(p("article_overlap_audit.json"))["prior_use"]["early_dev_eval"]["cohort_A_articles_flagged"])
    keep = np.asarray([a not in early for a in U["fresh_article"]])
    S = sets_for(L, "fresh_article")
    S2 = {k: {kk: (vv[keep] if isinstance(vv, np.ndarray) and vv.shape[:1] == keep.shape else vv) for kk, vv in v.items()}
          for k, v in S.items()}
    V["fresh_article_noearly"] = (S2, U["fresh_article"][keep], "article")
    V["fresh_article_window"] = (S, np.arange(len(U["fresh_article"])), "window")
    V["fresh_window_window"] = (sets_for(L, "fresh_window"), np.arange(len(U["fresh_window"])), "window")
    return V


# =====================================================================================================================
# statistics
# =====================================================================================================================
def variance(St, uid, key, nested_on=True):
    X, bl = St["X"], St["blocks"]
    U = E.Units(X, uid)
    th = X.mean(0)
    dr = U.theta(E.draw_counts(E.rng_for(100, *key), U.m, B_COV))
    vo, vs = E.vstats_from(th, dr, bl)
    vc = vo - vs
    cov = np.cov(dr, rowvar=False)
    r = dict(n_units=int(U.m), n_windows=int(len(uid)), V_obs=vo, V_samp=vs, V_corr=vc, V_corr_constrained=max(vc, 0.0),
             SD_corr=math.sqrt(vc) if vc > 0 else None, sampling_fraction=vs / vo if vo > 0 else None,
             reliability_lambda=vc / vo if vo > 0 else None, df=E.df_of(bl))
    extra = dict(cov=cov, theta=th, se=np.sqrt(np.diag(cov)))
    if nested_on:
        o, s_, c_ = E.nested(U, bl, E.rng_for(101, *key), B_OUT, B_IN)
        d = c_ - vo
        lo, hi = vc - np.quantile(d, .975), vc - np.quantile(d, .025)
        on, sn, cn = E.nested(U, bl, E.rng_for(102, *key), B_OUT, B_IN, null=True, theta_hat=th)
        r.update(V_obs_ci=[float(np.quantile(o, .025)), float(np.quantile(o, .975))],
                 V_samp_ci=[float(np.quantile(s_, .025)), float(np.quantile(s_, .975))],
                 V_corr_pivot_ci=[float(lo), float(hi)], null_p=float((cn >= vc).mean()),
                 SD_corr_ci_upper=math.sqrt(hi) if hi > 0 else 0.0, SD_corr_ci_lower=math.sqrt(lo) if lo > 0 else 0.0,
                 resolved=bool(lo > 0 and (cn >= vc).mean() < 0.05))
        extra["vcorr_star_minus_vobs"] = d
    return r, extra


def split_half(St, uid, key, null_on=True):
    X, C, bl = St["X"], St["C"], St["blocks"]
    rng = E.rng_for(103, *key)
    M = E.split_masks(rng, None, None, K_SPLIT, uid=uid)
    tA, tB = E.half_thetas(X, M)
    cA, cB = E.half_thetas(C, M)
    Uc = E.Units(C, uid)
    se_c = Uc.theta(E.draw_counts(E.rng_for(104, *key), Uc.m, B_COV)).std(0, ddof=1)
    thr = np.abs(C.mean(0)) > se_c
    st = E.reliability_stats(E.center(tA, bl), E.center(tB, bl), cA, cB, bl, se_c, thr)
    out = dict(n_partitions=K_SPLIT, n_units=int(len(np.unique(uid))),
               mean_half_windows=[float(M.sum(1).mean()), float((~M).sum(1).mean())],
               n_threshold_contrasts=int(thr.sum()))
    for k, v in st.items():
        out[k] = ({f"s{i}": E.summarize(v[:, i]) for i in range(v.shape[1])} if k == "per_seed_pearson" else E.summarize(v))
    raw = dict(pearson=st["pearson"])
    if null_on:
        th = X.mean(0)
        U = E.Units(X, uid)
        rn = E.rng_for(105, *key)
        med, allr = [], []
        for r in range(R_NULL):
            idx = rn.integers(0, U.m, U.m)
            rows, orig = [], []
            for cc, u in enumerate(idx):
                w = np.nonzero(U.inv == u)[0]
                rows.append(w); orig.append(np.full(len(w), cc))
            rows = np.concatenate(rows); orig = np.concatenate(orig)
            Xn = X[rows] - E.center(th, bl)[None, :]
            Mn = E.split_masks(rn, None, None, K_NULL, uid=orig)
            a, b = E.half_thetas(Xn, Mn)
            rr = E.pearson(E.center(a, bl), E.center(b, bl))
            med.append(np.nanmedian(rr)); allr.append(rr)
        med = np.asarray(med)
        obs = out["pearson"]["median"] if out["pearson"] else float("nan")
        out["null"] = dict(null_median_pearson=E.summarize(med), observed_median=obs, p_value=float((med >= obs).mean()))
        raw["null_med"] = med
    return out, raw


def centred(St, th=None):
    th = St["X"].mean(0) if th is None else th
    return E.center(th, St["blocks"])


def ranks(St, z):
    bl = St["blocks"]
    return np.concatenate([E.rankdata(z[b]) for b in bl]) if len(bl) > 1 else E.rankdata(z)


def transfer(Sa, ua, Sb, ub, key, lam_a, lam_b, res_a, res_b):
    za, zb = centred(Sa), centred(Sb)
    bl = Sa["blocks"]
    r = float(E.pearson(za, zb))
    sp = float(E.pearson(ranks(Sa, za), ranks(Sb, zb)))
    pl = E.pair_list(bl)
    po = float(np.mean([np.sign(za[i] - za[j]) * np.sign(zb[i] - zb[j]) > 0 for i, j in pl]))
    ca, cb = Sa["C"].mean(0), Sb["C"].mean(0)
    sg = float(np.mean(np.sign(ca) * np.sign(cb) > 0))
    rng = E.rng_for(106, *key)
    Ua, Ub = E.Units(Sa["X"], ua), E.Units(Sb["X"], ub)
    da = E.center(Ua.theta(E.draw_counts(rng, Ua.m, B_CROSS)), bl)
    db = E.center(Ub.theta(E.draw_counts(rng, Ub.m, B_CROSS)), bl)
    rb = E.pearson(da, db)
    exp_r = math.sqrt(lam_a * lam_b) if (lam_a and lam_b and lam_a > 0 and lam_b > 0) else None
    o = dict(pearson=r, pearson_ci95=[float(np.nanquantile(rb, .025)), float(np.nanquantile(rb, .975))],
             spearman=sp, pair_order_agreement=po, n_pairs=len(pl), sign_agreement=sg, n_contrasts=int(len(ca)),
             sign_disagreements=[Sa["contrast_ids"][i] for i in range(len(ca)) if np.sign(ca[i]) != np.sign(cb[i])],
             expected_r_if_identical_true_effects=exp_r,
             disattenuated_r=(r / exp_r if (exp_r and res_a and res_b) else None),
             disattenuation_reported=bool(exp_r and res_a and res_b))
    if "ref" in Sa and Sa["X"].shape[1] == 5:     # dense: rank of ORIGINAL among the 4 non-null groupings + ORIGINAL
        for nm, S in (("a", Sa), ("b", Sb)):
            v = np.append(S["X"][:, :4].mean(0), S["ref"].mean())
            o[f"ORIGINAL_rank_{nm}_of5"] = int(np.argsort(np.argsort(v))[-1] + 1)     # 1 = best (lowest NLL)
    return o, rb


def window_ci(St, uid, key):
    X = St["X"]
    U = E.Units(X, uid)
    C = E.draw_counts(E.rng_for(107, *key), U.m, B_COV)
    dD = U.theta(C)
    th = X.mean(0)
    se = dD.std(0, ddof=1)
    lo, hi = np.quantile(dD, [.025, .975], axis=0)
    pb = np.minimum(1.0, 2 * np.minimum((dD <= 0).mean(0), (dD >= 0).mean(0)))
    m = len(pb); order = np.argsort(pb); bh = np.empty(m); run = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]; run = min(run, pb[i] * m / (rank + 1)); bh[i] = run
    tmax = (np.abs(dD - th[None, :]) / se[None, :]).max(1)
    pmax = np.array([(tmax >= t).mean() for t in np.abs(th) / se])
    UR, UC = E.Units(St["FR"], uid), E.Units(St["FC"], uid)
    dP = np.exp(UC.theta(C)) - np.exp(UR.theta(C))
    pth = np.exp(St["FC"].mean(0)) - np.exp(St["FR"].mean(0))
    pairs = {c: dict(mean_nll_diff=float(th[g]), ci95=[float(lo[g]), float(hi[g])], se=float(se[g]),
                     excludes_zero=bool(lo[g] > 0 or hi[g] < 0), favored="FR" if th[g] > 0 else "FC",
                     p_boot=float(pb[g]), p_BH=float(bh[g]), p_maxT=float(pmax[g]),
                     ppl_diff=float(pth[g]), ppl_diff_ci95=[float(np.quantile(dP[:, g], .025)), float(np.quantile(dP[:, g], .975))])
             for g, c in enumerate(St["cols"])}
    sig = [c for c in St["cols"] if pairs[c]["excludes_zero"]]
    # execution-averaged FR/FC
    gm = dD.mean(1)
    seed_means = np.array([th[b].mean() for b in St["blocks"]])
    t975_2 = 4.302652729911275
    ex = dict(grand_mean=float(th.mean()), eval_only_ci95=[float(np.quantile(gm, .025)), float(np.quantile(gm, .975))],
              eval_only_se=float(gm.std(ddof=1)), seed_means=seed_means.tolist(),
              seed_level_t_ci95=[float(seed_means.mean() - t975_2 * seed_means.std(ddof=1) / math.sqrt(3)),
                                 float(seed_means.mean() + t975_2 * seed_means.std(ddof=1) / math.sqrt(3))])
    return dict(pairs=pairs, n_excluding_zero=len(sig), significant=[dict(pair=c, favored=pairs[c]["favored"]) for c in sig],
                directions=sorted({pairs[c]["favored"] for c in sig}),
                opposite_directions=len({pairs[c]["favored"] for c in sig}) > 1,
                n_BH_05=int((bh < 0.05).sum()), n_maxT_05=int((pmax < 0.05).sum()), execution_averaged=ex), dD


# =====================================================================================================================
# synth
# =====================================================================================================================
def stage_synth(a):
    W.check_protocol(code=False)
    t0 = time.time()
    # use ORIGINAL-WT2 data only (no fresh results) to build realistic cluster noise
    Lo = {"original": {"moe": {}, "dense4": {}, "dense3": {}}}
    for n in W.MOE_CONFIGS:
        Lo["original"]["moe"][n] = orig_pw(jload(os.path.join(HBV, "moe", f"{n}.json")))
    uid0 = np.asarray([r["article"] for r in E.jload(os.path.join(E.OUT, "window_structure_audit.json"))["WT2"]["rows"]])
    X0 = np.stack([Lo["original"]["moe"][f"FC_s{s}_{g}"] - Lo["original"]["moe"][f"FR_s{s}_{g}"] for s in SEEDS for g in GROUPS], 1)
    bl = [list(range(5 * i, 5 * i + 5)) for i in range(3)]
    th0 = X0.mean(0)
    Z = X0 - E.center(th0, bl)[None, :]                 # effect-free original windows (keeps window covariance)
    arts = np.unique(uid0)
    sizes = np.unique([r["article"] for r in jload(p("fresh_article_manifest.json"))["windows"]], return_counts=True)[1]
    vo0 = E.vstats_from(th0, E.Units(X0, uid0).theta(E.draw_counts(E.rng_for(110), len(arts), 1000)), bl)
    vc0 = vo0[0] - vo0[1]
    delta_true = E.center(th0, bl) * math.sqrt(max(vc0, 0) / vo0[0])   # true effects with variance = original V_corr
    rng = E.rng_for(111)

    def make(delta):
        rows, uid = [], []
        for k, sz in enumerate(sizes):
            a_ = arts[rng.integers(len(arts))]
            w = np.nonzero(uid0 == a_)[0]
            take = rng.choice(w, sz, replace=len(w) < sz)
            rows.append(Z[take] + delta[None, :]); uid += [k] * sz
        return np.concatenate(rows), np.asarray(uid)

    res = {}
    for label, delta in (("null", np.zeros(15)), ("effect_equal_original_Vcorr", delta_true)):
        ps, sp, rs, vcs = [], [], [], []
        for r in range(150):
            X, uid = make(delta)
            Uu = E.Units(X, uid)
            th = X.mean(0)
            dr = Uu.theta(E.draw_counts(rng, Uu.m, 400))
            vo, vs = E.vstats_from(th, dr, bl)
            on, sn, cn = E.nested(Uu, bl, rng, 200, 100, null=True, theta_hat=th)
            ps.append(float((cn >= vo - vs).mean())); vcs.append(vo - vs)
            M = E.split_masks(rng, None, None, 40, uid=uid)
            tA, tB = E.half_thetas(X, M)
            sp.append(float(np.nanmedian(E.pearson(E.center(tA, bl), E.center(tB, bl)))))
            rs.append(float(E.pearson(E.center(th, bl), E.center(th0, bl))))
        ps = np.asarray(ps)
        res[label] = dict(reps=150, frac_null_p_below_05=float((ps < 0.05).mean()), mean_V_corr=float(np.mean(vcs)),
                          true_V=float((delta_true ** 2).sum() / 12) if label != "null" else 0.0,
                          median_split_half_r=float(np.median(sp)), frac_split_half_ge_0p5=float((np.asarray(sp) >= 0.5).mean()),
                          median_r_with_original_estimate=float(np.median(rs)),
                          frac_r_with_original_gt_0=float((np.asarray(rs) > 0).mean()))
    out = dict(design=("semi-parametric: windows drawn from effect-free ORIGINAL-WT2 MoE paired-effect articles (keeps "
                       "real window/article covariance), arranged into clusters with the exact fresh_article sizes " +
                       str(sizes.tolist()) + "; 'effect' adds true within-seed effects with variance = original V_corr "
                       "and the original ranking"),
               moe=res, original_V_corr=vc0, original_V_obs=vo0[0],
               inherited_validation="runs_evaluation_uncertainty/statistical_validation.json (tests A-F, all passed)",
               interpretation=("frac_null_p_below_05 under 'null' = size of the common-effect null test with 10 article "
                               "clusters; under 'effect' = power to detect the original-magnitude effect"),
               seconds=time.time() - t0)
    out["passed_size"] = res["null"]["frac_null_p_below_05"] <= 0.10
    jdump(out, p("statistical_validation.json"))
    log(f"[synth] null size {res['null']['frac_null_p_below_05']:.3f}; power {res['effect_equal_original_Vcorr']['frac_null_p_below_05']:.3f}; "
        f"split-half r effect {res['effect_equal_original_Vcorr']['median_split_half_r']:.2f}; r with original "
        f"{res['effect_equal_original_Vcorr']['median_r_with_original_estimate']:.2f}")


# =====================================================================================================================
# analyze
# =====================================================================================================================
def stage_analyze(a):
    W.check_protocol(code=False)
    t0 = time.time()
    L, U = load_all()
    V = cohort_views(L, U)
    models = ["moe", "dense4", "dense3"]
    names = COH + COH_X
    var, spl, raw = {}, {}, {}
    for mi, mdl in enumerate(models):
        var[mdl], spl[mdl] = {}, {}
        for ci, c in enumerate(names):
            S, uid, sch = V[c]
            St = S[mdl]
            r, ex = variance(St, uid, (mi, ci), nested_on=True)
            r["scheme"] = sch
            var[mdl][c] = r
            raw[f"{mdl}|{c}|vstar"] = ex["vcorr_star_minus_vobs"]
            if c in COH + ["fresh_article_noearly"]:
                sh, rr = split_half(St, uid, (mi, ci), null_on=True)
                spl[mdl][c] = sh
                raw[f"{mdl}|{c}|pearson"] = rr["pearson"]; raw[f"{mdl}|{c}|null_med"] = rr["null_med"]
            log(f"[analyze] {mdl} {c}: Vobs {r['V_obs']:.2e} Vs {r['V_samp']:.2e} Vc {r['V_corr']:+.2e} CI {np.round(r['V_corr_pivot_ci'], 9)} "
                f"p {r['null_p']:.4f}" + (f" | split r {spl[mdl][c]['pearson']['median']:.2f} p {spl[mdl][c]['null']['p_value']:.3f}"
                                          if c in spl[mdl] else ""))
    # ---- descriptive per cohort
    desc = {}
    for mdl in models:
        desc[mdl] = {}
        for c in COH:
            S, uid, _ = V[c]
            St = S[mdl]
            th = St["X"].mean(0)
            d = dict(n_windows=int(len(uid)), n_units=int(len(np.unique(uid))), scored_targets=int(len(uid) * 511),
                     bf16_ppl=float(math.exp(St["bf16"].mean())))
            if mdl == "moe":
                d.update(mean_delta=float(th.mean()), seed_SD={f"s{i}": float(th[b].std(ddof=1)) for i, b in enumerate(St["blocks"])},
                         pooled_within_seed_SD=float(math.sqrt(sum(((th[b] - th[b].mean()) ** 2).sum() for b in St["blocks"]) / 12)),
                         FR_better=int((th > 0).sum()), FC_better=int((th < 0).sum()),
                         centred=dict(zip(St["cols"], E.center(th, St["blocks"]).tolist())),
                         paired=dict(zip(St["cols"], th.tolist())),
                         s0_ORIGINAL_delta=float(St["ref"].mean()),
                         FR_mean_ppl=float(np.exp(St["FR"].mean(0)).mean()), FC_mean_ppl=float(np.exp(St["FC"].mean(0)).mean()))
            else:
                ppl = np.exp(th)
                d.update(mean_ppl=float(ppl.mean()), SD_ppl=float(ppl.std(ddof=1)), SD_log_ppl=float(th.std(ddof=1)),
                         ORIGINAL_ppl=float(math.exp(St["ref"].mean())), penalty_Q=float(ppl.mean() - math.exp(St["bf16"].mean())),
                         deviation_from_ORIGINAL_nll=dict(zip(GROUPS, (th - St["ref"].mean()).tolist())),
                         V_obs_nll=float(((th - th.mean()) ** 2).sum() / 4), exact_null_G20261115=St["null_exact"])
            desc[mdl][c] = d
    # ---- transfer
    pairs = [("original", "fresh_article"), ("original", "fresh_window"), ("fresh_article", "fresh_window"),
             ("original", "C4HO"), ("original", "fresh_article_noearly")]
    trans = {}
    for mi, mdl in enumerate(models):
        trans[mdl] = {}
        for pi, (x, y) in enumerate(pairs):
            o, rb = transfer(V[x][0][mdl], V[x][1], V[y][0][mdl], V[y][1], (mi, pi),
                             var[mdl][x]["reliability_lambda"], var[mdl][y]["reliability_lambda"],
                             var[mdl][x]["resolved"], var[mdl][y]["resolved"])
            trans[mdl][f"{x}__vs__{y}"] = o
            raw[f"{mdl}|{x}__vs__{y}|rboot"] = rb
        # within-cohort stepped-up reliability for comparability
        for c in COH:
            r = spl[mdl][c]["pearson"]["median"]
            trans[mdl].setdefault("within_cohort_spearman_brown", {})[c] = 2 * r / (1 + r) if r > -1 else None
    # ---- window CIs (MoE)
    wci = {}
    for ci, c in enumerate(COH + ["fresh_article_noearly", "fresh_article_window", "fresh_window_window"]):
        o, dD = window_ci(V[c][0]["moe"], V[c][1], (ci,))
        wci[c] = o
        raw[f"moe|{c}|pair_se"] = dD.std(0, ddof=1)
    for c in ("fresh_article", "fresh_window"):
        wci[f"original_vs_{c}"] = dict(
            point_sign_agreement=int(sum(np.sign(wci["original"]["pairs"][k]["mean_nll_diff"]) == np.sign(wci[c]["pairs"][k]["mean_nll_diff"])
                                         for k in wci["original"]["pairs"])),
            opposite_significant=[k for k in wci["original"]["pairs"] if wci["original"]["pairs"][k]["excludes_zero"] and
                                  wci[c]["pairs"][k]["excludes_zero"] and wci["original"]["pairs"][k]["favored"] != wci[c]["pairs"][k]["favored"]],
            historical_significant_in_fresh={x["pair"]: dict(original=x["favored"], fresh_point=wci[c]["pairs"][x["pair"]]["favored"],
                                                             fresh_ci=wci[c]["pairs"][x["pair"]]["ci95"],
                                                             fresh_significant=wci[c]["pairs"][x["pair"]]["excludes_zero"])
                                             for x in wci["original"]["significant"]})
    for mdl in models:
        for c in COH:
            S = V[c][0]["moe"]
        for c in COH:
            if mdl == "moe":
                vc = max(var["moe"][c]["V_corr"], 0.0)
                ex = wci[c]["execution_averaged"]
                ex["combined_se"] = math.sqrt(ex["eval_only_se"] ** 2 + vc / 15)
                ex["combined_ci95"] = [ex["grand_mean"] - 1.96 * ex["combined_se"], ex["grand_mean"] + 1.96 * ex["combined_se"]]
                ex["distinguishable_from_zero"] = bool(ex["combined_ci95"][0] > 0 or ex["combined_ci95"][1] < 0 or
                                                       ex["seed_level_t_ci95"][0] > 0 or ex["seed_level_t_ci95"][1] < 0)
    # ---- decision
    def exec_verdict(mdl, c):
        v, s = var[mdl][c], spl[mdl][c]
        rel = s["pearson"]["median"] >= 0.5 and s["null"]["p_value"] < 0.05
        if v["resolved"]:
            return "EXECUTION-EFFECT-RESOLVED", rel
        if v["null_p"] >= 0.05 and not rel:
            return "EXECUTION-EFFECT-BELOW-EVALUATION-RESOLUTION", rel
        return "EXECUTION-EFFECT-INCONCLUSIVE", rel
    dec = dict(rules=jload(p("protocol.json"))["decision_rules"], settings={})
    for mdl in models:
        dd = {}
        for c in ("original", "fresh_article", "fresh_window", "C4HO"):
            ev, rel = exec_verdict(mdl, c)
            dd[c] = dict(execution=ev, reliable_split_half=rel, V_corr=var[mdl][c]["V_corr"],
                         V_corr_ci=var[mdl][c]["V_corr_pivot_ci"], null_p=var[mdl][c]["null_p"],
                         split_half_median_r=spl[mdl][c]["pearson"]["median"], split_half_null_p=spl[mdl][c]["null"]["p_value"])
        for c in ("fresh_article", "fresh_window"):
            t = trans[mdl][f"original__vs__{c}"]
            lam_o, lam_f = var[mdl]["original"]["reliability_lambda"], var[mdl][c]["reliability_lambda"]
            exp_r = t["expected_r_if_identical_true_effects"]
            if t["pearson"] > 0 and t["pearson_ci95"][0] > 0 and dd[c]["execution"] != "EXECUTION-EFFECT-BELOW-EVALUATION-RESOLUTION":
                g = "ARTICLE-GENERALIZATION-SUPPORTED"
            elif var[mdl][c]["null_p"] < 0.05 and exp_r is not None and t["pearson_ci95"][1] < exp_r:
                g = "ARTICLE-GENERALIZATION-NOT-SUPPORTED"
            else:
                g = "ARTICLE-GENERALIZATION-INCONCLUSIVE"
            dd[c]["generalization_from_original"] = g
            dd[c]["transfer_r"] = t["pearson"]; dd[c]["transfer_r_ci"] = t["pearson_ci95"]
        dec["settings"][mdl] = dd
    dec["verdicts"] = {mdl: dict(article_generalization=dec["settings"][mdl]["fresh_article"]["generalization_from_original"],
                                 execution_effect_fresh_article=dec["settings"][mdl]["fresh_article"]["execution"],
                                 secondary_same_article_generalization=dec["settings"][mdl]["fresh_window"]["generalization_from_original"],
                                 secondary_execution_effect_fresh_window=dec["settings"][mdl]["fresh_window"]["execution"])
                       for mdl in models}
    # ---- write
    jdump(dict(moe=desc["moe"]), p("moe_fresh_ppl_results.json"))
    jdump(dict(dense4=desc["dense4"]), p("dense4_fresh_ppl_results.json"))
    jdump(dict(dense3=desc["dense3"]), p("dense3_fresh_ppl_results.json"))
    jdump(dict(units="nats^2 (mean-NLL scale)", definition="V_corr = V_obs - tr(P Sigma)/df; pivot 95% CI; common-effect null p",
               cohorts=dict(original="WT2 historical 128 windows, 49 article clusters", fresh_article="cohort A",
                            fresh_window="cohort B", C4HO="historical reference (128 documents)",
                            fresh_article_noearly="sensitivity: cohort A without early-dev articles",
                            fresh_article_window="sensitivity: cohort A, naive window bootstrap",
                            fresh_window_window="sensitivity: cohort B, naive window bootstrap"),
               results=var), p("fresh_variance_components.json"))
    jdump(dict(note="article-level splits (C4: documents); partitions are not independent replicates", results=spl),
          p("fresh_split_half_reliability.json"))
    jdump(dict(note=("Pearson/Spearman of centred effects (MoE within-seed); 95% CI from independent cluster bootstraps; "
                     "expected r if identical true effects = sqrt(lam_a lam_b) when both > 0"), results=trans),
          p("cross_cohort_rank_transfer.json"))
    jdump(wci, p("fresh_window_ci_analysis.json"))
    jdump(dec, p("decision.json"))
    np.savez(p("logs", "figure_data.npz"), **{k.replace("|", "__"): v for k, v in raw.items()})
    # ---- correctness + inventory + cost
    corr = dict(dense_end=jload(p("logs", "dense_end_checks.json")), moe_end=jload(p("logs", "moe_end_checks.json")),
                per_config={})
    for sub in ("dense", "moe"):
        for f in sorted(os.listdir(p("results", sub))):
            r = jload(p("results", sub, f))
            ck = r["checks"]
            tok = {}
            for c in ("fresh_article", "fresh_window", "original_parity"):
                if c in r:
                    e = r[c]
                    nll = np.asarray(e["per_window_nll"]); cnt = np.asarray(e["per_window_tokens"])
                    tok[c] = bool(abs(nll.sum() - e["total_nll"]) < 1e-6 and abs(math.exp(nll.sum() / cnt.sum()) - e["ppl"]) < 1e-9 * e["ppl"]
                                  and (cnt == 511).all())
            corr["per_config"][f"{sub}/{f[:-5]}"] = dict(valid=r["valid"], checks=ck, token_accounting=tok)
    corr["all_valid"] = all(v["valid"] and all(v["token_accounting"].values()) for v in corr["per_config"].values())
    corr["isolation"] = {k: v for k, v in jload(p("article_overlap_audit.json")).items() if k.startswith(("token_overlap", "article_overlap", "exact_dup"))}
    corr["parity_configs"] = [k for k, v in corr["per_config"].items() if "original_parity_bitwise" in v["checks"]]
    corr["parity_all_bitwise"] = all(v["checks"]["original_parity_bitwise"] for v in corr["per_config"].values()
                                     if "original_parity_bitwise" in v["checks"])
    jdump(corr, p("evaluator_correctness.json"))
    inv = dict(moe_weight_manifest=os.path.join(W.HBV_DIR, "logs", "moe_weight_manifest.json"),
               configs={k: dict(installed_weights_sha256=jload(p("results", k.split("/")[0], k.split("/")[1] + ".json")).get("installed_weights_sha256"))
                        for k in corr["per_config"]},
               dense3_weights="~/moe_cache_heldout_bitwidth/dense3/<config>/L<l>.pt (tensor sha256 verified vs heldout run)",
               dense4_weights="~/moe_cache_dense_grouping_scope/weights/<config>/L<l>.pt (installed digest verified)",
               missing=[], not_used=["seed-1/2 ORIGINAL MoE weights (never saved)", "optional 5x5 reference matrix (not run)"])
    jdump(inv, p("artifact_inventory.json"))
    craw = jload(p("logs", "cost_raw.json")) or {}
    jdump(dict(raw=craw, dense_total_s=sum(v for k, v in craw.get("dense", {}).items() if isinstance(v, (int, float))),
               moe_total_s=sum(v["total_s"] for v in craw.get("moe", {}).values() if isinstance(v, dict)),
               moe_median_config_s=float(np.median([v["total_s"] for v in craw.get("moe", {}).values() if isinstance(v, dict)] or [0])),
               analysis_s=time.time() - t0, peak=dict(dense=(corr["dense_end"] or {}).get("peak_gpu_gb"),
                                                       moe_cpu_gb=(corr["moe_end"] or {}).get("peak_cpu_gb"),
                                                       moe_gpu_gb=(corr["moe_end"] or {}).get("peak_gpu_gb"))),
          p("cost_results.json"))
    log(f"[analyze] done {time.time() - t0:.0f}s; verdicts {dec['verdicts']}")
