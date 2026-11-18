"""Analysis, deliverable JSONs, figures for runs_dense_dose_response."""
from __future__ import annotations
import itertools, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_dose_response as S                                             # noqa: E402

LAY, PROJ, POL = S.LAY, S.PROJ, S.POLICIES
NB = 2000
RNG_SEED = 20261315


def jl(*a):
    return S.jload(S.p(*a))


# ---------------------------------------------------------------- per-config scalars
def load_all():
    pr = S.check_protocol()
    cfgs = {c["name"]: c for c in S.configs(pr)}
    res = {n: jl("results", f"{n}.json") for n in cfgs if os.path.exists(S.p("results", f"{n}.json"))}
    fun = jl("logs", "functional_raw.json") or {}
    ref = fun["_ref"]
    sq_quant = sum(ref[str(L)]["sq_quant"] for L in LAY)
    sq_bf16 = sum(ref[str(L)]["sq_bf16"] for L in LAY)
    rows = []
    for n, r in res.items():
        cf = cfgs[n]
        h = r["hessian"]
        DH = math.sqrt(sum(h[f"L{L}|{hk}"]["sq_diff"] for L in LAY for hk in ("gu", "down")) /
                       sum(h[f"L{L}|{hk}"]["sq_ref"] for L in LAY for hk in ("gu", "down")))
        dh = {hk: math.sqrt(sum(h[f"L{L}|{hk}"]["sq_diff"] for L in LAY) / sum(h[f"L{L}|{hk}"]["sq_ref"] for L in LAY)) for hk in ("gu", "down")}
        base = dict(name=n, family=cf["family"], k=cf["k"], rep=cf["rep"], optional=bool(cf.get("optional")),
                    historical=cf.get("historical"), D_H=DH, D_H_gu=dh["gu"], D_H_down=dh["down"],
                    D_H_layer_median=float(np.median([h[f"L{L}|{hk}"]["rel_fro_raw"] for L in LAY for hk in ("gu", "down")])),
                    D_H_layer_mean=float(np.mean([h[f"L{L}|{hk}"]["rel_fro_raw"] for L in LAY for hk in ("gu", "down")])),
                    D_H_trace_norm_median=float(np.median([h[f"L{L}|{hk}"]["rel_fro_trace_normalized"] for L in LAY for hk in ("gu", "down")])),
                    D_H_per_layer={f"L{L}|{hk}": h[f"L{L}|{hk}"]["rel_fro_raw"] for L in LAY for hk in ("gu", "down")},
                    diag_rel_change_median=float(np.median([h[f"L{L}|{hk}"]["diag_rel_change_median"] for L in LAY for hk in ("gu", "down")])),
                    actorder_pos_changed=float(np.mean([h[f"L{L}|{hk}"]["frac_actorder_pos_changed"] for L in LAY for hk in ("gu", "down")])),
                    group_changed=float(np.mean([h[f"L{L}|{hk}"]["frac_group_changed"] for L in LAY for hk in ("gu", "down")])),
                    act_rel={str(L): h.get(f"L{L}|act", {}).get("rel_l2_fr_rows", 0.0) for L in LAY})
        for pol in POL:
            pp = r["policies"][pol]
            row = dict(base, policy=pol, ppl=pp["ppl"], log_ppl=math.log(pp["ppl"]), per_window_nll=pp["per_window_nll"],
                       per_window_tokens=pp["per_window_tokens"], weights_sha=pp["installed_weights_sha256"],
                       repeat_bitwise=pp.get("repeat_per_window_bitwise"))
            f = fun.get(f"{n}|{pol}")
            if f:
                pl = f["per_layer"]
                row.update(R_func=math.sqrt(sum(pl[str(L)]["sq_diff_vs_orig"] for L in LAY) / sq_quant),
                           rel_vs_bf16_out=math.sqrt(sum(pl[str(L)]["sq_diff_vs_orig"] for L in LAY) / sq_bf16),
                           rmse=math.sqrt(sum(pl[str(L)]["sq_diff_vs_orig"] for L in LAY) / sum(pl[str(L)]["n_el"] for L in LAY)),
                           R_func_layer={str(L): math.sqrt(pl[str(L)]["sq_diff_vs_orig"] / ref[str(L)]["sq_quant"]) for L in LAY},
                           E_proj={k: math.sqrt(sum(pl[str(L)]["proj_sq_diff"][k] for L in LAY) / sum(ref[str(L)]["proj_den"][k] for L in LAY)) for k in PROJ},
                           E_layer_proj={f"L{L}|{k}": math.sqrt(pl[str(L)]["proj_sq_diff"][k] / ref[str(L)]["proj_den"][k]) for L in LAY for k in PROJ},
                           tok_p95=float(np.mean([pl[str(L)]["tok_p95"] for L in LAY])), tok_p99=float(np.mean([pl[str(L)]["tok_p99"] for L in LAY])),
                           tok_mean=float(np.mean([pl[str(L)]["tok_mean"] for L in LAY])),
                           code=f["codes"]["code"], level=f["codes"]["level"], code_per=f["codes"]["per"],
                           rel_frob=float(np.mean(list(f["rel_frob"].values()))), bf16_changed=float(np.mean(list(f["bf16_changed"].values()))))
            rows.append(row)
    return pr, cfgs, rows, fun, sq_quant, sq_bf16


def dose_table(rows, family, pol, k):
    return [r for r in rows if r["family"] == family and r["policy"] == pol and r["k"] == k and r["rep"] is not None]


def chi2_ci(s, n):
    from scipy.stats import chi2
    return [s * math.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), s * math.sqrt((n - 1) / chi2.ppf(0.025, n - 1))]


def boot_idx(n_rep):
    rng = np.random.default_rng(RNG_SEED)
    return [rng.integers(0, n_rep, n_rep) for _ in range(NB)]


def summarize(rows, ppl0, ppl_bf16):
    Q0 = ppl0 - ppl_bf16
    lQ0 = math.log(ppl0) - math.log(ppl_bf16)
    BI = boot_idx(S.N_REP)
    out = {}
    for fam in ("b1", "sp"):
        for pol in POL:
            ks = sorted({r["k"] for r in rows if r["family"] == fam and r["rep"] is not None})
            for k in ks:
                d = sorted(dose_table(rows, fam, pol, k), key=lambda r: r["rep"])
                if len(d) < 2:
                    continue
                ppl = np.array([r["ppl"] for r in d]); lp = np.log(ppl)
                rf = np.array([r.get("R_func", np.nan) for r in d])
                s, sl = float(ppl.std(ddof=1)), float(lp.std(ddof=1))
                bs_s = [ppl[i].std(ddof=1) for i in BI]
                bs_rms = [math.sqrt(np.mean(rf[i] ** 2)) for i in BI]
                bs_mean = [np.mean(rf[i]) for i in BI]
                o = dict(family=fam, policy=pol, k=k, n=len(d), reps=[r["rep"] for r in d],
                         ppl=ppl.tolist(), ppl_mean=float(ppl.mean()), ppl_min=float(ppl.min()), ppl_max=float(ppl.max()),
                         s=s, s_log=sl, s_chi2_ci=chi2_ci(s, len(d)), s_boot_ci=[float(np.percentile(bs_s, 2.5)), float(np.percentile(bs_s, 97.5))],
                         R_PPL=s / Q0, R_PPL_ci=[x / Q0 for x in chi2_ci(s, len(d))], R_PPL_mean=s / (ppl.mean() - ppl_bf16),
                         R_logPPL=sl / lQ0,
                         R_func_mean=float(np.mean(rf)), R_func_rms=float(math.sqrt(np.mean(rf ** 2))),
                         R_func_mean_ci=[float(np.percentile(bs_mean, 2.5)), float(np.percentile(bs_mean, 97.5))],
                         R_func_rms_ci=[float(np.percentile(bs_rms, 2.5)), float(np.percentile(bs_rms, 97.5))],
                         R_func_values=rf.tolist(),
                         D_H_mean=float(np.mean([r["D_H"] for r in d])), D_H_median=float(np.median([r["D_H"] for r in d])),
                         D_H_values=[r["D_H"] for r in d], D_H_gu_mean=float(np.mean([r["D_H_gu"] for r in d])),
                         D_H_down_mean=float(np.mean([r["D_H_down"] for r in d])),
                         D_H_trace_norm_median=float(np.median([r["D_H_trace_norm_median"] for r in d])),
                         code_mean=float(np.mean([r.get("code", np.nan) for r in d])), level_mean=float(np.mean([r.get("level", np.nan) for r in d])),
                         rel_frob_mean=float(np.mean([r.get("rel_frob", np.nan) for r in d])),
                         actorder_pos_changed=float(np.mean([r["actorder_pos_changed"] for r in d])),
                         group_changed=float(np.mean([r["group_changed"] for r in d])),
                         penalty_mean=float(ppl.mean() - ppl_bf16), dppl_mean_vs0=float(ppl.mean() - ppl0),
                         R_func_layer_rms={str(L): float(math.sqrt(np.mean([r["R_func_layer"][str(L)] ** 2 for r in d]))) for L in LAY} if "R_func_layer" in d[0] else None,
                         E_proj_rms={kk: float(math.sqrt(np.mean([r["E_proj"][kk] ** 2 for r in d]))) for kk in PROJ} if "E_proj" in d[0] else None,
                         E_layer_proj_rms={kk: float(math.sqrt(np.mean([r["E_layer_proj"][kk] ** 2 for r in d]))) for kk in d[0]["E_layer_proj"]} if "E_layer_proj" in d[0] else None)
                out[f"{fam}|{pol}|{k}"] = o
    return out


# ---------------------------------------------------------------- models
def powerlaw(rows, fam_set, pol):
    rs = [r for r in rows if r["family"] in fam_set and r["policy"] == pol and r["D_H"] > 0 and r.get("R_func", 0) > 0
          and r["historical"] is None]
    x, y = np.log([r["D_H"] for r in rs]), np.log([r["R_func"] for r in rs])
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ np.array([b, a])
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # trajectory bootstrap: replicate-level resampling; single endpoints always kept
    BI = boot_idx(S.N_REP)
    slopes = []
    for idx in BI:
        sel = [r for r in rs if r["rep"] is None] + [r for i in idx for r in rs if r["rep"] == i]
        xx, yy = np.log([r["D_H"] for r in sel]), np.log([r["R_func"] for r in sel])
        slopes.append(np.polyfit(xx, yy, 1)[0])
    from scipy.stats import spearmanr
    return dict(n=len(rs), b=float(b), a=float(math.exp(a)), r2=float(r2), b_ci=[float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
                spearman_DH_Rfunc=float(spearmanr([r["D_H"] for r in rs], [r["R_func"] for r in rs]).statistic),
                spearman_k_Rfunc=float(spearmanr([r["k"] for r in rs], [r["R_func"] for r in rs]).statistic),
                code_slope=float(np.polyfit(x, np.log([max(r["code"], 1e-12) for r in rs]), 1)[0]),
                relfrob_slope=float(np.polyfit(x, np.log([r["rel_frob"] for r in rs]), 1)[0]))


def ppl_models(dose, rows, fam_set, pol_set, Q0):
    """ML: PPL_k,r ~ N(mu_k, (Q0*Y(X_k))^2), mu_k free (profiled); X_k = dose RMS R_func of that policy/family."""
    from scipy.optimize import minimize
    groups = [(v["R_func_rms"], np.array(v["ppl"])) for v in dose.values() if v["family"] in fam_set and v["policy"] in pol_set]
    if not groups:
        return None

    def ll(sig):
        t = 0.0
        for (X, y), s in zip(groups, sig):
            s = max(s, 1e-12)
            t += -len(y) * math.log(s) - ((y - y.mean()) ** 2).sum() / (2 * s * s)
        return t
    models = {"M1_proportional": (lambda th, X: th[0] * X, [0.1]),
              "M2_sqrt": (lambda th, X: th[0] * math.sqrt(X), [0.1]),
              "M3_saturating": (lambda th, X: th[0] * X / (abs(th[1]) + X), [0.1, 0.1]),
              "M0_constant": (lambda th, X: th[0], [0.05])}
    out = {}
    for name, (fn, th0) in models.items():
        best = None
        for scale in (1e-3, 1e-2, 1e-1, 1, 10):
            r = minimize(lambda th: -ll([Q0 * abs(fn(th, X)) for X, _ in groups]), np.array(th0) * scale, method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-10, fatol=1e-10))
            if best is None or r.fun < best.fun:
                best = r
        k = len(th0)
        out[name] = dict(params=[float(abs(v)) for v in best.x], loglik=float(-best.fun), aic=float(2 * k + 2 * best.fun))
    free_ll = ll([float(np.sqrt(((y - y.mean()) ** 2).mean())) for _, y in groups])
    out["free_per_dose"] = dict(loglik=float(free_ll), aic=float(2 * len(groups) - 2 * free_ll))
    m = min((v["aic"] for kk, v in out.items() if kk.startswith("M")))
    for v in out.values():
        v["dAIC"] = v["aic"] - m
    out["n_doses"] = len(groups)
    out["X"] = [g[0] for g in groups]
    return out


# ---------------------------------------------------------------- mitigation
def mitigation(rows, dose, ppl0):
    BI = boot_idx(S.N_REP)
    out = {}
    for fam in ("b1", "sp"):
        ks = sorted({v["k"] for v in dose.values() if v["family"] == fam})
        for k in ks:
            nat = {r["rep"]: r for r in dose_table(rows, fam, "natural", k)}
            fix = {r["rep"]: r for r in dose_table(rows, fam, "fixed", k)}
            reps = sorted(set(nat) & set(fix))
            if len(reps) < 2:
                continue
            fn, ff = np.array([nat[i]["R_func"] for i in reps]), np.array([fix[i]["R_func"] for i in reps])
            pn, pf = np.array([nat[i]["ppl"] for i in reps]), np.array([fix[i]["ppl"] for i in reps])
            cn, cf = np.array([nat[i]["code"] for i in reps]), np.array([fix[i]["code"] for i in reps])
            dln, dlf = np.log(pn) - math.log(ppl0), np.log(pf) - math.log(ppl0)
            bs = [math.sqrt(np.mean(ff[i] ** 2)) / math.sqrt(np.mean(fn[i] ** 2)) for i in BI]
            out[f"{fam}|{k}"] = dict(
                family=fam, k=k, n=len(reps),
                R_mitigation=float(math.sqrt(np.mean(ff ** 2)) / math.sqrt(np.mean(fn ** 2))),
                R_mitigation_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                R_SD=float(pf.std(ddof=1) / pn.std(ddof=1)), code_ratio=float(cf.mean() / cn.mean()),
                paired_R_func_diff=(ff - fn).tolist(), frac_reps_fixed_lower_R_func=float(np.mean(ff < fn)),
                paired_ppl_diff=(pf - pn).tolist(), mean_ppl_diff=float((pf - pn).mean()),
                paired_sq_dlog_diff=((dln ** 2) - (dlf ** 2)).tolist(),
                mean_abs_dlog_nat=float(np.abs(dln).mean()), mean_abs_dlog_fix=float(np.abs(dlf).mean()),
                code_nat=float(cn.mean()), code_fix=float(cf.mean()), rms_nat=float(math.sqrt(np.mean(fn ** 2))),
                rms_fix=float(math.sqrt(np.mean(ff ** 2))), ppl_mean_nat=float(pn.mean()), ppl_mean_fix=float(pf.mean()),
                s_nat=float(pn.std(ddof=1)), s_fix=float(pf.std(ddof=1)))
    # pooled PPL-benefit test over replicates x doses (b1 main + optional)
    ks = sorted(v["k"] for v in out.values() if v["family"] == "b1")
    M = np.array([[out[f"b1|{k}"]["paired_sq_dlog_diff"][r] for k in ks] for r in range(S.N_REP)])
    P = np.array([[out[f"b1|{k}"]["paired_ppl_diff"][r] for k in ks] for r in range(S.N_REP)])
    bsm = [M[i].mean() for i in BI]
    bsp = [P[i].mean() for i in BI]
    out["pooled_b1"] = dict(doses=ks, mean_sq_dlog_benefit=float(M.mean()),
                            ci=[float(np.percentile(bsm, 2.5)), float(np.percentile(bsm, 97.5))],
                            mean_ppl_cost=float(P.mean()), ppl_cost_ci=[float(np.percentile(bsp, 2.5)), float(np.percentile(bsp, 97.5))],
                            note="benefit = mean over (replicate, dose) of dlogPPL_nat^2 - dlogPPL_fix^2 (dlog vs ORIGINAL); >0 favours fixed")
    return out


# ---------------------------------------------------------------- reference matrix
def refmatrix(fun, sq_quant, ppl_bf16):
    raw = jl("logs", "refmatrix_raw.json") or {}
    G = S.GROUPS
    if len(raw) < 25:
        return None
    import torch
    P = np.array([[raw[f"{i}|{j}"]["ppl"] for j in G] for i in G])
    Rf = np.full((5, 5), np.nan); Rr = np.full((5, 5), np.nan)
    for a, i in enumerate(G):
        for b, j in enumerate(G):
            f = fun.get(f"ref:{i}|{j}")
            if f:
                Rf[a, b] = math.sqrt(sum(f["per_layer"][str(L)]["sq_diff_vs_orig"] for L in LAY) / sq_quant)
                Rr[a, b] = math.sqrt(sum(f["per_layer"][str(L)]["sq_diff_vs_wref"] for L in LAY) / sq_quant)
    perms = {i: {(L, k): torch.load(os.path.join(S.DCACHE, "codes", i, f"L{L}.pt"))[k]["perm"] for L in LAY for k in PROJ} for i in G}
    dis = {f"{i}|{j}": float(np.mean([float((perms[i][x] != perms[j][x]).float().mean()) for x in perms[i]])) for i, j in itertools.combinations(G, 2)}
    nonnull = [0, 1, 2, 3]
    per_ref = {i: dict(mean_ppl=float(P[a].mean()), sd_ppl=float(P[a].std(ddof=1)), sd_ppl_4_nonnull_H=float(P[a, nonnull].std(ddof=1)),
                       penalty=float(P[a].mean() - ppl_bf16), mean_R_func_vs_ORIGINAL=float(np.nanmean(Rf[a])),
                       mean_R_func_vs_own_reference=float(np.nanmean(Rr[a, :4])))
               for a, i in enumerate(G)}
    return dict(ppl_matrix=P.tolist(), rows_reference_perm=G, cols_hessian=G, R_func_vs_ORIGINAL=Rf.tolist(),
                R_func_vs_own_reference=Rr.tolist(), per_reference=per_ref, actorder_disagreement=dis,
                within_reference_sd_mean=float(np.mean([P[a].std(ddof=1) for a in range(5)])),
                between_reference_sd_of_means=float(P.mean(1).std(ddof=1)), between_reference_range=float(P.mean(1).max() - P.mean(1).min()),
                within_hessian_sd_mean=float(np.mean([P[:, b].std(ddof=1) for b in range(5)])),
                natural_diagonal=np.diag(P).tolist(), checks={k: v.get("checks") for k, v in raw.items() if v.get("checks")})


# ---------------------------------------------------------------- figures
def figures(rows, dose, mit, models, ref, moe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(S.p("figures"), exist_ok=True)
    ks_col = {0: "k", 1: "tab:blue", 4: "tab:orange", 16: "tab:green", 32: "tab:olive", 64: "tab:red", 96: "tab:brown", 128: "tab:purple"}
    nat = [r for r in rows if r["policy"] == "natural"]
    # fig1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, mk in (("b1", "o"), ("sp", "^")):
        rs = [r for r in nat if r["family"] == fam and r["k"] > 0 and r["historical"] is None]
        ax.scatter([r["k"] * (1 + 0.03 * ((r["rep"] or 0) - 3.5)) for r in rs], [r["D_H"] for r in rs], marker=mk, s=18,
                   label=f"{fam} ({'batch-1' if fam == 'b1' else 'SDPA strong path'})")
    hs = [r for r in nat if r["historical"]]
    ax.scatter([1] * len(hs), [r["D_H"] for r in hs], marker="x", color="k", label="historical singletons")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("k perturbed calibration rows"); ax.set_ylabel("D_H (pooled rel. Frobenius)")
    ax.legend(fontsize=8); ax.set_title("Controlled perturbation dose", fontsize=9)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig1_hessian_dose.png"), dpi=140); plt.close(fig)
    # fig2
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pol, mk in (("natural", "o"), ("fixed", "s")):
        for fam in ("b1", "sp"):
            rs = [r for r in rows if r["policy"] == pol and r["family"] == fam and r["k"] > 0 and "R_func" in r and r["historical"] is None]
            ax.scatter([r["D_H"] for r in rs], [r["R_func"] for r in rs], c=[ks_col[r["k"]] for r in rs], marker=mk if fam == "b1" else "^",
                       s=20, alpha=.8, edgecolors="k" if pol == "fixed" else "none", linewidths=.5)
        pl = models.get(f"powerlaw_b1_{pol}")
        if pl:
            xx = np.logspace(np.log10(min(r["D_H"] for r in rows if r["D_H"] > 0)), np.log10(max(r["D_H"] for r in rows)), 50)
            ax.plot(xx, pl["a"] * xx ** pl["b"], "--" if pol == "fixed" else "-", color="gray", lw=.8, label=f"{pol}: b = {pl['b']:.2f}")
    ax.axhline(moe["R_func"], color="tab:red", ls=":", label=f"MoE block reference {moe['R_func']:.2f}")
    for k, col in ks_col.items():
        ax.scatter([], [], color=col, label=f"k={k}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("D_H"); ax.set_ylabel("R_func (vs ORIGINAL, / ORIGINAL quant error)")
    ax.legend(fontsize=6, ncol=2); ax.set_title("Functional dose response (o natural, s fixed, ^ strong path)", fontsize=9)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig2_functional_dose_response.png"), dpi=140); plt.close(fig)
    # fig3
    fig, ax = plt.subplots(figsize=(7, 5))
    for pol, col in (("natural", "tab:blue"), ("fixed", "tab:orange")):
        for fam, mk in (("b1", "o"), ("sp", "^")):
            ds = [v for v in dose.values() if v["policy"] == pol and v["family"] == fam]
            if not ds:
                continue
            x = [v["R_func_rms"] for v in ds]; y = [v["R_PPL"] for v in ds]
            yerr = np.array([[v["R_PPL"] - v["R_PPL_ci"][0], v["R_PPL_ci"][1] - v["R_PPL"]] for v in ds]).T
            xerr = np.array([[v["R_func_rms"] - v["R_func_rms_ci"][0], v["R_func_rms_ci"][1] - v["R_func_rms"]] for v in ds]).T
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=mk, color=col, ms=5, capsize=2, label=f"{pol} {fam}", alpha=.8)
            for v in ds:
                ax.annotate(f"k{v['k']}", (v["R_func_rms"], v["R_PPL"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.scatter([moe["R_func"]], [moe["R_PPL_Q0"]], marker="*", s=150, color="tab:red", label="MoE FR s0 (descriptive, Q0 denom.)")
    ax.set_xscale("log"); ax.set_xlabel("dose RMS R_func"); ax.set_ylabel("R_PPL = SD_r(PPL) / Q0 (chi-square 95 % CI)")
    ax.legend(fontsize=7); ax.set_title("Functional instability vs PPL variability", fontsize=9)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig3_functional_vs_ppl.png"), dpi=140); plt.close(fig)
    # fig4
    m = [v for v in mit.values() if isinstance(v, dict) and v.get("family") == "b1"]
    m.sort(key=lambda v: v["k"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ks = [v["k"] for v in m]
    ax.plot(ks, [v["code_ratio"] for v in m], "-o", label="code disagreement fixed / natural")
    ax.errorbar(ks, [v["R_mitigation"] for v in m], yerr=np.array([[v["R_mitigation"] - v["R_mitigation_ci"][0], v["R_mitigation_ci"][1] - v["R_mitigation"]] for v in m]).T,
                fmt="-s", capsize=2, label="R_mitigation (RMS R_func fixed / natural)")
    ax.plot(ks, [v["R_SD"] for v in m], "-^", label="SD(PPL) fixed / natural (descriptive)")
    for v in mit.values():
        if isinstance(v, dict) and v.get("family") == "sp":
            ax.scatter([v["k"]], [v["R_mitigation"]], marker="*", s=80, color="k", label=f"strong path k={v['k']} R_mitigation")
    ax.axhline(1, color="gray", lw=.6); ax.set_xscale("log", base=2); ax.set_xlabel("k"); ax.set_ylabel("ratio fixed / natural")
    ax.legend(fontsize=7); ax.set_title("Act-order mitigation across doses (batch-1 family)", fontsize=9)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig4_actorder_mitigation.png"), dpi=140); plt.close(fig)
    # fig5
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol, col in (("natural", "tab:blue"), ("fixed", "tab:orange")):
        rs = [r for r in rows if r["policy"] == pol and "R_func" in r and r["k"] > 0 and r["historical"] is None]
        ax.scatter([r["R_func"] for r in rs], [r["ppl"] for r in rs], s=12, color=col, marker="o", alpha=.6, label=pol)
    r0 = [r for r in rows if r["name"] == "b1_k000"][0]
    ax.axhline(r0["ppl"], color="k", lw=.6, ls=":", label="k=0 (ORIGINAL)")
    ax.set_xscale("log"); ax.set_xlabel("R_func"); ax.set_ylabel("PPL"); ax.legend(fontsize=7)
    ax.set_title("Quality vs functional instability (all configs, both families)", fontsize=9)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig5_quality_vs_instability.png"), dpi=140); plt.close(fig)
    # fig6
    if ref:
        fig, ax = plt.subplots(figsize=(6, 5))
        P = np.array(ref["ppl_matrix"])
        im = ax.imshow(P, cmap="viridis")
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{P[a, b]:.4f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(5)); ax.set_xticklabels([g[-2:] for g in S.GROUPS]); ax.set_yticks(range(5)); ax.set_yticklabels([g[-2:] for g in S.GROUPS])
        ax.set_xlabel("grouping Hessian H_j (G202611..)"); ax.set_ylabel("reference permutation pi_i (G202611..)")
        plt.colorbar(im, ax=ax); ax.set_title("PPL of GPTQ(H_j, pi_i); diagonal = natural; 15 = ORIGINAL", fontsize=8)
        fig.tight_layout(); fig.savefig(S.p("figures", "fig6_reference_matrix.png"), dpi=140); plt.close(fig)
    # fig7
    ds = sorted([v for v in dose.values() if v["family"] == "b1" and v["policy"] == "natural" and v["E_layer_proj_rms"]], key=lambda v: v["k"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for L in LAY:
        axs[0].plot([v["k"] for v in ds], [v["R_func_layer_rms"][str(L)] for v in ds], "-o", label=f"L{L} MLP")
    axs[0].set_title("per-layer R_func (RMS; / layer's own quant error), natural", fontsize=8)
    for key in ds[0]["E_layer_proj_rms"]:
        axs[1].plot([v["k"] for v in ds], [v["E_layer_proj_rms"][key] for v in ds], "-" if "down" in key else ":", label=key)
    axs[1].set_title("per layer x projection ||dW X|| / ||W_BF16 X|| (RMS), natural", fontsize=8)
    for ax in axs:
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("k"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig7_layer_projection.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    pr, cfgs, rows, fun, sq_quant, sq_bf16 = load_all()
    au = jl("logs", "audit.json")
    ppl_bf16 = au["bf16_ppl"]
    r0 = [r for r in rows if r["name"] == "b1_k000" and r["policy"] == "natural"][0]
    ppl0 = r0["ppl"]
    Q0 = ppl0 - ppl_bf16
    dose = summarize(rows, ppl0, ppl_bf16)
    # MoE reference (consistent definitions)
    afs = S.jload(os.path.join(S.ROOT, S.AFS_RUN, "functional_vs_code_comparison.json"))["ratios"]
    mg = {g: S.jload(os.path.join(S.ROOT, "runs_multigroup_robustness", "results", "CTRL_FR_ORIGINAL.json" if g == "ORIGINAL" else f"FR_{g}.json"))["ppl"]
          for g in S.GROUPS + ["ORIGINAL"]}
    mbf = S.jload(os.path.join(S.ROOT, "runs_partial_ppl", "bf16_baseline.json"))["ppl"]
    mv = np.array([mg[g] for g in S.GROUPS])
    moe = dict(R_func=afs["moe_primary_mean"]["moe_block"] / afs["quant_error_vs_bf16"]["moe_block"],
               R_func_definition="mean over 5 groupings of pooled block E(g vs ORIGINAL) / mean over configs of pooled block E(cfg vs BF16) "
                                 "(runs_actorder_functional_stability); dense analogue uses ||f_0 - f_BF16|| of ORIGINAL only",
               R_func_routed=afs["moe_primary_mean"]["moe_routed"] / afs["quant_error_vs_bf16"]["moe_routed"],
               s=float(mv.std(ddof=1)), Q0=mg["ORIGINAL"] - mbf, R_PPL_Q0=float(mv.std(ddof=1) / (mg["ORIGINAL"] - mbf)),
               R_PPL_mean=float(mv.std(ddof=1) / (mv.mean() - mbf)),
               R_logPPL=float(np.log(mv).std(ddof=1) / (math.log(mg["ORIGINAL"]) - math.log(mbf))),
               note="5 groupings, all perturbing nearly every calibration sequence; descriptive, not a replicate of the dense process")
    # dense historical 5-grouping reference
    dh = np.array([au["historical_ppl"][g] for g in S.GROUPS])
    dense_hist = dict(s=float(dh.std(ddof=1)), R_PPL_Q0=float(dh.std(ddof=1) / Q0), R_func_rms_4=float(math.sqrt(np.mean(
        [r["R_func"] ** 2 for r in rows if r["historical"] and r["policy"] == "natural" and "R_func" in r] or [np.nan]))))
    models = {}
    for pol in POL:
        models[f"powerlaw_b1_{pol}"] = powerlaw(rows, {"b1"}, pol)
        models[f"powerlaw_all_{pol}"] = powerlaw(rows, {"b1", "sp"}, pol)
        models[f"ppl_b1_{pol}"] = ppl_models(dose, rows, {"b1"}, {pol}, Q0)
        models[f"ppl_all_{pol}"] = ppl_models(dose, rows, {"b1", "sp"}, {pol}, Q0)
    models["ppl_b1_both"] = ppl_models(dose, rows, {"b1"}, set(POL), Q0)
    models["ppl_all_both"] = ppl_models(dose, rows, {"b1", "sp"}, set(POL), Q0)
    mit = mitigation(rows, dose, ppl0)
    ref = refmatrix(fun, sq_quant, ppl_bf16)
    # interpretation per protocol
    interp = {}
    for pol in POL:
        ks = sorted(v["k"] for v in dose.values() if v["family"] == "b1" and v["policy"] == pol)
        means = [dose[f"b1|{pol}|{k}"]["R_func_mean"] for k in ks]
        e128 = [r for r in rows if r["name"] == "b1_k128" and r["policy"] == pol]
        seq = means + ([e128[0]["R_func"]] if e128 and "R_func" in e128[0] else [])
        viol = sum(1 for x, y in zip(seq[:-1], seq[1:]) if y < x)
        pl = models[f"powerlaw_b1_{pol}"]
        interp[pol] = dict(monotonic=pl["spearman_DH_Rfunc"] >= 0.8 and viol <= 1, dose_mean_violations=viol,
                           power_law_supported=pl["r2"] >= 0.8, b=pl["b"], b_ci=pl["b_ci"])
        # solver saturation: two largest D_H dose levels (b1: k=64 mean vs k=128)
        top = sorted([(dose[f"b1|{pol}|{k}"]["D_H_mean"], dose[f"b1|{pol}|{k}"]["R_func_rms"]) for k in ks] +
                     ([(e128[0]["D_H"], e128[0]["R_func"])] if e128 and "R_func" in e128[0] else []))[-2:]
        slope = math.log(top[1][1] / top[0][1]) / math.log(top[1][0] / top[0][0]) if top[1][0] > top[0][0] else float("nan")
        interp[pol]["top_slope"] = slope
        interp[pol]["top_DH_ratio"] = top[1][0] / top[0][0]
        interp[pol]["solver_saturation"] = bool(slope < 0.25 and top[1][0] / top[0][0] >= 2)
    max_rms = max([v["R_func_rms"] for v in dose.values() if v["policy"] == "natural" and v["family"] == "b1"] +
                  [r["R_func"] for r in rows if r["name"] == "b1_k128" and r["policy"] == "natural" and "R_func" in r])
    max_rms_sp = max([v["R_func_rms"] for v in dose.values() if v["policy"] == "natural" and v["family"] == "sp"] +
                     [r["R_func"] for r in rows if r["name"] == "sp_k128" and r["policy"] == "natural" and "R_func" in r] or [float("nan")])
    interp["dynamic_range"] = dict(max_natural_b1=max_rms, max_natural_sp=max_rms_sp, threshold=0.41,
                                   sufficient_b1=max_rms >= 0.41, sufficient_any=max(max_rms, max_rms_sp if not math.isnan(max_rms_sp) else 0) >= 0.41,
                                   extrapolation_factor_to_moe=moe["R_func"] / max(max_rms, max_rms_sp if not math.isnan(max_rms_sp) else 0))
    pm = models["ppl_b1_both"]
    interp["evaluation_saturation"] = bool(interp["dynamic_range"]["sufficient_any"] and pm and
                                           pm["M3_saturating"]["aic"] <= min(pm["M1_proportional"]["aic"], pm["M2_sqrt"]["aic"]) - 2)
    interp["mitigation_effective"] = {k: v["R_mitigation_ci"][1] < 1 for k, v in mit.items() if isinstance(v, dict) and "R_mitigation_ci" in v}
    interp["ppl_benefit"] = bool(mit["pooled_b1"]["ci"][0] > 0)
    # ---- deliverables
    corr = jl("actorder_correctness.json")
    S.jdump(dict(per_config={r["name"]: {k: r[k] for k in ("family", "k", "rep", "historical", "D_H", "D_H_gu", "D_H_down", "D_H_layer_median",
                                                               "D_H_layer_mean", "D_H_trace_norm_median", "D_H_per_layer", "diag_rel_change_median",
                                                               "actorder_pos_changed", "group_changed", "act_rel")}
                                 for r in rows if r["policy"] == "natural"},
                 per_dose={k: {kk: v[kk] for kk in ("family", "k", "D_H_mean", "D_H_median", "D_H_values", "D_H_gu_mean", "D_H_down_mean",
                                                    "D_H_trace_norm_median", "actorder_pos_changed", "group_changed")}
                           for k, v in dose.items() if v["policy"] == "natural"},
                 note="D_H = pooled ||H - H0||_F / ||H0||_F over 4 layers x (gu, down); trace-normalized metric reported separately (not interchangeable)"),
            S.p("hessian_dose_results.json"))
    S.jdump({r["name"] + "|" + r["policy"]: {k: r.get(k) for k in ("family", "k", "rep", "policy", "weights_sha", "code", "level", "rel_frob",
                                                                     "bf16_changed", "code_per")} for r in rows}, S.p("dose_quantization_results.json"))
    S.jdump({r["name"] + "|" + r["policy"]: {k: r.get(k) for k in ("family", "k", "rep", "policy", "D_H", "R_func", "rel_vs_bf16_out", "rmse",
                                                                     "R_func_layer", "E_proj", "E_layer_proj", "tok_mean", "tok_p95", "tok_p99")}
             for r in rows} | {"_reference": dict(sq_quant=sq_quant, sq_bf16=sq_bf16, consistency=fun.get("_consistency_G20261111"))},
            S.p("dose_functional_results.json"))
    S.jdump(dict(bf16_ppl=ppl_bf16, ppl0=ppl0, Q0=Q0, configs={r["name"] + "|" + r["policy"]: {k: r[k] for k in (
        "family", "k", "rep", "policy", "ppl", "log_ppl", "per_window_nll", "per_window_tokens", "repeat_bitwise", "weights_sha")}
        for r in rows}), S.p("dose_ppl_results.json"))
    S.jdump(dict(Q0=Q0, ppl0=ppl0, bf16_ppl=ppl_bf16, per_dose=dose,
                 endpoints={r["name"] + "|" + r["policy"]: dict(ppl=r["ppl"], dppl=r["ppl"] - ppl0, R_func=r.get("R_func"), D_H=r["D_H"])
                            for r in rows if r["k"] == 128},
                 historical_singletons={r["name"] + "|" + r["policy"]: dict(ppl=r["ppl"], R_func=r.get("R_func"), D_H=r["D_H"]) for r in rows if r["historical"]},
                 moe_reference=moe, dense_historical_5_groupings=dense_hist,
                 note="k=128 endpoints have no execution SD; SDs use ddof=1 over 8 distinct subsets"), S.p("dose_variability.json"))
    S.jdump(dict(models=models, interpretation=interp), S.p("dose_response_models.json"))
    S.jdump(mit, S.p("actorder_mitigation.json"))
    if ref:
        S.jdump(ref, S.p("reference_order_matrix.json"))
    cst = jl("logs", "cost_raw.json") or {}
    S.jdump(dict(raw=cst, gpu_total_h=sum(cst.get("gpu", {}).values()) / 3600, per_config_median_s=float(np.median(list(cst.get("gpu", {}).values()) or [0])),
                 refmatrix_s=sum(v.get("seconds", 0) for v in (jl("logs", "refmatrix_raw.json") or {}).values()),
                 func_s=sum(v.get("seconds", 0) for k, v in fun.items() if not k.startswith("_"))), S.p("cost_results.json"))
    inv = {}
    for root, _, files in os.walk(S.CACHE):
        for fn in files:
            fp = os.path.join(root, fn)
            inv[os.path.relpath(fp, S.CACHE)] = dict(bytes=os.path.getsize(fp))
    S.jdump(dict(inputs=dict(audit=os.path.join(S.OUT, "logs", "audit.json"), all_ok=au["all_ok"]),
                 alt_caches=jl("activation_cache_validation.json"), cache_root=S.CACHE, n_files=len(inv),
                 total_gb=sum(v["bytes"] for v in inv.values()) / 1e9, files=inv,
                 weight_hashes={r["name"] + "|" + r["policy"]: r["weights_sha"] for r in rows}), S.p("artifact_inventory.json"))
    figures(rows, dose, mit, models, ref, moe)
    dig = dict(Q0=Q0, ppl0=ppl0, interp=interp, moe=moe, dense_hist=dense_hist,
               dose={k: {kk: v[kk] for kk in ("n", "ppl_mean", "s", "s_chi2_ci", "R_PPL", "R_PPL_ci", "R_PPL_mean", "R_logPPL", "R_func_mean",
                                              "R_func_rms", "R_func_rms_ci", "D_H_mean", "code_mean", "rel_frob_mean", "actorder_pos_changed",
                                              "group_changed", "dppl_mean_vs0")} for k, v in dose.items()},
               endpoints={r["name"] + "|" + r["policy"]: dict(ppl=r["ppl"], R_func=r.get("R_func"), D_H=r["D_H"], code=r.get("code")) for r in rows if r["k"] == 128},
               powerlaw={k: v for k, v in models.items() if k.startswith("powerlaw")},
               ppl_models={k: {m: (v[m]["dAIC"], v[m]["params"]) for m in v if m.startswith("M")} for k, v in models.items() if k.startswith("ppl") and v},
               mitigation={k: {kk: v[kk] for kk in v if not kk.startswith("paired")} for k, v in mit.items()},
               ref=None if not ref else {k: ref[k] for k in ("per_reference", "within_reference_sd_mean", "between_reference_sd_of_means",
                                                             "between_reference_range", "within_hessian_sd_mean", "natural_diagonal", "checks")},
               correctness=corr)
    S.jdump(dig, S.p("logs", "analysis_digest.json"))
    print(json.dumps(dig, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main()
