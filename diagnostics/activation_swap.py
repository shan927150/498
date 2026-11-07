"""Minimal activation-aware calibration swap pilot (layer 16, 8 experts, seed 0).

    --prepare   GPU forward only (no GPTQ): pick the 8 most-routed layer-16
                experts from the ORIGINAL pool routing cache (saved first),
                draw 64 incoming candidates and 16 outgoing sequences, profile
                per-(sequence, expert) diagonal second moments of the routed
                gate/up inputs for S0 + candidates and the tuning-split
                reference. Writes experts.json, stats.npz, prepare.json.
    --propose   CPU: all 16 x 64 swaps, predicted_gain, routing side metrics,
                the 9 prespecified swaps and the decision rule ->
                proposals.json (+ .sha256). No GPTQ.
    --evaluate  GPU: frozen GPTQ of the 8 experts for S0 and the 9 swaps,
                scored on the old test split (exploratory). Refuses to run
                if proposals.json changed.
    --report    summary.md / summary.json / figures.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

LAYER, N_EXPERTS, SEED, B = 16, 8, 0, 128
CAND_SEED, OUT_SEED, RAND_SEED = 20261108, 20261109, 20261110
N_IN, N_OUT = 64, 16
DECISION = {
    "PROMISING": "Spearman(predicted_gain, actual_gain) over the evaluated swaps >= 0.5 AND mean actual_gain "
                 "of the beneficial group > mean of the harmful group AND > mean of the random group AND >= 2 of "
                 "the 3 beneficial swaps have actual_gain > 0 AND the beneficial-group mean actual_gain exceeds "
                 "the SD of the random-group actual gains (effect not tiny relative to swap noise)",
    "NOT-SUPPORTED": "Spearman <= 0 AND mean actual_gain of the beneficial group <= mean of the harmful group",
    "INCONCLUSIVE": "otherwise, or fewer than 2 swaps in the beneficial or harmful group",
}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def frozen_and_s0(a):
    import e3_allocation as E3
    import report_typicality_matched as RTM
    from moe_calib.data_utils import load_meta, split_ids
    from moe_calib.frozen import load_frozen
    cfg = load_frozen(a.frozen)
    meta = load_meta(a.cache_dir)
    sp = split_ids(meta)
    assert sp["definition"] == cfg["split"]
    g = cfg["gptq"]
    assert (g["bits"], g["group_size"], float(g["percdamp"]), g["act_order"]) == (4, 128, 1.0, True)
    ws = json.load(open(os.path.join(a.matched_dir, "warm_starts.json")))
    f = RTM.run_path(a.matched_dir, "full", "random", SEED, ws)
    r = json.load(open(f))
    S0 = [int(s) for s in r["chosen"]]
    assert len(S0) == B == len(set(S0)) and set(S0) <= set(sp["pool"])
    assert not set(S0) & (set(sp["tune"]) | set(sp["test"]))
    assert S0[:32] == ws["seeds"][str(SEED)]["ids"] and E3.warm_hash(S0[:32]) == ws["seeds"][str(SEED)]["hash"]
    assert r["summary"]["strategy"] == "random" and r["summary"]["percdamp"] == 1.0
    return cfg, sp, S0, f


def prepare(a):
    import torch
    from moe_calib import activation_swap as AS
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    os.makedirs(a.outdir, exist_ok=True)
    cfg, sp, S0, s0_file = frozen_and_s0(a)
    z = np.load(os.path.join(a.matched_dir, "routing_profile_v2.npz"))
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"].astype(float)
    li = layers.index(LAYER)
    experts, tot = AS.top_experts(C, li, E, N_EXPERTS)
    json.dump(dict(layer=LAYER, experts=experts, rule="8 largest routed-token totals of layer 16 over the "
                   "original 2,000-sequence pool (routing_profile_v2.npz); ties -> lower id",
                   routed_tokens={str(e): float(tot[e]) for e in experts}),
              open(os.path.join(a.outdir, "experts.json"), "w"), indent=1)
    rest = sorted(set(sp["pool"]) - set(S0))
    cand = random.Random(CAND_SEED).sample(rest, N_IN)
    outs = random.Random(OUT_SEED).sample(sorted(S0), N_OUT)
    profile_ids = S0 + cand
    model, _ = MU.load_model()
    d = MU.expert_dims(model)[0]
    cache = load_layer_cache(a.cache_dir, LAYER)
    t0 = time.time()
    P = AS.DiagProfiler(model, LAYER, experts).run(cache, profile_ids)
    prof_s = time.time() - t0
    n, g = P.arrays(profile_ids, d)
    t1 = time.time()
    R = AS.DiagProfiler(model, LAYER, experts).run(cache, sp["tune"])
    ref_s = time.time() - t1
    nr, gr = R.arrays(sp["tune"], d)
    Nref = nr.sum(0)
    href = gr.sum(0) / Nref[:, None]
    # validation: routed-token counts vs the routing cache and the gate hook
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    cache_n = np.array([[C[row[s], li * E + e] for e in experts] for s in profile_ids])
    gate_n = np.array([[P.seq_counts.get(s, {}).get((LAYER, e), 0) for e in experts] for s in profile_ids])
    assert np.array_equal(n, cache_n), "expert-hook token counts differ from the routing cache"
    assert np.array_equal(n, gate_n), "expert-hook token counts differ from the gate hook"
    np.savez(os.path.join(a.outdir, "stats.npz"), profile_ids=np.array(profile_ids), n=n, g=g,
             href=href, nref=Nref, ref_ids=np.array(sp["tune"]), experts=np.array(experts))
    st = os.path.join(a.outdir, "stats.npz")
    res = dict(s0_file=s0_file, S0=S0, S0_hash=AS.ids_hash(S0), candidates_in=cand, outgoing=outs,
               cand_seed=CAND_SEED, out_seed=OUT_SEED, n_profiled=len(profile_ids),
               profile_s=prof_s, reference_s=ref_s, reference_split=sp["definition"]["tune"],
               reference_tokens={str(e): float(x) for e, x in zip(experts, Nref)},
               counts_match_routing_cache=True, counts_match_gate_hook=True,
               stats_bytes=os.path.getsize(st), stats_sha256=sha_file(st),
               full_forward_equiv_s=None, peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9
                                                           if torch.cuda.is_available() else None))
    json.dump(res, open(os.path.join(a.outdir, "prepare.json"), "w"), indent=1)
    print(f"experts {experts}; profiled {len(profile_ids)} seqs in {prof_s:.1f}s, reference {ref_s:.1f}s; "
          f"stats {res['stats_bytes'] / 1e6:.1f} MB")


def eff_sources(X):
    q = X / np.maximum(X.sum(0, keepdims=True), 1)
    return np.exp(-(np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0)).sum(0))


def propose(a):
    from moe_calib import activation_swap as AS
    from moe_calib import typicality as TY
    from moe_calib import aggregate_constraint as AG
    pth = os.path.join(a.outdir, "proposals.json")
    if os.path.exists(pth):
        raise SystemExit("proposals.json exists: proposals are frozen")
    prep = json.load(open(os.path.join(a.outdir, "prepare.json")))
    st = np.load(os.path.join(a.outdir, "stats.npz"))
    assert sha_file(os.path.join(a.outdir, "stats.npz")) == prep["stats_sha256"]
    ids = [int(s) for s in st["profile_ids"]]
    pos = {s: i for i, s in enumerate(ids)}
    n, g, href = st["n"], st["g"], st["href"]
    experts = [int(e) for e in st["experts"]]
    S0 = prep["S0"]
    rows0 = [pos[s] for s in S0]
    h0, N0 = AS.set_moments(n, g, rows0)
    D0, De0 = AS.mismatch(h0, href)
    Gs, Ns = g[rows0].sum(0), n[rows0].sum(0)
    # routing side (original pool routing cache)
    z = np.load(os.path.join(a.matched_dir, "routing_profile_v2.npz"))
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"].astype(float)
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    sc = TY.sequence_scores(C, len(layers), E, TY.pool_reference(C, len(layers), E))
    ref = AG.pool_reference_flat(C, len(layers), E)
    li = layers.index(LAYER)
    cols = [li * E + e for e in experts]
    c0 = C[[row[s] for s in S0]].sum(0)
    L1_0 = AG.aggregate_l1(c0, ref, B * 2048, len(layers), E)[li]
    eff0 = float(np.median(eff_sources(C[[row[s] for s in S0]][:, cols])))
    props = []
    for o in prep["outgoing"]:
        for i in prep["candidates_in"]:
            D1, De1 = AS.swap_mismatch(Ns, Gs, n[pos[o]], g[pos[o]], n[pos[i]], g[pos[i]], href)
            c1 = c0 - C[row[o]] + C[row[i]]
            S1 = [i if s == o else s for s in S0]
            props.append(dict(
                out=o, inn=i, predicted_gain=D0 - D1, D_swap=D1,
                predicted_gain_per_expert=(De0 - De1).tolist(),
                token_change_per_expert=(n[pos[i]] - n[pos[o]]).tolist(),
                d_set_mean_change=float((sc["d"][row[i]] - sc["d"][row[o]]) / B),
                d_in=float(sc["d"][row[i]]), d_out=float(sc["d"][row[o]]),
                aggregate_l1_L16_change=float(AG.aggregate_l1(c1, ref, B * 2048, len(layers), E)[li] - L1_0),
                eff_sources_median_change=float(np.median(eff_sources(C[[row[s] for s in S1]][:, cols])) - eff0)))
    for p in props:
        p["in"] = p.pop("inn")
    key = lambda p: (p["out"], p["in"])
    ben = sorted([p for p in props if p["predicted_gain"] > 0], key=lambda p: (-p["predicted_gain"], key(p)))
    har = sorted([p for p in props if p["predicted_gain"] < 0], key=lambda p: (p["predicted_gain"], key(p)))
    uo, ui = set(), set()
    b3, bu = AS.pick_unique(ben, 3, uo, ui)
    h3, hu = AS.pick_unique(har, 3, uo, ui)
    chosen = {key(p) for p in b3 + h3}
    remaining = sorted([p for p in props if key(p) not in chosen], key=key)
    rr = random.Random(RAND_SEED)
    order = rr.sample(remaining, len(remaining))       # uniform random order of the remaining pool
    r3, ru = AS.pick_unique(order, 3, uo, ui)
    nine = [dict(p, group="beneficial", rank=k) for k, p in enumerate(b3)] + \
           [dict(p, group="harmful", rank=k) for k, p in enumerate(h3)] + \
           [dict(p, group="random", rank=k) for k, p in enumerate(r3)]
    for p in nine:
        S = AS.swapped(S0, p["out"], p["in"])
        p["set_hash"] = AS.ids_hash(S)
        p["name"] = f"{p['group']}{p['rank']}"
    out = dict(layer=LAYER, experts=experts, seed=SEED, B=B, S0=S0, S0_hash=AS.ids_hash(S0),
               D_S0=D0, D_e_S0=De0.tolist(), reference="tuning split rows 2000-2127 (exploratory: used before)",
               reference_fingerprint=hashlib.sha256(href.tobytes()).hexdigest()[:16],
               stats_sha256=prep["stats_sha256"], metric="D_e = ||h - h_ref||^2 / (||h_ref||^2 + 1e-12); D = mean_e",
               n_proposals=len(props), n_positive=len(ben), n_negative=len(har),
               unique_ids_achieved=dict(beneficial=bu, harmful=hu, random=ru),
               rand_seed=RAND_SEED, nine=nine, decision_rule=DECISION,
               evaluation="frozen GPTQ (4-bit, g128, percdamp 1.0, act-order) of the 8 experts; old test split "
                          "(exploratory, previously inspected); swapped sequence replaces the outgoing one in place")
    json.dump(out, open(pth, "w"), indent=1)
    json.dump(props, open(os.path.join(a.outdir, "all_proposals.json"), "w"))
    open(pth + ".sha256", "w").write(sha_file(pth) + "\n")
    print(f"D(S0)={D0:.6g}; proposals {len(props)} (+{len(ben)} / -{len(har)}); nine:")
    for p in nine:
        print(f"  {p['name']}: out {p['out']} in {p['in']} predicted_gain {p['predicted_gain']:+.4g}")


def evaluate(a):
    import torch
    from moe_calib import activation_swap as AS
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import quantize_from_acc_or_fallback, per_seq_rel_errors
    pth = os.path.join(a.outdir, "proposals.json")
    h = sha_file(pth)
    if h != open(pth + ".sha256").read().strip():
        raise SystemExit("proposals.json changed after freezing")
    pr = json.load(open(pth))
    cfg, sp, S0, _ = frozen_and_s0(a)
    assert AS.ids_hash(S0) == pr["S0_hash"]
    g = cfg["gptq"]
    sets = [("S0", S0, None)] + [(p["name"], AS.swapped(S0, p["out"], p["in"]), p) for p in pr["nine"]]
    ev = os.path.join(a.outdir, "eval")
    os.makedirs(ev, exist_ok=True)
    model, _ = MU.load_model()
    cache = load_layer_cache(a.cache_dir, LAYER)
    HO = Profiler(model, LAYER, collect_hessian=False, collect_holdout=True).run(cache, sp["test"]).holdout
    for name, S, p in sets:
        f = os.path.join(ev, f"{name}.json")
        if os.path.exists(f) and json.load(open(f)).get("proposals_sha256") == h:
            print(f"[skip] {name}")
            continue
        assert len(S) == B == len(set(S))
        if p is not None:
            assert AS.ids_hash(S) == p["set_hash"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fin = Profiler(model, LAYER, K=cfg["e1"]["K"]).run(cache, S, weights=[1.0] * B)
        th = time.time() - t0
        res = {}
        t1 = time.time()
        for e in pr["experts"]:
            Wq, fb = quantize_from_acc_or_fallback(model, fin.acc, LAYER, e, g["bits"], g["group_size"],
                                                   float(g["percdamp"]))
            errs = per_seq_rel_errors(model, LAYER, e, Wq, HO)
            v = np.array(list(errs.values()))
            res[str(e)] = dict(mean=float(v.mean()), n_test_seqs=len(v), fallback=bool(fb),
                               tokens=fin.acc.tokens((LAYER, e, "gu")))
        tq = time.time() - t1
        fin.free()
        out = dict(name=name, set_ids=S, set_hash=AS.ids_hash(S), proposals_sha256=h, per_expert=res,
                   recon_mean=float(np.mean([r["mean"] for r in res.values()])), hessian_s=th, gptq_s=tq,
                   peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        json.dump(out, open(f, "w"), indent=1)
        print(f"[done] {name}", flush=True)          # scores not printed


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = json.load(open(os.path.join(a.outdir, "proposals.json")))
    prep = json.load(open(os.path.join(a.outdir, "prepare.json")))
    h = sha_file(os.path.join(a.outdir, "proposals.json"))
    ev = os.path.join(a.outdir, "eval")
    names = ["S0"] + [p["name"] for p in pr["nine"]]
    missing = [n for n in names if not os.path.exists(os.path.join(ev, f"{n}.json"))]
    if missing:
        res = dict(decision="RUN-BLOCKED", missing=missing)
        json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
        print(res)
        return
    R = {n: json.load(open(os.path.join(ev, f"{n}.json"))) for n in names}
    assert all(r["proposals_sha256"] == h for r in R.values())
    E0 = R["S0"]["recon_mean"]
    ex = [str(e) for e in pr["experts"]]
    # consistency: S0 per-expert errors vs the stored full matched run (same frozen evaluator)
    stored = json.load(open(prep["s0_file"]))
    st = {(x["layer"], x["expert"]): x["mean"] for x in stored["per_expert"]}
    s0_diff = max(abs(R["S0"]["per_expert"][e]["mean"] - st[(LAYER, int(e))]) for e in ex)
    rows = []
    for p in pr["nine"]:
        r = R[p["name"]]
        pe = {e: R["S0"]["per_expert"][e]["mean"] - r["per_expert"][e]["mean"] for e in ex}
        rows.append(dict(name=p["name"], group=p["group"], out=p["out"], **{"in": p["in"]},
                         predicted_gain=p["predicted_gain"], actual_gain=E0 - r["recon_mean"],
                         rel_change=(r["recon_mean"] - E0) / E0, E_swap=r["recon_mean"],
                         n_experts_improved=int(sum(v > 0 for v in pe.values())), actual_gain_per_expert=pe,
                         predicted_gain_per_expert=dict(zip(ex, p["predicted_gain_per_expert"])),
                         token_change_total=float(sum(p["token_change_per_expert"])),
                         token_change_per_expert=dict(zip(ex, p["token_change_per_expert"])),
                         d_set_mean_change=p["d_set_mean_change"], aggregate_l1_L16_change=p["aggregate_l1_L16_change"],
                         eff_sources_median_change=p["eff_sources_median_change"]))
    pg = [r["predicted_gain"] for r in rows]
    ag = [r["actual_gain"] for r in rows]
    rho = spearman(pg, ag)
    grp = {k: [r["actual_gain"] for r in rows if r["group"] == k] for k in ("beneficial", "harmful", "random")}
    gm = {k: float(np.mean(v)) if v else None for k, v in grp.items()}
    rand_sd = float(np.std(grp["random"], ddof=1)) if len(grp["random"]) > 1 else None
    n_ben_pos = sum(v > 0 for v in grp["beneficial"])
    # per-expert: within-expert Spearman of predicted per-expert gain vs actual per-expert gain
    pe_rho = {e: spearman([r["predicted_gain_per_expert"][e] for r in rows],
                          [r["actual_gain_per_expert"][e] for r in rows]) for e in ex}
    contrib = {e: float(np.mean([abs(r["actual_gain_per_expert"][e]) for r in rows])) for e in ex}
    tot = sum(contrib.values())
    if len(grp["beneficial"]) < 2 or len(grp["harmful"]) < 2:
        lab = "ACTIVATION-SIGNAL-INCONCLUSIVE"
    elif (rho >= 0.5 and gm["beneficial"] > gm["harmful"] and gm["beneficial"] > gm["random"] and n_ben_pos >= 2
          and rand_sd is not None and gm["beneficial"] > rand_sd):
        lab = "ACTIVATION-SIGNAL-PROMISING"
    elif rho <= 0 and gm["beneficial"] <= gm["harmful"]:
        lab = "ACTIVATION-SIGNAL-NOT-SUPPORTED"
    else:
        lab = "ACTIVATION-SIGNAL-INCONCLUSIVE"
    res = dict(decision=lab, E0=E0, D_S0=pr["D_S0"], experts=pr["experts"], spearman=rho, group_mean_actual_gain=gm,
               group_actual_gains=grp, random_group_sd=rand_sd, n_beneficial_improved=n_ben_pos,
               n_harmful_improved=sum(v > 0 for v in grp["harmful"]), n_random_improved=sum(v > 0 for v in grp["random"]),
               per_expert_spearman=pe_rho, per_expert_mean_abs_actual_change=contrib,
               per_expert_share_of_abs_change={e: c / tot for e, c in contrib.items()},
               s0_vs_stored_full_run_max_abs_diff=s0_diff, swaps=rows,
               per_expert_S0={e: R["S0"]["per_expert"][e]["mean"] for e in ex},
               cost=dict(profile_s=prep["profile_s"], reference_s=prep["reference_s"], n_profiled=prep["n_profiled"],
                         stats_mb=prep["stats_bytes"] / 1e6, hessian_s=[R[n]["hessian_s"] for n in names],
                         gptq_s=[R[n]["gptq_s"] for n in names],
                         peak_gpu_mem_gb=max(R[n]["peak_gpu_mem_gb"] for n in names)),
               n_gptq_evaluations=len(names), proposals_sha256=h)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
    # figures
    fd = os.path.join(a.outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    col = {"beneficial": "#1baf7a", "harmful": "#eb6834", "random": "#2a78d6"}
    mk = {"beneficial": "^", "harmful": "v", "random": "o"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#e6e6e6"})
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for k in col:
        rs = [r for r in rows if r["group"] == k]
        ax.scatter([1e4 * r["predicted_gain"] for r in rs], [1e6 * r["actual_gain"] for r in rs], s=48, color=col[k],
                   marker=mk[k], label=k, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(0, color="#777777", linewidth=1); ax.axvline(0, color="#777777", linewidth=1)
    ax.set_xlabel("predicted gain  D(S0) − D(S_swap)  (diagonal mismatch, ×1e-4)")
    ax.set_ylabel("actual gain  E0 − E_swap  (×1e-6)")
    ax.set_title(f"Predicted vs actual swap gain (Spearman {rho:+.2f}, n = {len(rows)})", loc="left")
    ax.grid(True); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for e_ in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"predicted_vs_actual.{e_}"), dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, k in enumerate(col):
        v = [1e6 * x for x in grp[k]]
        ax.scatter([i] * len(v), v, s=48, color=col[k], marker=mk[k], edgecolor="white", linewidth=1.5, zorder=3)
        if v:
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="#333333", linewidth=2)
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xticks(range(3), ["predicted\nbeneficial", "predicted\nharmful", "random"])
    ax.set_ylabel("actual gain  E0 − E_swap  (×1e-6)")
    ax.set_title("Actual gain by swap group (bar = mean)", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    for e_ in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"gain_by_group.{e_}"), dpi=200)
    plt.close(fig)
    write_md(res, pr, prep, os.path.join(a.outdir, "summary.md"))
    print(open(os.path.join(a.outdir, "summary.md")).read())


def write_md(res, pr, prep, path):
    L = []
    w = L.append
    w("# Activation-aware calibration swap pilot — summary\n")
    w("Layer 16, 8 experts, allocation seed 0, B = 128, frozen 4-bit GPTQ (g128, percdamp 1.0, act-order). "
      "Reconstruction on the OLD test split (rows 2128–2255; exploratory, previously inspected). Reference "
      "statistics from the tuning split (rows 2000–2127; previously used). **Exploratory pilot; no significance "
      "claims.**\n")
    w(f"**Decision (rule frozen in proposals.json): {res['decision']}**\n")
    w(f"- Experts (8 most-routed in layer 16 over the original pool): {res['experts']}")
    w(f"- Baseline S0 = matched Full Random seed 0 (hash {pr['S0_hash']}); D(S0) = {res['D_S0']:.6g}; "
      f"E0 = {res['E0']:.6f} (8-expert mean; per-expert values match the stored full run to "
      f"{res['s0_vs_stored_full_run_max_abs_diff']:.1e}).")
    w(f"- Proposals: {pr['n_proposals']} (16 out × 64 in); predicted gain > 0: {pr['n_positive']}, < 0: {pr['n_negative']}.")
    c = res["cost"]
    w(f"- Profiling: {c['n_profiled']} sequences in {c['profile_s']:.1f} s + reference (128 tuning seqs) "
      f"{c['reference_s']:.1f} s on one layer; statistics file {c['stats_mb']:.1f} MB. GPTQ evaluations: "
      f"{res['n_gptq_evaluations']} (Hessian replay ≈ {np.mean(c['hessian_s']):.1f} s, GPTQ of 8 experts ≈ "
      f"{np.mean(c['gptq_s']):.1f} s each; peak GPU {c['peak_gpu_mem_gb']:.1f} GB).\n")
    w("## Nine swaps\n")
    w("| swap | out → in | predicted gain | actual gain (×1e-6) | rel. change | experts improved | Δ tokens (8 experts) | Δ mean d(s) | Δ agg. L1 (L16) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in res["swaps"]:
        w(f"| {r['name']} | {r['out']} → {r['in']} | {r['predicted_gain']:+.3e} | {1e6 * r['actual_gain']:+.2f} | "
          f"{100 * r['rel_change']:+.3f} % | {r['n_experts_improved']}/8 | {r['token_change_total']:+.0f} | "
          f"{r['d_set_mean_change']:+.5f} | {r['aggregate_l1_L16_change']:+.5f} |")
    gm = res["group_mean_actual_gain"]
    w(f"\n**Spearman(predicted, actual) = {res['spearman']:+.3f}** (n = 9).\n")
    w("| group | mean actual gain (×1e-6) | swaps improving |")
    w("|---|---|---|")
    for k, nk in (("beneficial", "n_beneficial_improved"), ("harmful", "n_harmful_improved"), ("random", "n_random_improved")):
        w(f"| {k} | {1e6 * gm[k]:+.2f} | {res[nk]}/{len(res['group_actual_gains'][k])} |")
    w(f"\nRandom-group SD of actual gain: {1e6 * res['random_group_sd']:.2f} ×1e-6 (noise reference).\n")
    w("## Per-expert\n")
    w("| expert | S0 error | Spearman (per-expert predicted vs actual) | share of mean |Δ error| |")
    w("|---|---|---|---|")
    for e in res["per_expert_spearman"]:
        w(f"| {e} | {res['per_expert_S0'][e]:.6f} | {res['per_expert_spearman'][e]:+.2f} | "
          f"{100 * res['per_expert_share_of_abs_change'][e]:.1f} % |")
    w("\nFigures: `figures/predicted_vs_actual.{pdf,png}`, `figures/gain_by_group.{pdf,png}`.")
    open(path, "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_activation_swap")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    gp = ap.add_mutually_exclusive_group(required=True)
    for s in ("prepare", "propose", "evaluate", "report"):
        gp.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args(argv)
    {"prepare": prepare, "propose": propose, "evaluate": evaluate, "report": report}[
        next(s for s in ("prepare", "propose", "evaluate", "report") if getattr(a, s))](a)


if __name__ == "__main__":
    main()
