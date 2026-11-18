"""Loss-alignment diagnostic MVP A: does weighting experts by routing frequency (R1) or by squared applied router
weight (R2) change the stored reconstruction comparison (R0) and its relation to model-level NLL?

No GPTQ is re-run. Inputs (read-only):
  per-expert E3 test errors   runs_typicality_matched/*.json via runs_partial_ppl/selected_set_manifest.json (FR/TR/FC/TC),
                              runs_shared_covariance_e3/results/A_SC_s{0,1,2}.json (SC)
  ORIGINAL-extraction NLL     runs_partial_ppl/ppl_{C}_s{s}.json, runs_shared_covariance_e3/results/A_SC_s{s}.json
  grouped NLL + per-expert    runs_shared_covariance_e3/grouping_robustness/results (FR/TR/TC/SC, 5 groupings/seed)
                              runs_multigroup_robustness/results (FR/FC s0) + runs_cross_seed_grouping/results (FR/FC s1,s2)
                              -- the two studies use different grouping caches, so grouped deltas are paired WITHIN a study.
  C4 reference routing        runs_allocation_minimal/routing_profile_v2.npz (pool rows 0-1999, model top-k counts)
  C4 router weights           ~/moe_cache_loss_alignment/c4_router_stats.npz (stage c4router, layer replay of the pool)
  WikiText reference routing  ~/moe_cache_inference_routing/bf16_reference_routing.pt (BF16 model, the 128 frozen PPL
                              windows, top-4 ids + FP32 probabilities; applied weight = BF16 cast, norm_topk_prob=False)

Stages:  c4router (GPU, partial load)  |  analyze (CPU)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, resource, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

OUT = os.path.join(ROOT, "runs_loss_alignment", "mvp_a")
STORE = os.path.expanduser("~/moe_cache_loss_alignment")
LAYERS = [1, 8, 16, 23]
E = 60
SEEDS = (0, 1, 2)
STRATS = ("FR", "TR", "FC", "TC", "SC")
GROUPINGS = [20261111, 20261112, 20261113, 20261114, 20261115]
OCACHE = os.path.expanduser("~/moe_cache")
WIKI_REF = os.path.expanduser("~/moe_cache_inference_routing/bf16_reference_routing.pt")
C4_PROFILE = os.path.join(ROOT, "runs_allocation_minimal", "routing_profile_v2.npz")
C4_ROUTER = os.path.join(STORE, "c4_router_stats.npz")
METRICS = ("R0", "R1_c4", "R1_wiki", "R2_c4", "R2_wiki")
LABEL = {"R0": "R0 equal-expert", "R1_c4": "R1 frequency (C4 pool)", "R1_wiki": "R1 frequency (WikiText)",
         "R2_c4": "R2 router-weight² (C4 pool)", "R2_wiki": "R2 router-weight² (WikiText)"}


def jload(path):
    with open(path) as f:
        return json.load(f)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ===================================================================== GPU stage: C4 router weights
def stage_c4router(a):
    """Replay layers 1/8/16/23 on the cached BF16 inputs of pool rows 0..1999 and record, per expert, the routed
    token count and sum of squared APPLIED router weights (HF semantics). Counts must equal routing_profile_v2."""
    import torch
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, LAYERS))
    from moe_calib import model_utils as MU
    from moe_calib import loss_alignment as LA
    from moe_calib.data_utils import load_layer_cache
    model, _ = MU.load_model()
    k, norm = MU.top_k(model), bool(model.config.norm_topk_prob)
    prof = np.load(C4_PROFILE)
    assert list(prof["layers"]) == LAYERS and int(prof["E"]) == E
    pool = [int(x) for x in prof["pool_ids"]]
    assert pool == list(range(2000))
    cnt = np.zeros((len(LAYERS), E)); s1 = np.zeros_like(cnt); s2 = np.zeros_like(cnt)
    per_seq_cnt = np.zeros((len(pool), len(LAYERS) * E), dtype=np.float32)
    t0 = time.time()
    for li, L in enumerate(LAYERS):
        cache = load_layer_cache(OCACHE, L)
        got = []
        h = MU.moe_block(model, L).gate.register_forward_hook(lambda m, i, o: got.append(o.detach()))
        try:
            with MU.layer_on_gpu(model, L):
                for b in range(0, len(pool), a.batch):
                    rows = pool[b:b + a.batch]
                    got.clear()
                    MU.run_layer(model, L, cache[rows].to(MU.GPU))
                    assert len(got) == 1
                    logits = got[0]
                    idx, w32, wa = LA.router_topk(logits, k, norm_topk_prob=norm, out_dtype=MU.DTYPE)
                    c, a1, a2 = LA.routing_stats(idx.cpu(), wa.float().cpu(), E)
                    cnt[li] += c; s1[li] += a1; s2[li] += a2
                    T = cache.shape[1]
                    sid = torch.arange(len(rows)).repeat_interleave(T * k)
                    ix = idx.reshape(len(rows), T * k).reshape(-1).cpu()
                    pc = torch.zeros(len(rows), E).index_put_((sid, ix), torch.ones(ix.numel()), accumulate=True)
                    per_seq_cnt[b:b + len(rows), li * E:(li + 1) * E] = pc.numpy()
        finally:
            h.remove()
        print(f"[c4router] layer {L}: {cnt[li].sum():.0f} token-slots ({time.time() - t0:.0f} s)", flush=True)
    C = prof["C"]
    match = dict(per_seq_counts_equal_profile=bool(np.array_equal(per_seq_cnt, C)),
                 max_abs_per_seq_diff=float(np.abs(per_seq_cnt - C).max()),
                 total_counts_equal=bool(np.array_equal(cnt.reshape(-1), C.sum(0).astype(np.float64))))
    os.makedirs(STORE, exist_ok=True)
    np.savez(C4_ROUTER, counts=cnt, sum_w=s1, sum_w2=s2, layers=np.array(LAYERS), top_k=k, norm_topk_prob=norm)
    info = dict(file=C4_ROUTER, sha256=sha256_file(C4_ROUTER), profile_check=match, seconds=time.time() - t0,
                batch=a.batch, weight_definition="BF16-cast top-k softmax probability (norm_topk_prob=False), as applied",
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                peak_cpu_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    jdump(info, os.path.join(OUT, "c4_router_stats.json"))
    print(json.dumps(match), flush=True)
    if not match["per_seq_counts_equal_profile"]:
        raise SystemExit("C4 replay routing does not reproduce routing_profile_v2 counts")


# ===================================================================== loading
def reference_weights():
    """{name: [4, 60] row-normalized}, plus provenance / checks."""
    import torch
    from moe_calib import loss_alignment as LA
    prof = np.load(C4_PROFILE)
    assert list(prof["layers"]) == LAYERS
    c4_counts = prof["C"].sum(0).astype(np.float64).reshape(len(LAYERS), E)
    W = {"R0": LA.uniform_weights(len(LAYERS), E), "R1_c4": LA.normalize_rows(c4_counts, "R1_c4")}
    info = dict(R1_c4=dict(source=os.path.relpath(C4_PROFILE, ROOT), sha256=sha256_file(C4_PROFILE),
                           rule=str(prof["rule"]), n_seqs=int(prof["C"].shape[0]),
                           matches_stored_p=bool(np.allclose(W["R1_c4"].reshape(-1), prof["p"], rtol=0, atol=1e-12))))
    ref = torch.load(WIKI_REF, weights_only=False)
    prot = jload(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json"))
    assert ref["windows_sha256"] == prot["windows_sha256"], "WikiText routing reference is for different windows"
    idx, pk = ref["idx"], ref["pk"]
    wa = pk.to(torch.bfloat16).float()                         # the weight the BF16 model actually multiplies by
    wc, w2 = np.zeros((len(LAYERS), E)), np.zeros((len(LAYERS), E))
    for li, L in enumerate(LAYERS):
        c, _, s2 = LA.routing_stats(idx[L], wa[L], E)
        wc[li], w2[li] = c, s2
    W["R1_wiki"] = LA.normalize_rows(wc, "R1_wiki")
    W["R2_wiki"] = LA.normalize_rows(w2, "R2_wiki")
    info["wiki"] = dict(source=WIKI_REF, sha256=sha256_file(WIKI_REF), windows_sha256=ref["windows_sha256"],
                        n_windows=int(idx.shape[1]), tokens_per_window=int(idx.shape[2]), top_k=int(idx.shape[3]),
                        slot_sum_max=float(pk[LAYERS].sum(-1).max()), slot_sum_min=float(pk[LAYERS].sum(-1).min()),
                        indices_in_range=bool(int(idx[LAYERS].min()) >= 0 and int(idx[LAYERS].max()) < E),
                        no_repeated_ids=bool((idx[LAYERS].sort(-1).values.diff(dim=-1) != 0).all()),
                        weight="BF16 cast of the FP32 top-4 softmax probability (norm_topk_prob=False)")
    if os.path.exists(C4_ROUTER):
        z = np.load(C4_ROUTER)
        assert np.array_equal(z["counts"], c4_counts), "C4 router replay counts differ from the routing profile"
        W["R2_c4"] = LA.normalize_rows(z["sum_w2"], "R2_c4")
        info["R2_c4"] = dict(source=C4_ROUTER, sha256=sha256_file(C4_ROUTER), counts_equal_profile=True)
    for k, w in W.items():
        assert np.allclose(w.sum(1), 1.0, atol=1e-12), k
    return W, info


def per_expert_original():
    """{(C, s): dict(E [4,60], mean_nll, ppl, recon_stored, source)} for the ORIGINAL extraction."""
    from moe_calib import loss_alignment as LA
    man = jload(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json"))
    out = {}
    for c in STRATS:
        for s in SEEDS:
            if c == "SC":
                src = os.path.join(ROOT, "runs_shared_covariance_e3", "results", f"A_SC_s{s}.json")
                r = jload(src)
                assert r["install_ok"] and r["extraction"] == "ORIGINAL"
                Em = LA.expert_matrix(r["per_expert_test"], LAYERS, E)
                out[(c, s)] = dict(E=Em, recon_stored=r["recon_test"], mean_nll=r["mean_nll"], ppl=r["ppl"],
                                   total_tokens=r["total_tokens"], ids_hash=r["ids_hash"],
                                   recon_src=os.path.relpath(src, ROOT), nll_src=os.path.relpath(src, ROOT),
                                   usage_weighted_stored=None)
            else:
                ent = man["configs"][f"{c}_s{s}"]
                src = os.path.join(ROOT, ent["source"])
                assert sha256_file(src) == ent["source_sha256"]
                r = jload(src)
                pp = jload(os.path.join(ROOT, "runs_partial_ppl", f"ppl_{c}_s{s}.json"))
                assert pp["install_ok"] and pp["ids_hash"] == ent["ids_hash"]
                out[(c, s)] = dict(E=LA.expert_matrix(r["per_expert"], LAYERS, E), recon_stored=r["summary"]["recon_mean"],
                                   mean_nll=pp["mean_nll"], ppl=pp["ppl"], total_tokens=pp["total_tokens"],
                                   ids_hash=ent["ids_hash"], recon_src=ent["source"],
                                   nll_src=f"runs_partial_ppl/ppl_{c}_s{s}.json",
                                   usage_weighted_stored=r["summary"].get("recon_usage_weighted"))
    return out


def grouped():
    """{(study, C, s, g): dict(E, mean_nll, ppl)}; study 'B' = shared-covariance Stage B, 'FC' = FR/FC grouping studies."""
    from moe_calib import loss_alignment as LA
    out = {}
    d = os.path.join(ROOT, "runs_shared_covariance_e3", "grouping_robustness", "results")
    for c in ("FR", "TR", "TC", "SC"):
        for s in SEEDS:
            for g in GROUPINGS:
                r = jload(os.path.join(d, f"{c}_s{s}_G{g}.json"))
                assert r["install_ok"] and r["strategy"] == c and r["seed"] == s and r["grouping_seed"] == g
                out[("B", c, s, g)] = dict(E=LA.expert_matrix(r["per_expert_test"], LAYERS, E), mean_nll=r["mean_nll"],
                                           ppl=r["ppl"], total_tokens=r["total_tokens"],
                                           src=os.path.relpath(os.path.join(d, f"{c}_s{s}_G{g}.json"), ROOT))
    for c in ("FR", "FC"):
        for s in SEEDS:
            for g in GROUPINGS:
                if s == 0:
                    f = os.path.join(ROOT, "runs_multigroup_robustness", "results", f"{c}_G{g}.json")
                else:
                    f = os.path.join(ROOT, "runs_cross_seed_grouping", "results", f"{c}_s{s}_G{g}.json")
                r = jload(f)
                assert r["valid"] and r["strategy"] == c and r.get("allocation_seed", 0) == s
                out[("FC", c, s, g)] = dict(E=LA.expert_matrix(r["per_expert"], LAYERS, E), mean_nll=r["mean_nll"],
                                            ppl=r["ppl"], total_tokens=r["total_tokens"], src=os.path.relpath(f, ROOT))
    return out


# ===================================================================== analysis
def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(spearmanr(x, y).correlation) if len(x) > 2 else float("nan")


def stage_analyze(a):
    from moe_calib import loss_alignment as LA
    t0 = time.time()
    W, winfo = reference_weights()
    mets = [m for m in METRICS if m in W]
    orig = per_expert_original()
    grp = grouped()
    checks = {}
    # ---- R0 reproduces the stored macro metric; R1_c4 reproduces E3's stored recon_usage_weighted
    r0_err = max(abs(LA.weighted_reconstruction(v["E"], W["R0"]) - v["recon_stored"]) / v["recon_stored"]
                 for v in orig.values())
    uw = [abs(LA.weighted_reconstruction(v["E"], W["R1_c4"]) - v["usage_weighted_stored"]) / v["usage_weighted_stored"]
          for v in orig.values() if v["usage_weighted_stored"] is not None]
    checks.update(R0_reproduces_stored_recon_max_rel=r0_err, R0_ok=r0_err < 1e-12,
                  R1_c4_reproduces_stored_usage_weighted_max_rel=max(uw), R1_c4_matches_stored_p=winfo["R1_c4"]["matches_stored_p"],
                  weights_rows_sum_to_1={k: bool(np.allclose(w.sum(1), 1, atol=1e-12)) for k, w in W.items()},
                  nll_tokens_equal=len({v["total_tokens"] for v in orig.values()} |
                                       {v["total_tokens"] for v in grp.values()}) == 1)
    assert checks["R0_ok"], "R0 does not reproduce the stored macro reconstruction"
    # ---- weight concentration
    conc = {}
    for k, w in W.items():
        eff = 1.0 / (w ** 2).sum(1)
        conc[k] = dict(effective_experts_per_layer=eff.tolist(), max_weight_x_uniform=(w.max(1) * E).tolist(),
                       min_weight_x_uniform=(w.min(1) * E).tolist())
    corr_w = {f"{a_}|{b_}": spearman(W[a_].reshape(-1), W[b_].reshape(-1)) for a_ in mets for b_ in mets if a_ < b_ and "R0" not in (a_, b_)}
    # ---- metrics per config
    rows = []
    for (c, s), v in sorted(orig.items(), key=lambda kv: (STRATS.index(kv[0][0]), kv[0][1])):
        r = dict(strategy=c, seed=s, extraction="ORIGINAL", grouping="", ids_hash=v["ids_hash"],
                 mean_nll=v["mean_nll"], ppl=v["ppl"], recon_src=v["recon_src"], nll_src=v["nll_src"])
        for m in mets:
            r[m] = LA.weighted_reconstruction(v["E"], W[m])
            for li, L in enumerate(LAYERS):
                r[f"{m}_L{L}"] = float(LA.per_layer_weighted(v["E"], W[m])[li])
        rows.append(r)
    for (study, c, s, g), v in sorted(grp.items(), key=lambda kv: (kv[0][0], STRATS.index(kv[0][1]), kv[0][2], kv[0][3])):
        r = dict(strategy=c, seed=s, extraction=f"grouped[{'StageB' if study == 'B' else 'FRFC'}]", grouping=g,
                 ids_hash="", mean_nll=v["mean_nll"], ppl=v["ppl"], recon_src=v["src"], nll_src=v["src"])
        for m in mets:
            r[m] = LA.weighted_reconstruction(v["E"], W[m])
        rows.append(r)
    # ---- paired comparisons vs FR (within seed; grouped: within study, averaged over groupings FIRST)
    pairs = []
    for c in ("TR", "TC", "FC", "SC"):
        study = "FC" if c == "FC" else "B"
        for s in SEEDS:
            A, B = orig[(c, s)], orig[("FR", s)]
            r = dict(strategy=c, seed=s, grouped_study="runs_multigroup_robustness+runs_cross_seed_grouping"
                     if c == "FC" else "runs_shared_covariance_e3 Stage B")
            for m in mets:
                ra, rb = LA.weighted_reconstruction(A["E"], W[m]), LA.weighted_reconstruction(B["E"], W[m])
                r[f"d_{m}_rel"] = (ra - rb) / rb
                gd = [(LA.weighted_reconstruction(grp[(study, c, s, g)]["E"], W[m]) -
                       LA.weighted_reconstruction(grp[(study, "FR", s, g)]["E"], W[m])) /
                      LA.weighted_reconstruction(grp[(study, "FR", s, g)]["E"], W[m]) for g in GROUPINGS]
                r[f"d_{m}_rel_grouped_mean"] = float(np.mean(gd))
            r["d_nll_original"] = A["mean_nll"] - B["mean_nll"]
            r["d_ppl_original"] = A["ppl"] - B["ppl"]
            gn = [grp[(study, c, s, g)]["mean_nll"] - grp[(study, "FR", s, g)]["mean_nll"] for g in GROUPINGS]
            r["d_nll_grouped_mean"] = float(np.mean(gn))
            r["d_nll_grouped_sd"] = float(np.std(gn, ddof=1))
            r["d_nll_grouped_n_better"] = int(sum(x < 0 for x in gn))
            pairs.append(r)
    # ---- per-strategy summaries (3-seed means of relative deltas) + retention vs R0
    strat_sum = {}
    for c in ("TR", "TC", "FC", "SC"):
        P = [r for r in pairs if r["strategy"] == c]
        d = {m: [r[f"d_{m}_rel"] for r in P] for m in mets}
        strat_sum[c] = dict(
            per_seed={m: d[m] for m in mets}, mean={m: float(np.mean(d[m])) for m in mets},
            retention_vs_R0={m: float(np.mean(d[m]) / np.mean(d["R0"])) for m in mets},
            sign_consistent={m: bool(all(x < 0 for x in d[m]) or all(x > 0 for x in d[m])) for m in mets},
            d_nll_original=[r["d_nll_original"] for r in P], d_nll_grouped_mean=[r["d_nll_grouped_mean"] for r in P],
            d_nll_grouped_sd=[r["d_nll_grouped_sd"] for r in P],
            grouped_rel={m: [r[f"d_{m}_rel_grouped_mean"] for r in P] for m in mets})
    # ---- per-expert: does SC's improvement concentrate in low-weight experts?
    perexp = {}
    for c in ("SC", "FC", "TR", "TC"):
        imp = np.stack([(orig[("FR", s)]["E"] - orig[(c, s)]["E"]) / orig[("FR", s)]["E"] for s in SEEDS]).mean(0)
        perexp[c] = dict({f"spearman_improvement_vs_{m}_weight": spearman(imp.reshape(-1), W[m].reshape(-1))
                          for m in mets if m != "R0"},
                         mean_improvement_top_quartile_weight={m: float(imp.reshape(-1)[W[m].reshape(-1) >= np.quantile(
                             W[m].reshape(-1), 0.75)].mean()) for m in mets if m != "R0"},
                         mean_improvement_bottom_quartile_weight={m: float(imp.reshape(-1)[W[m].reshape(-1) <= np.quantile(
                             W[m].reshape(-1), 0.25)].mean()) for m in mets if m != "R0"})
    # ---- descriptive alignment with NLL over the 12 (strategy, seed) paired deltas
    align = {}
    for m in mets:
        align[m] = dict(spearman_vs_dnll_original=spearman([r[f"d_{m}_rel"] for r in pairs],
                                                           [r["d_nll_original"] for r in pairs]),
                        spearman_vs_dnll_grouped=spearman([r[f"d_{m}_rel_grouped_mean"] for r in pairs],
                                                          [r["d_nll_grouped_mean"] for r in pairs]),
                        n=len(pairs))
    # ---- write
    os.makedirs(OUT, exist_ok=True)
    cols = ["strategy", "seed", "extraction", "grouping", "ids_hash"] + mets + \
           [f"{m}_L{L}" for m in mets for L in LAYERS] + ["mean_nll", "ppl", "recon_src", "nll_src"]
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    pcols = ["strategy", "seed"] + [f"d_{m}_rel" for m in mets] + [f"d_{m}_rel_grouped_mean" for m in mets] + \
            ["d_nll_original", "d_ppl_original", "d_nll_grouped_mean", "d_nll_grouped_sd", "d_nll_grouped_n_better",
             "grouped_study"]
    with open(os.path.join(OUT, "paired_comparisons.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=pcols)
        wr.writeheader()
        for r in pairs:
            wr.writerow(r)
    np.savez(os.path.join(OUT, "reference_weights.npz"), layers=np.array(LAYERS), **W)
    res = dict(metrics=mets, labels=LABEL, weight_info=winfo, checks=checks, weight_concentration=conc,
               weight_spearman=corr_w, strategy_summary=strat_sum, per_expert_vs_weight=perexp, alignment=align,
               environment=dict(python=platform.python_version(), numpy=np.__version__, host=platform.node()),
               runtime_s=time.time() - t0,
               notes=["E[L,e] = stored per_expert['mean'] (mean over held-out sequences of relative squared "
                      "gate/up/down projection error); unchanged.",
                      "R2 is a heuristic: squared applied router weight, ignoring expert output scale, the "
                      "nonlinearity and cross-expert interaction.",
                      "Grouped deltas average the 5 groupings within an allocation seed before anything else; "
                      "groupings are not independent seeds.",
                      "Spearman over 12 (strategy, seed) deltas is descriptive only."])
    jdump(res, os.path.join(OUT, "results.json"))
    figures(rows, pairs, mets)
    write_summary(res, rows, pairs)
    print(f"[mvp_a] done in {time.time() - t0:.1f} s; checks {json.dumps({k: v for k, v in checks.items() if k != 'weights_rows_sum_to_1'})}")


# ===================================================================== figures + summary
COL = {"FR": "#2a78d6", "TR": "#eb6834", "FC": "#9b59b6", "TC": "#1baf7a", "SC": "#e5a000"}


def figures(rows, pairs, mets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(OUT, "figures")
    os.makedirs(fd, exist_ok=True)
    # fig1: relative change vs FR per metric, per strategy (bars = 3-seed mean, dots = seeds)
    strats = ("TR", "TC", "FC", "SC")
    fig, axs = plt.subplots(1, len(strats), figsize=(4.2 * len(strats), 3.8), sharey=False)
    for ax, c in zip(axs, strats):
        P = [r for r in pairs if r["strategy"] == c]
        for i, m in enumerate(mets):
            v = [100 * r[f"d_{m}_rel"] for r in P]
            ax.bar(i, np.mean(v), color=COL[c], alpha=0.85, width=0.65)
            ax.scatter([i] * len(v), v, color="white", edgecolor="black", zorder=3, s=22)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(range(len(mets)))
        ax.set_xticklabels([m.replace("_", "\n") for m in mets], fontsize=8)
        ax.set_title(f"{c} − FR")
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("relative change in weighted recon error vs FR (%)\n(< 0 = better than FR)")
    fig.suptitle("Expert-reconstruction aggregates: change vs Full Random (ORIGINAL extraction; bar = 3-seed mean, dots = seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig1_aggregates_vs_FR.png"), dpi=150)
    plt.close(fig)
    # fig2: paired metric change vs paired NLL change (grouping-averaged), one panel per metric
    fig, axs = plt.subplots(1, len(mets), figsize=(3.6 * len(mets), 3.6), sharey=True)
    for ax, m in zip(axs, mets):
        for c in strats:
            P = [r for r in pairs if r["strategy"] == c]
            ax.scatter([100 * r[f"d_{m}_rel_grouped_mean"] for r in P], [r["d_nll_grouped_mean"] for r in P],
                       color=COL[c], label=c, s=30)
        ax.axhline(0, color="gray", lw=0.7)
        ax.axvline(0, color="gray", lw=0.7)
        ax.set_xlabel(f"Δ {m} vs FR (%)", fontsize=9)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("Δ mean NLL vs FR\n(5-grouping mean within seed)")
    axs[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired (same allocation seed) reconstruction change vs NLL change; each dot = one (strategy, seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig2_delta_metric_vs_delta_nll.png"), dpi=150)
    plt.close(fig)


def fmt_pct(x):
    return f"{100 * x:+.2f} %"


def write_summary(res, rows, pairs):
    mets, S, ck = res["metrics"], res["strategy_summary"], res["checks"]
    L = []
    w = L.append
    w("# Loss-alignment MVP A — expert-reconstruction aggregations vs model-level NLL\n")
    w("No GPTQ was re-run. Every number comes from stored per-expert E3 test errors and stored partial-model "
      "WikiText-2 NLLs; only the expert aggregation weights are new. Reference weights are FIXED and identical for "
      "every strategy (never computed from a strategy's own calibration set); layers are always weighted equally.\n")
    w("## Metric definitions\n")
    w("- `E[L,e]`: stored `per_expert.mean` = mean over the 128 held-out C4 test sequences of the per-sequence relative "
      "squared gate/up/down projection error. Unchanged.")
    w("- **R0** = mean_L mean_e E[L,e] (project macro metric).")
    w("- **R1_c4** = mean_L Σ_e p[L,e] E[L,e], p = routed-token frequency over the 2,000-sequence C4 candidate pool "
      "(`routing_profile_v2.npz`).  **R1_wiki**: same with the BF16 model's routing on the 128 frozen WikiText-2 PPL windows.")
    w("- **R2** = mean_L Σ_e a[L,e] E[L,e], a ∝ Σ_t 1[e ∈ top-4(t)] · w_{t,e}², w = the router weight the BF16 model "
      "actually applies (FP32 softmax over 60 experts, top-4, **no renormalization** since `norm_topk_prob=False`, then cast "
      "to BF16). R2_c4 from a layer replay of the C4 pool; R2_wiki from the 128 WikiText windows. *Heuristic*: ignores "
      "expert output scale, the nonlinearity and cross-expert interaction.\n")
    w("## Checks\n")
    w(f"- R0 reproduces every stored macro reconstruction (max rel diff {ck['R0_reproduces_stored_recon_max_rel']:.1e}).")
    w(f"- R1_c4 reproduces E3's stored `recon_usage_weighted` for the 12 FR/TR/FC/TC runs (max rel diff "
      f"{ck['R1_c4_reproduces_stored_usage_weighted_max_rel']:.1e}); p equals the stored profile p: {ck['R1_c4_matches_stored_p']}.")
    w(f"- All weight rows sum to 1: {ck['weights_rows_sum_to_1']}. Same number of scored NLL tokens everywhere: {ck['nll_tokens_equal']}.")
    wi = res["weight_info"]["wiki"]
    w(f"- WikiText routing: {wi['n_windows']} windows × {wi['tokens_per_window']} tokens, top-{wi['top_k']}, ids valid "
      f"{wi['indices_in_range']}, no repeats {wi['no_repeated_ids']}, per-token sum of the 4 weights in "
      f"[{wi['slot_sum_min']:.3f}, {wi['slot_sum_max']:.3f}] (< 1, as expected without renormalization).")
    if "R2_c4" in res["weight_info"]:
        w("- C4 router replay reproduces the stored per-sequence routing counts exactly (`c4_router_stats.json`).")
    w("")
    w("Weight concentration (effective number of experts per layer, L1/L8/L16/L23; uniform = 60):\n")
    w("| weights | L1 | L8 | L16 | L23 | max weight ÷ uniform (per layer) |")
    w("|---|---|---|---|---|---|")
    for m in mets:
        c = res["weight_concentration"][m]
        w(f"| {m} | " + " | ".join(f"{x:.1f}" for x in c["effective_experts_per_layer"]) + " | " +
          " / ".join(f"{x:.2f}" for x in c["max_weight_x_uniform"]) + " |")
    w("")
    w("## 1. Aggregates per strategy (ORIGINAL extraction, 3-seed mean)\n")
    w("| strategy | " + " | ".join(mets) + " | mean NLL | PPL |")
    w("|---|" + "---|" * (len(mets) + 2))
    orig = [r for r in rows if r["extraction"] == "ORIGINAL"]
    for c in STRATS:
        R = [r for r in orig if r["strategy"] == c]
        w(f"| {c} | " + " | ".join(f"{np.mean([r[m] for r in R]):.6f}" for m in mets) +
          f" | {np.mean([r['mean_nll'] for r in R]):.6f} | {np.mean([r['ppl'] for r in R]):.4f} |")
    w("")
    w("## 2. Paired change vs FR (relative; < 0 = lower error than FR)\n")
    w("| strategy | " + " | ".join(mets) + " | ΔNLL ORIGINAL per seed | ΔNLL grouped (5-grouping mean ± SD per seed) |")
    w("|---|" + "---|" * (len(mets) + 2))
    for c in ("SC", "TR", "TC", "FC"):
        s = S[c]
        cells = []
        for m in mets:
            cells.append(f"{fmt_pct(s['mean'][m])} ({' / '.join(f'{100 * x:+.2f}' for x in s['per_seed'][m])})")
        w(f"| {c} | " + " | ".join(cells) + " | " + " / ".join(f"{x:+.5f}" for x in s["d_nll_original"]) + " | " +
          " / ".join(f"{m_:+.5f} ± {sd:.5f}" for m_, sd in zip(s["d_nll_grouped_mean"], s["d_nll_grouped_sd"])) + " |")
    w("\nRetention of the R0 change (mean Δ_metric ÷ mean Δ_R0; 1 = unchanged by weighting, < 1 = smaller, < 0 = reversed):\n")
    w("| strategy | " + " | ".join(mets) + " | sign consistent over 3 seeds |")
    w("|---|" + "---|" * (len(mets) + 1))
    for c in ("SC", "TR", "TC", "FC"):
        s = S[c]
        w(f"| {c} | " + " | ".join(f"{s['retention_vs_R0'][m]:.2f}" for m in mets) + " | " +
          ", ".join(m for m in mets if s["sign_consistent"][m]) + " |")
    w("\nGrouping-averaged paired change (per-expert errors of the grouped runs, same study as the grouped NLL):\n")
    w("| strategy | " + " | ".join(mets) + " |")
    w("|---|" + "---|" * len(mets))
    for c in ("SC", "TR", "TC", "FC"):
        w(f"| {c} | " + " | ".join(" / ".join(f"{100 * x:+.2f}" for x in S[c]["grouped_rel"][m]) + " %" for m in mets) + " |")
    w("")
    w("![](figures/fig1_aggregates_vs_FR.png)\n")
    w("## 3. Where does the change sit? (per-expert relative improvement over FR, 3-seed mean, vs expert weight)\n")
    w("| strategy | metric weight | Spearman(improvement, weight) | mean improvement, top-quartile-weight experts | bottom quartile |")
    w("|---|---|---|---|---|")
    for c in ("SC", "FC"):
        pe = res["per_expert_vs_weight"][c]
        for m in mets:
            if m == "R0":
                continue
            w(f"| {c} | {m} | {pe[f'spearman_improvement_vs_{m}_weight']:+.2f} | "
              f"{fmt_pct(pe['mean_improvement_top_quartile_weight'][m])} | {fmt_pct(pe['mean_improvement_bottom_quartile_weight'][m])} |")
    w("")
    w("## 4. Alignment with NLL (descriptive; 12 paired (strategy, seed) deltas, 4 strategies)\n")
    w("| metric | Spearman(Δmetric, ΔNLL) ORIGINAL | Spearman(Δmetric, ΔNLL) grouping-averaged |")
    w("|---|---|---|")
    for m in mets:
        a = res["alignment"][m]
        w(f"| {m} | {a['spearman_vs_dnll_original']:+.2f} | {a['spearman_vs_dnll_grouped']:+.2f} |")
    w("\nWith 12 points that are 4 strategies × 3 seeds (not independent), no correlation here is statistically "
      "meaningful; the table only shows whether a weighting changes the ranking.\n")
    w("![](figures/fig2_delta_metric_vs_delta_nll.png)\n")
    interp = os.path.join(OUT, "interpretation.md")
    if os.path.exists(interp):
        w(open(interp).read())
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["c4router", "analyze"])
    ap.add_argument("--batch", type=int, default=1, help="1 = the Profiler's own batch (routing_profile_v2 was built at batch 1; BF16 GEMMs and router ties depend on batch shape)")
    a = ap.parse_args(argv)
    dict(c4router=stage_c4router, analyze=stage_analyze)[a.stage](a)


if __name__ == "__main__":
    main()
