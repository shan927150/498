"""Final WikiText-2 fresh evaluation: article-level generalization of GPTQ execution effects.

Reuses the historical weights (no GPTQ, no recalibration) and the historical evaluator (EP.layerwise_nll, batch 8,
512-token windows on the flattened WikiText-2 test stream, i*512 grid).

Stages (python diagnostics/wt2_fresh_generalization.py --stage S), outputs in runs_wt2_fresh_generalization/:
  hist      historical significance-direction audit (CPU, saved results only) -> historical_significance_audit.json,
            abstract_wording_recommendation.md
  cohorts   article structure, unused-window inventory, cohorts A/B, overlap/usage audit (CPU tokenizer only)
  protocol  freeze protocol.json (+ read-only copy, sha256) BEFORE any model sees the fresh text
  dense     GPU: dense BF16, dense4 x6, dense3 x6 on cohorts A/B (+ parity on the original windows)
  moe       GPU: MoE BF16, 30 Family D + s0 ORIGINAL pair on cohorts A/B (+ parity subset)
  synth     small-cluster synthetic validation of the article bootstrap / null tests
  analyze   variance components, split-half, cross-cohort transfer, window CIs, decision, cost
"""
from __future__ import annotations
import argparse, bisect, gc, hashlib, json, math, os, platform, re, shutil, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
OUT = os.path.join(ROOT, "runs_wt2_fresh_generalization")
EU_DIR = os.path.join(ROOT, "runs_evaluation_uncertainty")
HBV_DIR = os.path.join(ROOT, "runs_heldout_bitwidth_validation")
SEQ, BATCH = 512, 8
GROUPS = ["G20261111", "G20261112", "G20261113", "G20261114", "G20261115"]
DENSE_CONFIGS = GROUPS + ["ORIGINAL"]
SEEDS = (0, 1, 2)
MOE_CONFIGS = [f"{st}_s{s}_{g}" for s in SEEDS for g in GROUPS for st in ("FR", "FC")]
MOE_REFS = ["FR_ORIGINAL", "FC_ORIGINAL"]
EARLY_DEV_TOKENS = 40 * 2048     # e3_allocation.py --ppl default: first 40 x 2048 test-stream tokens (early dev runs)
SEED = 20261119
# prespecified parity / repeat subsets
MOE_PARITY = ["FR_s0_G20261111", "FC_s2_G20261111"]
DENSE4_PARITY = ["G20261111", "ORIGINAL"]
DENSE3_PARITY = ["G20261111"]
MOE_REPEAT = ["FR_s0_G20261111"]
DENSE_REPEAT = ["G20261111"]
COHORTS = ["fresh_article", "fresh_window"]


def p(*a):
    return os.path.join(OUT, *a)


def jdump(o, f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(o, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, f)


def jload(f):
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def sha_file(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_ids(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


def log(msg):
    os.makedirs(p("logs"), exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(p("logs", "run.log"), "a") as fh:
        fh.write(line + "\n")


def cost_add(stage, key, val):
    f = p("logs", "cost_raw.json")
    d = jload(f) or {}
    d.setdefault(stage, {})[key] = val
    jdump(d, f)


# =====================================================================================================================
# stage hist: historical significance audit
# =====================================================================================================================
def stage_hist(a):
    import evaluation_uncertainty as E
    L = E.load_matrix()
    S = E.analysis_sets(L)
    moe = S["moe"]
    mci = jload(os.path.join(EU_DIR, "moe_window_ci_analysis.json"))
    rows = []
    res = {}
    for di, ds in enumerate(["WT2"]):
        for sj, sch in enumerate(E.WT2_SCHEMES):
            uid = E.units_for(sch, ds)
            U = E.Units(moe["X"][ds], uid)
            rng = E.rng_for(9, di, sj)               # identical draws to runs_evaluation_uncertainty
            dD = U.theta(E.draw_counts(rng, U.m, E.B_COV))
            th = moe["X"][ds].mean(0)
            se = dD.std(0, ddof=1)
            p_boot = np.minimum(1.0, 2 * np.minimum((dD <= 0).mean(0), (dD >= 0).mean(0)))
            # BH-FDR
            order = np.argsort(p_boot)
            m = len(p_boot)
            bh = np.empty(m)
            run = 1.0
            for rank in range(m - 1, -1, -1):
                i = order[rank]
                run = min(run, p_boot[i] * m / (rank + 1))
                bh[i] = run
            # max-|t| joint bootstrap (studentized, centred at the estimate)
            tmax = (np.abs(dD - th[None, :]) / se[None, :]).max(1)
            t_obs = np.abs(th) / se
            p_maxT = np.array([(tmax >= t).mean() for t in t_obs])
            crit = float(np.quantile(tmax, 0.95))
            ci_check = [mci["pairs"][c]["WT2"][sch]["ci"] for c in moe["cols"]]
            lo, hi = np.quantile(dD, [.025, .975], axis=0)
            res[sch] = dict(
                n_units=int(U.m), reproduces_stored_ci=bool(np.allclose(np.stack([lo, hi], 1), np.array(ci_check), atol=0, rtol=0)),
                maxT_critical_value=crit,
                pairs={c: dict(mean_nll_diff=float(th[g]), ci95=[float(lo[g]), float(hi[g])], se=float(se[g]),
                               excludes_zero=bool(lo[g] > 0 or hi[g] < 0),
                               favored=("FR" if th[g] > 0 else "FC"),
                               p_boot_two_sided=float(p_boot[g]), p_BH=float(bh[g]), p_maxT_familywise=float(p_maxT[g]),
                               simultaneous_ci95=[float(th[g] - crit * se[g]), float(th[g] + crit * se[g])])
                       for g, c in enumerate(moe["cols"])})
            sig = [c for c in moe["cols"] if res[sch]["pairs"][c]["excludes_zero"]]
            res[sch]["significant"] = [dict(pair=c, favored=res[sch]["pairs"][c]["favored"]) for c in sig]
            res[sch]["directions"] = sorted({res[sch]["pairs"][c]["favored"] for c in sig})
            res[sch]["n_BH_below_0p05"] = int(sum(res[sch]["pairs"][c]["p_BH"] < 0.05 for c in moe["cols"]))
            res[sch]["n_maxT_below_0p05"] = int(sum(res[sch]["pairs"][c]["p_maxT_familywise"] < 0.05 for c in moe["cols"]))
    for c in moe["cols"]:
        s, g = c.split("_")
        a_ = res["article"]["pairs"][c]; w_ = res["window"]["pairs"][c]
        rows.append(dict(seed=int(s[1:]), grouping=g, mean_nll_diff_FC_minus_FR=a_["mean_nll_diff"], article_ci=a_["ci95"],
                         article_significant=a_["excludes_zero"], naive_window_significant=w_["excludes_zero"],
                         favored=a_["favored"], p_boot_article=a_["p_boot_two_sided"], p_BH_article=a_["p_BH"],
                         p_maxT_article=a_["p_maxT_familywise"]))
    art_sig = res["article"]["significant"]
    win_sig = res["window"]["significant"]
    extra = [x for x in win_sig if x["pair"] not in {y["pair"] for y in art_sig}]
    answers = dict(
        q1_article_significant=[x["pair"] for x in art_sig],
        q2_directions=("opposite strategies: " + ", ".join(f"{x['pair']}->{x['favored']}" for x in art_sig))
        if len({x["favored"] for x in art_sig}) > 1 else f"all favor {art_sig[0]['favored'] if art_sig else '-'}",
        q3_naive_extra=[x["pair"] + "->" + x["favored"] for x in extra],
        q4_opposing_only_under_naive=(len({x["favored"] for x in art_sig}) < 2 and len({x["favored"] for x in win_sig}) > 1),
        multiplicity=dict(article_BH_q05=res["article"]["n_BH_below_0p05"], article_maxT_familywise_05=res["article"]["n_maxT_below_0p05"],
                          window_BH_q05=res["window"]["n_BH_below_0p05"], window_maxT_familywise_05=res["window"]["n_maxT_below_0p05"]),
        note=("marginal 95% CIs reproduced unchanged from runs_evaluation_uncertainty (same seeds/draws); p-values are "
              "two-sided percentile bootstrap p (resolution 1/2000); BH and max-|t| are exploratory multiplicity checks"))
    out = dict(source="runs_evaluation_uncertainty (stored per-window NLL; identical bootstrap seeds)",
               sign_convention="Delta = mean NLL(FC) - mean NLL(FR); positive = FR better",
               table=rows, per_scheme=res, answers=answers)
    jdump(out, p("historical_significance_audit.json"))
    # wording recommendation (separate from measurements)
    opp = len({x["favored"] for x in art_sig}) > 1
    fam = res["article"]["n_maxT_below_0p05"]
    txt = ["# Abstract-wording recommendation (historical significance audit)", "",
           "Generated from `historical_significance_audit.json`; this is a recommendation, not a measurement.", ""]
    txt.append(f"- Article-cluster bootstrap (49 articles): {len(art_sig)}/15 marginal 95% CIs exclude 0: " +
               ", ".join(f"{x['pair']} (favours {x['favored']})" for x in art_sig) + ".")
    txt.append(f"- Directions: {'opposite (both FR- and FC-favouring significant comparisons exist)' if opp else 'same direction'}.")
    txt.append(f"- Multiplicity: BH-FDR q<0.05: {res['article']['n_BH_below_0p05']}/15; max-|t| familywise 5%: {fam}/15.")
    if fam == 0:
        txt += ["", "**Recommendation.** The statement \"individual executions yield opposite statistically significant "
                "conclusions\" holds only at the *marginal* (per-comparison) 95% level, not familywise. Suggested wording:",
                "", "> Across 15 numerically equivalent calibration executions, individual FR-vs-FC comparisons on the "
                "same WikiText-2 windows reached marginally significant conclusions in opposite directions (2 favouring "
                "FR, 1 favouring FC at 95% article-cluster bootstrap; none survives a familywise correction), while the "
                "execution-averaged comparison is indistinguishable from zero."]
    else:
        txt += ["", "**Recommendation.** Opposite-direction significance survives the familywise correction for "
                f"{fam} comparison(s); the claim may be kept with the multiplicity qualification stated."]
    with open(p("abstract_wording_recommendation.md"), "w") as fh:
        fh.write("\n".join(txt) + "\n")
    log(f"[hist] article sig {answers['q1_article_significant']} dirs {answers['q2_directions']}; naive extra "
        f"{answers['q3_naive_extra']}; multiplicity {answers['multiplicity']}")


# =====================================================================================================================
# stage cohorts
# =====================================================================================================================
def stream():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from moe_calib import model_utils as MU
    prot = jload(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json"))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    assert hashlib.sha256(text.encode()).hexdigest() == prot["dataset"]["text_sha256"]
    tok = AutoTokenizer.from_pretrained(MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B"))
    enc = tok(text, return_offsets_mapping=True)
    ids = np.asarray(enc.input_ids, dtype=np.int64)
    starts = np.asarray([o[0] for o in enc.offset_mapping])
    assert len(ids) == prot["total_corpus_tokens"]
    heads, titles, pos, old_heads = [], [], 0, []
    lines = ds["text"]
    for i, line in enumerate(lines):
        if re.fullmatch(r" = [^=].*[^=] = \n", line):
            old_heads.append(pos)
            if i == 0 or lines[i - 1] == "":          # real article titles are preceded by an empty line
                heads.append(pos); titles.append(line.strip(" =\n"))
        pos += len(line) + 2
    art = np.maximum(np.searchsorted(np.asarray(heads), starts, side="right") - 1, 0)
    art_old = np.maximum(np.searchsorted(np.asarray(old_heads), starts, side="right") - 1, 0)
    return dict(ids=ids, art=art, art_old=art_old, n_old=len(old_heads), titles=titles, heads=heads, prot=prot,
                text_sha=prot["dataset"]["text_sha256"])


def stage_cohorts(a):
    if os.path.exists(p("protocol.json")):
        log("[cohorts] protocol frozen; cohorts not rebuilt")
        return
    import torch
    st = stream()
    ids, art, prot = st["ids"], st["art"], st["prot"]
    n_art = len(st["titles"])
    n_full = len(ids) // SEQ
    hist_idx = prot["window_indices"]
    win = torch.load(os.path.join(ROOT, "runs_partial_ppl", "windows.pt"))
    # ---- historical manifest
    hrows = []
    for i, w in enumerate(hist_idx):
        t0, t1 = w * SEQ, (w + 1) * SEQ
        assert np.array_equal(ids[t0:t1], win[i].numpy())
        arts = art[t0:t1]
        u, cnt = np.unique(arts[1:], return_counts=True)
        hrows.append(dict(window=i, stream_window_index=w, token_range=[t0, t1], token_sha256=sha_ids(ids[t0:t1]),
                          scored_targets=SEQ - 1, articles=[int(x) for x in np.unique(arts)],
                          major_article=int(u[np.argmax(cnt)]), crosses_article_boundary=bool(len(np.unique(arts)) > 1)))
    hist_major = sorted({r["major_article"] for r in hrows})
    # the historical cluster partition under the old (62-header) rule must equal the corrected one
    old_major = []
    for w in hist_idx:
        u, cnt = np.unique(st["art_old"][w * SEQ + 1:(w + 1) * SEQ], return_counts=True)
        old_major.append(int(u[np.argmax(cnt)]))
    new_major = [r["major_article"] for r in hrows]
    pair = lambda v: {(i, j) for i in range(len(v)) for j in range(len(v)) if v[i] == v[j]}
    hist_clusters_same = pair(old_major) == pair(new_major)
    hist_any = sorted({x for r in hrows for x in r["articles"]})
    art_ranges = {}
    for k in range(n_art):
        w = np.nonzero(art == k)[0]
        if len(w):
            art_ranges[k] = [int(w[0]), int(w[-1]) + 1]
    early = sorted({int(x) for x in np.unique(art[:EARLY_DEV_TOKENS])})
    hman = dict(n_windows=len(hrows), n_full_stream_windows=n_full, total_tokens=int(len(ids)),
                n_articles_in_test=n_art, articles_majority_rule=hist_major, n_articles_majority_rule=len(hist_major),
                articles_any_token=hist_any, n_articles_any_token=len(hist_any),
                n_boundary_crossing_windows=sum(r["crosses_article_boundary"] for r in hrows),
                verified_49_articles=len(hist_major) == 49, n_articles_corrected=n_art,
                article_rule=("article = WikiText-2 test top-level ' = Title = ' line PRECEDED BY AN EMPTY LINE; token belongs "
                              "to the last header at or before its first character; tokens before the first header -> "
                              "article 0"),
                article_rule_correction=dict(
                    old_rule_headers=st["n_old"], corrected_headers=n_art,
                    note=("the regex used by runs_evaluation_uncertainty also matched two formula lines inside 'Constant k "
                          "filter' (' = 1 rad / s and a nominal impedance k = ', ' = 1 henry and capacitance C = '), "
                          "giving 62; the test split has 60 articles"),
                    historical_window_clusters_identical=hist_clusters_same),
                article_titles={k: st["titles"][k] for k in range(n_art)}, article_token_ranges=art_ranges, windows=hrows)
    jdump(hman, p("historical_article_manifest.json"))
    # ---- unused grid windows inventory
    used = set(hist_idx)
    inv = {"A": [], "B": [], "C": []}
    for w in range(n_full):
        if w in used:
            continue
        t0, t1 = w * SEQ, (w + 1) * SEQ
        u = np.unique(art[t0:t1])
        r = dict(stream_window_index=w, token_range=[t0, t1], articles=[int(x) for x in u],
                 token_sha256=sha_ids(ids[t0:t1]), scored_targets=SEQ - 1,
                 early_dev_eval_tokens=bool(t0 < EARLY_DEV_TOKENS))
        if len(u) > 1:
            r["reason"] = "crosses article boundary"; inv["C"].append(r)
        elif int(u[0]) in hist_any:
            r["article"] = int(u[0]); inv["B"].append(r)
        else:
            r["article"] = int(u[0]); inv["A"].append(r)
    tail = len(ids) - n_full * SEQ
    never = [k for k in range(n_art) if k not in hist_any]
    # ---- evaluated cohorts: article-aligned windows inside unused article segments
    hist_mask = np.zeros(len(ids), bool)
    for w in hist_idx:
        hist_mask[w * SEQ:(w + 1) * SEQ] = True
    def segments(k):
        t0, t1 = art_ranges[k]
        free = ~hist_mask[t0:t1]
        segs, s = [], None
        for j, f in enumerate(free):
            if f and s is None:
                s = j
            if (not f or j == len(free) - 1) and s is not None:
                e = j + 1 if f else j
                segs.append((t0 + s, t0 + e)); s = None
        return segs
    def windows_for(arts, cohort):
        rows, dropped = [], 0
        for k in arts:
            for s0, s1 in segments(k):
                n = (s1 - s0) // SEQ
                dropped += (s1 - s0) - n * SEQ
                for j in range(n):
                    t0 = s0 + j * SEQ
                    rows.append(dict(window=len(rows), article=int(k), article_title=st["titles"][k],
                                     token_range=[int(t0), int(t0 + SEQ)], segment=[int(s0), int(s1)],
                                     token_sha256=sha_ids(ids[t0:t0 + SEQ]), scored_targets=SEQ - 1,
                                     on_historical_grid=bool(t0 % SEQ == 0), early_dev_eval_tokens=bool(t0 < EARLY_DEV_TOKENS)))
                    assert len(np.unique(art[t0:t0 + SEQ])) == 1 and not hist_mask[t0:t0 + SEQ].any()
        return rows, dropped
    A_rows, A_drop = windows_for(never, "A")
    B_rows, B_drop = windows_for(hist_any, "B")
    fresh_arts = sorted({r["article"] for r in A_rows})
    jdump(dict(cohort="A: fresh articles (article-disjoint)", rule=(
        "articles with NO token in any historical evaluation window (any-token rule, stricter than the majority rule); "
        "every complete 512-token window of the historical flattened-stream tokenization, non-overlapping, starting at "
        "the article's first token; article tails < 512 tokens are not evaluated (reported)"),
        unevaluated_tail_tokens=int(A_drop),
        n_windows=len(A_rows), n_articles=len(fresh_arts), articles=fresh_arts,
        article_titles={k: st["titles"][k] for k in fresh_arts},
        windows_per_article={k: sum(r["article"] == k for r in A_rows) for k in fresh_arts},
        scored_targets=len(A_rows) * (SEQ - 1), historical_scored_targets=65408,
        never_represented_articles=never, never_represented_without_eligible_window=[k for k in never if k not in fresh_arts],
        n_never_represented_articles=len(never), windows=A_rows), p("fresh_article_manifest.json"))
    bart = sorted({r["article"] for r in B_rows})
    jdump(dict(cohort="B: fresh windows from previously represented articles (window-disjoint, article-overlapping)",
               rule=("articles with >= 1 token in a historical window; every maximal run of article tokens not covered by "
                     "a historical window is cut into non-overlapping complete 512-token windows from the run start; "
                     "remainders < 512 not evaluated"), unevaluated_remainder_tokens=int(B_drop),
               n_windows=len(B_rows), n_articles=len(bart), articles=bart,
               windows_per_article={k: sum(r["article"] == k for r in B_rows) for k in bart},
               scored_targets=len(B_rows) * (SEQ - 1), windows=B_rows), p("fresh_window_manifest.json"))
    # ---- overlap / usage audit
    hist_tok = set()
    for w in hist_idx:
        hist_tok.update(range(w * SEQ, (w + 1) * SEQ))
    ov = dict(
        grid=dict(n_full_windows=n_full, historical=len(hist_idx), unused=n_full - len(hist_idx),
                  category_A=len(inv["A"]), category_B=len(inv["B"]), category_C=len(inv["C"]),
                  category_C_reasons={"crosses article boundary": len(inv["C"])}, stream_tail_tokens_unused=tail),
        verified_unused_456=(n_full - len(hist_idx)) == 456,
        grid_note="grid inventory (verification of the 456 unused i*512 windows); the evaluated cohorts use article-aligned "
                  "windows (see manifests) so that short never-represented articles are not lost to the grid",
        token_overlap_A_with_historical=int(sum(1 for r in A_rows for t in range(*r["token_range"]) if t in hist_tok)),
        token_overlap_B_with_historical=int(sum(1 for r in B_rows for t in range(*r["token_range"]) if t in hist_tok)),
        token_overlap_A_with_B=int(len({t for r in A_rows for t in range(*r["token_range"])} &
                                       {t for r in B_rows for t in range(*r["token_range"])})),
        article_overlap_A_with_historical_any=sorted(set(fresh_arts) & set(hist_any)),
        article_overlap_A_with_historical_major=sorted(set(fresh_arts) & set(hist_major)),
        exact_duplicate_windows_A_or_B_vs_historical=int(sum(r["token_sha256"] in {h["token_sha256"] for h in hrows}
                                                              for r in A_rows + B_rows)),
        prior_use=dict(
            early_dev_eval=dict(source="e3_allocation.py --ppl (moe_calib.data_utils.wikitext2_ppl, seqlen 2048, "
                                       "default --ppl-windows 40): first 81,920 test-stream tokens; runs_smoke result",
                                tokens=[0, EARLY_DEV_TOKENS], articles=early,
                                cohort_A_articles_flagged=sorted(set(fresh_arts) & set(early)),
                                cohort_A_windows_flagged=sum(r["early_dev_eval_tokens"] for r in A_rows),
                                cohort_B_windows_flagged=sum(r["early_dev_eval_tokens"] for r in B_rows)),
            calibration="WikiText-2 TEST never used for calibration (C4-train; domain ablation used WT2 train)",
            other_diagnostics=("all later WT2 evaluations / activation caches used the 128 historical windows (or a "
                               "32-window subset); no other test-stream text was used")),
        note="category A/B windows reuse the historical i*512 grid and tokenization slice (historical semantics); "
             "boundary-crossing unused windows (category C) are excluded so no cross-article transition is scored")
    jdump(ov, p("article_overlap_audit.json"))
    # window tensors (tokens only)
    import torch as T
    for nm, rows in (("fresh_article", A_rows), ("fresh_window", B_rows)):
        t = T.from_numpy(np.stack([ids[r["token_range"][0]:r["token_range"][1]] for r in rows]))
        T.save(t, p(f"windows_{nm}.pt"))
    log(f"[cohorts] grid {n_full}, unused {n_full - len(hist_idx)}: grid A {len(inv['A'])} B {len(inv['B'])} C {len(inv['C'])}; "
        f"cohort A {len(A_rows)} windows / {len(fresh_arts)} articles (never represented {len(never)}), cohort B "
        f"{len(B_rows)} / {len({r['article'] for r in B_rows})}; hist major {len(hist_major)} any {len(hist_any)} of {n_art}; "
        f"clusters same {hist_clusters_same}; early-dev A {sorted(set(fresh_arts) & set(early))}")


def load_cohort(nm):
    import torch
    t = torch.load(p(f"windows_{nm}.pt"))
    man = jload(p("fresh_article_manifest.json" if nm == "fresh_article" else "fresh_window_manifest.json"))
    assert sha_ids(t.numpy()) == jload(p("protocol.json"))["cohorts"][nm]["windows_tensor_sha256"]
    return t, man


# =====================================================================================================================
# stage protocol
# =====================================================================================================================
def code_hash():
    return {f: sha_file(os.path.join(ROOT, f)) for f in
            ["diagnostics/wt2_fresh_generalization.py", "diagnostics/evaluation_uncertainty.py",
             "diagnostics/heldout_bitwidth_validation.py", "diagnostics/eval_partial_ppl.py", "moe_calib/model_utils.py",
             "moe_calib/partial_model_quant.py", "moe_calib/dense_gptq.py", "diagnostics/dense_grouping_scope.py"]}


def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        log("[protocol] already frozen")
        return
    import torch
    A = jload(p("fresh_article_manifest.json")); B = jload(p("fresh_window_manifest.json"))
    H = jload(p("historical_article_manifest.json"))
    assert not os.path.exists(p("results")), "results exist before protocol"
    coh = {}
    for nm, man in (("fresh_article", A), ("fresh_window", B)):
        t = torch.load(p(f"windows_{nm}.pt"))
        coh[nm] = dict(n_windows=man["n_windows"], n_articles=man["n_articles"], articles=man["articles"],
                       windows_tensor_sha256=sha_ids(t.numpy()), manifest_sha256=sha_file(p(f"{'fresh_article' if nm == 'fresh_article' else 'fresh_window'}_manifest.json")),
                       window_token_starts=[r["token_range"][0] for r in man["windows"]],
                       window_article=[r["article"] for r in man["windows"]], scored_targets=man["scored_targets"])
    prot = dict(
        title="Final WikiText-2 fresh evaluation: article-level generalization of GPTQ execution effects",
        created=time.strftime("%F %T"),
        historical=dict(windows="runs_partial_ppl/windows.pt", windows_sha256=jload(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json"))["windows_sha256"],
                        n_articles_majority=H["n_articles_majority_rule"], n_articles_any=H["n_articles_any_token"],
                        manifest_sha256=sha_file(p("historical_article_manifest.json"))),
        cohorts=coh,
        cohort_definitions=dict(fresh_article=A["rule"], fresh_window=B["rule"],
                                optional_validation="omitted: WT2 validation was used by the calibration-domain ablation"),
        models=dict(moe=dict(configs=MOE_CONFIGS + MOE_REFS, weights="runs_heldout_bitwidth_validation/logs/moe_weight_manifest.json",
                             gptq="4-bit g128 percdamp 1.0 act-order, routed experts L1/8/16/23, natural routing"),
                    dense4=dict(configs=DENSE_CONFIGS, weights="~/moe_cache_dense_grouping_scope/weights"),
                    dense3=dict(configs=DENSE_CONFIGS, weights="~/moe_cache_heldout_bitwidth/dense3"),
                    optional_reference_matrix="not run (optional)"),
        evaluation=dict(evaluator="eval_partial_ppl.layerwise_nll, batch 8, 512-token windows, first token context only, "
                                  "FP32 CE / FP64 accumulation; windows in stream order",
                        parity=dict(moe=MOE_PARITY, dense4=DENSE4_PARITY, dense3=DENSE3_PARITY, bf16="both models"),
                        repeat=dict(moe=MOE_REPEAT, dense4=DENSE_REPEAT, dense3=DENSE_REPEAT, bf16="both models",
                                    cohort="fresh_article"),
                        exact_null="dense4/dense3 G20261115 == ORIGINAL per-window on both cohorts"),
        metrics=dict(primary="mean token NLL; MoE Delta = l_FC - l_FR (positive = FR better), 15 pairs, within-seed "
                             "centred (df 12); dense: 5 groupings centred (df 4), contrasts vs ORIGINAL (4 non-null)",
                     secondary="PPL presentation only"),
        statistics=dict(
            reused_code="diagnostics/evaluation_uncertainty.py (Units, nested, split_masks, reliability_stats); same B: "
                        "cov 2000, nested 2000 x 400, split 1000, null 500 x 40, cross bootstrap 2000",
            resampling=dict(original_WT2="article (majority rule, 49)", fresh_article="article", fresh_window="article",
                            C4HO="document (historical reference)"),
            variance="V_corr = V_obs - tr(P Sigma)/df; pivot 95% interval; common-effect null p; negative values raw",
            split_half="balanced random article split, 1000 partitions; common-effect null (500 x 40)",
            cross_cohort=("Pearson/Spearman of centred effects, sign agreement of contrasts, pairwise order agreement, "
                          "ORIGINAL rank (dense); 95% CI from independent article bootstraps of the two cohorts; "
                          "comparisons: orig vs fresh_article, orig vs fresh_window, fresh_article vs fresh_window, orig vs C4HO"),
            window_ci="per pair article-cluster 95% percentile CI; BH and max-|t| exploratory",
            sensitivity=dict(
                exclude_early_dev_articles=("fresh_article cohort without articles touched by the early e3_allocation "
                                            "--ppl smoke evaluation (first 81,920 stream tokens): articles " +
                                            str(jload(p("article_overlap_audit.json"))["prior_use"]["early_dev_eval"]["cohort_A_articles_flagged"]) +
                                            "; exploratory"),
                window_bootstrap="fresh cohorts with naive window resampling (comparison only; never primary)"),
            synthetic_validation=("cluster-structured simulation with the exact fresh_article article sizes: (a) no-effect "
                                  "size of the null test and split-half null, (b) power to resolve V_corr equal to the "
                                  "original-WT2 estimate and to detect r(original, fresh) under identical true effects"),
            fr_fc_execution_level=("grand mean of the 15 paired effects; (i) eval-only article CI; (ii) seed-level t "
                                   "interval from the 3 seed means (2 df); (iii) combined SE sqrt(SE_eval^2 + "
                                   "max(V_corr,0)/15)")),
        decision_rules=dict(
            execution_effect=dict(
                RESOLVED="fresh-cohort V_corr pivot 95% lower bound > 0 AND common-effect null p < 0.05",
                BELOW_EVALUATION_RESOLUTION="null p >= 0.05 AND split-half not reliable (median r < 0.5 or null p >= 0.05)",
                INCONCLUSIVE="otherwise"),
            article_generalization=dict(
                SUPPORTED="r(original, fresh_article) > 0 with 95% CI lower bound > 0 AND fresh_article execution "
                          "effect not BELOW-EVALUATION-RESOLUTION",
                NOT_SUPPORTED="fresh_article null p < 0.05 AND r CI upper bound < sqrt(lam_orig * lam_fresh) (lam = "
                              "V_corr/V_obs, both > 0)",
                INCONCLUSIVE="otherwise"),
            primary_cohort="fresh_article; fresh_window is secondary and reported with the same rules"),
        seeds=dict(master=SEED, stats="evaluation_uncertainty.rng_for with key prefix 100+"),
        code_sha256=code_hash())
    jdump(prot, p("protocol.json"))
    h = sha_file(p("protocol.json"))
    shutil.copy(p("protocol.json"), p("logs", "protocol_readonly_copy.json"))
    with open(p("logs", "protocol_sha256.txt"), "w") as fh:
        fh.write(h + "\n")
    os.chmod(p("protocol.json"), 0o444); os.chmod(p("logs", "protocol_readonly_copy.json"), 0o444)
    log(f"[protocol] frozen sha256 {h}")


def check_protocol(code=True):
    pr = jload(p("protocol.json"))
    assert sha_file(p("protocol.json")) == open(p("logs", "protocol_sha256.txt")).read().strip(), "protocol modified"
    if code:
        now = code_hash()
        ch = [f for f in now if f != "diagnostics/wt2_fresh_generalization.py" and now[f] != pr["code_sha256"][f]]
        assert not ch, f"evaluation code changed since freeze: {ch}"
        if now["diagnostics/wt2_fresh_generalization.py"] != pr["code_sha256"]["diagnostics/wt2_fresh_generalization.py"]:
            log("[note] driver file changed since freeze (recorded in logs/code_state.json)")
            d = jload(p("logs", "code_state.json")) or {}
            d[time.strftime("%F %T")] = now
            jdump(d, p("logs", "code_state.json"))
    return pr


def software():
    import torch, transformers
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                numpy=np.__version__)


# =====================================================================================================================
# GPU stages
# =====================================================================================================================
def stage_dense(a):
    check_protocol()
    import torch
    import heldout_bitwidth_validation as H
    import eval_partial_ppl as EP
    wt, _ = EP.load_windows()
    cw = {nm: load_cohort(nm)[0] for nm in COHORTS}
    R = lambda *x: p("results", "dense", *x)
    os.makedirs(R(), exist_ok=True)
    rig = H.DenseRig()
    cost_add("dense", "model_load_s", rig.load_s)
    hist = lambda f: jload(os.path.join(HBV_DIR, "results", f))
    if not (jload(R("BF16.json")) or {}).get("valid"):
        t0 = time.time()
        n0, k0, s0 = H.dense_eval(rig.model, wt)
        rec = dict(config="BF16", original_parity=H.rec_of(n0, k0, s0))
        ck = dict(original_parity_bitwise=bool(np.array_equal(n0, np.array(hist("dense/BF16.json")["wikitext"]["per_window_nll"]))))
        for nm in COHORTS:
            n1, k1, s1 = H.dense_eval(rig.model, cw[nm])
            n2, _, _ = H.dense_eval(rig.model, cw[nm])
            rec[nm] = H.rec_of(n1, k1, s1)
            ck[f"{nm}_repeat_bitwise"] = bool(np.array_equal(n1, n2))
        rec.update(checks=ck, valid=all(ck.values()), seconds=time.time() - t0)
        jdump(rec, R("BF16.json"))
        log(f"[dense BF16] {', '.join(f'{nm} {rec[nm][chr(112)+chr(112)+chr(108)]:.4f}' for nm in COHORTS)} checks {ck}")
        if not rec["valid"]:
            raise SystemExit("dense BF16 parity/repeat failed -> BLOCKED")
    for bits in (4, 3):
        for g in DENSE_CONFIGS:
            f = R(f"dense{bits}_{g}.json")
            if (jload(f) or {}).get("valid"):
                continue
            t0 = time.time()
            hrec = hist(f"dense{'' if bits == 4 else '3'}/dense{bits}_{g}.json")
            if bits == 4:
                wq = H.load_dense_hist_weights(g)
                tsha = True
            else:
                wq, tsha = {}, True
                for L in H.DENSE_LAYERS:
                    d = torch.load(H.d3dir(g, f"L{L}.pt"))
                    wq[L] = d["weights"]
                    for k in H.PROJ:
                        tsha &= H.tensor_sha(wq[L][k]) == hrec["per_layer"][str(L)]["weight_sha256"][k]
                    ok = all(m["codes_exact"] and m["codes_in_range"] and m["n_levels_used"] == 8 for m in d["meta"].values())
                    tsha &= ok
                    del d
            parity = g in (DENSE4_PARITY if bits == 4 else DENSE3_PARITY)
            sets = ([("original_parity", wt)] if parity else []) + [(nm, cw[nm]) for nm in COHORTS]
            rep = ("fresh_article",) if g in DENSE_REPEAT else ()
            out, ck = rig.run(wq, sets, repeat=rep, expect_digest=hrec["checks"]["installed_digest"])
            ck["weights_match_history"] = bool(tsha)
            if parity:
                ck["original_parity_bitwise"] = bool(np.array_equal(np.array(out["original_parity"]["per_window_nll"]),
                                                                    np.array(hrec["wikitext"]["per_window_nll"])))
            if rep:
                ck["fresh_article_repeat_bitwise"] = out["fresh_article"]["repeat_per_window_bitwise"]
            rec = dict(config=g, bits=bits, installed_weights_sha256=ck["installed_digest"], **out, checks=ck,
                       valid=H.ok_checks(ck), seconds=time.time() - t0)
            jdump(rec, f)
            cost_add("dense", f"dense{bits}_{g}", rec["seconds"])
            log(f"[dense{bits} {g}] FA {out['fresh_article']['ppl']:.5f} FW {out['fresh_window']['ppl']:.5f} valid {rec['valid']} "
                f"{rec['seconds']:.0f}s")
            if not rec["valid"]:
                raise SystemExit(f"dense{bits} {g} integrity failed -> BLOCKED")
    # end checks
    bf = jload(R("BF16.json"))
    end = dict(bf16_original_bitwise=bool(np.array_equal(H.dense_eval(rig.model, wt)[0], np.array(bf["original_parity"]["per_window_nll"]))),
               bf16_fresh_article_bitwise=bool(np.array_equal(H.dense_eval(rig.model, cw["fresh_article"])[0],
                                                              np.array(bf["fresh_article"]["per_window_nll"]))),
               targets_pristine=rig.pristine(), untouched_unchanged=rig.DG.untouched_digests(rig.model, H.DENSE_LAYERS) == rig.unt0,
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=H.peak_cpu_gb(), software=software(),
               at=time.strftime("%F %T"))
    for bits in (4, 3):
        A_, B_ = jload(R(f"dense{bits}_G20261115.json")), jload(R(f"dense{bits}_ORIGINAL.json"))
        end[f"null_dense{bits}"] = {nm: A_[nm]["per_window_nll"] == B_[nm]["per_window_nll"] for nm in COHORTS}
        end[f"null_dense{bits}"]["installed_digest_equal"] = A_["installed_weights_sha256"] == B_["installed_weights_sha256"]
    jdump(end, p("logs", "dense_end_checks.json"))
    log(f"[dense end] {end}")


def stage_moe(a):
    check_protocol()
    import torch
    import heldout_bitwidth_validation as H
    import eval_partial_ppl as EP
    from moe_calib import model_utils as MU, partial_model_quant as PQ
    wt, _ = EP.load_windows()
    cw = {nm: load_cohort(nm)[0] for nm in COHORTS}
    wm = jload(os.path.join(HBV_DIR, "logs", "moe_weight_manifest.json"))
    R = lambda *x: p("results", "moe", *x)
    os.makedirs(R(), exist_ok=True)
    t0 = time.time()
    model, _ = MU.load_model()
    cost_add("moe", f"model_load_s@{time.strftime('%T')}", time.time() - t0)
    snap = PQ.snapshot_experts(model, H.LAYERS_Q)
    unt0 = PQ.untouched_digests(model, H.LAYERS_Q)
    light0 = H.moe_light_digest(model)
    ev = lambda w: H.dense_eval(model, w)
    hist = lambda n: jload(os.path.join(HBV_DIR, "results", "moe", f"{n}.json"))
    if not (jload(R("BF16.json")) or {}).get("valid"):
        t0 = time.time()
        n0, k0, s0 = ev(wt)
        rec = dict(config="BF16", original_parity=H.rec_of(n0, k0, s0))
        ck = dict(original_parity_bitwise=bool(np.array_equal(n0, np.array(hist("BF16")["wikitext"]["per_window_nll"]))))
        for nm in COHORTS:
            n1, k1, s1 = ev(cw[nm])
            rec[nm] = H.rec_of(n1, k1, s1)
            if nm == "fresh_article":
                ck["fresh_article_repeat_bitwise"] = bool(np.array_equal(n1, ev(cw[nm])[0]))
        rec.update(checks=ck, valid=all(ck.values()), seconds=time.time() - t0)
        jdump(rec, R("BF16.json"))
        log(f"[moe BF16] FA {rec['fresh_article']['ppl']:.4f} FW {rec['fresh_window']['ppl']:.4f} checks {ck}")
        if not rec["valid"]:
            raise SystemExit("MoE BF16 parity failed -> BLOCKED")
    order = MOE_PARITY + [n for n in MOE_CONFIGS + MOE_REFS if n not in MOE_PARITY]
    for n in order:
        if (jload(R(f"{n}.json")) or {}).get("valid"):
            continue
        t0 = time.time()
        ent = wm["configs"][n]
        ick = H.moe_install(model, ent, snap)
        out = {}
        if n in MOE_PARITY:
            nw, kw, sw = ev(wt)
            out["original_parity"] = H.rec_of(nw, kw, sw)
        for nm in COHORTS:
            n1, k1, s1 = ev(cw[nm])
            out[nm] = H.rec_of(n1, k1, s1)
            if nm == "fresh_article" and n in MOE_REPEAT:
                out[nm]["repeat_per_window_bitwise"] = bool(np.array_equal(n1, ev(cw[nm])[0]))
        ck = dict(**{k: v for k, v in ick.items() if k != "install_s"},
                  installed_unchanged_during_eval=PQ.installed_digest(model, H.LAYERS_Q) == ent["installed_weights_sha256"],
                  light_non_target_unchanged=H.moe_light_digest(model) == light0)
        if n in MOE_PARITY:
            ck["original_parity_bitwise"] = bool(np.array_equal(nw, np.array(ent["historical_per_window_nll"], dtype=np.float64)))
        if n in MOE_REPEAT:
            ck["fresh_article_repeat_bitwise"] = out["fresh_article"]["repeat_per_window_bitwise"]
        PQ.restore_experts(model, snap)
        eq, tot = PQ.experts_match(model, snap)
        ck["restored_pristine"] = eq == tot
        rec = dict(config=n, strategy=ent["strategy"], allocation_seed=ent.get("allocation_seed"), grouping=ent.get("grouping"),
                   installed_weights_sha256=ent["installed_weights_sha256"], **out, checks=ck, valid=H.ok_checks(ck),
                   install_s=ick["install_s"], seconds=time.time() - t0, peak_cpu_gb=H.peak_cpu_gb())
        jdump(rec, R(f"{n}.json"))
        cost_add("moe", n, dict(install_s=ick["install_s"], total_s=rec["seconds"]))
        log(f"[moe {n}] FA {out['fresh_article']['ppl']:.5f} FW {out['fresh_window']['ppl']:.5f} valid {rec['valid']} "
            f"{rec['seconds']:.0f}s")
        if not rec["valid"]:
            raise SystemExit(f"MoE {n} integrity failed -> BLOCKED")
        gc.collect()
    bf = jload(R("BF16.json"))
    eq, tot = PQ.experts_match(model, snap)
    end = dict(untouched_unchanged=PQ.untouched_digests(model, H.LAYERS_Q) == unt0, n_untouched=len(unt0),
               targets_pristine=eq == tot, light_non_target_unchanged=H.moe_light_digest(model) == light0,
               bf16_original_bitwise=bool(np.array_equal(ev(wt)[0], np.array(bf["original_parity"]["per_window_nll"]))),
               bf16_fresh_article_bitwise=bool(np.array_equal(ev(cw["fresh_article"])[0], np.array(bf["fresh_article"]["per_window_nll"]))),
               peak_cpu_gb=H.peak_cpu_gb(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, software=software(),
               at=time.strftime("%F %T"))
    jdump(end, p("logs", "moe_end_checks.json"))
    log(f"[moe end] {end}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["hist", "cohorts", "protocol", "dense", "moe", "synth", "analyze"])
    a = ap.parse_args()
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage in ("synth", "analyze"):
        import wt2_fresh_analysis as WA
        return getattr(WA, f"stage_{a.stage}")(a)
    dict(hist=stage_hist, cohorts=stage_cohorts, protocol=stage_protocol, dense=stage_dense, moe=stage_moe)[a.stage](a)


if __name__ == "__main__":
    main()
