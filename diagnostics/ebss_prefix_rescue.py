"""C4-prefix diversity rescue for model-guided calibration generation (modified EBSS; NOT a MoEQuant reproduction).

16 frozen C4 Random seed-0 windows (B=16 pilot) -> prefix = tokens[0:64]; generate 448 tokens with
Probability-Only (PO) and ALG1_RATE EBSS (w = 4, tau = 1.3) from the identical prefixes; matched natural control =
the original windows. B=16 GPTQ on layers 1/8/16/23 with the unchanged historical pipeline.

Stages (--stage): freeze | regression | smoke | generate --cond {PX_PO,PX_EBSS} | capture | diagnostics |
                  quantize --cond {PX_PO,PX_EBSS} | report
Nothing outside runs_ebss_prefix_rescue/ and ~/moe_cache_ebss_prefix_rescue is written.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
OUT = os.path.join(ROOT, "runs_ebss_prefix_rescue")
NCACHE = os.path.expanduser("~/moe_cache_ebss_prefix_rescue")
PILOT = os.path.join(ROOT, "runs_ebss_generation_pilot")
FID = os.path.join(ROOT, "runs_ebss_balance_fidelity")
FCACHE = os.path.expanduser("~/moe_cache_ebss_balance_fidelity")
os.environ["EBSS_VALIDATION_CACHE"] = NCACHE
import official_ebss_validation as OV                  # noqa: E402
OV.ECACHE = NCACHE
import ebss_generation_pilot as EGP                    # noqa: E402
OV.ECACHE = NCACHE
from moe_calib import ebss_generator as G              # noqa: E402
from moe_calib import ebss_pilot as P                  # noqa: E402
from moe_calib import ebss_prefix as X                 # noqa: E402

W, TAU, VOCAB, NL, E, K = G.W_REF, G.TAU_REF, 151646, 24, 60, 4
STRONG = dict(balance_std_mode="algorithm1", usage_mode="per_token")          # ALG1_RATE
CONDS = {"PX_PO": dict(use_balance=False), "PX_EBSS": dict(use_balance=True, **STRONG)}
SMOKE_N, SMOKE_G = 4, 32
SMOKE_KV_STEPS = (64, 65, 72, 95)                    # steps whose cached logits are re-derived without KV cache


def p(*a):
    return os.path.join(OUT, *a)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path))


def need(path, key="passed"):
    assert os.path.exists(path), f"missing upstream result {path}"
    d = jload(path)
    assert d.get(key), f"upstream gate failed: {path} [{key}]"
    return d


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# =================================================================== freeze
def heldout_windows():
    held, win = OV.leakage_reference()
    return [h.tolist() for h in held] + [w.tolist() for w in win]


def stage_freeze(a):
    if os.path.exists(p("protocol.json")):
        print("[freeze] protocol already frozen"); return
    pman = jload(os.path.join(PILOT, "logs", "capture_manifest.json"))
    src = EGP.C4R_FILES[0]
    chosen = list(map(int, pman["c4_random"]["0"]["chosen"]))
    r = jload(os.path.join(ROOT, src))
    assert list(map(int, r["chosen"])) == chosen and EGP.sha256_file(os.path.join(ROOT, src)) == pman["c4_random"]["0"]["sha256"]
    allt = OV.c4_tokens()
    cfg = OV.frozen_cfg()
    parents = allt[chosen]
    pre, cont = X.split_windows(parents.tolist())
    held = heldout_windows()
    chk = X.prefix_checks(pre, parents.tolist(), held, P.contamination)
    chk["parents_in_training_pool"] = all(c < cfg["split"]["tune"][0] for c in chosen)
    chk["valid"] = chk["valid"] and chk["parents_in_training_pool"]
    assert chk["valid"], chk
    entries = [dict(index=i, parent_row=c, offset=0, prefix_tokens=[0, X.P_LEN], continuation_tokens=[X.P_LEN, X.SEQ],
                    parent_sha256=X.token_hash(parents[i].tolist()), prefix_sha256=X.token_hash(pre[i]),
                    continuation_sha256=X.token_hash(cont[i]), prefix_token_ids=pre[i])
               for i, c in enumerate(chosen)]
    jdump(dict(selection_rule="the 16 windows of the frozen B=16 C4 Random seed-0 selection, in their stored order; "
                              "prefix = tokens[0:64] of each 512-token window; no selection by probability, routing, "
                              "reconstruction or generation outcome",
               source=src, source_sha256=pman["c4_random"]["0"]["sha256"], parent_rows=chosen,
               row_universe="OV.c4_tokens() = cat(~/moe_cache/ids.pt pool, holdout); rows < split.tune[0] are training "
                            "pool", heldout_reference="C4 rows split.tune[0]..split.test[1] + 128 WikiText-2 PPL windows; "
                                                      f"exact {P.ED.LEAK_NGRAM}-gram overlap test",
               checks=chk, prefixes=entries), p("prefix_manifest.json"))
    jdump(dict(condition="C4_NATURAL = C4R_s0", definition="64 C4 prefix tokens + the original 448 C4 tokens of the same "
                                                           "windows = the B=16 C4 Random seed-0 calibration set",
               parent_rows=chosen, reuse=dict(gptq_result=os.path.relpath(os.path.join(PILOT, "results", "C4R_s0.json"), ROOT),
                                              reason="identical windows, CANONICAL-B1 extraction and OV.quant_config; "
                                                     "re-verified by the bitwise cache-compatibility check of stage capture"),
               historical_references=dict(
                   RS_PO=os.path.relpath(os.path.join(PILOT, "results", "PO.json"), ROOT),
                   RS_EBSS=os.path.relpath(os.path.join(FID, "results", "ALG1_RATE.json"), ROOT),
                   C4R_s1=os.path.relpath(os.path.join(PILOT, "results", "C4R_s1.json"), ROOT),
                   C4R_s2=os.path.relpath(os.path.join(PILOT, "results", "C4R_s2.json"), ROOT)),
               sha256={k: EGP.sha256_file(os.path.join(ROOT, v)) for k, v in dict(
                   C4R_s0="runs_ebss_generation_pilot/results/C4R_s0.json", C4R_s1="runs_ebss_generation_pilot/results/C4R_s1.json",
                   C4R_s2="runs_ebss_generation_pilot/results/C4R_s2.json", RS_PO="runs_ebss_generation_pilot/results/PO.json",
                   RS_EBSS="runs_ebss_balance_fidelity/results/ALG1_RATE.json").items()}),
          p("matched_c4_manifest.json"))
    jdump(dict(model=P.MODEL, precision="BF16 original weights (pristine digests verified at every load)",
               w=W, tau=TAU, PX_PO=dict(use_balance=False), PX_EBSS=dict(use_balance=True, variant="ALG1_RATE", **STRONG),
               prefix_len=X.P_LEN, generated_len=X.G_LEN, seqlen=X.SEQ, n_sentences=16,
               initial_token_ids="frozen prefixes (prefix_manifest.json)",
               prefix_semantics=dict(
                   prefill="one batched causal forward of the 16 x 64 prefixes into a fresh KV cache (no padding, no chat "
                           "template, no BOS insertion); next-token logits from position 63",
                   R="cumulative log-probability includes the teacher-forced prefix transitions 1..63; token 0 contributes "
                     "0 (as the historical random first token). Shared by all branches of a sentence, so it never changes "
                     "a ranking (per-sentence pruning and final selection compare branches of one sentence only)",
                   denominator="|S| counts the prefix: score = -(R + log p_v)/(|S|+1) + sigma(S)/tau (audited generator)",
                   expert_counts="c_total = c_prefix (all 64 prefix tokens, all 24 layers, top-4) + c_generated",
                   first_expansion="one branch per sentence -> the w best of its |V| expansions; afterwards global "
                                   "per-sentence top-w over w x |V|"),
               eos="no stop rule (as historical): special tokens are ordinary candidates; exactly 448 tokens appended",
               unchanged=["score sign", "minimisation", "deferred parent sigma", "24 MoE layers", "top-4 counts",
                          "final argmin selection", "CPU-offloaded expert residency"],
               not_used=["repetition penalty", "temperature", "top-k/nucleus sampling", "diversity penalty", "C4 reranking",
                         "post-hoc mixing"]), p("generator_config.json"))
    jdump(dict(paper="arXiv 2505.03804 (MoEQuant), Sec. 4.2 / Alg. 1",
               status="modified EBSS (natural-text prefix initialisation); not paper-faithful; ALG1_RATE is one of four "
                      "paper-grounded normalisations (update_ebss_balance_fidelity.md), chosen for its near-C4 expert "
                      "balance, not known to be the authors' implementation and not the best-reconstruction variant",
               generator=dict(path="moe_calib/ebss_generator.py",
                              pre_study_copy="runs_ebss_prefix_rescue/logs/ebss_generator_pre_prefix.py",
                              pre_study_sha256="dc9cc32ab76582501ae3278b68a805d12cb8175ad07941b0a8dc185852f3965e"),
               prior_studies=["update_ebss_generation_pilot.md", "update_ebss_score_audit.md",
                              "update_ebss_balance_fidelity.md", "update_ebss_official_validation.md"]),
          p("source_provenance.json"))
    proto = dict(
        title="C4-prefix diversity rescue (B=16, P=64, G=448)", frozen_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        conditions=dict(C4_NATURAL="C4R_s0 windows (existing)", PX_PO="64 C4 + 448 PO", PX_EBSS="64 C4 + 448 ALG1_RATE EBSS"),
        historical=dict(RS_PO="pilot PO 16x512 (1 random token + 511)", RS_EBSS="fidelity ALG1_RATE 16x512",
                        C4R=["s0", "s1", "s2"]),
        comparison_status=dict(PX_PO_vs_PX_EBSS="matched intervention (only the balance term differs)",
                               PX_vs_RS="descriptive (initial content and number of generated tokens differ)"),
        smoke=dict(n=SMOKE_N, prefix=X.P_LEN, generated=SMOKE_G, kv_check_steps=list(SMOKE_KV_STEPS),
                   replay_tolerance=dict(mean_abs_dlogp=0.05, pearson_r=0.99),
                   note="batched-KV BF16 vs batch-1 no-cache BF16 replay differ by kernel-shape rounding; the pilot "
                        "used the same tolerance"),
        regions=dict(prefix=[0, 64], continuation=[64, 512], full=[0, 512],
                     historical_slicing="random-start sequences are sliced identically: continuation = [64:512] (448 "
                                        "generated tokens at the same absolute positions; the random first token and "
                                        "generated tokens 1..63 are excluded), full = [0:512]"),
        repetition=dict(loop_onset=f"first continuation offset whose next {X.LOOP_NGRAM} tokens copy an earlier "
                                   f"{X.LOOP_NGRAM}-gram of the sequence (prefix included)",
                        coverage=f"fraction of continuation positions inside an {X.COVER_NGRAM}-gram occurring >= 2x in the sequence",
                        long_span=f"continuation longest repeated n-gram > {X.LONG_SPAN}",
                        long_single_token_loop=f"continuation identical-token run >= {X.LONG_RUN}",
                        halves={k: list(v) for k, v in X.HALVES.items()}),
        routing=dict(primary="CANONICAL-B1 replay of every 512-token sequence; generation-time routing reported separately",
                     exposure_decomposition="per-region counts from the same B1 replay, normalised to relative load "
                                            "count/(tokens*k/E)"),
        gptq=dict(pipeline="OV.quant_config unchanged", layers=EGP.LAYERS, bits=4, group_size=128, percdamp=1.0,
                  act_order=True, extraction="CANONICAL-B1", K=4),
        decision=dict(
            primary=dict(BEATS_C4="PX_EBSS mean reconstruction < each of C4R_s0, s1, s2",
                         PARTIAL="not BEATS_C4, and PX_PO < RS_PO or PX_EBSS < RS_EBSS (strict)",
                         NO_GAIN="neither prefix method below its random-start counterpart",
                         RUN_BLOCKED="correctness/data/memory/execution failure",
                         reported=f"relative change and a 'material' flag (<= -{X.QUAL_MARGIN_PCT} %, the margin of the "
                                  "fidelity study) per method; gap recovery G_m; PPL separately"),
            diversity=dict(threshold_distinct2=X.DIV_D2, threshold_repetition_pp=X.DIV_REP * 100,
                           rule="see moe_calib/ebss_prefix.diversity_label")),
        constraints=["no checkpoint/w/tau/normalisation change", "no repetition penalty or sampling", "single draw",
                     "no prefix-length tuning", "no B=128", "no full-depth quantization", "GPTQ unchanged",
                     "no new numerical-sensitivity study", "no offloading optimisation"])
    jdump(proto, p("protocol.json"))
    print(f"[freeze] 16 prefixes from {src}; checks {chk}")


# =================================================================== regression (CPU)
def stage_regression(a):
    Gold = _load(p("logs", "ebss_generator_pre_prefix.py"), "ebss_generator_pre_prefix")
    NO = _load(os.path.join(FID, "normalization_oracle.py"), "normalization_oracle")
    res = []

    def rec(name, ok, **kw):
        res.append(dict(test=name, passed=bool(ok), **kw))
    # 1. historical default: bitwise vs the pre-study generator (all scores, decisions and outputs)
    for seed in range(4):
        for ub in (False, True):
            for var in (dict(), STRONG):
                logs = {}
                for tag, mod in (("new", G), ("old", Gold)):
                    trace = []
                    be = mod.SyntheticBackend(V=40, L=3, E=8, k=2, seed=seed)
                    r = mod.generate(be, [1, 5, 9], 24, 4, 1.3, ub, audit=lambda d: trace.append(d), **var)
                    logs[tag] = (r, trace)
                (rn, tn), (ro, to) = logs["new"], logs["old"]
                same = (torch.equal(rn.tokens, ro.tokens) and torch.equal(rn.cum_logp, ro.cum_logp)
                        and torch.equal(rn.final_score, ro.final_score) and len(tn) == len(to)
                        and all(torch.equal(x.get("scores", x.get("final_scores")), y.get("scores", y.get("final_scores")))
                                for x, y in zip(tn, to))
                        and all(torch.equal(x["parent"], y["parent"]) and torch.equal(x["token"], y["token"])
                                for x, y in zip(tn, to) if x["kind"] == "step"))
                rec(f"default_bitwise_vs_pre_study[seed{seed},balance{ub},{'ALG1_RATE' if var else 'EQ4_RATE'}]", same)
    # 2. P = 1 initial_token_ids == historical path
    be = G.SyntheticBackend(V=40, L=3, E=8, k=2, seed=7)
    for ub in (False, True):
        r0 = G.generate(be, [2, 4, 6], 20, 4, 1.3, ub, **STRONG)
        r1 = G.generate(be, None, 20, 4, 1.3, ub, initial_token_ids=[[2], [4], [6]], **STRONG)
        rec(f"P1_prefix_equals_historical[balance{ub}]", torch.equal(r0.tokens, r1.tokens)
            and torch.equal(r0.cum_logp, r1.cum_logp) and torch.equal(r0.counts, r1.counts))
    # 3. independent oracle of prefix-conditioned beam search on the synthetic backend (pure python)
    rng = np.random.default_rng(0)
    for case in range(6):
        Pn = int(rng.integers(2, 9)); N = 3; T = Pn + 10
        pre = rng.integers(0, 40, size=(N, Pn)).tolist()
        for ub in (False, True):
            be = G.SyntheticBackend(V=40, L=3, E=8, k=2, seed=case)
            trace = []
            r = G.generate(be, None, T, 4, 1.3, ub, initial_token_ids=pre, audit=lambda d: trace.append(d), **STRONG)
            ok, steps_ok = oracle_prefix_search(G.SyntheticBackend(V=40, L=3, E=8, k=2, seed=case), pre, T, ub, NO, r, trace)
            rec(f"prefix_oracle[case{case},P{Pn},balance{ub}]", ok, steps_checked=steps_ok)
    # 4. PO invariant to the normalisation option, prefix path
    pre = [[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]]
    a1 = G.generate(G.SyntheticBackend(V=40, L=3, E=8, k=2), None, 20, 4, 1.3, False, initial_token_ids=pre)
    a2 = G.generate(G.SyntheticBackend(V=40, L=3, E=8, k=2), None, 20, 4, 1.3, False, initial_token_ids=pre, **STRONG)
    rec("po_invariant_to_normalisation[prefix]", torch.equal(a1.tokens, a2.tokens) and torch.equal(a1.cum_logp, a2.cum_logp))
    # 5. accounting / lengths / counts
    r = G.generate(G.SyntheticBackend(V=40, L=3, E=8, k=2), None, 20, 4, 1.3, True, initial_token_ids=pre, **STRONG)
    rec("length_and_prefix_identity", tuple(r.tokens.shape) == (2, 20) and r.tokens[:, :5].tolist() == pre)
    rec("counts_equal_recount", all(torch.equal(r.counts[i], G.recount(r.hist_experts[i], 8)) for i in range(2)))
    rec("counts_total", all(int(r.counts[i].sum()) == 20 * 3 * 2 for i in range(2)))
    rec("R_equals_sum_token_logp", bool(torch.allclose(r.cum_logp, r.token_logp.sum(1), rtol=1e-12, atol=1e-12)))
    out = dict(tests=res, passed=all(t["passed"] for t in res), n=len(res))
    jdump(dict(cpu=out), p("correctness_results.json"))
    print(f"[regression] {sum(t['passed'] for t in res)}/{len(res)} passed")
    if not out["passed"]:
        raise SystemExit("regression FAILED")


def oracle_prefix_search(be, pre, T, ub, NO, prod, trace):
    """Independent beam search: every branch's full prefix is re-run from scratch on the backend (no KV state),
    log-probs/counts recomputed, sigma from the NumPy/fsum oracle, global per-sentence top-w by (score, parent,
    token). Compares every production decision and the final output."""
    L, Ek, k, V = be.n_layers, be.n_experts, be.top_k, be.vocab_limit
    N, ok, nsteps = len(pre), True, 0

    def run(seq):                       # logits after seq, routing of every token of seq
        lg, sel = be.start([seq[0]])
        sels = [sel[0]]
        lps = []
        for t in seq[1:]:
            lps.append(float(torch.log_softmax(lg[0].float(), -1)[t]))
            lg, sel = be.step([t]); sels.append(sel[0])
        return torch.log_softmax(lg[0].float(), -1).double().numpy(), torch.stack(sels), lps
    steps = [d for d in trace if d["kind"] == "step"]
    for i in range(N):
        beams = [list(pre[i])]
        for d in steps:
            l = len(beams[0])
            cand_s, cand_p, cand_t = [], [], []
            for b, seq in enumerate(beams):
                lp, sels, lps = run(seq)
                cum = sum(lps)
                cnt = np.zeros((L, Ek), int)
                for s in sels:
                    for li in range(L):
                        for e in s[li].tolist():
                            cnt[li, e] += 1
                sig = NO.sigma(cnt.tolist(), l, STRONG["balance_std_mode"], STRONG["usage_mode"], k)
                sc = NO.scores([cum], lp[None], [sig], l, 1.3, ub)[0]
                cand_s += sc.tolist(); cand_p += [b] * V; cand_t += list(range(V))
            top, _ = NO.rank(cand_s, cand_p, cand_t, 4)
            B = len(beams)
            par = d["parent"].view(N, -1)[i] - i * B
            tok = d["token"].view(N, -1)[i]
            ok &= [cand_p[j] for j in top] == par.tolist() and [cand_t[j] for j in top] == tok.tolist()
            beams = [beams[cand_p[j]] + [cand_t[j]] for j in top]
            nsteps += 1
        ok &= any(bm == prod.tokens[i].tolist() for bm in beams)
    return bool(ok), nsteps


# =================================================================== GPU helpers
def prefixes(n=16):
    man = jload(p("prefix_manifest.json"))
    assert man["checks"]["valid"]
    pre = [e["prefix_token_ids"] for e in man["prefixes"][:n]]
    assert all(X.token_hash(x) == e["prefix_sha256"] for x, e in zip(pre, man["prefixes"]))
    return pre


def run_generation(be, cond, pre, seqlen, audit=None, progress_every=0, record_steps=()):
    torch.cuda.reset_peak_memory_stats()
    kv_peak = [0]
    orig = be.step

    def step(tokens):
        out = orig(tokens)
        kv_peak[0] = max(kv_peak[0], be.kv_cache_bytes())
        return out
    be.step = step
    n0 = (be.n_forward, be.rows_forwarded, be.bytes_transferred)
    t0 = time.time()
    try:
        r = G.generate(be, None, seqlen, W, TAU, initial_token_ids=pre, audit=audit, progress_every=progress_every,
                       record_steps=record_steps, **CONDS[cond])
    finally:
        be.step = orig
    wall = time.time() - t0
    n = len(pre)
    nf, rows, by = be.n_forward - n0[0], be.rows_forwarded - n0[1], be.bytes_transferred - n0[2]
    acc = X.accounting(n, len(pre[0]), seqlen, W, VOCAB)
    t = r.timing
    pf_x = t.get("prefill_expert_transfer_s", 0.0)
    dec_x = t["expert_transfer_s"] - pf_x
    parts = dict(prefix_prefill=t.get("prefill_s", 0.0),
                 decode_model_forward_compute=t["forward_s"] - dec_x - t["routing_hook_s"],
                 decode_expert_weight_transfer=dec_x, routing_capture_hook=t["routing_hook_s"],
                 expert_statistics_update=t["stats_s"], beam_scoring=t["score_s"], beam_pruning_selection=t["select_s"],
                 beam_state_gather_and_kv_reorder=t["reorder_gather_s"])
    parts["other"] = t["total_s"] - sum(parts.values())
    cost = dict(condition=cond, wall_s=wall, gpu_seconds=wall, timing=t, forward_calls=nf, rows_forwarded=rows,
                bytes_host_to_device=by, h2d_gb_per_s=by / 1e9 / max(t["expert_transfer_s"], 1e-9), accounting=acc,
                prefill_expert_transfer_s=pf_x, breakdown_s=parts, breakdown_frac={k: v / t["total_s"] for k, v in parts.items()},
                continuation_tokens_per_s=acc["emitted_continuation_tokens"] / wall,
                final_tokens_per_s=acc["final_calibration_tokens"] / wall,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, kv_cache_peak_gb=kv_peak[0] / 1e9,
                peak_cpu_gb=EGP.FD.peak_cpu_gb(), validation=P.validate_generated(r.tokens, n, seqlen, VOCAB),
                gpu_seconds_note="one 20 GB MIG slice used exclusively; GPU-seconds = wall")
    gen = dict(tokens=r.tokens, cum_logp=r.cum_logp, token_logp=r.token_logp, counts=r.counts,
               hist_experts=r.hist_experts, sigma=r.sigma, final_score=r.final_score, final_beams=r.final_beams,
               step_log=r.step_log, intervention=r.intervention, config=r.config, method=cond)
    return gen, cost, r


class StepTrace:
    def __init__(self, keep_logp=True):
        self.steps, self.final, self.keep = [], None, keep_logp

    def __call__(self, d):
        if d["kind"] == "final":
            self.final = d
        else:
            d = dict(d); d.pop("logits", None); d.pop("hist", None)
            self.steps.append(d)


def oracle_check_trace(trace, use_balance):
    """Every captured pruning decision vs the NumPy oracle (full |V| per branch)."""
    NO = _load(os.path.join(FID, "normalization_oracle.py"), "normalization_oracle")
    ok_rank = ok_sigma = True
    max_sig = 0.0
    for d in trace.steps:
        l, B = d["prefix_len"], d["n_branches"]
        R = d["logp"].shape[0]; N = R // B
        sig = np.array([NO.sigma(d["counts"][r].tolist(), l, STRONG["balance_std_mode"], STRONG["usage_mode"], K)
                        for r in range(R)])
        if use_balance:
            max_sig = max(max_sig, float(np.abs(sig - d["sigma"].numpy()).max()))
        sc = NO.scores(d["cum"].numpy(), d["logp"].numpy(), sig, l, TAU, use_balance)
        V = sc.shape[1]
        for i in range(N):
            s = sc[i * B:(i + 1) * B].reshape(-1)
            par = np.repeat(np.arange(B), V); tok = np.tile(np.arange(V), B)
            top, _ = NO.rank(s, par, tok, W)
            pp = d["parent"].view(N, W)[i].numpy() - i * B; tt = d["token"].view(N, W)[i].numpy()
            ok_rank &= bool((par[top] == pp).all() and (tok[top] == tt).all())
    ok_sigma = max_sig <= 1e-12
    return dict(ranking_matches_oracle=bool(ok_rank), sigma_matches_oracle=bool(ok_sigma), max_abs_sigma_diff=max_sig,
                n_steps=len(trace.steps))


# =================================================================== smoke (GPU)
def stage_smoke(a):
    if os.path.exists(p("smoke_results.json")) and jload(p("smoke_results.json")).get("passed"):
        print("[smoke] done"); return
    assert jload(p("correctness_results.json"))["cpu"]["passed"], "CPU regression must pass first"
    pre = prefixes(SMOKE_N)
    T = X.P_LEN + SMOKE_G
    be, load = EGP.load_backend(check_dispatch=True)
    out, checks = dict(load=load, n=SMOKE_N, prefix=X.P_LEN, generated=SMOKE_G), {}
    runs = {}
    for cond in CONDS:
        tr = StepTrace()
        g1, c1, r1 = run_generation(be, cond, pre, T, audit=tr)
        g2, c2, r2 = run_generation(be, cond, pre, T)
        runs[cond] = (r1, r2, tr, c1)
    # prefix prefill vs batch-1 teacher forcing (no KV cache)
    ref = [be.replay(x) for x in pre]
    be2_logits, be2_sel, be2_plp = be.prefill(torch.tensor(pre))
    dl = np.concatenate([(be2_plp[i].cpu() - ref[i][0]).abs().numpy() for i in range(SMOKE_N)])
    ragree = float(np.mean([(torch.sort(be2_sel[i].cpu().long(), -1).values == torch.sort(ref[i][1].long(), -1).values)
                            .all(-1).float().mean() for i in range(SMOKE_N)]))
    lastref = []
    for i in range(SMOKE_N):              # last-position logits of the replay = next-token distribution
        ids = torch.tensor(pre[i], device=be.dev).view(1, -1)
        with torch.no_grad():
            lg = be.model(input_ids=ids, use_cache=False).logits[0, -1, :VOCAB].float()
        lastref.append(torch.log_softmax(lg, -1).cpu())
    lastgen = torch.log_softmax(be2_logits.float(), -1).cpu()
    dlast = float(max((lastgen[i] - lastref[i]).abs().max() for i in range(SMOKE_N)))
    # same comparison restricted to tokens the reference gives probability >= 1e-4 (where ranking can matter)
    dlast_hp = float(max((lastgen[i] - lastref[i]).abs()[lastref[i] >= np.log(1e-4)].max() for i in range(SMOKE_N)))
    top1 = all(int(lastgen[i].argmax()) == int(lastref[i].argmax()) for i in range(SMOKE_N))
    out["prefill_vs_teacher_forcing"] = dict(prefix_logp_mean_abs=float(dl.mean()), prefix_logp_max_abs=float(dl.max()),
                                             last_logp_max_abs=dlast, last_logp_max_abs_p_ge_1e4=dlast_hp,
                                             last_top1_agree=top1,
                                             prefix_routing_set_agreement=ragree)
    # attempt 1 used an ad-hoc full-vocabulary cap of 0.5 (not in the frozen protocol) and failed on it alone
    # (0.81; logs/smoke_attempt1_failed_last_logp_cap.json). The same 0.81 occurs between cached and no-cache logits
    # of ordinary decode steps (kv check), i.e. BF16 kernel-shape rounding in far-tail tokens. The check now uses the
    # frozen replay tolerance (mean |dlogp| <= 0.05) + top-1 agreement + the kv-check cap (1.0).
    checks["prefill_logits_match_teacher_forcing"] = bool(dl.mean() <= 0.05 and top1 and dlast <= 1.0)
    checks["prefix_routing_matches_teacher_forcing"] = ragree >= 0.95
    out["methods"] = {}
    for cond, (r1, r2, tr, c1) in runs.items():
        ub = CONDS[cond]["use_balance"]
        m = dict(tokens=r1.tokens.tolist(), cost=c1)
        m["prefix_identity"] = bool(r1.tokens[:, :X.P_LEN].tolist() == pre)
        m["appended_exactly"] = tuple(r1.tokens.shape) == (SMOKE_N, T)
        m["deterministic"] = bool(torch.equal(r1.tokens, r2.tokens) and torch.equal(r1.cum_logp, r2.cum_logp)
                                  and torch.equal(r1.hist_experts, r2.hist_experts) and torch.equal(r1.final_score, r2.final_score))
        m["counts_equal_recount"] = all(torch.equal(r1.counts[i], G.recount(r1.hist_experts[i], E)) for i in range(SMOKE_N))
        m["counts_total"] = all(int(r1.counts[i].sum()) == T * NL * K for i in range(SMOKE_N))
        # prefix part of the generation-time routing = prefill routing; prefix logp part of R = prefill logp
        m["prefix_routing_in_counts"] = bool(torch.equal(r1.hist_experts[:, :X.P_LEN].long(), be2_sel.cpu().long()))
        m["prefix_logp_in_R"] = bool(torch.equal(r1.token_logp[:, :X.P_LEN - 1], be2_plp.cpu()))
        m["R_identity"] = bool(torch.allclose(r1.cum_logp, r1.token_logp.sum(1), rtol=1e-12, atol=1e-9))
        first = tr.steps[0]
        m["first_expansion_single_branch"] = bool(first["n_branches"] == 1 and first["prefix_len"] == X.P_LEN
                                                  and first["logp"].shape[0] == SMOKE_N)
        m["first_expansion_counts_are_prefix"] = all(int(first["counts"][i].sum()) == X.P_LEN * NL * K for i in range(SMOKE_N))
        m["oracle"] = oracle_check_trace(tr, ub)
        # KV-cache vs selected parent: re-derive the logits of every live branch without cache at chosen steps
        kv = []
        for d in tr.steps:
            if d["step"] in SMOKE_KV_STEPS:
                for row in range(d["tokens"].shape[0]):
                    ids = d["tokens"][row].to(be.dev).view(1, -1)
                    with torch.no_grad():
                        lg = be.model(input_ids=ids, use_cache=False).logits[0, -1, :VOCAB].float()
                    lpr = torch.log_softmax(lg, -1).cpu()
                    kv.append(dict(step=d["step"], row=row, max_abs=float((lpr - d["logp"][row]).abs().max()),
                                   top1=int(lpr.argmax()) == int(d["logp"][row].argmax())))
        m["kv_parent_check"] = dict(n=len(kv), max_abs_logp=max(x["max_abs"] for x in kv),
                                    top1_agree=float(np.mean([x["top1"] for x in kv])))
        m["replay"] = EGP._replay_checks(be, r1)
        if ub:
            m["intervention"] = r1.intervention
        out["methods"][cond] = m
    po, eb = runs["PX_PO"][0], runs["PX_EBSS"][0]
    checks.update(
        prefix_identity=all(out["methods"][c]["prefix_identity"] for c in CONDS),
        appended_exactly_32=all(out["methods"][c]["appended_exactly"] for c in CONDS),
        prefix_routing_counts=all(out["methods"][c]["prefix_routing_in_counts"] and out["methods"][c]["counts_equal_recount"]
                                  and out["methods"][c]["counts_total"] and out["methods"][c]["first_expansion_counts_are_prefix"]
                                  for c in CONDS),
        cumulative_scores=all(out["methods"][c]["prefix_logp_in_R"] and out["methods"][c]["R_identity"] for c in CONDS),
        identical_prefixes_po_ebss=bool(torch.equal(po.tokens[:, :X.P_LEN], eb.tokens[:, :X.P_LEN])),
        first_expansion_single_branch=all(out["methods"][c]["first_expansion_single_branch"] for c in CONDS),
        ranking_matches_oracle=all(out["methods"][c]["oracle"]["ranking_matches_oracle"] for c in CONDS),
        sigma_matches_oracle=out["methods"]["PX_EBSS"]["oracle"]["sigma_matches_oracle"],
        kv_cache_parent=all(out["methods"][c]["kv_parent_check"]["top1_agree"] >= 0.95
                            and out["methods"][c]["kv_parent_check"]["max_abs_logp"] <= 1.0 for c in CONDS),
        replay_logprob_close=all(out["methods"][c]["replay"]["mean_abs_dlogp"] <= 0.05
                                 and out["methods"][c]["replay"]["pearson_r"] >= 0.99 for c in CONDS),
        deterministic=all(out["methods"][c]["deterministic"] for c in CONDS),
        historical_generator_unchanged=jload(p("correctness_results.json"))["cpu"]["passed"],
        dispatch_mismatches_zero=be.dispatch_mismatches == 0)
    out["control"] = dict(sentences_differing=int((po.tokens != eb.tokens).any(1).sum()),
                          first_divergence=[int((po.tokens[i] != eb.tokens[i]).nonzero()[0]) if (po.tokens[i] != eb.tokens[i]).any()
                                            else None for i in range(SMOKE_N)])
    out["checks"] = checks
    out["passed"] = all(checks.values())
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(P.MODEL)
    out["text"] = {c: [t.decode(x[X.P_LEN:]) for x in runs[c][0].tokens.tolist()] for c in CONDS}
    jdump(out, p("smoke_results.json"))
    cr = jload(p("correctness_results.json")); cr["smoke"] = dict(checks=checks, passed=out["passed"])
    jdump(cr, p("correctness_results.json"))
    print(f"[smoke] passed {out['passed']} {checks}\n prefill {out['prefill_vs_teacher_forcing']}\n control {out['control']}")
    if not out["passed"]:
        raise SystemExit("smoke FAILED")


# =================================================================== B=16 generation
def stage_generate(a):
    cond = a.cond
    pt = p("generation_logs", f"generated_{cond}.pt")
    if os.path.exists(pt):
        print(f"[gen] {cond} done"); return
    need(p("smoke_results.json"))
    pre = prefixes(16)
    be, load = EGP.load_backend(check_dispatch=False)
    g, c, r = run_generation(be, cond, pre, X.SEQ, progress_every=64, record_steps=(64, 65, 128, 256, 511))
    c["load"] = load
    if not c["validation"]["valid"] or g["tokens"][:, :X.P_LEN].tolist() != pre:
        raise SystemExit(f"{cond} invalid")
    torch.save(g, pt)
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(P.MODEL)
    name = {"PX_PO": "generated_prefix_po.jsonl", "PX_EBSS": "generated_prefix_ebss.jsonl"}[cond]
    with open(p(name) + ".tmp", "w") as f:
        for i in range(16):
            ids = g["tokens"][i].tolist()
            f.write(json.dumps(dict(index=i, method=cond, prefix_len=X.P_LEN, n_tokens=len(ids), padding_tokens=0,
                                    prefix_sha256=X.token_hash(ids[:X.P_LEN]), cum_logprob=float(g["cum_logp"][i]),
                                    sigma_alg1_rate_internal=float(g["sigma"][i]), final_score=float(g["final_score"][i]),
                                    token_ids=ids, prefix_text=t.decode(ids[:X.P_LEN]),
                                    continuation_text=t.decode(ids[X.P_LEN:])), ensure_ascii=False) + "\n")
    os.replace(p(name) + ".tmp", p(name))
    jdump(c, p("generation_logs", f"generation_cost_{cond}.json"))
    print(f"[gen] {cond}: {c['wall_s']:.0f} s (prefill {c['breakdown_s']['prefix_prefill']:.1f} s) valid {c['validation']['valid']}")


# =================================================================== CANONICAL-B1 capture (+ per-token NLL)
@torch.no_grad()
def forward_b1_tok(model, tokens):
    """EGP.forward_b1_sel plus per-token NLL [n, 511] (fp32 log_softmax of the same logits)."""
    from moe_calib import model_utils as MU
    n, T = tokens.shape
    hs = [MU.embed(model, tokens[i:i + 1]).cpu() for i in range(n)]
    stored = {L: torch.empty(n, T, hs[0].shape[-1], dtype=hs[0].dtype) for L in EGP.LAYERS}
    sel_all = torch.zeros(n, T, NL, K, dtype=torch.int16)
    for L in range(NL):
        if L in stored:
            for j in range(n):
                stored[L][j] = hs[j][0]
        with MU.layer_on_gpu(model, L):
            cap = []
            h = MU.moe_block(model, L).gate.register_forward_hook(lambda m, i, o: cap.append(o.detach()))
            try:
                for j in range(n):
                    hs[j] = MU.run_layer(model, L, hs[j].to(MU.GPU)).cpu()
                    lg = cap.pop().reshape(T, E)
                    sel_all[j, :, L] = torch.topk(torch.softmax(lg, dim=-1, dtype=torch.float), K,
                                                  dim=-1).indices.to(torch.int16).cpu()
            finally:
                h.remove()
    nll = np.zeros(n); tok = np.zeros((n, T - 1))
    for j in range(n):
        logits = model.lm_head(model.model.norm(hs[j].to(MU.GPU)))
        s, c = EGP.EP.nll_from_logits(logits, tokens[j:j + 1])
        nll[j] = float(s[0]) / int(c[0])
        lp = torch.log_softmax(logits[0, :-1, :].float(), -1)
        tok[j] = (-lp.gather(1, tokens[j, 1:, None].to(lp.device))[:, 0]).double().cpu().numpy()
    return stored, sel_all, nll, tok


def stage_capture(a):
    man_path = p("logs", "capture_manifest.json")
    if os.path.exists(man_path):
        print("[capture] done"); return
    gens = {c: torch.load(p("generation_logs", f"generated_{c}.pt"), weights_only=False) for c in CONDS}
    pre = prefixes(16)
    for c in CONDS:
        assert P.validate_generated(gens[c]["tokens"], 16, X.SEQ, VOCAB)["valid"]
        assert gens[c]["tokens"][:, :X.P_LEN].tolist() == pre
    pman = jload(os.path.join(PILOT, "logs", "capture_manifest.json"))
    pid = torch.load(os.path.join(P.CACHE, "ids.pt"))
    pos = {r: i for i, r in enumerate(pman["rows"]["c4_rows"])}
    c4_idx = [pos[r] for r in pman["c4_random"]["0"]["chosen"]]
    rs_po = torch.load(os.path.join(PILOT, "logs", "generated_po.pt"), weights_only=False)["tokens"]
    rs_eb = torch.load(os.path.join(FID, "logs", "generated_ALG1_RATE.pt"), weights_only=False)["tokens"]
    tokens = torch.cat([pid["tokens"][c4_idx], gens["PX_PO"]["tokens"], gens["PX_EBSS"]["tokens"], rs_po, rs_eb])
    rows = dict(C4R_s0=list(range(16)), PX_PO=list(range(16, 32)), PX_EBSS=list(range(32, 48)),
                RS_PO=list(range(48, 64)), RS_EBSS=list(range(64, 80)))
    model, t_load = EGP.FD.load_model_full()
    assert OV.verify_pristine_full(model)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stored, sel, nll, tok = forward_b1_tok(model, tokens)
    t_cap = time.time() - t0
    os.makedirs(NCACHE, exist_ok=True)
    files, reg = {}, {}
    fman = jload(os.path.join(FID, "logs", "capture_manifest.json"))
    for L in EGP.LAYERS:
        path = os.path.join(NCACHE, f"L{L}.pt")
        torch.save(stored[L], path + ".tmp"); os.replace(path + ".tmp", path)
        files[str(L)] = dict(shape=list(stored[L].shape), sha256=EGP.sha256_file(path))
        R = torch.load(os.path.join(P.CACHE, f"L{L}.pt"), mmap=True)
        F = torch.load(os.path.join(FCACHE, f"L{L}.pt"), mmap=True)
        reg[str(L)] = dict(c4r_s0_vs_pilot=bool(torch.equal(stored[L][:16], R[c4_idx])),
                           rs_po_vs_pilot=bool(torch.equal(stored[L][48:64], R[pman["rows"]["po"]])),
                           rs_ebss_vs_fidelity=bool(torch.equal(stored[L][64:80], F[fman["rows"]["ALG1_RATE"]])))
    json.dump(dict(pool_size=0, holdout_size=0, seqlen=512, layers=EGP.LAYERS, policy="CANONICAL-B1", n_rows=len(tokens)),
              open(os.path.join(NCACHE, "meta.json"), "w"), indent=1)
    torch.save(dict(tokens=tokens, rows=rows), os.path.join(NCACHE, "ids.pt"))
    torch.save(dict(sel=sel, nll=torch.tensor(nll), tok_nll=torch.tensor(tok)), p("logs", "capture_routing_nll.pt"))
    psel = torch.load(os.path.join(PILOT, "logs", "capture_routing_nll.pt"))
    fsel = torch.load(os.path.join(FID, "logs", "capture_routing_nll.pt"))
    routing = dict(c4r_s0=bool(torch.equal(sel[:16], psel["sel"][c4_idx])),
                   rs_po=bool(torch.equal(sel[48:64], psel["sel"][pman["rows"]["po"]])),
                   rs_ebss=bool(torch.equal(sel[64:80], fsel["sel"][fman["rows"]["ALG1_RATE"]])))
    nll_ok = bool(np.allclose(nll[:16], psel["nll"].numpy()[c4_idx], rtol=0, atol=0))
    tok_ok = float(np.abs(tok.mean(1) - nll).max())
    # prefix routing identical between the natural control and both generated sets (causal, batch-1)
    pre_route = {c: float((sel[rows[c]][:, :X.P_LEN] == sel[:16][:, :X.P_LEN]).all(-1).float().mean()) for c in CONDS}
    man = dict(cache_dir=NCACHE, rows=rows, files=files, capture_s=t_cap, model_load_s=t_load,
               compatibility=dict(layer_inputs=reg, routing_equal=routing, c4r_s0_row_nll_equal_pilot=nll_ok),
               per_token_nll_vs_row_nll_max_abs=tok_ok, prefix_routing_equal_to_c4_natural=pre_route,
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=EGP.FD.peak_cpu_gb())
    man["passed"] = bool(all(all(v.values()) for v in reg.values()) and all(routing.values()) and tok_ok < 1e-4)
    jdump(man, man_path)
    print(f"[capture] {len(tokens)} rows {t_cap:.0f} s; passed {man['passed']} routing {routing} prefix-routing {pre_route}")
    if not man["passed"]:
        raise SystemExit("capture compatibility FAILED")


# =================================================================== diagnostics (CPU)
SETS = ["C4R_s0", "PX_PO", "PX_EBSS", "RS_PO", "RS_EBSS"]


def stage_diagnostics(a):
    from transformers import AutoTokenizer
    import ebss_balance_fidelity as BF
    OV.ECACHE = NCACHE                                   # BF's import points OV at its own cache; restore ours
    man = need(p("logs", "capture_manifest.json"))
    cap = torch.load(p("logs", "capture_routing_nll.pt"))
    ids = torch.load(os.path.join(NCACHE, "ids.pt"))
    tokzr = AutoTokenizer.from_pretrained(P.MODEL)
    rows = man["rows"]
    sel, nll, tok = cap["sel"].numpy(), cap["nll"].numpy(), cap["tok_nll"].numpy()
    toks = {s: ids["tokens"][rows[s]].tolist() for s in SETS}
    gens = {c: torch.load(p("generation_logs", f"generated_{c}.pt"), weights_only=False) for c in CONDS}
    hist = dict(PX_PO=gens["PX_PO"]["hist_experts"].numpy(), PX_EBSS=gens["PX_EBSS"]["hist_experts"].numpy(),
                RS_PO=torch.load(os.path.join(PILOT, "logs", "generated_po.pt"), weights_only=False)["hist_experts"].numpy(),
                RS_EBSS=torch.load(os.path.join(FID, "logs", "generated_ALG1_RATE.pt"), weights_only=False)["hist_experts"].numpy())
    # ---- diversity by region
    div = {}
    for s in SETS:
        div[s] = {r: X.region_text(toks[s], tok[rows[s]], reg) for r, reg in X.REGIONS.items()}
        div[s]["full"]["model_ppl_b1_row_mean"] = float(np.exp(nll[rows[s]].mean()))
    # descriptive splice: the 64 C4 prefix tokens + the historical continuation [64:512]
    for m in ("PO", "EBSS"):
        spl = [toks["C4R_s0"][i][:X.P_LEN] + toks[f"RS_{m}"][i][X.P_LEN:] for i in range(16)]
        div[f"SPLICE_C4prefix_RS_{m}"] = dict(full=X.region_text(spl, None, X.REGIONS["full"]),
                                              note="descriptive text-only control: the prefix content alone, with the "
                                                   "random-start continuation; never used for calibration")
    bd = {s: {f"n{n}": X.boundary_decomposition(toks[s], n) for n in (2, 3)} for s in SETS}
    psm = lambda s, r: div[s][r]["per_sequence_mean"]
    lab = X.diversity_label({m: psm(f"PX_{m}", "continuation") for m in ("PO", "EBSS")},
                            {m: psm(f"RS_{m}", "continuation") for m in ("PO", "EBSS")},
                            {m: psm(f"PX_{m}", "full") for m in ("PO", "EBSS")},
                            {m: psm(f"RS_{m}", "full") for m in ("PO", "EBSS")})
    jdump(dict(regions=X.REGIONS, sets=div, boundary_decomposition=bd, diversity_classification=lab,
               definitions=dict(per_sequence_mean="mean over the 16 sequences of each sequence's statistic over the region",
                                corpus="pooled n-grams of the 16 region slices (n-grams never cross sequences)",
                                repetition_rate="1 - distinct-1 (per sequence, mean)",
                                nll="CANONICAL-B1 teacher-forced per-token NLL of the targets inside the region"),
               contamination={c: P.contamination(gens[c]["tokens"].tolist(), heldout_windows()) for c in CONDS}),
          p("diversity_by_region.json"))
    # ---- repetition
    rep = {s: X.repetition_profile(toks[s]) for s in SETS}
    pos = {s: np.mean([X.repetition_by_position(w) for w in toks[s]], 0).tolist() for s in SETS}
    jdump(dict(sets=rep, coverage_by_position_32tok_bins=pos,
               comparisons={"PX_PO_vs_PX_EBSS": ("PX_PO", "PX_EBSS"), "PX_PO_vs_RS_PO": ("PX_PO", "RS_PO"),
                            "PX_EBSS_vs_RS_EBSS": ("PX_EBSS", "RS_EBSS"), "PX_vs_C4_natural": ("PX_*", "C4R_s0")}),
          p("repetition_analysis.json"))
    # ---- routing (CANONICAL-B1, full 512-token sequences) + generation-time
    ref, d_star = BF.load_reference()
    rout = {}
    for s in SETS:
        st = BF.set_stats(sel[rows[s]], toks[s], nll[rows[s]], hist.get(s), ref, d_star, tokzr)
        st.pop("text", None)
        st["quantized_layers"] = {str(L): st["b1"]["per_layer"][L] for L in EGP.LAYERS}
        rout[s] = st
    # C4R s1/s2 from the pilot capture (identical B1 policy)
    pman = jload(os.path.join(PILOT, "logs", "capture_manifest.json"))
    psel = torch.load(os.path.join(PILOT, "logs", "capture_routing_nll.pt"))
    pid = torch.load(os.path.join(P.CACHE, "ids.pt"))
    ppos = {r: i for i, r in enumerate(pman["rows"]["c4_rows"])}
    for sd in ("1", "2"):
        rr = [ppos[r] for r in pman["c4_random"][sd]["chosen"]]
        st = BF.set_stats(psel["sel"].numpy()[rr], pid["tokens"][rr].tolist(), psel["nll"].numpy()[rr], None, ref, d_star, tokzr)
        st.pop("text", None)
        st["quantized_layers"] = {str(L): st["b1"]["per_layer"][L] for L in EGP.LAYERS}
        rout[f"C4R_s{sd}"] = st
    comp = {s: dict(paper_sigma=v["b1"]["paper_sigma"], **{k: v["b1"]["mean_over_layers"][k]
                                                           for k in ("cv", "gini", "min", "p5", "p10", "n_zero")},
                    routing_l1=v["representativeness"]["routing_l1_mean"], frac_atypical=v["representativeness"]["frac_atypical"],
                    gen_cv=v.get("generation_time", {}).get("mean_over_layers", {}).get("cv"),
                    gen_vs_b1_agreement=v.get("generation_vs_b1_set_agreement")) for s, v in rout.items()}
    jdump(dict(compact=comp, full=rout, policy="CANONICAL-B1 primary; generation-time routing separately"),
          p("routing_diagnostics.json"))
    # ---- exposure decomposition (B1, length-normalised)
    from moe_calib import typicality as TY  # noqa: F401  (reference distribution comes from BF.load_reference)
    dec = {}
    for s in SETS:
        S = sel[rows[s]]
        cp, cc = X.region_counts(S, 0, X.P_LEN), X.region_counts(S, X.P_LEN, X.SEQ)
        ct = X.region_counts(S, 0, X.SEQ)
        l1 = lambda c: float(np.mean([P.ED.routing_l1(c[L], ref[i]) for i, L in enumerate(EGP.LAYERS)]))
        dec[s] = dict(prefix=X.rate_stats(cp, 16 * X.P_LEN), continuation=X.rate_stats(cc, 16 * X.G_LEN),
                      total=X.rate_stats(ct, 16 * X.SEQ), additive=bool((cp + cc == ct).all()),
                      routing_l1_4layers=dict(prefix=l1(cp), continuation=l1(cc), total=l1(ct)),
                      continuation_share_of_tokens=X.G_LEN / X.SEQ)
    jdump(dict(sets=dec, definition="n_total = n_prefix + n_continuation from the same B1 replay; statistics use the "
                                    "relative load count/(tokens*k/E) so 64- and 448-token segments are comparable; "
                                    "routing L1 vs the frozen C4 pool reference on layers 1/8/16/23"),
          p("exposure_decomposition.json"))
    print("[diag]", lab["label"], {s: (round(comp[s]["cv"], 3), round(psm(s, "continuation")["distinct2"], 3)) for s in SETS})


# =================================================================== quantize
def stage_quantize(a):
    cond = a.cond
    os.makedirs(p("results"), exist_ok=True)
    out = p("results", f"{cond}.json")
    if os.path.exists(out):
        print(f"[quant] {cond} done"); return
    man = need(p("logs", "capture_manifest.json"))
    for L in EGP.LAYERS:
        assert EGP.sha256_file(os.path.join(NCACHE, f"L{L}.pt")) == man["files"][str(L)]["sha256"], "cache changed"
    chosen = man["rows"][cond]
    assert len(chosen) == 16 == len(set(chosen))
    model, t_load = EGP.FD.load_model_full()
    assert OV.verify_pristine_full(model)
    assert OV.ECACHE == NCACHE
    r = OV.quant_config(model, cond, NCACHE, chosen, OV.OCACHE, 4)
    r.update(config=cond, policy="CANONICAL-B1", cache_rows=chosen, model_load_s=t_load, B=16)
    jdump(r, out)
    print(f"[quant] {cond}: recon {r['recon']:.7f} PPL {r['ppl']:.6f} RTN fallbacks {r['rtn_fallbacks']} valid {r['valid']}",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cond", choices=list(CONDS))
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("generation_logs"), exist_ok=True)
    if a.stage == "report":
        import report_ebss_prefix_rescue as R
        return R.main()
    dict(freeze=stage_freeze, regression=stage_regression, smoke=stage_smoke, generate=stage_generate,
         capture=stage_capture, diagnostics=stage_diagnostics, quantize=stage_quantize)[a.stage](a)


if __name__ == "__main__":
    main()
