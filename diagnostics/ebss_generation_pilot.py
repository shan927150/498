"""Model-matched MoEQuant EBSS reproduction + Probability-Only ablation (small-budget pilot).

Generator: moe_calib/ebss_generator.py (Algorithm 1 / Eq. 13 of MoEQuant, BF16 Qwen1.5-MoE-A2.7B base).
Stages (--stage): preflight | source_review | smoke | feasibility | generate --method {po,ebss} |
                  diagnostics | quantize --config {C4R_s0,C4R_s1,C4R_s2,PO,EBSS} | report
Every stage is resumable (skips finished work) and refuses to run if an upstream gate failed.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import ebss_pilot as P                  # noqa: E402
os.environ["EBSS_VALIDATION_CACHE"] = P.CACHE         # quant_config writes weights under OUR cache dir
from moe_calib import ebss_generator as G              # noqa: E402
from moe_calib import ebss_official_data as ED         # noqa: E402
import official_ebss_validation as OV                  # noqa: E402
import eval_partial_ppl as EP                          # noqa: E402
import full_depth_pilot as FD                          # noqa: E402

OUT = P.OUT
LAYERS = [1, 8, 16, 23]
N_LAYERS, E, K_TOP = 24, 60, 4
VOCAB = 151646
METHODS = {"po": "PROBABILITY_ONLY", "ebss": "EBSS"}
QCONFIGS = ["C4R_s0", "C4R_s1", "C4R_s2", "PO", "EBSS"]
C4R_FILES = {0: "runs_low_budget/e3_random_B16_b4_d1_s0_warm-b7ef3d2f.json",
             1: "runs_low_budget/e3_random_B16_b4_d1_s1_warm-7c1251cd.json",
             2: "runs_low_budget/e3_random_B16_b4_d1_s2_warm-1bcd0994.json"}
SMOKE_RECORD_STEPS = (1, 2, 16, 31)
PILOT_RECORD_STEPS = (1, 2, 64, 256, 511)
jdump, sha256_file = FD.jdump, FD.sha256_file


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(P.MODEL)


def need(path, key="passed"):
    assert os.path.exists(path), f"missing upstream result {path}"
    d = json.load(open(path))
    assert d.get(key), f"upstream gate failed: {path} [{key}]"
    return d


def load_backend(check_dispatch):
    model, t_load = FD.load_model_full()
    t0 = time.time()
    pristine = OV.verify_pristine_full(model)
    t_dig = time.time() - t0
    assert pristine, "checkpoint parameters differ from the pristine digests"
    be = G.HFOffloadBackend(model, vocab_limit=VOCAB, check_dispatch=check_dispatch)
    return be, dict(model_load_s=t_load, pristine_all_parameters=pristine, digest_s=t_dig, place_s=be.place_s,
                    gpu_resident_gb=torch.cuda.memory_allocated() / 1e9, expert_layer_bytes=be.layer_bytes)


# ================================================================ preflight
def stage_preflight(a):
    from moe_calib.model_utils import _snapshot_dir
    snap = _snapshot_dir(P.MODEL)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    free, total = torch.cuda.mem_get_info()
    t = tok()
    rec = dict(model=P.MODEL, snapshot_dir=snap, snapshot=os.path.basename(snap),
               snapshot_matches=os.path.basename(snap) == P.SNAPSHOT, base_not_chat="chat" not in snap.lower(),
               config=dict((k, cfg[k]) for k in ("num_hidden_layers", "num_experts", "num_experts_per_tok",
                                                  "hidden_size", "num_key_value_heads", "num_attention_heads",
                                                  "vocab_size", "torch_dtype", "norm_topk_prob")),
               tokenizer_len=len(t), special_ids=t.all_special_ids, bos_token_id=t.bos_token_id,
               gpu_free_gb=free / 1e9, gpu_total_gb=total / 1e9,
               host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, environment=OV.env_record(),
               gptq_sha256=sha256_file(os.path.join(ROOT, "moe_calib", "gptq.py")))
    rec["passed"] = bool(rec["snapshot_matches"] and rec["base_not_chat"] and len(t) == VOCAB)
    os.makedirs(p("logs"), exist_ok=True)
    jdump(rec, p("logs", "preflight.json"))
    print(f"[preflight] snapshot {rec['snapshot']} ok={rec['passed']}; GPU free {free / 1e9:.1f} GB; host "
          f"{rec['host_mem_available_gb']:.0f} GB")


# ================================================================ source review (writes the frozen protocol)
SOURCES = {
    "paper": "MoEQuant, ICML 2025, PMLR 267:8245-8260 (https://proceedings.mlr.press/v267/chen25aa.html); text read "
             "from arXiv:2505.03804v1 HTML (https://arxiv.org/html/2505.03804)",
    "algorithm_1_image": "github.com/chenzx921020/MoEQuant @ d77ba1b asserts/Algorithm_1_EBSS.png (rebuttal "
                         "supplement, read directly)",
    "rebuttal_R1": "Tables_for_R1_eTCn.md (Table R2: tau sweep; Table R3: AGQ c; Table R4: AGQ gate layer)",
    "rebuttal_R2": "Tables_for_R2_NkUk.md (Table R3: time cost GPTQ vs MoEQuant, '128 sequences of 512 length', "
                   "batching; Table R7)",
    "rebuttal_R3": "Tables_for_R3_dN5E.md (Table R1 expert-balance STD by calibration set; Table R2 tau; Table R3 "
                   "width w; Table R6 EBSS time; Algorithm 1 image)",
    "rebuttal_R4": "Tables_for_R4_ZiDj.md (Table R3 time cost; Table R4 generated-sample snapshot)",
    "openreview_note": "https://openreview.net/forum?id=0epuNvt5Dj&noteId=aKw8ICecFU -- NOT ACCESSIBLE (bot "
                       "verification interstitial; API endpoints redirect to a challenge). Its content is not used "
                       "and not inferred.",
    "official_code": "the released repository contains NO EBSS generation code (only the EBSS_data/*.jsonl outputs "
                     "and a calibration loader)",
}

DECISIONS = [
    ("objective", "minimise -(1/N) sum log P + sigma/tau (Eq. 9-10)", "paper Eq. 9-10", "documented"),
    ("pruning score", "score(S||v) = -(R_S + log P(v|S))/(l+1) + sigma(M,S)/tau", "paper Eq. 13; Alg. 1 line 14",
     "documented"),
    ("score direction", "keep the w SMALLEST scores; final = argmin Score(c)", "Eq. 9-10 argmin + minus sign of "
     "Eq. 13; Eq. 14 text 'maximize' and Alg. 1 line 20 'arg max' conflict", "resolved (inconsistency)"),
    ("length normalisation", "divide by l+1 (= tokens of the candidate incl. the first token)", "Eq. 13; Alg. 1 "
     "line 14 (j+1)", "documented"),
    ("balance uses prefix or candidate", "prefix S ('deferred expert balance': the pruning metric excludes v)",
     "paper Sec. 4.2 text + Eq. 13 sigma(M,S); Alg. 1 line 14 writes sigma(M,new_candidate)", "resolved "
     "(inconsistency; v's routing is unknowable without a forward of v, the paper text is explicit)"),
    ("sigma definition", "sigma = mean over all MoE layers of sigma_l; sigma_l = sqrt(1/(E-1) sum (u - mean)^2)",
     "paper Eq. 3/4", "documented (Alg. 1 line 30 writes E/(E-1): typo-level conflict, paper equations used)"),
    ("usage frequency u", "routed tokens of the branch per expert / tokens of the branch", "rebuttal R3 Table R1 "
     "scale (0.0427 wikitext2 / 0.0052 EBSS on DeepSeek-MoE-16B)", "ASSUMPTION"),
    ("routed counts vs affinities", "top-k membership counts", "paper 'frequency of expert usage'", "assumption "
                                                                                                  "(consistent)"),
    ("tau meaning", "divides sigma (expert-balance coefficient, called 'temperature'); never applied to logits",
     "Eq. 13; Alg. 1 input list 'Expert balance temperature'; App. A.1 'controls the significance of expert "
     "balance'", "documented"),
    ("tau value", "1.3 (task protocol)", "rebuttal R2/R3 Table R2 Qwen: 1.2 -> 49.60, 1.3 -> 49.61; paper sets 1.2",
     "protocol choice (paper default is 1.2)"),
    ("w", "4", "paper Sec. 5.2 / App. Table 7; rebuttal R3 Table R3", "documented"),
    ("first token", "Random(V)", "Alg. 1 line 4", "documented; distribution and seed undocumented -> uniform over "
                                                  "non-special ids, seed 20261110 (ASSUMPTION)"),
    ("number / length of sequences", "128 x 512 (full budget); this pilot 16 x 512", "rebuttal R2 Table R3, R3 "
     "Table R6 ('128 sequences of 512 length'); paper does not state it", "rebuttal clarification"),
    ("stop rule / EOS", "none: Seqlen-1 expansion steps over all of V", "Alg. 1 lines 6-18", "documented by "
                                                                                           "omission"),
    ("batching", "sentences stitched along the batch dimension", "rebuttal R2 Table R3 / R3 Table R6", "rebuttal "
                                                                                                      "clarification"),
    ("generation hardware / time", "A800, Qwen-MoE-14B 18 min (128 x 512)", "rebuttal R3 Table R6", "reference "
                                                                                                    "only"),
    ("checkpoint", "Qwen/Qwen1.5-MoE-A2.7B base @ 1a758c50", "paper cites huggingface.co/Qwen/Qwen1.5-MoE-A2.7B; "
     "task protocol", "documented for the paper; the released file's generator stays undocumented"),
]

AMBIGUITIES = {
    "score_direction": "Eq. 14 defines argtopk as maximising and Alg. 1 line 20 uses arg max, while Eq. 9-10 "
                       "minimise and Eq. 13 carries a leading minus (negative log-likelihood + imbalance). Resolved: "
                       "minimisation everywhere.",
    "balance_prefix_vs_candidate": "Alg. 1 line 14 writes sigma(M, new_candidate); Eq. 13 writes sigma(M, S) and "
                                   "Sec. 4.2 describes deferred expert balance that excludes v. Resolved: prefix S.",
    "sigma_scale": "Alg. 1 line 30 has sqrt(E/(E-1) sum ...), Eq. 4 has sqrt(1/(E-1) sum ...). Eq. 4 used (the "
                   "factor E only rescales sigma, i.e. it would act like tau/E).",
    "usage_frequency_normalisation": "per-token rate (counts / tokens) vs share (counts / (tokens * k)) is not "
                                     "stated; the share would shrink the balance term by k = 4 (equivalent to "
                                     "tau x 4). Per-token rate adopted (rebuttal STD scale).",
    "eq12_missing_log": "Eq. 12 writes P(v|S) without log; Eq. 11/13 use log. Eq. 13 used.",
    "first_token_distribution": "Random(V): distribution, special-token handling and seed undocumented. The "
                                "released EBSS_qwen.jsonl instead starts record i with token id i (sequential), "
                                "which contradicts Random(V) or indicates a different script.",
    "sequence_count_and_length": "paper silent; rebuttal: 128 x 512; released file: 600 records up to ~2048 tokens.",
    "eos_and_specials": "no stop rule; whether specials / EOS were masked is undocumented. Not masked here.",
    "per_sentence_vs_global_balance": "sigma(M,S) is a property of one branch; no cross-sentence balance state is "
                                      "described. Per-branch adopted.",
    "precision_and_batching_numerics": "authors' precision undocumented; BF16 used (our checkpoint dtype). Batched "
                                       "KV decoding is not bitwise equal to a batch-1 replay (measured in smoke).",
    "openreview_response": "not accessible; nothing from it is used.",
}

REBUTTAL_NOTES = """# Author rebuttal material incorporated

Sources: the four `Tables_for_R*.md` files and `asserts/Algorithm_1_EBSS.png` of github.com/chenzx921020/MoEQuant
@ d77ba1b (local clone ~/ebss_official_src/moequant). The OpenReview note aKw8ICecFU could NOT be accessed (bot
verification wall; API redirects to a challenge) -- nothing from it is used or inferred.

| item | rebuttal value | file / table | used as |
|---|---|---|---|
| search width w (Qwen 7-task avg) | w=2 49.19, 3 49.45, **4 49.59**, 5 49.59, 6 49.61, 10 49.60, 20 49.63 | R3 Table R3 | w = 4 (paper default; diminishing returns) |
| tau (Qwen) | 1.0 49.47/49.49, 1.1 49.53, **1.2 49.60**, 1.3 49.61, 1.4 49.59, 1.5 49.55 | R1 Table R2, R3 Table R2 | tau = 1.3 (task protocol; paper default 1.2; difference 0.01 points) |
| tau name | "Temperature Parameter (tau)"; Alg. 1: "Expert balance temperature" | R1/R3 Table R2, Alg. 1 | divides sigma only; NOT a logit temperature |
| calibration budget | "MoEQuant generates 128 sequences of 512 length" | R2 Table R3, R3 Table R6, R4 Table R3 | full target 128 x 512 = 65,536 tokens; this pilot 16 x 512 |
| batching | "we stitch multiple sequences together in the batch dimension" | same | all 16 sentences x w branches decoded in one batch |
| EBSS generation time (A800) | Qwen-MoE-14B 18 min, DeepSeek-MoE-16B 12 min, Mixtral-8x7B 42 min | R3 Table R6 | reference only; not normalised to our MIG slice |
| total quantization time (A800) | Qwen GPTQ 37 min vs MoEQuant 54 min | R2 Table R3 | motivation for overhead analysis |
| expert balance STD | DeepSeek-MoE-16B: wikitext2 0.0427, humaneval 0.0877, gsm8k 0.0928, EBSS 0.0052 | R3 Table R1 (asserts/inter-expert_balance.md) | scale evidence for u = per-token usage rate |
| generated samples | Qwen sample begins '! ### GMAT Club Daily Prep ...' (mixed English/Chinese, exam items, code, web text, some repetition) | R4 Table R4 | context only; not evidence of a defect |
| Algorithm 1 | score = -(RS+log prob)/(j+1) + sigma(M,new_candidate)/tau; Topk(w, key=score); best = argmax Score | Alg. 1 image | see source_ambiguities.json for the two conflicts with the paper |

The rebuttal reports only downstream accuracy for w/tau; no tables of generated-data balance for Qwen exist.
"""


def stage_source_review(a):
    if os.path.exists(p("protocol.json")):
        print("[source_review] protocol already frozen"); return
    starts = G.sample_first_tokens(P.N_PILOT)
    t = tok()
    init = dict(policy="uniform Random(V) over non-special ids 0..151642 (Alg. 1 line 4; distribution assumed)",
                generator="numpy.random.default_rng(seed).integers(0, 151643, size=16)", seed=G.INIT_SEED,
                pilot_start_ids=starts, smoke_start_ids=starts[:P.N_SMOKE],
                start_tokens_decoded=[t.decode([x]) for x in starts], shared_by_both_methods=True,
                frozen_before_generation=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(init, p("initialization_manifest.json"))
    gcfg = dict(model=P.MODEL, snapshot=P.SNAPSHOT, precision="bf16", attention="eager", w=G.W_REF, tau=G.TAU_REF,
                seqlen=G.SEQLEN, n_sentences=P.N_PILOT, vocab_limit=VOCAB, choices=G.GENERATOR_CHOICES,
                methods={"EBSS": "Eq. 13 score with sigma(M,S)/tau", "PROBABILITY_ONLY": "Eq. 13 score without the "
                         "balance term (identical candidates, normalisation, scheduling, starts)"},
                no_tuning="w, tau, starts, seed fixed before any generation; no repetition penalty, no sampling "
                          "temperature, no top-k/nucleus sampling, no prompt template")
    jdump(gcfg, p("generator_config.json"))
    jdump(dict(ambiguities=AMBIGUITIES, decisions=[dict(item=i, implementation=v, source=s, status=st)
                                                   for i, v, s, st in DECISIONS]), p("source_ambiguities.json"))
    open(p("rebuttal_notes.md"), "w").write(REBUTTAL_NOTES)
    lines = ["# Source review: MoEQuant EBSS (what each implementation decision rests on)\n",
             "## Sources\n"] + [f"- **{k}**: {v}" for k, v in SOURCES.items()] + [
        "\n## Decisions\n", "| item | implementation | source | status |", "|---|---|---|---|"] + [
        f"| {i} | {v} | {s} | {st} |" for i, v, s, st in DECISIONS] + [
        "\n## Ambiguities and their resolution\n"] + [f"- **{k}**: {v}" for k, v in AMBIGUITIES.items()] + [
        "\n## Not documented anywhere accessible\n",
        "- the authors' generation script, its first-token distribution, special-token handling, precision, "
        "batching numerics and stop rule;",
        "- the generating checkpoint of the released EBSS_qwen.jsonl (see update_ebss_official_validation.md);",
        "- any per-step balance diagnostics of generated Qwen data.\n"]
    open(p("source_review.md"), "w").write("\n".join(lines) + "\n")
    proto = dict(
        experiment="Model-matched EBSS reproduction + Probability-Only ablation (pilot 16 x 512, one seed)",
        hypotheses=dict(H1="EBSS balancing reduces expert-exposure inequality vs Probability-Only (tested)",
                        H2="balance guidance may improve the lower tail but change distribution/diversity (tested)",
                        H3="probability-only beam generation may be repetitive; EBSS may mitigate or worsen (tested)",
                        H4="better balance need not improve GPTQ reconstruction (B=16 pilot only)",
                        H5="generation overhead may dominate calibration cost (measured)"),
        generator=gcfg, initialization=dict(seed=G.INIT_SEED, pilot_start_ids=starts),
        smoke=dict(n=P.N_SMOKE, seqlen=P.SMOKE_LEN, w=G.W_REF, record_steps=list(SMOKE_RECORD_STEPS),
                   checks=list(P.SMOKE_CHECKS) + ["dispatch_mismatches == 0"],
                   replay_tolerance="batch-1 no-cache replay of every chosen smoke sequence: mean |delta log p| <= "
                                    "0.05 nats and Pearson r >= 0.99 (BF16 batch-shape rounding; exact equality is "
                                    "not expected, see update_batch_sensitivity.md)"),
        pilot=dict(n=P.N_PILOT, seqlen=G.SEQLEN, tokens_per_method=P.N_PILOT * G.SEQLEN,
                   progress_positions=list(P.PROGRESS_POSITIONS)),
        gptq_pilot=dict(configs=QCONFIGS, c4_random_sources=C4R_FILES, B=16, K=4, gptq=OV.frozen_cfg()["gptq"],
                        layers=LAYERS, extraction="CANONICAL-B1 (each sequence alone, BF16)",
                        reconstruction="E3 held-out rows 2128-2255 of ~/moe_cache (unchanged)",
                        ppl="frozen 128 x 512 WikiText-2 windows (secondary)",
                        rtn_fallback="E3 semantics preserved (zero-token experts fall back to RTN); counted and "
                                     "reported, never presented as GPTQ"),
        representativeness=dict(reference="runs_low_budget/routing_profile_v2.npz C4-pool reference, layers "
                                           "1/8/16/23 (moe_calib.typicality)", routing_source="CANONICAL-B1 replay"),
        decision_rule=P.DECISION_RULE,
        hard_constraints="no checkpoint change, no chat model, no quantized generator, no AGQ, no GPTQ change, no new "
                         "sampling, no repetition penalty, no tau/w tuning, same starts for both methods, no padding, "
                         "no C4 mixing, no held-out-based sample selection, no automatic B=128 run",
        code_sha256={f: sha256_file(os.path.join(ROOT, f)) for f in
                     ("moe_calib/ebss_generator.py", "moe_calib/ebss_pilot.py", "diagnostics/ebss_generation_pilot.py",
                      "moe_calib/gptq.py")},
        frozen_before_any_generation=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(proto, p("protocol.json"))
    print(f"[source_review] protocol frozen; starts {starts}")


# ================================================================ smoke
def _replay_checks(be, res):
    d_all, r_lp, r_gen, agree, total = [], [], [], 0, 0
    for i in range(res.tokens.shape[0]):
        lp, sel = be.replay(res.tokens[i].tolist())
        d_all.append((lp - res.token_logp[i]).abs().numpy())
        r_lp.append(lp.numpy()); r_gen.append(res.token_logp[i].numpy())
        a = torch.sort(sel.long(), -1).values == torch.sort(res.hist_experts[i].long(), -1).values
        agree += int(a.all(-1).sum()); total += a.shape[0] * a.shape[1]
    d = np.concatenate(d_all)
    r = float(np.corrcoef(np.concatenate(r_lp), np.concatenate(r_gen))[0, 1])
    return dict(mean_abs_dlogp=float(d.mean()), max_abs_dlogp=float(d.max()), pearson_r=r,
                routing_set_agreement=agree / total, n_token_layers=total)


def _step_log_checks(log, w, use_balance):
    ok_formula = ok_rank = ok_bal = True
    for s in log:
        sc = [c["score"] for c in s["candidates"]]
        ok_rank &= all(x <= y for x, y in zip(sc[:-1], sc[1:])) and all(c["kept"] == (c["rank"] < w)
                                                                         for c in s["candidates"])
        for c in s["candidates"]:
            ok_formula &= abs(c["prob_term"] + c["balance_term"] - c["score"]) <= 1e-12
            if use_balance:
                ok_bal &= abs(c["balance_term"] - c["sigma_parent"] / s["tau"]) <= 1e-12 and c["balance_term"] > 0
            else:
                ok_bal &= c["balance_term"] == 0.0
    return ok_formula and ok_rank, ok_bal


def stage_smoke(a):
    if os.path.exists(p("smoke_test_results.json")):
        print("[smoke] done"); return
    need(p("logs", "preflight.json"))
    assert os.path.exists(p("protocol.json")), "run source_review first"
    starts = json.load(open(p("initialization_manifest.json")))["smoke_start_ids"]
    be, load = load_backend(check_dispatch=True)
    runs = {}
    for name, ub in (("PROBABILITY_ONLY", False), ("EBSS", True)):
        r1 = G.generate(be, starts, P.SMOKE_LEN, G.W_REF, G.TAU_REF, ub, record_steps=SMOKE_RECORD_STEPS)
        r2 = G.generate(be, starts, P.SMOKE_LEN, G.W_REF, G.TAU_REF, ub)
        runs[name] = (r1, r2)
    out, checks = dict(load=load, starts=starts, seqlen=P.SMOKE_LEN, w=G.W_REF, tau=G.TAU_REF, methods={}), {}
    inc = det = lpi = rank = length = ids_ok = intact = True
    replay, sigma_err = {}, {}
    for name, (r1, r2) in runs.items():
        ub = name == "EBSS"
        inc &= all(torch.equal(r1.counts[i], G.recount(r1.hist_experts[i], E)) for i in range(len(starts)))
        det &= (torch.equal(r1.tokens, r2.tokens) and torch.equal(r1.cum_logp, r2.cum_logp)
                and torch.equal(r1.hist_experts, r2.hist_experts) and torch.equal(r1.final_score, r2.final_score))
        lpi &= bool(torch.allclose(r1.cum_logp, r1.token_logp.sum(1), rtol=1e-12, atol=0))
        fr, fb = _step_log_checks(r1.step_log, G.W_REF, ub)
        rank &= fr
        checks["ebss_includes_balance" if ub else "po_ignores_balance"] = fb
        length &= tuple(r1.tokens.shape) == (len(starts), P.SMOKE_LEN)
        ids_ok &= int(r1.tokens.max()) < VOCAB and int(r1.tokens.min()) >= 0
        fbm = r1.final_beams
        # sigma was reduced on the GPU; the CPU recompute differs only by reduction order (attempt 1 used atol=0
        # and failed on that alone: logs/smoke_attempt1_failed_parents_intact.json)
        sig_err = max(abs(float(G.balance_sigma(fbm["counts"][i, b], P.SMOKE_LEN)) - float(fbm["sigma"][i, b]))
                      for i in range(len(starts)) for b in range(G.W_REF))
        sigma_err[name] = sig_err
        intact &= all(int(fbm["counts"][i, b].sum()) == P.SMOKE_LEN * N_LAYERS * K_TOP
                      for i in range(len(starts)) for b in range(G.W_REF)) and sig_err <= 1e-12
        intact &= bool(torch.equal(r1.tokens[:, 0], torch.tensor(starts)))
        replay[name] = _replay_checks(be, r1)
        t = tok()
        out["methods"][name] = dict(
            tokens=r1.tokens.tolist(), text=[t.decode(x) for x in r1.tokens.tolist()],
            cum_logp=r1.cum_logp.tolist(), sigma=r1.sigma.tolist(), final_score=r1.final_score.tolist(),
            final_beam_scores=r1.final_beams["score"].tolist(), final_beam_sigma=r1.final_beams["sigma"].tolist(),
            final_pick=r1.final_beams["pick"].tolist(), step_log=r1.step_log, intervention=r1.intervention,
            special_tokens_generated=int(sum((r1.tokens[:, 1:] == s).sum() for s in (151643, 151644, 151645))),
            timing=r1.timing, replay=replay[name])
    checks.update(incremental_equals_recount=inc, deterministic=det, logprob_identity=lpi, ranking_recomputed=rank,
                  length_exact=length, token_ids_valid=ids_ok, parents_intact=intact,
                  no_chat_template=True,           # inputs are exactly [start id] + generated ids (checked above)
                  replay_logprob_close=all(v["mean_abs_dlogp"] <= 0.05 and v["pearson_r"] >= 0.99
                                           for v in replay.values()),
                  dispatch_mismatches=be.dispatch_mismatches)
    po, eb = runs["PROBABILITY_ONLY"][0], runs["EBSS"][0]
    out["control"] = dict(
        identical_outputs=bool(torch.equal(po.tokens, eb.tokens)),
        sentences_differing=int((po.tokens != eb.tokens).any(1).sum()),
        first_divergence_position=[int((po.tokens[i] != eb.tokens[i]).nonzero()[0]) if (po.tokens[i] != eb.tokens[i]).any()
                                   else None for i in range(len(starts))],
        same_candidate_universe="both score all w x |V| expansions of their live branches with the identical "
                                "probability term; EBSS adds sigma(prefix)/tau per parent",
        ebss_intervention=eb.intervention)
    out["checks"] = checks
    out["final_beam_sigma_cpu_vs_gpu_max_abs"] = sigma_err
    out["gate"] = P.smoke_gate(checks)
    out["passed"] = out["gate"]["passed"]
    jdump(out, p("smoke_test_results.json"))
    print(f"[smoke] gate {out['gate']}; replay {replay}; control {out['control']}")
    if not out["passed"]:
        raise SystemExit("smoke gate FAILED")


# ================================================================ feasibility
def stage_feasibility(a):
    if os.path.exists(p("feasibility.json")):
        print("[feasibility] done"); return
    need(p("smoke_test_results.json"))
    starts = json.load(open(p("initialization_manifest.json")))["pilot_start_ids"]
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    be, load = load_backend(check_dispatch=False)
    torch.cuda.reset_peak_memory_stats()
    probe_len = 12
    r = G.generate(be, starts, probe_len, G.W_REF, G.TAU_REF, True)
    steps = probe_len - 1
    per_step = (r.timing["total_s"]) / steps
    cfg = be.model.config
    est = P.memory_estimate(P.N_PILOT * G.W_REF, G.SEQLEN, cfg.num_hidden_layers, cfg.num_key_value_heads,
                            cfg.hidden_size // cfg.num_attention_heads)
    free, total = torch.cuda.mem_get_info()
    gpu_static = load["gpu_resident_gb"]
    need_gb = gpu_static + est["kv_cache_gb"] + 1.0               # + logits/activation headroom
    rec = dict(load=load, probe=dict(n_sentences=P.N_PILOT, w=G.W_REF, steps=steps, rows=P.N_PILOT * G.W_REF,
                                     total_s=r.timing["total_s"], per_step_s=per_step, timing=r.timing,
                                     peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9),
               gpu_total_gb=total / 1e9, kv_cache_at_512_gb=est["kv_cache_gb"], gpu_needed_estimate_gb=need_gb,
               host_mem_available_gb_at_start=int(mem["MemAvailable"].split()[0]) / 1e6,
               peak_cpu_gb=FD.peak_cpu_gb(),
               estimated_generation_s_per_method=per_step * (G.SEQLEN - 1) * 1.15,
               note="per-step time measured with 64 rows at short context; x1.15 allowance for attention growth "
                    "to 512. Generator precision BF16, weights unchanged (experts streamed from pinned host memory).")
    rec["passed"] = bool(need_gb < total / 1e9 * 0.95 and rec["estimated_generation_s_per_method"] < 4 * 3600)
    rec["status"] = "FEASIBLE" if rec["passed"] else "RESOURCE-BLOCKED"
    jdump(rec, p("feasibility.json"))
    print(f"[feasibility] {rec['status']}: {per_step:.2f} s/step -> ~{rec['estimated_generation_s_per_method'] / 60:.0f}"
          f" min per method; GPU need ~{need_gb:.1f} of {total / 1e9:.1f} GB")
    if not rec["passed"]:
        raise SystemExit("RESOURCE-BLOCKED")


# ================================================================ generate
def stage_generate(a):
    name = METHODS[a.method]
    pt = p("logs", f"generated_{a.method}.pt")
    if os.path.exists(pt):
        print(f"[generate] {name} done"); return
    need(p("feasibility.json"))
    starts = json.load(open(p("initialization_manifest.json")))["pilot_start_ids"]
    be, load = load_backend(check_dispatch=False)
    torch.cuda.reset_peak_memory_stats()
    kv_peak = [0]
    orig_step = be.step

    def step(tokens):
        out = orig_step(tokens)
        kv_peak[0] = max(kv_peak[0], be.kv_cache_bytes())
        return out
    be.step = step
    t0 = time.time()
    r = G.generate(be, starts, G.SEQLEN, G.W_REF, G.TAU_REF, name == "EBSS", record_steps=PILOT_RECORD_STEPS,
                   progress_every=64)
    wall = time.time() - t0
    t = tok()
    val = P.validate_generated(r.tokens, P.N_PILOT, G.SEQLEN, VOCAB)
    acc = P.token_accounting(P.N_PILOT, G.SEQLEN, G.W_REF, VOCAB)
    cost = dict(method=name, wall_s=wall, gpu_seconds=wall,
                gpu_seconds_note="one MIG slice used exclusively for the whole generation; GPU-seconds = wall",
                timing=r.timing, forward_calls=be.n_forward, rows_forwarded=be.rows_forwarded,
                avg_forward_batch=be.rows_forwarded / be.n_forward, bytes_host_to_device=be.bytes_transferred,
                h2d_gb_per_s=be.bytes_transferred / 1e9 / max(r.timing["expert_transfer_s"], 1e-9),
                accounting=acc, emitted_tokens_per_s=acc["emitted_tokens"] / wall,
                forward_rows_per_s=be.rows_forwarded / wall, scored_candidates_per_s=acc["scored_candidates"] / wall,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, kv_cache_peak_gb=kv_peak[0] / 1e9,
                peak_cpu_gb=FD.peak_cpu_gb(), load=load, validation=val)
    torch.save(dict(tokens=r.tokens, cum_logp=r.cum_logp, token_logp=r.token_logp, counts=r.counts,
                    hist_experts=r.hist_experts, sigma=r.sigma, final_score=r.final_score,
                    final_beams=r.final_beams, step_log=r.step_log, intervention=r.intervention,
                    config=r.config, method=name), pt)
    P.write_jsonl(r, p(f"generated_{'probability_only' if a.method == 'po' else 'ebss'}.jsonl"), t.decode)
    jdump(cost, p("logs", f"generation_cost_{a.method}.json"))
    print(f"[generate] {name}: {wall:.0f} s, valid {val['valid']}, sigma mean {r.sigma.mean():.5f}")
    if not val["valid"]:
        raise SystemExit(f"{name} generated set invalid")


# ================================================================ diagnostics (B1 capture + statistics)
def c4r_rows():
    out = {}
    for s, f in C4R_FILES.items():
        r = json.load(open(os.path.join(ROOT, f)))
        assert r["summary"]["B"] == 16 and r["summary"]["K"] == 4 and len(set(r["chosen"])) == 16
        out[s] = dict(chosen=list(map(int, r["chosen"])), source=f, sha256=sha256_file(os.path.join(ROOT, f)))
    return out


@torch.no_grad()
def forward_b1_sel(model, tokens):
    """CANONICAL-B1 forward (each row alone) through layers 0..23 + lm_head: stores layer inputs of 1/8/16/23,
    per-token top-4 experts at ALL 24 layers [n, T, 24, 4] and per-row NLL (511 targets)."""
    from moe_calib import model_utils as MU
    n, T = tokens.shape
    hs = [MU.embed(model, tokens[i:i + 1]).cpu() for i in range(n)]
    stored = {L: torch.empty(n, T, hs[0].shape[-1], dtype=hs[0].dtype) for L in LAYERS}
    sel_all = torch.zeros(n, T, N_LAYERS, K_TOP, dtype=torch.int16)
    for L in range(N_LAYERS):
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
                    sel_all[j, :, L] = torch.topk(torch.softmax(lg, dim=-1, dtype=torch.float), K_TOP,
                                                  dim=-1).indices.to(torch.int16).cpu()
            finally:
                h.remove()
    nll = np.zeros(n)
    for j in range(n):
        logits = model.lm_head(model.model.norm(hs[j].to(MU.GPU)))
        s, c = EP.nll_from_logits(logits, tokens[j:j + 1])
        nll[j] = float(s[0]) / int(c[0])
    return stored, sel_all, nll


def stage_capture(a):
    man_path = p("logs", "capture_manifest.json")
    if os.path.exists(man_path):
        print("[capture] done"); return
    gens = {m: torch.load(p("logs", f"generated_{m}.pt")) for m in METHODS}
    for m in METHODS:
        assert P.validate_generated(gens[m]["tokens"], P.N_PILOT, G.SEQLEN, VOCAB)["valid"]
    c4 = c4r_rows()
    allt = OV.c4_tokens()
    c4_union = sorted({r for v in c4.values() for r in v["chosen"]})
    tokens = torch.cat([allt[c4_union], gens["po"]["tokens"], gens["ebss"]["tokens"]])
    rows = dict(c4_rows=c4_union, n_c4=len(c4_union), po=list(range(len(c4_union), len(c4_union) + 16)),
                ebss=list(range(len(c4_union) + 16, len(c4_union) + 32)))
    model, t_load = FD.load_model_full()
    assert OV.verify_pristine_full(model)
    t0 = time.time()
    stored, sel_all, nll = forward_b1_sel(model, tokens)
    t_cap = time.time() - t0
    os.makedirs(P.CACHE, exist_ok=True)
    files = {}
    for L in LAYERS:
        path = os.path.join(P.CACHE, f"L{L}.pt")
        torch.save(stored[L], path + ".tmp"); os.replace(path + ".tmp", path)
        files[str(L)] = dict(shape=list(stored[L].shape), sha256=sha256_file(path))
    json.dump(dict(pool_size=0, holdout_size=0, seqlen=512, layers=LAYERS, policy="CANONICAL-B1", n_rows=len(tokens)),
              open(os.path.join(P.CACHE, "meta.json"), "w"), indent=1)
    torch.save(dict(tokens=tokens, rows=rows), os.path.join(P.CACHE, "ids.pt"))
    torch.save(dict(sel=sel_all, nll=torch.tensor(nll)), p("logs", "capture_routing_nll.pt"))
    # extraction regression: C4 rows also present in the official-artifact study's CANONICAL-B1 cache must match
    pol = json.load(open(os.path.join(ROOT, "runs_ebss_official_validation", "execution_policy.json")))
    opos = {r: i for i, r in enumerate(pol["rows"]["c4_rows"])}
    shared = [(i, opos[r]) for i, r in enumerate(c4_union) if r in opos]
    reg = {}
    for L in LAYERS:
        R = torch.load(os.path.join(os.path.expanduser("~/moe_cache_ebss_official_validation"), f"L{L}.pt"), mmap=True)
        reg[str(L)] = bool(all(torch.equal(stored[L][i], R[j]) for i, j in shared)) if shared else None
    man = dict(cache_dir=P.CACHE, rows=rows, files=files, capture_s=t_cap, model_load_s=t_load,
               c4_random=c4, extraction_regression=dict(n_shared_rows=len(shared), bitwise_equal=reg,
                                                        passed=bool(shared) and all(reg.values())),
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb())
    man["passed"] = man["extraction_regression"]["passed"]
    jdump(man, man_path)
    print(f"[capture] {len(tokens)} rows in {t_cap:.0f} s; regression {man['extraction_regression']}")
    if not man["passed"]:
        raise SystemExit("CANONICAL-B1 extraction regression FAILED")


def stage_diagnostics(a):
    stage_capture(a)
    import diagnostics_ebss_pilot as DG
    DG.run(ROOT)


# ================================================================ quantize
def stage_quantize(a):
    name = a.config
    assert name in QCONFIGS
    os.makedirs(p("results"), exist_ok=True)
    if os.path.exists(p("results", f"{name}.json")):
        print(f"[quant] {name} done"); return
    man = need(p("logs", "capture_manifest.json"))
    need(p("logs", "diagnostics_gate.json"))
    for L in LAYERS:
        assert sha256_file(os.path.join(P.CACHE, f"L{L}.pt")) == man["files"][str(L)]["sha256"], "cache changed"
    rows = man["rows"]
    if name.startswith("C4R"):
        s = int(name[-1])
        pos = {r: i for i, r in enumerate(rows["c4_rows"])}
        chosen = [pos[r] for r in man["c4_random"][str(s)]["chosen"]]
    else:
        chosen = rows["po" if name == "PO" else "ebss"]
    assert len(chosen) == 16 == len(set(chosen))
    model, t_load = FD.load_model_full()
    assert OV.verify_pristine_full(model)
    r = OV.quant_config(model, name, P.CACHE, chosen, OV.OCACHE, 4)
    r.update(config=name, policy="CANONICAL-B1", cache_rows=chosen, model_load_s=t_load, B=16)
    jdump(r, p("results", f"{name}.json"))
    print(f"[quant] {name}: recon {r['recon']:.7f} PPL {r['ppl']:.6f} RTN fallbacks {r['rtn_fallbacks']} "
          f"valid {r['valid']} ({r['quantize_total_s']:.0f} s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--method", choices=list(METHODS))
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage == "report":
        import report_ebss_generation_pilot as R
        return R.main()
    dict(preflight=stage_preflight, source_review=stage_source_review, smoke=stage_smoke,
         feasibility=stage_feasibility, generate=stage_generate, diagnostics=stage_diagnostics,
         quantize=stage_quantize)[a.stage](a)


if __name__ == "__main__":
    main()
