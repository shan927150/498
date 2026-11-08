"""Partial-model WikiText-2 perplexity for the frozen B = 128 matched selections.

PARTIAL model: only the 240 routed experts of layers 1/8/16/23 are GPTQ
fake-quantized (frozen E3 solver and settings, non-sequential calibration on
cached BF16 layer inputs); attention, routers, shared experts, embeddings,
norms, lm_head and the other 20 layers stay BF16. Inference is a normal
SEQUENTIAL forward through all 24 layers (one decoder layer on the GPU at a
time; the model lives in CPU RAM), so quantization error propagates.

Stages (python diagnostics/eval_partial_ppl.py --stage S):
  protocol   freeze the WikiText-2 windows (128 x 512 tokens), fingerprints and
             the decision rule -> evaluation_protocol.json, windows.pt
  manifest   validate the 12 frozen selections -> selected_set_manifest.json
  bf16       BF16 baseline, evaluated twice in one process -> bf16_baseline.json
  reference  independent HF forward (accelerate offload, labels=...) on 2 windows
  quant      [--only FR:0 ...] quantize + install + verify + PPL per config,
             resumable; then the FR s0 repeatability rerun and a final BF16 check
PPL = exp(sum NLL / number of scored targets); targets are input_ids[:, 1:],
logits[:, :-1]; the first token of every window is context only.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, sys, time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

OUT = "runs_partial_ppl"
MATCHED = "runs_typicality_matched"
POOLDIR = "runs_typicality"
CONFIRM = "runs_aggregate_constraint/eval"
FROZEN = "configs/frozen_b4.json"
SEQLEN, N_WIN, WIN_SEED, BATCH = 512, 128, 20261108, 8
STRATS = {"FR": ("full", "random"), "TR": ("typical_half", "random"),
          "FC": ("full", "coverage_lp"), "TC": ("typical_half", "coverage_lp")}
NAMES = {"FR": "Full Random", "TR": "Typical Random", "FC": "Full Coverage-LP", "TC": "Typical Coverage-LP"}
SEEDS = (0, 1, 2)
ORDER = [("FR", 0)] + [(c, s) for s in SEEDS for c in STRATS if (c, s) != ("FR", 0)]
CHECK_EXPERTS = [(L, e) for L in (1, 8, 16, 23) for e in (0, 30)]     # prespecified, not outcome-based
N_CHECK_SEQS = 4
# Installed-module vs E3 FP32 output tolerance (amended after smoke attempt 1, before any quantized PPL;
# logs/amendment_install_tolerance.json): the deviation may not exceed what BF16 weight storage plus BF16
# arithmetic produce -- weight-rounding-only deviation + 1.5 x the BF16-compute deviation of the SAME module
# with its ORIGINAL weights on the same inputs. Weights must also be bitwise BF16(W_q).
INSTALL_TOL = lambda ck: ck["weight_rounding_only_rel"] + 1.5 * ck["bf16_compute_noise_rel"]

DECISION_RULE = {
    "tau": "3 x max(|BF16 run1 - run2|, |BF16 across processes|, |FR s0 - FR s0 repeat|) in PPL units, "
           "floored at 1e-4 x PPL_BF16",
    "PPL-RECON-ALIGNED": "PPL(FC) - PPL(FR) > tau on 3/3 seeds AND FC has the highest mean PPL of the four "
                         "AND Spearman(recon, PPL) over the 12 configs >= 0.5",
    "PPL-RECON-DISAGREE": "PPL(FC) - PPL(FR) < -tau on 3/3 seeds OR Spearman(recon, PPL) <= -0.5",
    "PPL-RECON-WEAKLY-RELATED": "otherwise (differences within tau, inconsistent across seeds, or ranking "
                                "not reproduced)",
    "PPL-RUN-BLOCKED": "BF16 baseline or any of the 12 configurations missing or failing a correctness check",
    "recon_reference": "matched-warm-start old-test-split reconstruction (runs_typicality_matched); the locked "
                       "confirmation reconstruction is reported alongside",
}


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tensor_fp(t):
    return hashlib.sha256(t.contiguous().numpy().astype(np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------- loss
def nll_from_logits(logits, ids, mask=None):
    """Per-row (sum NLL [float64], n scored targets). Target t+1 is scored iff
    both positions t and t+1 are real tokens (mask = 1); padding never counts."""
    B, T, V = logits.shape
    lp = logits[:, :-1].float()
    tgt = ids[:, 1:].to(lp.device)
    nll = F.cross_entropy(lp.reshape(-1, V), tgt.reshape(-1), reduction="none").view(B, T - 1).double()
    if mask is None:
        valid = torch.ones_like(nll, dtype=torch.bool)
    else:
        m = mask.to(lp.device).bool()
        valid = m[:, 1:] & m[:, :-1]
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return nll.sum(1), valid.sum(1)


@torch.no_grad()
def layerwise_nll(model, windows, batch=BATCH):
    """Sequential full-model forward, one decoder layer on the GPU at a time.
    windows: [N, T] long (full, unpadded). Returns per-window (nll, n)."""
    from moe_calib import model_utils as MU
    N, T = windows.shape
    hs = [MU.embed(model, windows[i:i + batch]) for i in range(0, N, batch)]
    for L in range(MU.num_layers(model)):
        with MU.layer_on_gpu(model, L):
            hs = [MU.run_layer(model, L, h) for h in hs]
    nll, cnt = [], []
    for bi, h in enumerate(hs):
        for j in range(h.shape[0]):                      # one window of logits at a time (152k vocab)
            logits = model.lm_head(model.model.norm(h[j:j + 1]))
            s, n = nll_from_logits(logits, windows[bi * batch + j:bi * batch + j + 1])
            nll.append(float(s[0])); cnt.append(int(n[0]))
        hs[bi] = None
    return np.array(nll, dtype=np.float64), np.array(cnt, dtype=np.int64)


def ppl_record(nll, cnt, seconds):
    tot, n = float(nll.sum()), int(cnt.sum())
    return dict(total_nll=tot, total_tokens=n, mean_nll=tot / n, ppl=math.exp(tot / n),
                per_window_nll=nll.tolist(), per_window_tokens=cnt.tolist(), eval_s=seconds)


# ---------------------------------------------------------------- protocol
def stage_protocol(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import datasets, transformers
    from moe_calib import model_utils as MU
    path = MU._snapshot_dir(MU.MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(path)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids[0]
    n_full = enc.shape[0] // SEQLEN
    assert n_full >= N_WIN, f"only {n_full} complete windows"
    idx = sorted(random.Random(WIN_SEED).sample(range(n_full), N_WIN))
    win = torch.stack([enc[i * SEQLEN:(i + 1) * SEQLEN] for i in idx])
    cache_files = [f["filename"] for f in getattr(ds, "cache_files", [])]
    prot = dict(
        benchmark="WikiText-2 raw (wikitext-2-raw-v1), test split",
        dataset=dict(hf_id="wikitext", config="wikitext-2-raw-v1", split="test", n_rows=len(ds),
                     datasets_version=datasets.__version__, fingerprint=getattr(ds, "_fingerprint", None),
                     cache_revision=[c for c in cache_files[0].split(os.sep) if len(c) == 40][0] if cache_files else None,
                     text_sha256=hashlib.sha256(text.encode()).hexdigest()),
        tokenizer=dict(name=MU.MODEL_NAME, snapshot=os.path.basename(path), cls=type(tok).__name__,
                       transformers_version=transformers.__version__, bos_token=tok.bos_token,
                       eos_token=tok.eos_token, adds_special_tokens=tok("a").input_ids != tok("a", add_special_tokens=False).input_ids),
        preprocessing='"\\n\\n".join(test["text"]); one tokenizer call on the whole text; no special tokens are added',
        packing=f"non-overlapping {SEQLEN}-token windows i*{SEQLEN}..(i+1)*{SEQLEN}; {n_full} complete windows; "
                f"{N_WIN} sampled by random.Random({WIN_SEED}).sample, sorted; tail discarded",
        context_length=SEQLEN, n_windows=N_WIN, window_indices=idx,
        token_offsets=[i * SEQLEN for i in idx], total_corpus_tokens=int(enc.shape[0]),
        scored_targets_per_window=SEQLEN - 1, scored_targets_total=N_WIN * (SEQLEN - 1),
        scoring="logits[:, :-1] vs input_ids[:, 1:]; the first token of each window is context only; "
                "FP32 cross entropy, FP64 accumulation; PPL = exp(total NLL / total targets); no attention mask "
                "(complete windows, no padding); positions restart at 0 in every window",
        windows_sha256=tensor_fp(win), decision_rule=DECISION_RULE,
        model_scope="PARTIAL: GPTQ fake-quantized routed experts (gate/up/down) of layers 1/8/16/23 only; "
                    "everything else BF16",
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(p("logs"), exist_ok=True)
    if os.path.exists(p("evaluation_protocol.json")):
        old = json.load(open(p("evaluation_protocol.json")))
        assert old["windows_sha256"] == prot["windows_sha256"], "protocol windows changed; refusing"
        print("protocol already frozen; identical windows"); return
    torch.save(win, p("windows.pt"))
    json.dump(prot, open(p("evaluation_protocol.json"), "w"), indent=1)
    print(f"protocol frozen: {N_WIN} windows of {SEQLEN}, {prot['scored_targets_total']} targets, "
          f"sha {prot['windows_sha256'][:16]}")


def load_windows():
    prot = json.load(open(p("evaluation_protocol.json")))
    win = torch.load(p("windows.pt"))
    assert tensor_fp(win) == prot["windows_sha256"] and tuple(win.shape) == (N_WIN, SEQLEN)
    return win, prot


# ---------------------------------------------------------------- selections
def stage_manifest(a):
    import e3_allocation as E3
    import report_typicality_matched as RTM
    cfg = json.load(open(os.path.join(ROOT, FROZEN)))
    ws = json.load(open(os.path.join(ROOT, MATCHED, "warm_starts.json")))
    typ = set(json.load(open(os.path.join(ROOT, POOLDIR, "pool_typical_half.json")))["ids"])
    lo, hi = cfg["split"]["pool"]
    man = dict(frozen_config=cfg, frozen_config_sha256=sha256_file(os.path.join(ROOT, FROZEN)), configs={})
    for c, (pool, strat) in STRATS.items():
        for s in SEEDS:
            path = RTM.run_path(os.path.join(ROOT, MATCHED), pool, strat, s, ws)
            r = json.load(open(path))
            S, ch, W = r["summary"], r["chosen"], ws["seeds"][str(s)]["ids"]
            checks = dict(
                n_unique_128=len(ch) == 128 == len(set(ch)),
                warm_ids_and_order=r["warm_start"] == W and ch[:len(W)] == W,
                warm_hash=S["warm_start_fixed"]["hash"] == ws["seeds"][str(s)]["hash"] == E3.warm_hash(W),
                pool_membership=all(lo <= x < hi for x in ch) and (pool == "full" or set(ch) <= typ),
                no_tune_test_overlap=all(x < cfg["split"]["tune"][0] for x in ch),
                frozen_gptq=(S["bits"], S.get("group_size", 128), S["percdamp"], S["layers"]) ==
                            (cfg["gptq"]["bits"], cfg["gptq"]["group_size"], cfg["gptq"]["percdamp"], cfg["layers"]),
                strategy=S["strategy"] == strat, seed=S["seed"] == s, B=S["B"] == 128,
                no_rtn_fallback=S["n_zero_token_quantization_fallback"] == 0)
            conf = os.path.join(ROOT, CONFIRM, f"{c}_s{s}.json")
            cr = json.load(open(conf)) if os.path.exists(conf) else None
            if cr:
                checks["confirmation_run_same_ids"] = cr["ids_hash"] == E3.warm_hash(ch)
            man["configs"][f"{c}_s{s}"] = dict(
                strategy=c, name=NAMES[c], seed=s, source=os.path.relpath(path, ROOT), source_sha256=sha256_file(path),
                ids_hash=E3.warm_hash(ch), K=S["K"], recon_test=S["recon_mean"],
                recon_confirmation=cr["recon_mean"] if cr else None, checks=checks, valid=all(checks.values()),
                chosen=ch)
    bad = [k for k, v in man["configs"].items() if not v["valid"]]
    json.dump(man, open(p("selected_set_manifest.json"), "w"), indent=1)
    assert not bad, f"invalid selections: {bad}"
    print("manifest ok: 12 selections valid (ids, warm start, pools, no leakage, frozen GPTQ)")


# ---------------------------------------------------------------- model stages
def load_full_model():
    os.environ.pop("MOE_LOAD_LAYERS", None)
    from moe_calib import model_utils as MU
    t0 = time.time()
    model, tok = MU.load_model()
    return model, time.time() - t0


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def stage_bf16(a):
    win, prot = load_windows()
    model, t_load = load_full_model()
    runs = []
    for r in range(2):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        nll, cnt = layerwise_nll(model, win)
        rec = ppl_record(nll, cnt, time.time() - t0)
        rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        runs.append(rec)
        print(f"[bf16] run {r}: PPL {rec['ppl']:.6f} ({rec['eval_s']:.0f} s)", flush=True)
    d = dict(protocol_windows_sha256=prot["windows_sha256"], model_load_s=t_load, peak_cpu_gb=peak_cpu_gb(),
             runs=runs, ppl=runs[0]["ppl"], repeat_abs_diff_ppl=abs(runs[0]["ppl"] - runs[1]["ppl"]),
             repeat_max_abs_window_nll_diff=float(np.max(np.abs(np.array(runs[0]["per_window_nll"]) -
                                                                np.array(runs[1]["per_window_nll"])))),
             precision="BF16 weights and activations; FP32 logits/loss; FP64 accumulation",
             offload="model in CPU RAM; one decoder layer moved to the GPU at a time; embed/norm/lm_head on GPU")
    json.dump(d, open(p("bf16_baseline.json"), "w"), indent=1)


def stage_reference(a):
    """Independent path: HF AutoModelForCausalLM with accelerate offload and its
    own loss (labels=input_ids), on the first 2 protocol windows."""
    from transformers import AutoModelForCausalLM
    from moe_calib import model_utils as MU
    win, prot = load_windows()
    base = json.load(open(p("bf16_baseline.json")))
    model = AutoModelForCausalLM.from_pretrained(
        MU._snapshot_dir(MU.MODEL_NAME), torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="auto", max_memory={0: "15GiB", "cpu": "120GiB"}, low_cpu_mem_usage=True).eval()
    out = []
    for i in range(2):
        ids = win[i:i + 1].to("cuda")
        with torch.no_grad():
            r = model(input_ids=ids, labels=ids, use_cache=False)
        hf_mean = float(r.loss)                                    # HF: mean over 511 shifted targets
        s, n = nll_from_logits(r.logits, ids)                       # our scorer on HF's logits
        lw = base["runs"][0]["per_window_nll"][i] / base["runs"][0]["per_window_tokens"][i]
        out.append(dict(window=i, hf_loss_mean_nll=hf_mean, our_scorer_on_hf_logits=float(s[0]) / int(n[0]),
                        layerwise_mean_nll=lw, abs_diff_layerwise_vs_hf=abs(lw - hf_mean),
                        rel_diff_layerwise_vs_hf=abs(lw - hf_mean) / hf_mean))
        print(out[-1], flush=True)
    json.dump(dict(windows=out, note="independent HF forward with device_map offload (batch 1) vs the layer-wise "
                                     "evaluator (batch 8)"), open(p("reference_check.json"), "w"), indent=1)


def run_name(c, s):
    return f"ppl_{c}_s{s}"


def result_path(c, s, tag=""):
    return p(f"{run_name(c, s)}{tag}.json")


def quantize_one(model, c, s, man, win, prot, cache_dir, snap, untouched0, tag=""):
    from moe_calib import partial_model_quant as PQ
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_meta, split_ids, load_layer_cache
    cfg = man["frozen_config"]
    ent = man["configs"][f"{c}_s{s}"]
    assert ent["valid"]
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot, f"model not pristine before {c} s{s}: {eq}/{tot}"
    meta = load_meta(cache_dir)
    test_ids = split_ids(meta, cfg["split"]["tune"][1] - cfg["split"]["tune"][0])["test"]
    assert test_ids == list(range(*cfg["split"]["test"]))
    layers, g = cfg["layers"], cfg["gptq"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    q = PQ.quantize_install(model, cache_dir, layers, ent["chosen"], test_ids, ent["K"], g["bits"],
                            g["group_size"], g["percdamp"], capture=CHECK_EXPERTS)
    t_quant = time.time() - t0
    # 1. E3 scores of the installed weights reproduce the frozen run exactly
    stored = json.load(open(os.path.join(ROOT, ent["source"])))
    sm = {(r["layer"], r["expert"]): r["mean"] for r in stored["per_expert"]}
    now = {(r["layer"], r["expert"]): r["mean"] for r in q["results"]}
    assert set(sm) == set(now)
    max_rel = max(abs(now[k] - sm[k]) / sm[k] for k in sm)
    recon_now = float(np.mean(list(now.values())))
    # 2. untouched parameters bitwise unchanged
    untouched = PQ.untouched_digests(model, layers)
    n_changed = sum(untouched[k] != untouched0[k] for k in untouched0)
    # 3. targets all changed
    eq_after, _ = PQ.experts_match(model, snap)
    # 4. expert-level output check on identical inputs (test-split layer inputs)
    checks = []
    for L, e in CHECK_EXPERTS:
        X = PQ.expert_inputs(model, L, e, load_layer_cache(cache_dir, L)[test_ids[:N_CHECK_SEQS]])
        checks.append(PQ.expert_output_check(model, L, e, q["captured"][(L, e)], snap[(L, e)], X))
    install_ok = (max_rel <= 1e-6 and n_changed == 0 and eq_after == 0 and q["counts"]["n_zero_token_quantization_fallback"] == 0
                  and all(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                          and ck["installed_vs_e3_rel"] <= INSTALL_TOL(ck) for ck in checks))
    digest = PQ.installed_digest(model, layers)
    if not install_ok:
        json.dump(dict(strategy=c, seed=s, install_ok=False, recon_max_rel_diff=max_rel, untouched_changed=n_changed,
                       experts_still_pristine=eq_after, expert_checks=checks), open(result_path(c, s, tag + "_FAILED"), "w"), indent=1)
        raise SystemExit(f"installation check failed for {c} s{s}; not evaluating PPL")
    # 5. sequential full-model PPL
    t1 = time.time()
    nll, cnt = layerwise_nll(model, win)
    rec = ppl_record(nll, cnt, time.time() - t1)
    peak = torch.cuda.max_memory_allocated() / 1e9
    # 6. restore pristine and verify
    PQ.restore_experts(model, snap)
    eq_rest, tot = PQ.experts_match(model, snap)
    assert eq_rest == tot, "restore failed"
    out = dict(strategy=c, name=NAMES[c], seed=s, config_id=f"{run_name(c, s)}|ids:{ent['ids_hash']}|"
               f"frozen:{man['frozen_config_sha256'][:12]}|win:{prot['windows_sha256'][:12]}",
               ids_hash=ent["ids_hash"], model_scope=prot["model_scope"],
               quantized_experts=q["counts"]["n_experts_quantized_total"],
               rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"], install_ok=True,
               recon_recomputed=recon_now, recon_stored=ent["recon_test"], recon_max_rel_diff=max_rel,
               recon_confirmation=ent["recon_confirmation"], untouched_params_changed=n_changed,
               n_untouched_params_checked=len(untouched0), installed_weights_sha256=digest,
               expert_checks=checks, restored_pristine=True,
               hessian_s=q["hessian_s"], gptq_s=q["gptq_s"], quantize_wall_s=t_quant, peak_gpu_gb=peak,
               peak_cpu_gb=peak_cpu_gb(), quant_artifact_bytes_if_packed=int(240 * 3 * 2048 * 1408 * 0.5 +
                                                                         240 * 3 * (2048 * 1408 // 128) * 2 * 2),
               **rec)
    json.dump(out, open(result_path(c, s, tag), "w"), indent=1)
    print(f"[quant] {c} s{s}{tag}: PPL {rec['ppl']:.6f}  recon {recon_now:.6f} (stored {ent['recon_test']:.6f}, "
          f"max rel diff {max_rel:.1e}); quant {t_quant:.0f} s, PPL {rec['eval_s']:.0f} s", flush=True)
    return out


def valid_result(c, s, man, prot, tag=""):
    f = result_path(c, s, tag)
    if not os.path.exists(f):
        return False
    r = json.load(open(f))
    return (r.get("install_ok") and r.get("ids_hash") == man["configs"][f"{c}_s{s}"]["ids_hash"]
            and prot["windows_sha256"][:12] in r.get("config_id", "") and r.get("total_tokens") == N_WIN * (SEQLEN - 1))


def stage_quant(a):
    from moe_calib import partial_model_quant as PQ
    win, prot = load_windows()
    man = json.load(open(p("selected_set_manifest.json")))
    base = json.load(open(p("bf16_baseline.json")))
    todo = ORDER if not a.only else [(x.split(":")[0], int(x.split(":")[1])) for x in a.only]
    pending = [(c, s) for c, s in todo if not valid_result(c, s, man, prot)]
    rep_needed = not a.only and not valid_result("FR", 0, man, prot, "_repeat")
    if not pending and not rep_needed:
        print("all requested configurations already valid"); return
    model, t_load = load_full_model()
    layers = man["frozen_config"]["layers"]
    snap = PQ.snapshot_experts(model, layers)
    untouched0 = PQ.untouched_digests(model, layers)
    t0 = time.time()
    nll, cnt = layerwise_nll(model, win)
    chk = ppl_record(nll, cnt, time.time() - t0)
    cross = abs(chk["ppl"] - base["ppl"])
    print(f"[quant] process BF16 check: PPL {chk['ppl']:.6f} vs baseline {base['ppl']:.6f} (|d| {cross:.2e})", flush=True)
    log = dict(model_load_s=t_load, bf16_check_ppl=chk["ppl"], bf16_cross_process_abs_diff=cross,
               bf16_check_eval_s=chk["eval_s"])
    for c, s in pending:
        quantize_one(model, c, s, man, win, prot, a.cache_dir, snap, untouched0)
    if rep_needed:
        r = quantize_one(model, "FR", 0, man, win, prot, a.cache_dir, snap, untouched0, tag="_repeat")
        first = json.load(open(result_path("FR", 0)))
        print(f"[quant] repeat FR s0: PPL diff {abs(r['ppl'] - first['ppl']):.2e}, identical weights "
              f"{r['installed_weights_sha256'] == first['installed_weights_sha256']}", flush=True)
    if not a.only:
        nll, cnt = layerwise_nll(model, win)
        end = ppl_record(nll, cnt, 0.0)
        log.update(bf16_end_ppl=end["ppl"], bf16_end_abs_diff=abs(end["ppl"] - base["ppl"]),
                   untouched_changed_at_end=sum(v != untouched0[k] for k, v in PQ.untouched_digests(model, layers).items()))
        print(f"[quant] BF16 after all runs: PPL {end['ppl']:.6f}", flush=True)
    name = "quant_process_" + ("smoke" if a.only else "main") + ".json"
    json.dump(log, open(p("logs", name), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["protocol", "manifest", "bf16", "reference", "quant"])
    ap.add_argument("--only", nargs="*", default=None, help="e.g. FR:0")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    os.makedirs(p("figures"), exist_ok=True)
    dict(protocol=stage_protocol, manifest=stage_manifest, bf16=stage_bf16, reference=stage_reference,
         quant=stage_quant)[a.stage](a)


if __name__ == "__main__":
    main()
