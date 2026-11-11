"""Statistics of the EBSS generation pilot: expert exposure (all 24 layers), balance progression over positions,
diversity / repetition, representativeness vs the frozen C4 reference, contamination, generation cost.
Called by `ebss_generation_pilot.py --stage diagnostics` after the CANONICAL-B1 capture."""
from __future__ import annotations

import json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import ebss_generator as G           # noqa: E402
from moe_calib import ebss_official_data as ED      # noqa: E402
from moe_calib import ebss_pilot as P               # noqa: E402
from moe_calib import typicality as TY              # noqa: E402

LAYERS = [1, 8, 16, 23]
NL, E, K = 24, 60, 4


def p(*a):
    return os.path.join(ROOT, P.OUT, *a)


def counts_from_sel(sel, upto=None):
    """sel [n, T, L, k] -> summed counts [L, E] over the first `upto` positions of every row."""
    s = sel[:, :upto] if upto else sel
    n, T, L, k = s.shape
    out = np.zeros((L, E), dtype=np.int64)
    for l in range(L):
        out[l] = np.bincount(s[:, :, l, :].reshape(-1).astype(np.int64), minlength=E)
    return out


def row_counts_4layers(sel):
    """per-row counts at layers 1/8/16/23 in the [n, 4 * E] layout of routing_profile_v2."""
    n = sel.shape[0]
    C = np.zeros((n, len(LAYERS) * E))
    for i in range(n):
        for li, L in enumerate(LAYERS):
            C[i, li * E:(li + 1) * E] = np.bincount(sel[i, :, L, :].reshape(-1).astype(np.int64), minlength=E)
    return C


def exposure_block(cnt):
    se = G.set_exposure(cnt)
    four = {kk: float(np.mean([se["per_layer"][L][kk] for L in LAYERS]))
            for kk in ("sd", "cv", "gini", "min", "p5", "p10", "max", "n_zero")}
    four["paper_sigma"] = float(np.mean([se["paper_sigma_per_layer"][L] for L in LAYERS]))
    se["layers_1_8_16_23"] = four
    se["histogram"] = cnt.tolist()
    return se


def longest_repeated_ngram(ids):
    """largest n such that some n-gram occurs at least twice in the sequence (overlaps allowed)."""
    ids = list(map(int, ids))
    lo, hi = 0, len(ids) - 1

    def has(n):
        seen = set()
        for i in range(len(ids) - n + 1):
            t = tuple(ids[i:i + n])
            if t in seen:
                return True
            seen.add(t)
        return False
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def text_block(W, nll, tokzr):
    W = [list(map(int, w)) for w in W]
    td = ED.text_diagnostics(W)
    lr = np.array([longest_repeated_ngram(w) for w in W])
    cjk = sum(any("一" <= ch <= "鿿" for ch in tokzr.decode(w)) for w in W)
    spec = sum(sum(1 for t in w[1:] if t in (151643, 151644, 151645)) for w in W)
    return dict(td, distinct3_mean=P.distinct3(W), set_distinct3=ED.set_distinct_n(W, 3),
                longest_repeated_ngram=dict(mean=float(lr.mean()), median=float(np.median(lr)), max=int(lr.max()),
                                            per_sequence=lr.tolist()),
                near_duplicates=ED.near_duplicate_stats(W),
                model_nll=dict(per_sequence=[float(x) for x in nll], mean=float(np.mean(nll)),
                               corpus_ppl=float(np.exp(np.mean(nll))), median=float(np.median(nll)),
                               min=float(np.min(nll)), max=float(np.max(nll))),
                sequence_length=dict(all_equal=len({len(w) for w in W}) == 1, lengths=sorted({len(w) for w in W})),
                windows_with_cjk=int(cjk), special_tokens_generated=int(spec))


def representativeness(sel, ref, d_star):
    C = row_counts_4layers(sel)
    sc = TY.sequence_scores(C, len(LAYERS), E, ref=ref)
    agg = C.sum(0)
    per = {str(L): ED.routing_l1(agg[li * E:(li + 1) * E], ref[li]) for li, L in enumerate(LAYERS)}
    return dict(routing_l1_per_layer=per, routing_l1_mean=float(np.mean(list(per.values()))),
                frac_atypical=float(np.mean(sc["d"] > d_star)), mean_d=float(np.mean(sc["d"])),
                sequence_routing_entropy_mean=float(np.mean(sc["entropy"])),
                set_js=TY.selected_set_js(C, ref, len(LAYERS), E))


def run(root=ROOT):
    from transformers import AutoTokenizer
    import official_ebss_validation as OV
    import eval_partial_ppl as EP
    tokzr = AutoTokenizer.from_pretrained(P.MODEL)
    man = json.load(open(p("logs", "capture_manifest.json")))
    cap = torch.load(p("logs", "capture_routing_nll.pt"))
    sel_b1, nll = cap["sel"].numpy(), cap["nll"].numpy()
    rows = man["rows"]
    toks = torch.load(os.path.join(P.CACHE, "ids.pt"))["tokens"]
    gens = {m: torch.load(p("logs", f"generated_{m}.pt")) for m in ("po", "ebss")}
    names = {"po": "PROBABILITY_ONLY", "ebss": "EBSS"}
    pos = {r: i for i, r in enumerate(rows["c4_rows"])}
    sets = {"PROBABILITY_ONLY": rows["po"], "EBSS": rows["ebss"]}
    for s, v in man["c4_random"].items():
        sets[f"C4R_s{s}"] = [pos[r] for r in v["chosen"]]
    prof = np.load(os.path.join(root, "runs_low_budget", "routing_profile_v2.npz"))
    assert list(prof["layers"]) == LAYERS
    ref = TY.pool_reference(prof["C"], len(LAYERS), E)
    pool = TY.sequence_scores(prof["C"], len(LAYERS), E, ref=ref)
    typ_ids = json.load(open(os.path.join(root, "runs_typicality", "pool_typical_half.json")))["ids"]
    d_star = float(np.nanmax(pool["d"][np.array(typ_ids)]))

    routing = dict(definition=dict(
        exposure="routed tokens per expert (top-4 membership, each token once per selected expert) summed over the "
                 "16 x 512 tokens of a set; CANONICAL-B1 replay routing (decision source) and generation-time "
                 "batched-KV routing", paper_sigma="Eq. 3/4: mean over layers of std_ddof1(counts / tokens)",
        cv="std_ddof0 / mean of per-expert counts, mean over layers", uniform_mean=16 * 512 * K / E,
        progression_positions=list(P.PROGRESS_POSITIONS)), sets={})
    progress = {}
    for name, rr in sets.items():
        s = sel_b1[rr]
        blk = dict(b1=exposure_block(counts_from_sel(s)))
        prog = {}
        for t in P.PROGRESS_POSITIONS:
            se = G.set_exposure(counts_from_sel(s, t))
            per_seq = [float(G.balance_sigma(torch.as_tensor(counts_from_sel(s[i:i + 1], t)), t))
                       for i in range(s.shape[0])]
            prog[str(t)] = dict(set_paper_sigma=se["paper_sigma"], set_cv=se["mean_over_layers"]["cv"],
                                set_min=se["mean_over_layers"]["min"], per_sequence_sigma_mean=float(np.mean(per_seq)))
        blk["progression_b1"] = prog
        blk["representativeness"] = representativeness(s, ref, d_star)
        routing["sets"][name] = blk
    agree = {}
    for m in ("po", "ebss"):
        g = gens[m]
        gh = g["hist_experts"].numpy()
        name = names[m]
        routing["sets"][name]["generation_time"] = exposure_block(counts_from_sel(gh))
        routing["sets"][name]["progression_generation_time"] = {
            str(t): dict(set_paper_sigma=G.set_exposure(counts_from_sel(gh, t))["paper_sigma"],
                         set_cv=G.set_exposure(counts_from_sel(gh, t))["mean_over_layers"]["cv"])
            for t in P.PROGRESS_POSITIONS}
        a = np.sort(gh, -1) == np.sort(sel_b1[sets[name]], -1)
        agree[name] = dict(token_layer_set_agreement=float(a.all(-1).mean()),
                           per_layer=[float(a.all(-1)[:, :, l].mean()) for l in range(NL)])
        routing["sets"][name]["generation_sigma_per_sequence"] = g["sigma"].tolist()
        routing["sets"][name]["intervention"] = g["intervention"]
    routing["generation_vs_b1_routing"] = agree
    # official artifact (4 layers only, B=128) as descriptive context
    od = json.load(open(os.path.join(root, "runs_ebss_official_validation", "routing_diagnostics.json")))["configs"]
    routing["context_official_artifact_B128_4layers"] = {
        c: dict(mean_over_layers=od[c]["mean_over_layers"], typicality=od[c]["typicality"],
                model_ppl=od[c]["model_nll"]["ppl"]) for c in ("EBSS", "FR_s0")}
    po, eb = routing["sets"]["PROBABILITY_ONLY"]["b1"], routing["sets"]["EBSS"]["b1"]
    layer_cv = [(eb["per_layer"][l]["cv"] - po["per_layer"][l]["cv"]) for l in range(NL)]
    routing["ebss_minus_po"] = dict(
        paper_sigma=eb["paper_sigma"] - po["paper_sigma"], cv=eb["mean_over_layers"]["cv"] - po["mean_over_layers"]["cv"],
        gini=eb["mean_over_layers"]["gini"] - po["mean_over_layers"]["gini"],
        min=eb["mean_over_layers"]["min"] - po["mean_over_layers"]["min"],
        p10=eb["mean_over_layers"]["p10"] - po["mean_over_layers"]["p10"],
        n_zero=eb["mean_over_layers"]["n_zero"] - po["mean_over_layers"]["n_zero"],
        layers_cv_improved=int(sum(d < 0 for d in layer_cv)), per_layer_cv_diff=layer_cv,
        generation_time_paper_sigma=(routing["sets"]["EBSS"]["generation_time"]["paper_sigma"]
                                     - routing["sets"]["PROBABILITY_ONLY"]["generation_time"]["paper_sigma"]))
    FD_jdump(routing, p("routing_statistics.json"))

    # diversity
    div = dict(definition="ED.text_diagnostics (same as the official-artifact audit) + distinct-3, longest repeated "
                          "n-gram, near-duplicates (13-gram Jaccard), CANONICAL-B1 model NLL per 511 targets", sets={})
    for name, rr in sets.items():
        W = toks[rr].tolist()
        div["sets"][name] = text_block(W, nll[rr], tokzr)
    for m in ("po", "ebss"):
        g = gens[m]
        div["sets"][names[m]]["generation_time_mean_nll"] = float((-g["cum_logp"] / 511).mean())
    au = json.load(open(os.path.join(root, "runs_ebss_official_validation", "official_dataset_audit.json")))
    ofr = od["EBSS"]
    div["context_official_artifact_128_windows"] = dict(text=ofr["text"], model_ppl=ofr["model_nll"]["ppl"],
                                                        near_duplicates=ofr["near_duplicates"],
                                                        audit_600_records=au["text_diagnostics_first512_of_long_records"])
    # contamination against held-out evaluation data
    allt = OV.c4_tokens()
    win, _ = EP.load_windows()
    held = [allt[i].tolist() for i in range(2000, 2256)] + [w.tolist() for w in win]
    div["contamination"] = {names[m]: P.contamination(gens[m]["tokens"].tolist(), held) for m in ("po", "ebss")}
    FD_jdump(div, p("diversity_statistics.json"))

    # generation cost
    cost = {}
    for m in ("po", "ebss"):
        c = json.load(open(p("logs", f"generation_cost_{m}.json")))
        t = c["timing"]
        compute = t["forward_s"] - t["expert_transfer_s"] - t["routing_hook_s"]
        parts = dict(model_forward_compute=compute, expert_weight_transfer=t["expert_transfer_s"],
                     routing_capture_hook=t["routing_hook_s"], expert_statistics_update=t["stats_s"],
                     beam_scoring=t["score_s"], beam_pruning_selection=t["select_s"],
                     beam_state_gather_and_kv_reorder=t["reorder_gather_s"])
        parts["other"] = t["total_s"] - sum(parts.values())
        c["breakdown_s"] = parts
        c["breakdown_frac"] = {k: v / t["total_s"] for k, v in parts.items()}
        c["balance_specific_s"] = t["stats_s"] + t["routing_hook_s"]
        cost[names[m]] = c
    cost["reference_author_A800"] = dict(qwen_ebss_128x512_min=18, qwen_gptq_total_min=37, qwen_moequant_total_min=54,
                                         note="author-reported (rebuttal R3 Table R6 / R2 Table R3); not normalised")
    FD_jdump(cost, p("generation_cost.json"))

    smoke = json.load(open(p("smoke_test_results.json")))
    val = {names[m]: P.validate_generated(gens[m]["tokens"], P.N_PILOT, G.SEQLEN, 151646) for m in ("po", "ebss")}
    gate = dict(smoke=smoke["passed"], capture_regression=man["passed"], generated_valid=val,
                contamination_free=all(v["windows_with_32gram_match"] == 0 for v in div["contamination"].values()))
    gate["passed"] = bool(gate["smoke"] and gate["capture_regression"] and all(v["valid"] for v in val.values())
                          and gate["contamination_free"])
    FD_jdump(gate, p("logs", "diagnostics_gate.json"))
    d = routing["ebss_minus_po"]
    print(f"[diagnostics] EBSS-PO: sigma {d['paper_sigma']:+.5f} cv {d['cv']:+.4f} min {d['min']:+.1f}; gate {gate['passed']}")


def FD_jdump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
