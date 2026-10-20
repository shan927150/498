"""Support-coverage diagnostic (no GPTQ, no predictor changes).

Question: when E3 evaluates the benefit predictor, do the experts' routed-
token counts n_e fall inside the count support [lo, hi] that E2's model was
fitted on? E3 never extrapolates outside that support, so a large
out-of-support fraction at the DECISION CHECKPOINTS means the E1 n0 grid
must be extended.

Decision checkpoints come from e3_allocation.round_schedule (the exact
schedule E3 runs): for each round, the chosen-set size at which delta_hat is
evaluated. Final-B coverage is reported too, labelled FINAL_STATE_ONLY,
because the predictor is never evaluated on the final set.

Calibration sets per (B, seed): the warm start is the exact draw E3's
tlp/oracle make (first call of random.Random(seed)); each later checkpoint
extends it by the round budget with uniform random sequences, a proxy for
the adaptive choice. Counts use the cached routing matrix C[s, (layer,
expert)] (router-only replay; cached in <outdir>/routing_profile.npz).

Outputs: <outdir>/support_diagnostic_b{bits}.json and .csv.
"""
from __future__ import annotations
import argparse, csv, json, os, random

import numpy as np

from e3_allocation import DEFAULT_ROUNDS, round_schedule
from moe_calib import model_utils as MU
from moe_calib.runid import damp_tag
from moe_calib.data_utils import load_meta, split_ids

FINAL = "FINAL_STATE_ONLY"
CSV_FIELDS = ["target_final_B", "stage", "round", "decision_chosen_size", "seed", "layer",
              "n_experts", "n_zero", "n_below_nonzero", "n_inside", "n_above"]


def classify(n: np.ndarray, lo: float, hi: float) -> dict:
    n = np.asarray(n, float)
    return dict(n_experts=int(n.size), n_zero=int((n == 0).sum()),
                n_below_nonzero=int(((n > 0) & (n < lo)).sum()),
                n_inside=int(((n >= lo) & (n <= hi)).sum()), n_above=int((n > hi).sum()))


def _fractions(rows) -> dict:
    tot = {k: sum(r[k] for r in rows) for k in CSV_FIELDS[6:]}
    t = max(tot["n_experts"], 1)
    return dict(n_experts=tot["n_experts"], fraction_zero=tot["n_zero"] / t,
                fraction_below_support_nonzero=tot["n_below_nonzero"] / t,
                fraction_inside_support=tot["n_inside"] / t,
                fraction_above_support=tot["n_above"] / t)


def _summary(rows, layers, seeds) -> dict:
    return dict(overall=_fractions(rows),
                by_layer={str(L): _fractions([r for r in rows if r["layer"] == L]) for L in layers},
                by_seed={str(s): _fractions([r for r in rows if r["seed"] == s]) for s in seeds})


def support_coverage(C: np.ndarray, pool_ids: list[int], layers: list[int], E: int,
                     lo: float, hi: float, budgets, seeds, K: int,
                     rounds: int = DEFAULT_ROUNDS):
    """C: [len(pool_ids), len(layers)*E], layer-major columns. Returns
    (rows, report). rows: one per (B, stage/round, seed, layer). report[B]:
    the E3 schedule, a summary per decision checkpoint, and FINAL_STATE_ONLY."""
    row_of = {s: i for i, s in enumerate(pool_ids)}
    rows, report = [], {}

    def add(B, stage, rd, size, seed, chosen):
        n = C[[row_of[s] for s in chosen]].sum(axis=0)
        for li, L in enumerate(layers):
            rows.append(dict(target_final_B=B, stage=stage, round=rd,
                             decision_chosen_size=size, seed=seed, layer=L,
                             **classify(n[li * E:(li + 1) * E], lo, hi)))

    for B in budgets:
        if B > len(pool_ids):
            report[f"B={B}"] = dict(skipped=f"B exceeds pool size {len(pool_ids)}")
            continue
        warm, sched = round_schedule(B, K, rounds)
        for seed in seeds:
            rng = random.Random(seed)
            chosen = rng.sample(pool_ids, warm)          # == E3 tlp/oracle warm start
            for rd, size, budget in sched:
                assert len(chosen) == size
                add(B, "decision", rd, size, seed, chosen)
                taken = set(chosen)
                chosen = chosen + rng.sample([s for s in pool_ids if s not in taken], budget)
            add(B, FINAL, None, len(chosen), seed, chosen)
        rb = [r for r in rows if r["target_final_B"] == B]
        report[f"B={B}"] = dict(
            schedule=dict(warm_start=warm, rounds=rounds, K=K,
                          decision_checkpoints=[size for _, size, _ in sched],
                          round_budgets=[b for _, _, b in sched]),
            checkpoints=[dict(round=rd, decision_chosen_size=size, round_budget=b,
                              **_summary([r for r in rb if r["stage"] == "decision"
                                          and r["round"] == rd], layers, seeds))
                         for rd, size, b in sched],
            **{FINAL: dict(decision_chosen_size=B, note="predictor not evaluated here",
                           **_summary([r for r in rb if r["stage"] == FINAL], layers, seeds))})
    return rows, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--signal-model", default=None,
                    help="E2 artifact (default: <outdir>/e2_model_b{bits}.json)")
    ap.add_argument("--signal", default="auto", help="whose count support to use")
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="must match E3 --rounds")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--percdamp", type=float, default=None,
                    help="use e2_model_b{bits}_d{percdamp}.json (omit = legacy name)")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args(argv)

    dtag = f"_{damp_tag(args.percdamp)}" if args.percdamp is not None else ""
    sm_path = args.signal_model or os.path.join(args.outdir, f"e2_model_b{args.bits}{dtag}.json")
    sigmodel = json.load(open(sm_path))
    signal = sigmodel["selected_signal"] if args.signal == "auto" else args.signal
    lo, hi = sigmodel["predictors"][signal]["count_range"]
    K = sigmodel["K"]                                    # E3 uses the E2 model's K

    meta = load_meta(args.cache_dir)
    layers = args.layers or meta["layers"]
    pool_ids = split_ids(meta, None)["pool"]
    # v2 cache only (model topk(k) rule); the pre-fix routing_profile.npz is ignored
    from e3_allocation import ROUTING_CACHE, cached_routing_profile
    rp_path = os.path.join(args.outdir, ROUTING_CACHE)
    C = None
    if os.path.exists(rp_path):
        z = np.load(rp_path)
        if (z["layers"].tolist() == list(layers) and z["pool_ids"].tolist() == pool_ids
                and str(z["rule"]) == "model_topk"):
            C, E = z["C"], int(z["E"])
    if C is None:
        model, _ = MU.load_model()
        E = MU.num_experts(model)
        C, _ = cached_routing_profile(model, args.cache_dir, layers, pool_ids, args.outdir)

    rows, report = support_coverage(C, pool_ids, layers, E, lo, hi, args.budgets,
                                    args.seeds, K, args.rounds)
    out = dict(signal_model=sm_path, signal=signal, support=[lo, hi], K=K, rounds=args.rounds,
               proxy="warm start = exact E3 tlp/oracle draw; later checkpoints extend it "
                     "with uniform random sequences (proxy for adaptive selection)",
               budgets=args.budgets, seeds=args.seeds, layers=layers, report=report)
    base = os.path.join(args.outdir, f"support_diagnostic_b{args.bits}{dtag}")
    json.dump(out, open(base + ".json", "w"), indent=2)
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"[support] signal={signal} fitted count support [{lo:.0f}, {hi:.0f}], "
          f"K={K}, rounds={args.rounds}")
    print(f"  {'target_B':>8s} {'round':>16s} {'size':>5s} {'zero':>6s} "
          f"{'below>0':>7s} {'inside':>6s} {'above':>6s}")
    for k, v in report.items():
        if "skipped" in v:
            print(f"  {k:>8s} {v['skipped']}")
            continue
        for cp in v["checkpoints"] + [dict(round=FINAL, **v[FINAL])]:
            o = cp["overall"]
            print(f"  {k[2:]:>8s} {str(cp['round']):>16s} {cp['decision_chosen_size']:5d} "
                  f"{o['fraction_zero']:6.3f} {o['fraction_below_support_nonzero']:7.3f} "
                  f"{o['fraction_inside_support']:6.3f} {o['fraction_above_support']:6.3f}")
    print(f"[support] wrote {base}.json / .csv")
    return out


if __name__ == "__main__":
    main()
