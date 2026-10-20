"""P0 — extended fixed-damping study + per-projection threshold diagnostic.

Two questions, one GPU pass (tuning split only; production solver untouched):

 (1) Which ONE GPTQ damping to freeze? Extends diagnostics/damping_selection.py
     to a wider grid (default 0.01 0.1 0.2 0.3 0.5 1.0) on the same nested
     prefixes 16..256, seeds, layers, tuning split and RTN reference, and
     reports whether the best value is interior to the grid.

 (2) Why does 16 -> 32 hurt at 1% damping? Hypothesis (sample-wise double
     descent of weakly-ridged least squares): held-out error peaks when an
     expert's routed tokens approach the projection's INPUT dimension
     (gate/up: 2048, down: 1408), and damping (= ridge) removes the peak.
     Test: errors are recorded PER PROJECTION GROUP (gate+up vs down) and
     binned by tokens / d_in; the peak should sit near ratio 1 for BOTH groups
     if the mechanism is right. The effective rank of each Hessian
     (exp-entropy of its spectrum, Roy & Vetterli 2007) is recorded too,
     because tokens inside a sequence are correlated (erank <= tokens).

  python diagnostics/damping_threshold.py run      # resumable per (seed, layer, prefix, damp)
  python diagnostics/damping_threshold.py report   # -> runs_diag/damping_threshold.{json,md}
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from low_data_diag import (OUT, PREFIXES, gpu_weights, hessians, nested_order,  # noqa: E402
                           quant, read_jsonl, rtn, splits, tune_holdout)
from damping_selection import prep_tune                            # noqa: E402
from moe_calib import model_utils as MU                            # noqa: E402
from moe_calib.data_utils import load_layer_cache                  # noqa: E402
from moe_calib.quant_eval import per_seq_rel_errors                # noqa: E402

DEFAULT_DAMPS = [0.01, 0.1, 0.2, 0.3, 0.5, 1.0]
RAW = f"{OUT}/damping_threshold_raw.jsonl"
GROUPS = {"gu": ("gate_proj", "up_proj"), "down": ("down_proj",)}
D_IN = {"gu": 2048, "down": 1408}           # Qwen1.5-MoE-A2.7B expert input dims


# ------------------------------------------------------------------ metrics

@torch.no_grad()
def tune_err_groups(W, Q, T):
    """Mean over tuning sequences of the relative error, overall and per
    projection group. 'all' equals E1's metric (sum over the three
    projections in numerator and denominator)."""
    n = T["n"]
    if n == 0:
        return {k: float("nan") for k in ("all", "gu", "down")}
    num = {g: torch.zeros(n, device=MU.GPU, dtype=torch.float64) for g in GROUPS}
    den = {g: torch.zeros(n, device=MU.GPU, dtype=torch.float64) for g in GROUPS}
    for g, names in GROUPS.items():
        if T[g] is None:
            continue
        X, seg = T[g]
        Xf = X.float()
        for nm in names:
            num[g].index_add_(0, seg, (Xf @ (W[nm] - Q[nm].to(MU.GPU)).T).pow(2).sum(1).double())
            den[g].index_add_(0, seg, (Xf @ W[nm].T).pow(2).sum(1).double())
    out = {}
    tot_n = num["gu"] + num["down"]
    tot_d = den["gu"] + den["down"]
    out["all"] = float((tot_n / (tot_d + 1e-12)).mean())
    for g in GROUPS:
        m = den[g] > 0                       # sequences that reach this group
        out[g] = float((num[g][m] / den[g][m]).mean()) if m.any() else float("nan")
    return out


@torch.no_grad()
def spectrum_stats(H):
    """Effective rank exp(entropy of normalized eigenvalues) and participation
    ratio (sum l)^2 / sum l^2 of a PSD matrix."""
    lam = torch.linalg.eigvalsh(H.to(MU.GPU, torch.float64)).clamp(min=0)
    s = lam.sum()
    if s <= 0:
        return 0.0, 0.0
    p = lam / s
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    return float(ent.exp()), float(s * s / (lam * lam).sum())


# ---------------------------------------------------------------------- run

def done_damps(raw):
    out = {}
    for r in raw:
        out.setdefault((r["seed"], r["layer"], r["n"]), set()).update(map(float, r["err"].keys()))
    return out


def cmd_run(a):
    meta, sp = splits(a.cache_dir)
    os.makedirs(OUT, exist_ok=True)
    have = done_damps(read_jsonl(RAW))
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    checked = False
    for L in a.layers:
        todo_seeds = [s for s in a.seeds if any(
            not set(a.damps) <= have.get((s, L, n), set()) for n in PREFIXES)]
        if not todo_seeds:
            continue
        cache = load_layer_cache(a.cache_dir, L)
        HO = tune_holdout(model, L, cache, sp["tune"])
        Ws = {e: gpu_weights(model, L, e) for e in range(E)}
        Ts = {e: prep_tune(HO, L, e) for e in range(E)}
        rtn_err = {e: tune_err_groups(Ws[e], rtn(Ws[e]), Ts[e]) for e in range(E)}
        if not checked:                           # 'all' == E1's metric
            for e in range(min(3, E)):
                ref = np.mean(list(per_seq_rel_errors(model, L, e, rtn(Ws[e]), HO).values()))
                assert abs(rtn_err[e]["all"] - ref) <= 1e-4 * ref, (e, rtn_err[e]["all"], ref)
            checked = True
        for seed in todo_seeds:
            order = nested_order(sp["pool"], seed)
            run, tok, lo = None, None, 0
            for n in PREFIXES:
                Hc, tc = hessians(model, L, cache, order[lo:n], K=1)
                if run is None:
                    run, tok = Hc, tc
                else:
                    for e in range(E):
                        for p in ("gu", "down"):
                            run[e][p] += Hc[e][p]
                        tok[e] += tc[e]
                    del Hc
                lo = n
                prior = have.get((seed, L, n), set())
                need = [d for d in a.damps if float(d) not in prior]
                if not need:
                    continue
                rec = dict(seed=seed, layer=L, n=n, err={str(d): [] for d in need})
                if not prior:                     # first record for this key: context
                    d_gu, d_down = MU.expert_dims(model)
                    rec["d_in"] = {"gu": d_gu, "down": d_down}
                    rec["tokens"] = [tok[e] for e in range(E)]
                    rec["rtn"] = [rtn_err[e] for e in range(E)]
                    rec["erank"] = {"gu": [], "down": []}
                    rec["pratio"] = {"gu": [], "down": []}
                    for e in range(E):
                        for g in ("gu", "down"):
                            er, pr = spectrum_stats(run[e][g])
                            rec["erank"][g].append(er)
                            rec["pratio"][g].append(pr)
                for e in range(E):
                    for d in need:
                        q = quant(Ws[e], run[e], percdamp=d, actorder=True)
                        rec["err"][str(d)].append(tune_err_groups(Ws[e], q, Ts[e]))
                with open(RAW, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                have.setdefault((seed, L, n), set()).update(map(float, need))
                msg = " ".join(f"{d}:{np.mean([x['all'] for x in rec['err'][str(d)]]):.5f}"
                               for d in need)
                print(f"[damp-thr] seed={seed} L={L} n={n} tok_med={np.median([tok[e] for e in range(E)]):.0f} "
                      f"RTN={np.mean([rtn_err[e]['all'] for e in range(E)]):.5f} {msg}", flush=True)
            del run
        del cache, HO, Ws, Ts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------- report

def merged(raw):
    """(seed, layer, n) -> dict(tokens, rtn, erank, pratio, err{damp: [per-expert dict]})."""
    by = {}
    for r in raw:
        k = (r["seed"], r["layer"], r["n"])
        m = by.setdefault(k, dict(err={}))
        for f in ("tokens", "rtn", "erank", "pratio", "d_in"):
            if f in r:
                m[f] = r[f]
        for d, v in r["err"].items():
            m["err"][float(d)] = v
    return {k: v for k, v in by.items() if "tokens" in v}


def frame(by, damps):
    """Long arrays over (seed, layer, n, expert)."""
    rows = []
    for (s, L, n), m in by.items():
        for e, t in enumerate(m["tokens"]):
            row = dict(seed=s, layer=L, n=n, expert=e, tokens=float(t),
                       rtn_all=m["rtn"][e]["all"], rtn_gu=m["rtn"][e]["gu"],
                       rtn_down=m["rtn"][e]["down"],
                       erank_gu=m["erank"]["gu"][e], erank_down=m["erank"]["down"][e])
            for d in damps:
                if d in m["err"]:
                    for g in ("all", "gu", "down"):
                        row[f"e{d}_{g}"] = m["err"][d][e][g]
            rows.append(row)
    keys = sorted({k for r in rows for k in r})
    return {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}


RATIO_BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf]


def cmd_report(a):
    by = merged(read_jsonl(RAW))
    if not by:
        raise SystemExit(f"no complete records in {RAW}; run first")
    damps = sorted({d for m in by.values() for d in m["err"]})
    D_IN.update(next(iter(by.values())).get("d_in", {}))
    F = frame(by, damps)
    seeds, layers = sorted(set(F["seed"].astype(int))), sorted(set(F["layer"].astype(int)))
    out = dict(design=dict(damps=damps, prefixes=PREFIXES, seeds=seeds, layers=layers,
                           d_in=D_IN, split="tuning only",
                           rtn="H=I, act-order off (production fallback)"),
               summary={}, per_prefix={}, threshold={}, within_prefix={}, erank={})
    M = ["# P0 — extended damping study + per-projection threshold test", "",
         f"seeds {seeds}; layers {layers}; nested prefixes {PREFIXES}; tuning split only.", "",
         "## 1. Damping grid (all seeds x layers)", "",
         "| percdamp | mean err (avg over prefixes) | % experts better than RTN | "
         + " | ".join(f"{a_}->{b_} improved %" for a_, b_ in zip(PREFIXES[:-1], PREFIXES[1:])) + " |",
         "|---|---|---|" + "---|" * (len(PREFIXES) - 1)]
    for d in damps:
        col = F.get(f"e{d}_all")
        if col is None:
            continue
        per_n = {n: col[F["n"] == n] for n in PREFIXES}
        better = np.mean([np.mean(per_n[n] < F["rtn_all"][F["n"] == n]) for n in PREFIXES])
        trans = []
        for n0, n1 in zip(PREFIXES[:-1], PREFIXES[1:]):
            m0 = F["n"] == n0
            m1 = F["n"] == n1
            key0 = list(zip(F["seed"][m0], F["layer"][m0], F["expert"][m0]))
            key1 = dict(zip(zip(F["seed"][m1], F["layer"][m1], F["expert"][m1]), col[m1]))
            diff = np.array([v - key1[k] for k, v in zip(key0, col[m0]) if k in key1])
            trans.append(float(np.mean(diff > 0)))
        s = dict(mean_err_over_prefixes=float(np.mean([per_n[n].mean() for n in PREFIXES])),
                 frac_better_than_rtn=float(better), frac_improved_by_transition=trans)
        out["summary"][str(d)] = s
        M.append(f"| {d} | {s['mean_err_over_prefixes']:.5f} | {100 * better:.1f}% | "
                 + " | ".join(f"{100 * t:.0f}%" for t in trans) + " |")
    best = min(out["summary"], key=lambda k: out["summary"][k]["mean_err_over_prefixes"])
    interior = float(best) not in (min(damps), max(damps))
    out["best_overall"] = dict(percdamp=float(best), interior=interior)
    M += ["", f"**Best overall: percdamp = {best}** — "
          + ("interior to the grid, safe to freeze." if interior else
             "AT THE GRID EDGE: extend the grid before freezing.")]
    M += ["", "Best percdamp per prefix (does the optimum depend on budget?):", "",
          "| prefix | best | its err | RTN err |", "|---|---|---|---|"]
    for n in PREFIXES:
        m = F["n"] == n
        errs = {d: float(np.nanmean(F[f"e{d}_all"][m])) for d in damps if f"e{d}_all" in F}
        bd = min(errs, key=errs.get)
        out["per_prefix"][str(n)] = dict(best=bd, err=errs, rtn=float(np.nanmean(F["rtn_all"][m])))
        M.append(f"| {n} | {bd} | {errs[bd]:.5f} | {np.nanmean(F['rtn_all'][m]):.5f} |")

    # ---- 2. threshold test per projection group ----
    show = [d for d in damps if d in (0.01, float(best))] or damps[:1]
    M += ["", "## 2. Per-projection error vs routed tokens / input dim", "",
          "GPTQ/RTN error ratio (mean over expert-prefix points). Double-descent "
          "prediction: peak near tokens/d_in ~ 1 for BOTH groups; damping flattens it.", ""]
    for d in show:
        out["threshold"][str(d)] = {}
        M += [f"### percdamp = {d}", "", "| tokens / d_in | gate+up ratio (n) | down ratio (n) |",
              "|---|---|---|"]
        peaks = {}
        tab = {}
        for g in ("gu", "down"):
            r = F["tokens"] / D_IN[g]
            ratio = F[f"e{d}_{g}"] / F[f"rtn_{g}"]
            tab[g] = []
            for lo, hi in zip(RATIO_BINS[:-1], RATIO_BINS[1:]):
                m = (r >= lo) & (r < hi) & np.isfinite(ratio)
                tab[g].append((float(np.mean(ratio[m])) if m.sum() >= 20 else float("nan"), int(m.sum())))
            vals = [v for v, _ in tab[g]]
            peaks[g] = (None if np.all(np.isnan(vals)) else
                        f"[{RATIO_BINS[int(np.nanargmax(vals))]}, {RATIO_BINS[int(np.nanargmax(vals)) + 1]})")
            out["threshold"][str(d)][g] = dict(bins=RATIO_BINS[:-1], ratio=vals,
                                               n=[c for _, c in tab[g]], peak_bin=peaks[g])
        for i, (lo, hi) in enumerate(zip(RATIO_BINS[:-1], RATIO_BINS[1:])):
            cell = lambda g: (f"{tab[g][i][0]:.3f} ({tab[g][i][1]})" if np.isfinite(tab[g][i][0])
                              else f"– ({tab[g][i][1]})")
            M.append(f"| [{lo}, {hi}) | {cell('gu')} | {cell('down')} |")
        M += ["", f"Peak bin: gate+up {peaks['gu']}, down {peaks['down']}.", ""]

    # ---- 3. within-prefix check (decouples tokens from prefix size) ----
    d0 = show[0]
    M += [f"## 3. Within a fixed prefix (percdamp = {d0}): token quintile -> GPTQ/RTN ratio", "",
          "Rising with tokens below the threshold and falling above it supports a peak "
          "in tokens rather than in the number of sequences.", "",
          "| prefix | group | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|---|"]
    for n in PREFIXES[:3]:
        for g in ("gu", "down"):
            m = (F["n"] == n) & np.isfinite(F[f"e{d0}_{g}"])
            t, ratio = F["tokens"][m], F[f"e{d0}_{g}"][m] / F[f"rtn_{g}"][m]
            qs = np.percentile(t, [0, 20, 40, 60, 80, 100])
            vals = [float(ratio[(t >= a_) & (t <= b_)].mean()) for a_, b_ in zip(qs[:-1], qs[1:])]
            out["within_prefix"][f"n={n},{g}"] = dict(quintile_edges=qs.tolist(), ratio=vals)
            M.append(f"| {n} | {g} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    # ---- 4. effective rank ----
    M += ["", "## 4. Effective rank of H vs routed tokens (median over experts)", "",
          "| prefix | median tokens | gate+up erank / 2048 | down erank / 1408 |", "|---|---|---|---|"]
    for n in PREFIXES:
        m = F["n"] == n
        row = dict(tokens=float(np.median(F["tokens"][m])),
                   erank_frac_gu=float(np.median(F["erank_gu"][m]) / D_IN["gu"]),
                   erank_frac_down=float(np.median(F["erank_down"][m]) / D_IN["down"]))
        out["erank"][str(n)] = row
        M.append(f"| {n} | {row['tokens']:.0f} | {row['erank_frac_gu']:.3f} | {row['erank_frac_down']:.3f} |")

    base = f"{OUT}/damping_threshold"
    json.dump(out, open(base + ".json", "w"), indent=1, default=float)
    open(base + "_report.md", "w").write("\n".join(M) + "\n")
    print("\n".join(M))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 23])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--damps", type=float, nargs="+", default=DEFAULT_DAMPS)
    a = ap.parse_args(argv)
    return dict(run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
