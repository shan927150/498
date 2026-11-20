"""Damping-grid extension: locate the interior optimum of R_ideal(nu) for down (and gate+up as control).

Follow-up to update_theorem1_check.md (T1-PASS-BOTH). Part B's grid stopped at nu = 5.0, which was the argmin for
down at B16 / B32. This script re-uses diagnostics/theorem1_check.py (imported, never copied) to
  E1: evaluate R_ideal(nu) = tr(M^T S M) / tr(S) on a fine wide grid (33 log-spaced 1e-2..1e2 + the 12 Part B
      points) for all 60 experts x {gu, down} x 5 nested prefixes x layers {1, 8, 16, 23} x seeds {0, 1};
  E2: run the frozen GPTQ solver at nu in {5, 7, 10, 15, 20, 30, 50} on the B2 experts (down: all prefixes;
      gu: prefixes 16, 32) and measure tune_err_groups / e_meas, with nu = 5.0 asserted against Part B.

  python diagnostics/damping_grid_extension.py protocol   # freeze runs_grid_ext/protocol.json (refuses to overwrite)
  python diagnostics/damping_grid_extension.py run        # E1 + E2, resumable, checkpoint per (seed, layer)
  python diagnostics/report_damping_grid_extension.py     # report / figures

Tuning split only. moe_calib/gptq.py is frozen and only called.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theorem1_check as T1                                         # noqa: E402

OUT = "runs_grid_ext"
PROTOCOL = f"{OUT}/protocol.json"
RAW_E1 = f"{OUT}/raw_e1.jsonl"
RAW_E2 = f"{OUT}/raw_e2.jsonl"
EVENTS = f"{OUT}/events.jsonl"
GPTQ_SHA_PREFIX = "58e40d1c"

NU_LOG33 = [float(x) for x in np.logspace(-2, 2, 33)]


def union_grid(a, b, rtol=1e-9):
    """Sorted union; near-duplicates (<= rtol relative) merged, keeping the value from b (the exact Part B point)."""
    out = [float(v) for v in b]
    for v in a:
        if all(abs(v - w) > rtol * w for w in out):
            out.append(float(v))
    return sorted(out)


NU_E1 = union_grid(NU_LOG33, T1.NU_GRID)                          # 42 points, 1e-2 .. 1e2
NU_E2 = [5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
NU_MERGED = T1.NU_GRID + NU_E2[1:]                                  # 18 points (5.0 from Part B, re-measured as anchor)
E2_PREFIXES = {"gu": [16, 32], "down": [16, 32, 64, 128, 256]}
SEEDS, LAYERS, PREFIXES = [0, 1], [1, 8, 16, 23], [16, 32, 64, 128, 256]
T1_OUTLIERS = {1: [0, 42, 33], 23: [57, 26, 25]}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_protocol():
    return dict(
        name="damping_grid_extension",
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        follows="update_theorem1_check.md (T1-PASS-BOTH), runs_theorem1_check/ (read only)",
        solver=f"moe_calib/gptq.py gptq_quantize (FROZEN, sha256 prefix {GPTQ_SHA_PREFIX}), 4-bit g128 act-order",
        split="tuning split only; test / confirmation splits, configs/frozen_b4.json and existing outputs untouched",
        reuse=("all math and data paths imported from diagnostics/theorem1_check.py: solver_coords, m_matrix, "
               "col_quad, sweep_curve (argmin + parabola refinement), b2_cell, layer_context (Sigma = tuning X^T X / "
               "n_tune fp64, RTN, B2 selection), prefix_hessians (nested low_data_diag.hessians running sums), "
               "damping_threshold.tune_err_groups"),
        precision="fp64 for H_hat, S, M, R (H_hat fp32 accumulator cast to fp64 exactly as Part B); Cholesky failure "
                  "-> cell recorded as failed, no jitter",
        E1=dict(
            seeds=SEEDS, layers=LAYERS, prefixes=PREFIXES, experts="all 60", groups=["gu", "down"],
            nu_grid=NU_E1,
            nu_grid_def="sorted union of logspace(-2, 2, 33) and the Part B grid " + str(T1.NU_GRID) +
                        " (duplicates within 1e-9 relative merged) = 42 points",
            per_cell=("full R_ideal curve over nu_grid; nu* = argmin over nu_grid refined by the vertex of the parabola "
                      "in log nu through the argmin and its two grid neighbours, R_ideal re-evaluated at the vertex and "
                      "kept only if not above the grid minimum (theorem1_check.sweep_curve); edge flag if argmin is "
                      "the first (1e-2) or last (1e2) grid point; R(1.0), R(5.0) read from the curve. Cells whose "
                      "curve is constant (no tokens) are flagged 'flat' and excluded from per-expert nu* statistics"),
            invariant=("first (seed, layer) processed: R_ideal at the 12 Part B grid points equals "
                       "runs_theorem1_check/raw_grid.jsonl R_ideal for every expert and group to <= 1e-6 relative; "
                       "abort otherwise"),
            output=RAW_E1),
        E2=dict(
            experts=("B2 set of theorem1_check (12 per layer: ranks {0-3}, {28-31}, {56-59} of seed-0 B32 routed "
                     "tokens, ascending, ties to lower id); recomputed and asserted equal to "
                     "runs_theorem1_check/grid_events.jsonl"),
            nu=NU_E2, prefixes=E2_PREFIXES,
            per_cell=("theorem1_check.b2_cell (R_ideal, R_w, e_meas, e_rtn, clamp_frac_W / clamp_frac_work, gaps) "
                      "+ tune_err_groups(W, Q, T)[g] / rtn_err[g] where Q = GPTQ for group g and the FP weights for "
                      "the other group (tune_err_groups' per-group value depends only on that group's projections)"),
            anchor=("nu = 5.0 re-measured: tune_err_groups[g], e_meas and R_ideal must equal the stored Part B "
                    "values (runs_theorem1_check/raw_grid.jsonl, b2 cell) to <= 1e-6 relative; abort otherwise"),
            merged_grid=NU_MERGED,
            merged=("per (seed, layer, prefix, expert, group) cell: Part B's 12 stored points (nu <= 5.0) + the 6 new "
                    "points = 18; measured ratio and R_ideal over the 18 points"),
            output=RAW_E2),
        report=dict(
            Q1=("per (prefix, group): budget-level nu* = vertex (log nu) of the parabola through the argmin of the "
                "expert-mean R_ideal curve (mean over all (seed, layer, expert) cells with a finite curve) and its two "
                "grid neighbours (grid point if the argmin is at an edge); interior = argmin not at the first/last "
                "grid point; 95% CI by bootstrap over clusters (layer, expert) [both seeds in one cluster], 2000 "
                "resamples, numpy default_rng(0); shown with the old Part B grid argmin (T1 report) and the Part C "
                "power law exp(a) (d/n)^b (runs_theorem1_check/report.json T4, n<d or n>d fit by the side of the "
                "prefix's median n/d, median over all cells of tokens/d_g)"),
            Q2=("down, B2 cells over the 18-point merged grid: measured argmin vs predicted (R_ideal) argmin, "
                "Spearman, argmin agreement within +-1 merged-grid step; gate = frac agreement >= 0.70 AND median "
                "Spearman >= 0.8 (same as T1); clamp fractions vs nu; gu (prefixes 16, 32) reported the same way, "
                "not gated"),
            Q3=("per (prefix, group): predicted excess = Rbar(nu0)/Rbar(nu*) - 1 for nu0 in {5.0, 1.0} on the "
                "expert-mean curve (Rbar(nu*) = parabola value at the vertex), and per-cell median of "
                "R(nu0)/R(nu*) - 1; measured excess on B2 cells = mean measured curve at nu0 / its minimum over the "
                "merged grid - 1 (12-point grid for gu at prefixes >= 64), plus per-cell median; 'good enough' = "
                "excess < 1%"),
            Q4=("nu*_down / nu*_gu: budget-level per prefix; per cell matched by (seed, layer, prefix, expert), both "
                "nu* interior and non-flat: median, IQR; trend = OLS slope of log ratio on log(tokens) with cluster "
                "bootstrap CI (clusters (layer, expert)) and per-prefix medians"),
            Q5=("per (prefix, group) distribution of per-cell nu*: median, IQR, 5th / 95th percentiles, fraction at "
                "the 1e2 edge and at the 1e-2 edge; edge experts listed and compared with the T1 outlier experts "
                f"{T1_OUTLIERS}"),
            verdict=dict(labels={
                "EXT-INTERIOR-CONFIRMED": "every (prefix, group) budget-level nu* interior AND measured down gate passes",
                "EXT-INTERIOR-PREDICTED-ONLY": "every budget-level nu* interior, measured down gate fails",
                "EXT-EDGE": "some budget-level nu* at a grid edge (named)"},
                order="EXT-EDGE takes precedence")),
        cost=("checkpoint after each (seed, layer); wall-clock estimate printed after the first (seed, layer); "
              "expected E1 1-2 GPU-h, E2 less"),
    )


def cmd_protocol(a):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(PROTOCOL):
        raise SystemExit(f"{PROTOCOL} exists and is frozen; not overwriting")
    json.dump(build_protocol(), open(PROTOCOL, "w"), indent=1)
    print(f"froze {PROTOCOL} ({len(NU_E1)} E1 nu points)")


def stored_partb():
    """(seed, layer, n, nu) -> Part B record (runs_theorem1_check/raw_grid.jsonl), read only."""
    return {(r["seed"], r["layer"], r["n"], float(r["nu"])): r for r in T1.read_jsonl(T1.RAW_GRID)}


def stored_b2_sets():
    return {e["layer"]: e["b2_experts"] for e in T1.read_jsonl(f"{T1.OUT}/grid_events.jsonl") if "b2_experts" in e}


def e1_cell(H, S, d):
    Hp, perm, dead = T1.solver_coords(H)
    rec = T1.sweep_curve(Hp, S[perm][:, perm], None, d, nus=NU_E1, curve=[], gs=False)
    rec.pop("R_curve", None)
    fine = rec["R_fine"]
    ok = [v for v in fine if v is not None]
    rec["flat"] = bool(ok) and (max(ok) - min(ok)) == 0.0
    rec["n_dead"] = int(dead.sum())
    return rec


def cmd_run(a):
    if not sha256("moe_calib/gptq.py").startswith(GPTQ_SHA_PREFIX):
        raise SystemExit("moe_calib/gptq.py hash changed")
    T1._lazy_imports()
    LD, MU, DT = T1.LD, T1.MU, T1.DT
    P = json.load(open(PROTOCOL)) if os.path.exists(PROTOCOL) else None
    if P is None:
        raise SystemExit("freeze the protocol first")
    assert P["E1"]["nu_grid"] == NU_E1 and P["E2"]["nu"] == NU_E2
    meta, sp = LD.splits(a.cache_dir)
    done1 = {(r["seed"], r["layer"], r["n"]) for r in T1.read_jsonl(RAW_E1)}
    done2 = {(r["seed"], r["layer"], r["n"], r["nu"]) for r in T1.read_jsonl(RAW_E2)}
    ev = T1.read_jsonl(EVENTS)
    inv1_done = any(e.get("e1_invariant") == "pass" for e in ev)
    est_done = any("est_total_sec" in e for e in ev)
    partb = stored_partb()
    b2_ref = stored_b2_sets()
    model, _ = MU.load_model()
    E = MU.num_experts(model)
    D = dict(zip(("gu", "down"), MU.expert_dims(model)))
    t_all = time.time()
    n_units = len(a.seeds) * len(a.layers)
    for L in a.layers:
        def need(seed):
            return (any((seed, L, n) not in done1 for n in PREFIXES) or
                    any((seed, L, n, nu) not in done2 for n in PREFIXES for nu in NU_E2))
        todo = [s for s in a.seeds if need(s)]
        if not todo:
            continue
        cache = T1.load_layer_cache(a.cache_dir, L)
        cx = T1.layer_context(model, L, cache, sp, E)
        if cx["b2"] != b2_ref[L]:
            raise SystemExit(f"B2 set mismatch L={L}: {cx['b2']} vs stored {b2_ref[L]}")
        Ws, Ts, Sig, rtnQ, rtn_err, b2 = (cx[k] for k in ("Ws", "Ts", "Sig", "rtnQ", "rtn_err", "b2"))
        rtn_e = {(e, g): sum(T1.quad_err(Ws[e][nm] - rtnQ[e][nm], Sig[(e, g)]) for nm in T1.GROUPS[g])
                 if Sig[(e, g)] is not None else float("nan") for e in b2 for g in T1.GROUPS}
        for seed in todo:
            t_unit = time.time()
            sec1 = sec2 = 0.0
            order = LD.nested_order(sp["pool"], seed)
            for n, run, tok in T1.prefix_hessians(model, L, cache, order, E):
                # ---------------- E1
                if (seed, L, n) not in done1:
                    t0 = time.time()
                    rec = dict(seed=seed, layer=L, n=n, tokens=[float(tok[e]) for e in range(E)], cells={})
                    for e in range(E):
                        for g in T1.GROUPS:
                            S = Sig[(e, g)]
                            if S is None:
                                rec["cells"][f"{e}:{g}"] = dict(failed="no tuning tokens")
                                continue
                            rec["cells"][f"{e}:{g}"] = e1_cell(run[e][g], S, D[g])
                    if not inv1_done:
                        worst = 0.0
                        for nu in T1.NU_GRID:
                            pb = partb[(seed, L, n, nu)]["R_ideal"]
                            j = NU_E1.index(nu) if nu in NU_E1 else int(np.argmin([abs(x - nu) for x in NU_E1]))
                            assert abs(NU_E1[j] - nu) <= 1e-9 * nu
                            for e in range(E):
                                for g in T1.GROUPS:
                                    c, ref = rec["cells"][f"{e}:{g}"], pb[g].get(str(e))
                                    if not isinstance(ref, float):
                                        continue
                                    x = c["R_fine"][j]
                                    rel = abs(x - ref) / abs(ref)
                                    worst = max(worst, rel)
                                    if rel > 1e-6:
                                        T1.append(EVENTS, dict(e1_invariant="FAIL", seed=seed, layer=L, n=n, nu=nu,
                                                               expert=e, group=g, got=x, stored=ref))
                                        raise SystemExit(f"E1 invariant FAIL s{seed} L{L} n{n} nu{nu} e{e} {g}")
                        rec["invariant_max_rel"] = worst
                    rec["sec"] = time.time() - t0
                    sec1 += rec["sec"]
                    T1.append(RAW_E1, rec)
                    done1.add((seed, L, n))
                    st = {g: [rec["cells"][f"{e}:{g}"].get("nu_star") for e in range(E)] for g in T1.GROUPS}
                    print(f"[ext-E1] s{seed} L{L} n{n}: median nu* gu={np.nanmedian([v for v in st['gu'] if v]):.3g} "
                          f"down={np.nanmedian([v for v in st['down'] if v]):.3g} edges(1e2) down="
                          f"{sum(rec['cells'][f'{e}:down'].get('argmin_idx') == len(NU_E1) - 1 for e in range(E))} "
                          f"({rec['sec']:.0f}s)", flush=True)
                # ---------------- E2
                for nu in NU_E2:
                    if (seed, L, n, nu) in done2:
                        continue
                    t0 = time.time()
                    rec = dict(seed=seed, layer=L, n=n, nu=nu, b2_experts=b2, cells={})
                    for e in b2:
                        for g in T1.GROUPS:
                            if n not in E2_PREFIXES[g]:
                                continue
                            if Sig[(e, g)] is None:
                                rec["cells"][f"{e}:{g}"] = dict(failed="no tuning tokens")
                                continue
                            r, Qg = T1.b2_cell(Ws[e], run[e][g], Sig[(e, g)], g, nu)
                            Qd = {nm: Ws[e][nm] for nm in Ws[e]}
                            Qd.update(Qg)
                            r["e_rtn"] = rtn_e[(e, g)]
                            r["gptq_err"] = DT.tune_err_groups(Ws[e], Qd, Ts[e])[g]
                            r["rtn_err"] = rtn_err[e][g]
                            r["ratio"] = r["gptq_err"] / r["rtn_err"]
                            if nu == 5.0:
                                pb = partb[(seed, L, n, 5.0)]["b2"][str(e)]
                                chk = dict(gptq_err=(r["gptq_err"], pb["gptq_err"][g]),
                                           e_meas=(r["e_meas"], pb[g]["e_meas"]),
                                           R_ideal=(r["R_ideal"], pb[g]["R_ideal"]))
                                rels = {k: abs(x - y) / max(abs(y), 1e-300) for k, (x, y) in chk.items()}
                                r["anchor_rel"] = rels
                                if max(rels.values()) > 1e-6:
                                    T1.append(EVENTS, dict(e2_anchor="FAIL", seed=seed, layer=L, n=n, expert=e,
                                                           group=g, rels=rels))
                                    raise SystemExit(f"E2 anchor FAIL s{seed} L{L} n{n} e{e} {g}: {rels}")
                            rec["cells"][f"{e}:{g}"] = r
                    rec["sec"] = time.time() - t0
                    sec2 += rec["sec"]
                    T1.append(RAW_E2, rec)
                    done2.add((seed, L, n, nu))
                    dv = [c for k, c in rec["cells"].items() if k.endswith(":down") and "ratio" in c]
                    print(f"[ext-E2] s{seed} L{L} n{n} nu={nu}: down mean ratio {np.mean([c['ratio'] for c in dv]):.4f}"
                          f" clampW {np.mean([c['clamp_frac_W'] for c in dv]):.2e} ({rec['sec']:.0f}s)", flush=True)
            if not inv1_done:
                T1.append(EVENTS, dict(e1_invariant="pass", seed=seed, layer=L,
                                       max_rel=max(r.get("invariant_max_rel", 0) for r in T1.read_jsonl(RAW_E1)
                                                   if r["seed"] == seed and r["layer"] == L)))
                inv1_done = True
                print("[ext] E1 invariant (Part B R_ideal reproduced) PASSED", flush=True)
            el = time.time() - t_unit
            T1.append(EVENTS, dict(checkpoint=True, seed=seed, layer=L, sec=el, sec_e1=sec1, sec_e2=sec2))
            if not est_done:
                T1.append(EVENTS, dict(est_total_sec=el * n_units, est_e1_sec=sec1 * n_units,
                                       est_e2_sec=sec2 * n_units, first_unit_sec=el))
                est_done = True
                print(f"[ext] first (seed, layer) took {el / 60:.1f} min -> estimated full run "
                      f"{el * n_units / 3600:.2f} h (E1 {sec1 * n_units / 3600:.2f} h, E2 {sec2 * n_units / 3600:.2f} h)",
                      flush=True)
        del cache, cx, Ws, Ts, Sig, rtnQ
        torch.cuda.empty_cache()
    print(f"[ext] done in {(time.time() - t_all) / 3600:.2f} h", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "run"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args(argv)
    dict(protocol=cmd_protocol, run=cmd_run)[a.cmd](a)


if __name__ == "__main__":
    main()
