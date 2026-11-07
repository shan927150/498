"""Fit P2c's M1 / Dlb / M4 on the full characterization table and export them
as the E3 controller artifact (CPU, seconds; no GPU, no E1/E2 regeneration).

Input: the P2 observation table (default runs/e2_signals_b4_d1.csv), built
from E1 labels on the candidate pool + TUNING split only. The final test
split is never read. Every E1 run behind the table is checked against the
frozen config, and the table's seeds / layers / budgets must match those runs.

    python diagnostics/export_allocation_models.py \
        --frozen configs/frozen_b4.json --signals-csv runs/e2_signals_b4_d1.csv \
        --out runs/allocation_models_b4_d1.json

The artifact also records each model's LOSO MSE recomputed with P2c's own
cross-validation code, so it can be checked against runs/p2c_layer_exposure.json.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib.alloc_models import STRATEGY_MODEL, file_sha256, fit_export, load_table  # noqa: E402
from moe_calib.frozen import check_e1_fp, load_frozen                                  # noqa: E402
from moe_calib.runid import load_e1_runs                                               # noqa: E402


def loso_mse(d, name):
    import p2c_layer_exposure as P2C
    folds = P2C.loso_folds(d)
    return float(np.mean((d["y"] - P2C.cv_pred(d, name, folds)) ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--signals-csv", default="runs/e2_signals_b4_d1.csv")
    ap.add_argument("--e1-outdir", default="runs")
    ap.add_argument("--out", default="runs/allocation_models_b4_d1.json")
    a = ap.parse_args(argv)
    cfg = load_frozen(a.frozen)
    runs, _ = load_e1_runs(a.e1_outdir, cfg["gptq"]["bits"], percdamp=cfg["gptq"]["percdamp"])
    for r in runs:
        check_e1_fp(cfg, r["fp"])
    d, n_drop = load_table(a.signals_csv)
    if n_drop:
        raise SystemExit(f"{n_drop} non-finite rows in {a.signals_csv}")
    seeds = sorted(set(d["seed"].tolist()))
    if seeds != sorted(float(r["char_seed"]) for r in runs):
        raise SystemExit(f"table seeds {seeds} != E1 runs {[r['char_seed'] for r in runs]}")
    if sorted(set(d["layer"].tolist())) != [float(L) for L in cfg["layers"]]:
        raise SystemExit("table layers differ from the frozen layers")
    if sorted(set(d["budget"].tolist())) != [float(b) for b in cfg["e1"]["n0"]]:
        raise SystemExit("table budgets differ from the frozen n0")
    prov = dict(frozen=dict(name=cfg["name"], model=cfg["model"], gptq=cfg["gptq"],
                            split=cfg["split"], layers=list(cfg["layers"])),
                training_table=dict(path=a.signals_csv, sha256=file_sha256(a.signals_csv)),
                e1_runs=[dict(dir=r["dir"], char_seed=r["char_seed"]) for r in runs],
                data_used="E1 labels on candidate pool + tuning split; test split never read",
                selected_layer_model="M4 (P2c LA-A, runs/p2c_layer_exposure.md)")
    art = fit_export(d, prov)
    art["loso_mse_recomputed"] = {m: loso_mse(d, m) for m in STRATEGY_MODEL.values()}
    from moe_calib.alloc_models import artifact_fingerprint
    art["fingerprint"] = artifact_fingerprint(art)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(art, open(a.out, "w"), indent=2)
    print(f"wrote {a.out} fingerprint={art['fingerprint']} n_obs={art['n_obs']}")
    for m, v in art["loso_mse_recomputed"].items():
        print(f"  {m}: LOSO MSE {v:.4e}  in-sample {art['models'][m]['train_mse']:.4e}  "
              f"k={len(art['models'][m]['beta'])}")
    return a.out


if __name__ == "__main__":
    main()
