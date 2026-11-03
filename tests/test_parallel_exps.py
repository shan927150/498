"""CPU tests for the parallel-plan experiments (Exp1/2/4/5/6) and the partial
model loader."""
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import model_utils as MU

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import adaptive_damping as AD          # noqa: E402
import depth_generalization as DG      # noqa: E402
import exposure_diversity as ED        # noqa: E402
import headroom as HR                  # noqa: E402
import within_layer_ranking as WL      # noqa: E402


# ------------------------------------------------------------- partial load

def test_partial_loader_matches_full(tmp_path):
    model, _ = tiny_model()
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="200KB")
    part, tok = MU._load_partial(str(tmp_path), [2])   # the conftest patches load_model itself
    assert part.partial_layers == [2] and tok is None
    assert part.model.layers[0].mlp.experts[0].gate_proj.weight.is_meta
    for (n1, p1), (n2, p2) in zip(model.model.layers[2].named_parameters(),
                                  part.model.layers[2].named_parameters()):
        assert torch.equal(p1, p2), n1
    with pytest.raises(ValueError):
        MU._load_partial(str(tmp_path), [9])


# ------------------------------------------------------------- Exp1

def _labels(seed_layers_budgets, rng, layer_effect=True):
    rows = []
    for s in (0, 1, 2):
        for L in (1, 8, 16, 23):
            for b in (16, 32, 64, 128):
                n = rng.lognormal(np.log(34 * b), 0.15, 60)
                base = (0.08 if L == 1 else 0.05) * (16 / b) ** (0.6 if L == 1 else 0.3)
                y = base * (n / n.mean()) ** -0.3 + 0.002 * rng.normal(size=60)
                rows += [(s, L, e, b, n[e], y[e]) for e in range(60)]
    a = np.array(rows)
    return dict(seed=a[:, 0], layer=a[:, 1], expert=a[:, 2], budget=a[:, 3], count=a[:, 4], y=a[:, 5])


def test_within_layer_equals_coverage_but_pooled_differs():
    d = _labels(None, np.random.default_rng(0))
    agg, wins, ncells = WL.summarize(WL.run(d))
    # within a layer a decreasing f_L(n)/n orders experts like 1/n
    assert agg["layer_aware"]["same_order_as_coverage"] > 0.99
    assert abs(agg["layer_aware"]["within_layer"] - agg["coverage"]["within_layer"]) < 0.01
    assert agg["oracle"]["pooled"] == pytest.approx(1.0)
    assert ncells == 12


# ------------------------------------------------------------- Exp5

def test_adaptive_damping_rules(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rng = np.random.default_rng(1)
    with open(raw, "w") as f:
        for s in (0, 1):
            for L in (1, 8):
                for n in (16, 32, 64):
                    tok = (34 * n * rng.uniform(0.8, 1.2, 10)).tolist()
                    best = 2.0 if n < 64 else 1.0
                    err = {str(d): [{"all": 0.005 + 0.001 * abs(np.log(d / best)) / n + 1e-5 * rng.random()}
                                    for _ in range(10)] for d in (0.5, 1.0, 2.0, 3.0)}
                    f.write(json.dumps(dict(seed=s, layer=L, n=n, tokens=tok,
                                            d_in={"gu": 2048, "down": 1408}, err=err)) + "\n")
    res = AD.analyse(AD.load(str(raw)), 1.0)
    r16 = next(r for r in res["rows"] if r["n"] == 16)
    assert r16["per_prefix"] > 0 and r16["best_per_prefix_all_seeds"] == 2.0
    assert r16["per_expert_oracle"] >= r16["per_prefix"] - 1e-9
    assert next(r for r in res["rows"] if r["n"] == 64)["per_prefix"] == pytest.approx(0, abs=1e-6)
    assert "doubled" in r16 and r16["doubled"] is not None


# ------------------------------------------------------------- Exp6

def test_depth_generalization_interpolates():
    rng = np.random.default_rng(2)

    def make(layers, seeds):
        rows = []
        for s in seeds:
            for L in layers:
                for b in (16, 32, 64, 128):
                    n = rng.lognormal(np.log(34 * b), 0.15, 40)
                    slope = 0.2 + 0.02 * L                      # smooth in depth
                    y = 0.06 - slope * 0.01 * np.log(n / 500) + 0.001 * rng.normal(size=40)
                    rows += [(s, L, e, b, n[e], y[e]) for e in range(40)]
        a = np.array(rows)
        return dict(seed=a[:, 0], layer=a[:, 1], expert=a[:, 2], budget=a[:, 3],
                    count=a[:, 4], y=a[:, 5])
    old, new = make((1, 8, 16, 23), (0, 1, 2)), make((4, 12, 20), (0,))
    res = DG.evaluate(old, new)
    assert set(res) == {4, 12, 20}
    for r in res.values():
        assert r["gain_vs_global"]["interpolate"] > 0
        assert r["bracket"][0] < r["bracket"][1]
    prof = DG.depth_profile(old, new)
    assert prof[12]["src"] == "new" and prof[8]["src"] == "char"


# ------------------------------------------------------------- Exp4 (tiny model)

def test_exposure_diversity_run_report(cache_dir, tmp_path):
    out = str(tmp_path / "div")
    args = ["--cache-dir", cache_dir, "--out", out, "--layers", *map(str, LAYERS),
            "--seeds", "0", "--regimes", "8", "16", "--pool-n", "30", "--percdamp", "1.0"]
    ED.main(["run", *args])
    recs = [json.loads(l) for l in open(os.path.join(out, "exposure_diversity_raw.jsonl"))]
    assert len(recs) == 2 * len(LAYERS)
    v = next(iter(recs[0]["experts"].values()))
    assert set(v) <= set(ED.CONDS) and "natural" in v
    assert v["spread"]["tokens"] == v["natural"]["tokens"] == v["conc"]["tokens"]
    assert v["conc"]["sources"] <= v["natural"]["sources"] <= v["spread"]["sources"]
    assert v["half_seq"]["sources"] <= v["half_tok"]["sources"]
    ED.main(["run", *args])                                    # resume: nothing new
    assert len(open(os.path.join(out, "exposure_diversity_raw.jsonl")).readlines()) == len(recs)
    ED.main(["report", "--out", out])
    assert os.path.exists(os.path.join(out, "exposure_diversity_report.md"))


# ------------------------------------------------------------- Exp2 (tiny model)

def test_headroom_run_resume_report(cache_dir, tmp_path):
    out = str(tmp_path / "hr")
    common = ["--cache-dir", cache_dir, "--outdir", out, "--seeds", "0", "--budgets", "4", "8",
              "--percdamp", "1.0", "--no-ppl"]
    HR.main(["run", *common])
    files = sorted(os.listdir(out))
    for B in (0, 4, 8):
        assert f"e3_random_B{B}_b4_d1_s0.json" in files
    s0 = json.load(open(os.path.join(out, "e3_random_B0_b4_d1_s0.json")))["summary"]
    assert s0["n_selected"] == 0 and s0["n_zero_token_quantization_fallback"] == len(LAYERS) * E
    HR.main(["run", *common])                                  # resume: skips all
    HR.main(["report", "--outdir", out])
    rep = json.load(open(os.path.join(out, "headroom_report.json")))
    assert [r["B"] for r in rep["rows"]] == [0, 4, 8]


def test_within_layer_single_seed_uses_expert_folds():
    d = _labels(None, np.random.default_rng(3))
    one = {k: v[d["seed"] == 0] for k, v in d.items()}
    group, how = WL.folds_of(one)
    assert how.startswith("expert-grouped") and len(set(group.tolist())) == 5
    for g in set(group.tolist()):                       # an expert never spans folds
        assert set(one["expert"][group == g].tolist()).isdisjoint(one["expert"][group != g].tolist())
    agg, wins, ncells = WL.summarize(WL.run(one))
    assert ncells == 5 * 4
    assert abs(agg["layer_aware"]["within_layer"] - agg["coverage"]["within_layer"]) < 0.01
