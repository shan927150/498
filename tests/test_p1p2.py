"""Tests for the frozen-damping pipeline (P0-P2): GPTQ damping is threaded
through E1/E2/E3 and recorded in run identity; the E1 / E2 gates; the
extended damping + per-projection threshold diagnostic; E3 routing-shift
logging."""
import json
import os
import sys

import numpy as np
import pytest
import torch

import e1_gate
import e1_label_sufficiency as E1
import e2_gate
import e2_signals as E2
import e3_allocation as E3
from conftest import E, LAYERS, tiny_model
from moe_calib import model_utils as MU
from moe_calib.quant_eval import quantize_from_H
from moe_calib.runid import load_e1_runs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "diagnostics"))

DAMP = 0.2


@pytest.fixture(scope="module")
def damped(cache_dir, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("runs_damped"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        common = ["--cache-dir", cache_dir, "--outdir", out]
        for seed in (0, 1, 2):
            E1.main(common + ["--n0", "4", "8", "--reps", "2", "--min-paired-seqs", "3",
                              "--char-seed", str(seed), "--percdamp", str(DAMP)])
        # a legacy (unrecorded = 1%) run in the same outdir must not be mixed in
        E1.main(common + ["--n0", "4", "8", "--reps", "2", "--min-paired-seqs", "3",
                          "--char-seed", "0"])
        art = E2.main(common + ["--min-char-seeds", "3", "--percdamp", str(DAMP)])
        e3 = json.load(open(E3.main(common + ["--B", "12", "--strategy", "tlp",
                                              "--percdamp", str(DAMP)])))
        e3r = json.load(open(E3.main(common + ["--B", "12", "--strategy", "random",
                                               "--percdamp", str(DAMP)])))
    return dict(out=out, art=art, e3=e3, e3r=e3r, cache_dir=cache_dir)


def test_percdamp_changes_quantization(cache_dir):
    model, _ = tiny_model()
    d = MU.expert_dims(model)
    g = torch.Generator().manual_seed(0)
    Xg, Xd = torch.randn(40, d[0], generator=g), torch.randn(40, d[1], generator=g)
    a = quantize_from_H(model, LAYERS[0], 0, Xg.T @ Xg, Xd.T @ Xd, percdamp=0.01)
    b = quantize_from_H(model, LAYERS[0], 0, Xg.T @ Xg, Xd.T @ Xd, percdamp=1.0)
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_damping_recorded_and_runs_separated(damped):
    runs, common = load_e1_runs(damped["out"], 4, None, DAMP)
    assert len(runs) == 3 and common["percdamp"] == DAMP
    assert all("_d0.2_" in os.path.basename(r["dir"]) for r in runs)
    legacy, lc = load_e1_runs(damped["out"], 4)                      # legacy selection
    assert len(legacy) == 1 and lc["percdamp"] == 0.01
    assert damped["art"]["percdamp"] == DAMP
    assert os.path.exists(os.path.join(damped["out"], "e2_model_b4_d0.2.json"))


def test_e3_uses_and_checks_damping(damped):
    s = damped["e3"]["summary"]
    assert s["percdamp"] == DAMP
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        with pytest.raises(ValueError, match="differs from the E2 model"):
            E3.main(["--cache-dir", damped["cache_dir"], "--outdir", damped["out"], "--B", "12",
                     "--strategy", "tlp", "--percdamp", str(DAMP),
                     "--signal-model", os.path.join(damped["out"], "e2_model_b4_d0.2.json"),
                     "--percdamp", "0.05"])


def test_routing_shift_logged(damped):
    for k in ("e3", "e3r"):
        sh = damped[k]["summary"]["routing_shift"]
        assert set(sh["per_layer"]) == {str(L) for L in LAYERS}
        assert 0 <= sh["mean_l1"] <= 2 and 0 <= sh["mean_js"] <= np.log(2) + 1e-9
    assert os.path.exists(os.path.join(damped["out"], E3.ROUTING_CACHE))


def test_routing_shift_known_values():
    C = np.array([[2, 0], [0, 2], [2, 0], [0, 2]], float)          # 1 layer, 2 experts
    same = E3.routing_shift(C, [0, 1, 2, 3], [0, 1], [7], 2)
    assert same["mean_l1"] == pytest.approx(0) and same["mean_js"] == pytest.approx(0)
    skew = E3.routing_shift(C, [0, 1, 2, 3], [0, 2], [7], 2)
    assert skew["mean_l1"] == pytest.approx(1.0)                  # |1-.5|+|0-.5|
    assert skew["per_layer"]["7"]["n_experts_unseen"] == 1


def test_e1_gate_runs_and_flags(damped):
    code = e1_gate.main(["--outdir", damped["out"], "--percdamp", str(DAMP)])
    res = json.load(open(os.path.join(damped["out"], "e1_gate_b4_d0.2.json")))
    assert code == (0 if res["passed"] else 1)
    assert sorted(int(k) for k in res["per_n0"]) == [4, 8]
    rows = [dict(seed=s, layer=1, expert=e, n0=16, rel=-0.1, abs=-1e-3)
            for s in (0, 1) for e in range(10)]
    assert not e1_gate.evaluate(rows)["passed"]
    rows = [dict(seed=s, layer=1, expert=e, n0=n, rel=0.2, abs=1e-3 / n)
            for s in (0, 1) for e in range(10) for n in (16, 32)]
    assert e1_gate.evaluate(rows)["passed"]


def _synth(kind, rng):
    rows = []
    for seed in range(3):
        for L in (1, 8, 16):
            for n0 in (16, 32, 64):
                cnt = rng.lognormal(np.log(35 * n0), 0.3, 40)
                lc = np.log(cnt)
                hidden = rng.normal(size=40)
                y = 0.3 * np.sin(lc - 7.0) + 0.02 * rng.normal(size=40)
                if kind == "real":
                    y = y + 0.03 * hidden
                    sig = hidden + 0.5 * rng.normal(size=40)
                else:                                               # knows only count
                    sig = np.sin(lc - 7.0) + 0.05 * rng.normal(size=40)
                rows += [(seed, L, n0, cnt[e], y[e], sig[e]) for e in range(40)]
    a = np.array(rows)
    d = dict(seed=a[:, 0], layer=a[:, 1], budget=a[:, 2], count=a[:, 3], y=a[:, 4])
    for s in e2_gate.STATE_SIGNALS:
        d[s] = a[:, 5] if s == "relse" else rng.normal(size=len(a))
    return d


def test_e2_gate_detects_real_signal_and_rejects_count_proxy():
    rng = np.random.default_rng(0)
    real = e2_gate.run_gate(_synth("real", rng), R=30)
    assert real["case"] == "A" and real["selected_signal"] == "relse"
    proxy = e2_gate.run_gate(_synth("proxy", rng), R=30)
    # a count-only proxy wins against LINEAR count but not against flexible exposure
    assert proxy["signals"]["relse"]["old_style_LOSO_gain_vs_linear_count"] > 0.2
    assert proxy["case"] == "C"


def test_e2_gate_cli(damped):
    code = e2_gate.main(["--outdir", damped["out"], "--percdamp", str(DAMP),
                         "--permutations", "5"])
    res = json.load(open(os.path.join(damped["out"], "e2_gate_b4_d0.2.json")))
    assert code in (0, 1, 2) and res["case"] in ("A", "A_budget", "B", "C")
    assert res["noise_ceiling"] is not None


@pytest.fixture(scope="module")
def big_cache(tmp_path_factory):
    """Pool >= 256 so the nested prefixes 16..256 exist."""
    from conftest import D
    d = tmp_path_factory.mktemp("bigcache")
    P, H, T = 270, 24, 8
    g = torch.Generator().manual_seed(3)
    torch.save({"pool": torch.zeros(P, T, dtype=torch.long),
                "holdout": torch.zeros(H, T, dtype=torch.long)}, d / "ids.pt")
    for L in LAYERS:
        torch.save(torch.randn(P + H, T, D, generator=g).to(torch.bfloat16), d / f"L{L}.pt")
    (d / "meta.json").write_text(json.dumps(dict(
        pool_size=P, holdout_size=H, tune_size=12, seqlen=T, data_seed=0, layers=LAYERS,
        model="tiny", num_layers=3, cache_seconds=0.0, forward_s_per_seq=0.01)))
    return str(d)


def test_damping_threshold_run_resume_report(big_cache, tmp_path, monkeypatch):
    import damping_threshold as DT
    import low_data_diag as LD
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MU, "load_model", lambda *a, **k: tiny_model())
    DT.cmd_run(type("A", (), dict(cache_dir=big_cache, layers=[LAYERS[0]], seeds=[0],
                                  damps=[0.01, 0.5]))())
    raw = LD.read_jsonl(DT.RAW)
    assert len(raw) == len(LD.PREFIXES)
    # resume with one extra damping: only the missing value is computed
    DT.cmd_run(type("A", (), dict(cache_dir=big_cache, layers=[LAYERS[0]], seeds=[0],
                                  damps=[0.01, 0.5, 1.0]))())
    raw = LD.read_jsonl(DT.RAW)
    assert len(raw) == 2 * len(LD.PREFIXES)
    assert all(list(r["err"]) == ["1.0"] for r in raw[len(LD.PREFIXES):])
    out = DT.cmd_report(None)
    assert set(out["summary"]) == {"0.01", "0.5", "1.0"}
    assert out["best_overall"]["percdamp"] in (0.01, 0.5, 1.0)
    r0 = raw[0]
    assert r0["d_in"] == dict(zip(("gu", "down"), MU.expert_dims(tiny_model()[0])))
    assert 0 < r0["erank"]["gu"][0] <= r0["d_in"]["gu"]
    for e in r0["rtn"]:
        assert set(e) == {"all", "gu", "down"}


def test_spectrum_stats_known():
    import damping_threshold as DT
    er, pr = DT.spectrum_stats(torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.0])))
    assert er == pytest.approx(3.0, rel=1e-6) and pr == pytest.approx(3.0, rel=1e-6)
