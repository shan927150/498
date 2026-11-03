"""Exp2 — headroom: how much do expert reconstruction and PPL move with the
calibration budget under the frozen GPTQ?  (Random selection only.)

Any allocation method can at best recover part of what a larger random
budget buys. If doubling B barely moves PPL beyond seed noise, allocation
gains will not be visible at model level and E3 should be reported on
expert reconstruction.

Runs e3_allocation.py (strategy random) in ONE process with the model loaded
once: B = 0 (no calibration -> every expert falls back to RTN; also the FP
PPL), then B in 32/128/512/64/256 for each seed. Each finished run is a JSON
file; re-running skips it (resumable). Test split only, as in E3.
Note: only the profiled layers' routed experts are quantized (4 of 24
layers), so PPL differences are diluted; compare across B, not to papers.

  python diagnostics/headroom.py run    --cache-dir ~/moe_cache --outdir runs_headroom
  python diagnostics/headroom.py report --outdir runs_headroom
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import e3_allocation as E3                                   # noqa: E402
from moe_calib import model_utils as MU                      # noqa: E402
from moe_calib.runid import damp_tag                         # noqa: E402

ORDER = [32, 128, 512, 64, 256]


def out_name(outdir, B, bits, d, seed):
    return os.path.join(outdir, f"e3_random_B{B}_b{bits}_{damp_tag(d)}_s{seed}.json")


def cmd_run(a):
    os.makedirs(a.outdir, exist_ok=True)
    model, tok = MU.load_model()
    MU.load_model = lambda *x, **k: (model, tok)              # reuse for every E3 run
    plan = [(0, 0)] + [(B, s) for s in a.seeds for B in a.budgets]
    for B, s in plan:
        path = out_name(a.outdir, B, a.bits, a.percdamp, s)
        if os.path.exists(path):
            print(f"[headroom] skip B={B} seed={s} (done)", flush=True)
            continue
        argv = ["--strategy", "random", "--B", str(B), "--seed", str(s), "--K", "1",
                "--bits", str(a.bits), "--percdamp", str(a.percdamp),
                "--cache-dir", a.cache_dir, "--outdir", a.outdir]
        if a.layers:
            argv += ["--layers", *map(str, a.layers)]
        if not a.no_ppl:
            argv += ["--ppl", "--ppl-windows", str(a.ppl_windows)]
            if B == 0:
                argv.append("--fp-ppl")
        print(f"[headroom] B={B} seed={s}", flush=True)
        got = E3.main(argv)
        assert os.path.abspath(got) == os.path.abspath(path), (got, path)


def cmd_report(a):
    runs = [json.load(open(f))["summary"] for f in glob.glob(os.path.join(a.outdir, "e3_random_B*.json"))]
    if not runs:
        raise SystemExit(f"no runs in {a.outdir}")
    by = {}
    for s in runs:
        by.setdefault(s["B"], []).append(s)
    fp = next((s.get("wikitext2_ppl_fp") for s in by.get(0, []) if s.get("wikitext2_ppl_fp")), None)
    rows, M = [], ["# Exp2 — budget headroom (random selection, frozen GPTQ, test split)", ""]
    if fp:
        M.append(f"FP PPL (no quantization): {fp:.4f}")
    M += ["", "| B | seeds | expert recon mean | p90 | rel. change vs previous B | PPL mean ± sd |",
          "|---|---|---|---|---|---|"]
    prev = None
    for B in sorted(by):
        rs = by[B]
        rec = np.array([r["recon_mean"] for r in rs])
        p90 = np.array([r["recon_p90"] for r in rs])
        ppl = np.array([r["wikitext2_ppl"] for r in rs if "wikitext2_ppl" in r])
        rel = "" if prev is None or prev == 0 else f"{100 * (rec.mean() / prev - 1):+.2f}%"
        pstr = (f"{ppl.mean():.4f} ± {ppl.std(ddof=1) if len(ppl) > 1 else 0:.4f}" if len(ppl) else "–")
        label = "0 (RTN)" if B == 0 else str(B)
        M.append(f"| {label} | {len(rs)} | {rec.mean():.5f} | {p90.mean():.5f} | {rel} | {pstr} |")
        rows.append(dict(B=B, n=len(rs), recon_mean=float(rec.mean()), recon_sd=float(rec.std()),
                         p90=float(p90.mean()), ppl_mean=float(ppl.mean()) if len(ppl) else None,
                         ppl_sd=float(ppl.std(ddof=1)) if len(ppl) > 1 else None))
        prev = rec.mean()
    M += ["", "Reading: compare each step's PPL change with the seed sd at the same B. "
          "If a doubling is within ~2 sd, model-level gains from allocation (a fraction of a "
          "doubling) will not be detectable; report E3 on expert reconstruction instead."]
    json.dump(dict(fp_ppl=fp, rows=rows), open(os.path.join(a.outdir, "headroom_report.json"), "w"),
              indent=2)
    open(os.path.join(a.outdir, "headroom_report.md"), "w").write("\n".join(M) + "\n")
    print("\n".join(M))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--outdir", default="runs_headroom")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budgets", type=int, nargs="+", default=ORDER)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layers to quantize (default: every layer in the cache)")
    ap.add_argument("--ppl-windows", type=int, default=40)
    ap.add_argument("--no-ppl", action="store_true", help="reconstruction only (tests)")
    a = ap.parse_args(argv)
    return dict(run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
