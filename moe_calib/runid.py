"""E1 run identity: never silently reuse results across configurations.

Every E1 run lives in its own directory named by a hash of its fingerprint
(model, cache/data seed, characterization seed, layers, n0, reps, K, bits,
group size, min tokens, holdout split). Re-running with the same config
resumes; any change gets a new directory; a directory whose stored
fingerprint disagrees with the requested one is rejected loudly.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os

FP_FILE = "fingerprint.json"


def fingerprint_hash(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:10]


DEFAULT_PERCDAMP = 0.01   # value used by every run made before percdamp was recorded


def damp_tag(percdamp) -> str:
    return f"d{float(percdamp):g}"


def fp_percdamp(fp: dict) -> float:
    return float(fp.get("percdamp", DEFAULT_PERCDAMP))


def e1_run_dir(outdir: str, fp: dict) -> str:
    """Create or resume the run directory for `fp`; raise on any mismatch.
    Runs that record GPTQ damping are named e1_b{bits}_d{damp}_s{seed}_{hash};
    older runs (no 'percdamp' key, i.e. 1% damping) keep e1_b{bits}_s{seed}_{hash}."""
    mid = f"_{damp_tag(fp['percdamp'])}" if "percdamp" in fp else ""
    path = os.path.join(outdir, f"e1_b{fp['bits']}{mid}_s{fp['char_seed']}_{fingerprint_hash(fp)}")
    fpath = os.path.join(path, FP_FILE)
    if os.path.exists(fpath):
        with open(fpath) as f:
            stored = json.load(f)
        if stored != fp:
            raise RuntimeError(f"E1 run dir {path} has a different fingerprint; "
                               f"refusing to resume.\nstored={stored}\nrequested={fp}")
    else:
        os.makedirs(path, exist_ok=True)
        with open(fpath, "w") as f:
            json.dump(fp, f, indent=2, sort_keys=True)
    return path


def _finished(pattern):
    return sorted(d for d in glob.glob(pattern) if os.path.exists(os.path.join(d, "labels.csv")))


def load_e1_runs(outdir: str, bits: int, dirs: list[str] | None = None,
                 percdamp: float | None = None):
    """Finished E1 runs (one per characterization seed). All fingerprints must
    agree on everything except char_seed. Returns (runs, common_fp).
    percdamp=None keeps the legacy behaviour (unnamed-damping runs only);
    a value selects runs made with that GPTQ damping (legacy runs count as
    1% damping)."""
    if not dirs:
        if percdamp is None:
            dirs = _finished(os.path.join(outdir, f"e1_b{bits}_s*_*"))
        else:
            dirs = _finished(os.path.join(outdir, f"e1_b{bits}_{damp_tag(percdamp)}_s*_*"))
            if float(percdamp) == DEFAULT_PERCDAMP:
                dirs += _finished(os.path.join(outdir, f"e1_b{bits}_s*_*"))
    if not dirs:
        raise FileNotFoundError(f"no finished E1 runs for bits={bits} in {outdir}")
    runs, common = [], None
    for d in dirs:
        with open(os.path.join(d, FP_FILE)) as f:
            fp = json.load(f)
        rest = {k: v for k, v in fp.items() if k != "char_seed"}
        rest["percdamp"] = fp_percdamp(fp)
        if common is None:
            common = rest
        elif rest != common:
            raise RuntimeError(f"E1 run {d} is incompatible with {dirs[0]} (only "
                               f"char_seed may differ); pass --e1-dirs explicitly.\n"
                               f"{rest}\nvs\n{common}")
        with open(os.path.join(d, "draws.json")) as f:
            draws = {int(k): v for k, v in json.load(f).items()}
        runs.append(dict(dir=d, fp=fp, char_seed=fp["char_seed"], draws=draws,
                         labels=os.path.join(d, "labels.csv")))
    seeds = [r["char_seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate characterization seeds in {dirs}")
    return runs, common
