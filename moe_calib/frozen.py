"""Frozen production experiment configuration (configs/frozen_*.json).

The config is the single source of truth for the GPTQ setting and the
characterization design once P0 has frozen them. Consumers do not read their
own defaults for these values; they validate what they are about to run (an
E1 fingerprint) against the config and refuse any mismatch.
"""
from __future__ import annotations
import inspect
import json


def load_frozen(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _act_order_default() -> bool:
    from .gptq import gptq_quantize                 # production path relies on this default
    return inspect.signature(gptq_quantize).parameters["actorder"].default


def check_e1_fp(cfg: dict, fp: dict, check_seed: bool = True) -> None:
    """Raise ValueError if an E1 fingerprint deviates from the frozen config."""
    g, e1 = cfg["gptq"], cfg["e1"]
    want = dict(model=cfg["model"], bits=g["bits"], group_size=g["group_size"],
                percdamp=float(g["percdamp"]), layers=list(cfg["layers"]),
                n0=list(e1["n0"]), reps=e1["reps"], K=e1["K"])
    bad = [f"{k}: run={fp.get(k)!r} frozen={v!r}" for k, v in want.items() if fp.get(k) != v]
    split = {k: fp.get("split", {}).get(k) for k in cfg["split"]}
    if split != cfg["split"]:
        bad.append(f"split: run={split} frozen={cfg['split']}")
    if check_seed and fp.get("char_seed") not in e1["char_seeds"]:
        bad.append(f"char_seed {fp.get('char_seed')} not in frozen {e1['char_seeds']}")
    if _act_order_default() != g["act_order"]:
        bad.append(f"act_order: gptq default={_act_order_default()} frozen={g['act_order']}")
    if bad:
        raise ValueError(f"run deviates from frozen config {cfg.get('name')}:\n  "
                         + "\n  ".join(bad))
