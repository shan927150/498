"""Per-expert Hessian early-stopping feasibility MVP: pure helpers (no model code).

Checkpoints are CHRONOLOGICAL prefixes of the frozen ordered calibration manifest (S16 c S32 c S64 c S128), never the
hashed statistical blocks of moe_calib.hessian.block_of.
"""
from __future__ import annotations

import numpy as np
import torch

CHECKPOINTS = (16, 32, 64, 128)
DECISION_POINTS = (32, 64)           # B = 16 is warm-up (no earlier checkpoint for drift)


def prefixes(order, checkpoints=CHECKPOINTS):
    """{B: first B entries of the ordered manifest}; asserts nesting and no duplicates."""
    order = list(order)
    assert len(set(order)) == len(order), "duplicate sequence in manifest"
    assert max(checkpoints) <= len(order)
    out = {B: order[:B] for B in checkpoints}
    for a, b in zip(checkpoints, checkpoints[1:]):
        assert set(out[a]) < set(out[b])
    return out


def contiguous_blocks(n, K):
    """Profiler.run's default block of position i (production E3): (i * K) // n."""
    return [(i * K) // n for i in range(n)]


# ----------------------------------------------------------------------------- drift statistics
def trace_drift(H_b, n_b, H_half, n_half, eps=1e-12):
    """|tr(H_b/n_b) - tr(H_half/n_half)| / (|tr(H_b/n_b)| + eps); NaN when either count is 0."""
    if n_b <= 0 or n_half <= 0:
        return float("nan")
    tb = float(torch.trace(H_b.double())) / n_b
    th = float(torch.trace(H_half.double())) / n_half
    return abs(tb - th) / (abs(tb) + eps)


def frob_drift(H_b, n_b, H_half, n_half, eps=1e-12):
    """||H_b/n_b - H_half/n_half||_F / (||H_b/n_b||_F + eps); NaN when either count is 0."""
    if n_b <= 0 or n_half <= 0:
        return float("nan")
    A = H_b.double() / n_b
    Bm = H_half.double() / n_half
    return float(torch.linalg.norm(A - Bm) / (torch.linalg.norm(A) + eps))


# ----------------------------------------------------------------------------- stopping rules
def rule_ok(stats_e: dict, rule: str, thr: dict) -> bool:
    """stats_e: {"gu": {"trace":..,"frob":..}, "down": {...}, "n": tokens}. Both projection groups must pass;
    NaN statistics (zero exposure) never pass."""
    def ok(v, t):
        return bool(np.isfinite(v) and v < t)
    if rule == "trace":
        return ok(stats_e["gu"]["trace"], thr["tau_trace"]) and ok(stats_e["down"]["trace"], thr["tau_trace"])
    if rule == "frob":
        return ok(stats_e["gu"]["frob"], thr["tau_frob"]) and ok(stats_e["down"]["frob"], thr["tau_frob"])
    if rule == "frob_exposure":
        return (ok(stats_e["gu"]["frob"], thr["tau_frob"]) and ok(stats_e["down"]["frob"], thr["tau_frob"])
                and stats_e["n"] >= thr["n_min"])
    if rule == "exposure":
        return stats_e["n"] >= thr["n_min"]
    raise ValueError(rule)


def assign_stop_budgets(stats: dict, rule: str, thr: dict, decision_points=DECISION_POINTS, final=128) -> dict:
    """stats[B][key] = statistics available AT checkpoint B only (no future data). Returns {key: B_stop}.
    An expert stopped at an earlier decision point is never re-evaluated."""
    keys = list(stats[decision_points[0]].keys())
    out = {}
    for k in keys:
        out[k] = final
        for B in decision_points:
            if rule_ok(stats[B][k], rule, thr[B] if B in thr else thr):
                out[k] = B
                break
    return out


# ----------------------------------------------------------------------------- outcomes
def degradation(err_stop, err_full):
    return (np.asarray(err_stop, float) - np.asarray(err_full, float)) / np.asarray(err_full, float)


def false_safe(err_stop, err_full, stopped_mask, tol=0.01):
    """Among STOPPED experts, fraction with E_stop > (1 + tol) E_128."""
    d = degradation(err_stop, err_full)
    m = np.asarray(stopped_mask, bool)
    if m.sum() == 0:
        return dict(n_stopped=0, n_false=0, rate=0.0)
    nf = int((d[m] > tol).sum())
    return dict(n_stopped=int(m.sum()), n_false=nf, rate=nf / int(m.sum()))


def workload(tokens_by_B: dict, stop: dict, d_gu: int, d_down: int, final=128) -> dict:
    """Hessian accumulation work ~ routed rows x (d_gu^2 + d_down^2) per expert. tokens_by_B[B][key] = rows through B."""
    full = sum(tokens_by_B[final][k] for k in stop) * (d_gu ** 2 + d_down ** 2)
    done = sum(tokens_by_B[stop[k]][k] for k in stop) * (d_gu ** 2 + d_down ** 2)
    pairs_full = sum(tokens_by_B[final][k] for k in stop)
    pairs_done = sum(tokens_by_B[stop[k]][k] for k in stop)
    return dict(work_fraction_done=done / full, work_fraction_skipped=1 - done / full,
                token_expert_pairs_full=int(pairs_full), token_expert_pairs_done=int(pairs_done),
                mean_effective_budget=float(np.mean([stop[k] for k in stop])),
                n_stop={B: int(sum(v == B for v in stop.values())) for B in sorted(set(list(stop.values()) + [final]))})


def frontier_interp(points: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation of (skipped_work, mean_degradation) fixed-budget points at skipped-work x."""
    pts = sorted(points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(x, xs, ys))


def project_time(cost: dict, work_fraction_done: float) -> dict:
    """Cost model from MEASURED components: replay = forward + hook overhead + work_fraction_done x accumulation;
    total = replay + solve. Output is a PROJECTION (labelled)."""
    acc = cost["accumulation_s"] * work_fraction_done
    replay = cost["forward_s"] + cost["hook_overhead_s"] + acc
    return dict(kind="PROJECTED (cost model from measured components; not a measured run)",
                replay_s=replay, total_s=replay + cost["solve_s"] + cost.get("other_s", 0.0))


def classify(science: dict, systems_ok: bool, blocked: bool = False) -> str:
    """Frozen precedence: BLOCKED > SIGNAL-FAIL (no frozen rule passes the scientific gate) > NO-ADVANTAGE (a rule
    passes but none beats the fixed-budget frontier) > SYSTEMS-FAIL (science + advantage, systems gate fails) > PROMISING."""
    if blocked:
        return "EARLY-STOP-BLOCKED"
    if not science["any_rule_passes"]:
        return "EARLY-STOP-SIGNAL-FAIL"
    if not science["any_passing_rule_beats_frontier"]:
        return "EARLY-STOP-NO-ADVANTAGE"
    if not systems_ok:
        return "EARLY-STOP-SYSTEMS-FAIL"
    return "EARLY-STOP-PROMISING"
