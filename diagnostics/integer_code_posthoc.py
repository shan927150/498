"""POST-HOC (not prespecified; does not enter the protocol classification): code-flip rate along GPTQ execution order.

Motivated by Fig. 1 of runs_integer_code_mechanism (flips concentrate in late execution groups). For each primary dense
pair (4 x vs ORIGINAL) and each MoE primary pair (5 x vs ORIGINAL, 20 sampled experts), per projection: in the
REFERENCE's act-order, the code-flip rate, the zero-relative level (q - zero) change rate, and the fraction of
(row, group) cells whose zero-point changed, by execution-position decile; and the share of flips inside whole-group
shifts (all 128 codes of a (row, group) cell flipped). Execution position == rank of diag(H) (act-order), so position
effects cannot be separated from diag(H) magnitude here.
"""
from __future__ import annotations
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integer_code_mechanism as S                                  # noqa: E402
from moe_calib import code_spatial as CS                            # noqa: E402

NB = 10


def profile(ra, rb):
    F = ra["q"] != rb["q"]
    Sa, Za = CS.element_grid(ra)
    Sb, Zb = CS.element_grid(rb)
    lv = (ra["q"].float() - Za) != (rb["q"].float() - Zb)
    perm = rb["perm"].long()
    Fe, Le, Ze = F[:, perm], lv[:, perm], (Za != Zb)[:, perm]
    R, C = F.shape
    edges = np.linspace(0, C, NB + 1).astype(int)
    ng = C // 128
    G = Fe.view(R, ng, 128)
    full = G.all(-1)
    return dict(flip=[float(Fe[:, a:b].float().mean()) for a, b in zip(edges[:-1], edges[1:])],
                level=[float(Le[:, a:b].float().mean()) for a, b in zip(edges[:-1], edges[1:])],
                zero_changed=[float(Ze[:, a:b].float().mean()) for a, b in zip(edges[:-1], edges[1:])],
                flips=int(F.sum()), flips_in_whole_group_shifts=int(full.sum()) * 128,
                group_flip_rate_by_exec_group=G.float().mean((0, 2)).tolist())


def agg(ps):
    w = np.array([p["flips"] for p in ps], float)
    return dict(flip=np.mean([p["flip"] for p in ps], 0).tolist(), level=np.mean([p["level"] for p in ps], 0).tolist(),
                zero_changed=np.mean([p["zero_changed"] for p in ps], 0).tolist(),
                share_flips_in_whole_group_shifts=float(sum(p["flips_in_whole_group_shifts"] for p in ps) / max(w.sum(), 1)))


def main():
    out = dict(note=__doc__, dense={}, moe={})
    for pj in S.PROJ:
        ps = []
        for va, rf in S.DENSE_PRIMARY:
            for L in S.DLAY:
                ps.append(profile(S.dense_rec(va, L)[pj], S.dense_rec(rf, L)[pj]))
        out["dense"][pj] = agg(ps)
        out["dense"][pj + "_by_layer"] = {str(L): agg([ps[i] for i in range(len(ps)) if i % len(S.DLAY) == j])
                                          for j, L in enumerate(S.DLAY)}
    em = S.jload(S.p("moe_expert_manifest.json"))
    R = {g: {L: S.moe_rec(g, L) for L in S.MLAY} for g in S.GROUPS + ["ORIGINAL"]}
    for pj in S.PROJ:
        ps = [profile(R[va][x["layer"]][x["expert"]][pj], R[rf][x["layer"]][x["expert"]][pj])
              for va, rf in S.MOE_PRIMARY for x in em["experts"]]
        out["moe"][pj] = agg(ps)
    S.jdump(out, S.p("logs", "posthoc_execution_profile.json"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    x = (np.arange(NB) + .5) / NB
    cols = {"gate_proj": "tab:blue", "up_proj": "tab:orange", "down_proj": "tab:green"}
    for ax, m in ((axs[0], "dense"), (axs[1], "moe")):
        for pj in S.PROJ:
            d = out[m][pj]
            ax.plot(x, d["flip"], "-o", ms=3, color=cols[pj], label=f"{pj} code flip")
            ax.plot(x, d["level"], ":", color=cols[pj], label=f"{pj} level (q-zero) change")
        ax.set_yscale("log"); ax.set_xlabel("execution position (reference act-order; 0 = largest diag H)")
        ax.set_title(f"{m}: post-hoc profile along GPTQ order", fontsize=9); ax.legend(fontsize=6)
    axs[0].set_ylabel("rate")
    fig.tight_layout(); fig.savefig(S.p("figures", "fig7_posthoc_execution_profile.png"), dpi=140); plt.close(fig)
    for m in ("dense", "moe"):
        for pj in S.PROJ:
            d = out[m][pj]
            print(m, pj, "flip", [f"{v:.4f}" for v in d["flip"]], "whole-group share", round(d["share_flips_in_whole_group_shifts"], 3))


if __name__ == "__main__":
    main()
