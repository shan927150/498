"""Weight + integer-code differences for runs_dense_grouping_scope (supersedes dense_grouping_scope --stage weights).

The frozen driver's stage_weights called grouping_routing_amplification.weight_diff with dense layer keys; that
historical function hard-codes the MoE layer indices (LAYERS_Q = 1/8/16/23) for its per-layer tally and raised
KeyError: 9 before writing anything. Neither file is modified. Here the SAME weight_diff is called with the dense
layers relabelled onto its slots in order (1->1, 9->8, 19->16, 27->23) and the per-layer labels mapped back; every
other quantity is independent of the layer label. The integer-code block is copied verbatim from the driver.
"""
from __future__ import annotations
import os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_grouping_scope as S                                         # noqa: E402
import grouping_routing_amplification as GRA                              # noqa: E402


def main():
    pr = S.check_protocol()
    LAY = [int(x) for x in pr["quantization"]["layer_mapping"].values()]
    slot = dict(zip(LAY, GRA.LAYERS_Q))
    back = {str(v): str(k) for k, v in slot.items()}
    PROJ, CONFIGS = S.PROJ, S.CONFIGS

    W = {g: {(slot[L], 0, k): torch.load(S.c("weights", g, f"L{L}.pt"))[k] for L in LAY for k in PROJ} for g in CONFIGS}
    C = {g: {L: torch.load(S.c("codes", g, f"L{L}.pt")) for L in LAY} for g in CONFIGS}
    out = dict(weights={}, codes={})
    for a_, b_ in S.comparison_pairs():
        key = f"{a_}__{b_}"
        wd = GRA.weight_diff(W[a_], W[b_])
        wd["frac_changed_per_layer"] = {back[k]: v for k, v in wd["frac_changed_per_layer"].items()}
        out["weights"][key] = wd
        cd = dict(per_layer_proj={})
        n = nc = nb = nb_c = nc_nb = ngc = 0
        for L in LAY:
            for k in PROJ:
                qa, qb = C[a_][L][k]["q"], C[b_][L][k]["q"]
                ba, bb = W[a_][(slot[L], 0, k)], W[b_][(slot[L], 0, k)]
                code_d, bf_d = qa != qb, ba != bb
                grp_d = C[a_][L][k]["group"] != C[b_][L][k]["group"]
                n += qa.numel(); nc += int(code_d.sum()); nb += int(bf_d.sum())
                nb_c += int((bf_d & code_d).sum()); nc_nb += int((code_d & ~bf_d).sum())
                ngc += int(grp_d.sum()) * qa.shape[0]
                cd["per_layer_proj"][f"{L}.{k}"] = dict(
                    frac_codes_changed=float(code_d.float().mean()), frac_bf16_changed=float(bf_d.float().mean()),
                    frac_columns_group_changed=float(grp_d.float().mean()),
                    mean_abs_code_shift_changed=float((qa.int() - qb.int())[code_d].abs().float().mean()) if code_d.any() else 0.0)
        cd.update(frac_codes_changed=nc / n, frac_bf16_changed=nb / n,
                  frac_bf16_changed_with_code_change=nb_c / max(nb, 1), frac_bf16_changed_same_code=(nb - nb_c) / max(nb, 1),
                  frac_code_changed_same_bf16=nc_nb / max(nc, 1), frac_elements_in_group_changed_columns=ngc / n)
        out["codes"][key] = cd
        print(f"[weights] {key}: bf16 {wd['frac_changed']:.4f} codes {cd['frac_codes_changed']:.4f}", flush=True)
    out["note"] = ("weights: grouping_routing_amplification.weight_diff (identical Family D definition; dense layers relabelled "
                   "onto its 1/8/16/23 slots and mapped back); codes: integer q in original column order from the solver info "
                   "capture (bitwise-verified)")
    S.jdump(out, S.p("logs", "weight_differences_raw.json"))


if __name__ == "__main__":
    main()
