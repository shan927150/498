"""POST-HOC sensitivity (not prespecified; does not enter the F1/F2/F3 classification): layer-matched MoE/dense functional
ratios. The prespecified R_MLP pools squared norms over the four layers, which is dominated by the layers with the largest
output norms; here the depth-matched layer pairs (1-1, 8-9, 16-19, 23-27) are compared individually, and the mean of
per-layer relative errors is reported alongside. Also: grouping-induced difference relative to each model's own
quantization error (variant vs BF16) per layer."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actorder_functional_stability as S

def main():
    d = S.jload(S.p("dense_functional_difference.json"))["per_layer"]
    m = S.jload(S.p("moe_functional_difference.json"))
    agg, pp = m["per_layer_aggregate"], m["per_layer_pooled_projection"]
    DP = [S.pk(*x) for x in S.DENSE_PRIMARY]; MP = [S.pk(*x) for x in S.MOE_PRIMARY]
    out = dict(note=__doc__, layers={})
    for Ld, Lm in zip(S.DLAY, S.MLAY):
        r = dict(dense_layer=Ld, moe_layer=Lm)
        dm = np.mean([d[str(Ld)][k]["mlp"]["E_func"] for k in DP])
        mb = np.mean([agg[str(Lm)][k]["vs_block"]["E_func"] for k in MP])
        mr = np.mean([agg[str(Lm)][k]["vs_routed"]["E_func"] for k in MP])
        r.update(dense_E_mlp=dm, moe_E_block=mb, moe_E_routed=mr, R_block=mb / dm, R_routed=mr / dm)
        for k in S.PROJ:
            de = np.mean([d[str(Ld)][x][k]["E_func"] for x in DP]); me = np.mean([pp[str(Lm)][x][k]["E_func"] for x in MP])
            r[f"R_{k}"] = me / de
        dq = d[str(Ld)]["ORIGINAL__BF16"]["mlp"]["E_func"]
        r.update(dense_quant_E=dq, moe_quant_E_block=agg[str(Lm)]["ORIGINAL__BF16"]["vs_block"]["E_func"],
                 moe_quant_E_routed=agg[str(Lm)]["ORIGINAL__BF16"]["vs_routed"]["E_func"],
                 dense_grouping_over_quant=dm / dq, moe_grouping_over_quant_block=mb / agg[str(Lm)]["ORIGINAL__BF16"]["vs_block"]["E_func"],
                 moe_grouping_over_quant_routed=mr / agg[str(Lm)]["ORIGINAL__BF16"]["vs_routed"]["E_func"])
        out["layers"][f"{Ld}-{Lm}"] = r
    L = out["layers"].values()
    out["mean_of_layer_E"] = dict(dense=float(np.mean([x["dense_E_mlp"] for x in L])), moe_block=float(np.mean([x["moe_E_block"] for x in L])),
                                  moe_routed=float(np.mean([x["moe_E_routed"] for x in L])))
    out["mean_of_layer_E"]["R_block"] = out["mean_of_layer_E"]["moe_block"] / out["mean_of_layer_E"]["dense"]
    out["mean_of_layer_E"]["R_routed"] = out["mean_of_layer_E"]["moe_routed"] / out["mean_of_layer_E"]["dense"]
    S.jdump(out, S.p("logs", "posthoc_layer_matched_ratios.json"))
    for k, v in out["layers"].items():
        print(k, {a: round(b, 3) for a, b in v.items() if isinstance(b, float)})
    print(out["mean_of_layer_E"])

if __name__ == "__main__":
    main()
