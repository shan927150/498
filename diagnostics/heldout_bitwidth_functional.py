"""OPTIONAL (protocol section 14, run after the 3-bit PPL results showed a change in relative variability):
3-bit vs 4-bit functional disagreement of the dense grouping variants, with the historical pooled full-MLP norm.

R_func,b(g) = sqrt( sum_L ||f_b,g(X) - f_b,ORIG(X)||^2 / sum_L ||f_b,ORIG(X) - f_BF16(X)||^2 )
X = the saved BF16 MLP inputs on the WikiText-2 evaluation windows (runs_actorder_functional_stability act/dense; not
calibration data). Reuses diagnostics/dense_dose_response.functional_one / func_cache unchanged (FP32, TF32 off,
FP64 accumulation). Integer-code disagreement is compared WITHIN each bit-width only.
"""
from __future__ import annotations
import json, math, os, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import heldout_bitwidth_validation as H        # noqa: E402
import dense_dose_response as DDR              # noqa: E402
from moe_calib import functional_diff as FD    # noqa: E402

LAY, PROJ, G = H.DENSE_LAYERS, H.PROJ, H.GROUPS


def weights(bits, g):
    if bits == 4:
        return ({L: torch.load(os.path.join(H.DCACHE, "weights", g, f"L{L}.pt")) for L in LAY},
                {L: torch.load(os.path.join(H.DCACHE, "codes", g, f"L{L}.pt")) for L in LAY})
    d = {L: torch.load(H.d3dir(g, f"L{L}.pt")) for L in LAY}
    return {L: d[L]["weights"] for L in LAY}, {L: d[L]["codes"] for L in LAY}


@torch.no_grad()
def f0_of(W):
    """f_ORIG for a bit-width, computed exactly like DDR.func_cache's f0."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    D = DDR.dev()
    out = {}
    for L in LAY:
        act = torch.load(os.path.join(DDR.ACACHE, "act", "dense", f"L{L}.pt"), mmap=True)
        Wo = {k: v.to(D).float() for k, v in W[L].items()}
        f0 = []
        for i in range(0, act["X"].shape[0], DDR.CHUNK):
            x = act["X"][i:i + DDR.CHUNK].to(D).float()
            f0.append(FD.mlp_fp32(x, Wo["gate_proj"], Wo["up_proj"], Wo["down_proj"]).cpu())
        out[L] = torch.cat(f0)
    return out


def main():
    t0 = time.time()
    ref4 = DDR.func_cache()                                         # historical: f0 = 4-bit ORIGINAL, fb = BF16
    out = dict(definition=__doc__, bits={})
    for bits in (4, 3):
        Wo, Co = weights(bits, "ORIGINAL")
        if bits == 4:
            ref = ref4
        else:
            f3 = f0_of(Wo)
            ref = {L: dict(f0=f3[L], fb=ref4[L]["fb"]) for L in LAY}
        sq_q = sum(float((ref[L]["f0"].double() - ref[L]["fb"].double()).pow(2).sum()) for L in LAY)
        sq_b = sum(float(ref4[L]["fb"].double().pow(2).sum()) for L in LAY)
        rec = dict(quant_error_rel=math.sqrt(sq_q / sq_b), configs={})
        for g in G:
            W, C = weights(bits, g)
            r = DDR.functional_one(W, ref, Wo)
            num = sum(r[str(L)]["sq_diff_vs_orig"] for L in LAY)
            n_el = n_ch = 0
            for L in LAY:
                for k in PROJ:
                    a, b = C[L][k]["q"], Co[L][k]["q"]
                    n_el += a.numel(); n_ch += int((a != b).sum())
            rec["configs"][g] = dict(R_func=math.sqrt(num / sq_q), E_mlp_rel_bf16=math.sqrt(num / sq_b),
                                     code_disagreement_vs_original=n_ch / n_el,
                                     bf16_elements_changed=sum(int((W[L][k] != Wo[L][k]).sum()) for L in LAY for k in PROJ) /
                                     sum(W[L][k].numel() for L in LAY for k in PROJ),
                                     per_layer_R_func={str(L): math.sqrt(r[str(L)]["sq_diff_vs_orig"] /
                                                       float((ref[L]["f0"].double() - ref[L]["fb"].double()).pow(2).sum())) for L in LAY})
            print(f"[func {bits}b {g}] R_func {rec['configs'][g]['R_func']:.4f} code {n_ch / n_el:.4f}", flush=True)
        prim = [rec["configs"][g] for g in G[:4]]                   # 4 perturbations (G20261115 = exact null)
        rec["mean_R_func_4_perturbations"] = sum(x["R_func"] for x in prim) / 4
        rec["rms_R_func_4_perturbations"] = math.sqrt(sum(x["R_func"] ** 2 for x in prim) / 4)
        rec["mean_E_mlp_4_perturbations"] = sum(x["E_mlp_rel_bf16"] for x in prim) / 4
        rec["mean_code_disagreement_4_perturbations"] = sum(x["code_disagreement_vs_original"] for x in prim) / 4
        rec["null_G20261115_R_func"] = rec["configs"]["G20261115"]["R_func"]
        out["bits"][str(bits)] = rec
    b4, b3 = out["bits"]["4"], out["bits"]["3"]
    hist = H.jload(os.path.join(ROOT, "runs_dense_dose_response/logs/functional_raw.json"))["_consistency_G20261111"]
    out["comparison"] = dict(
        quant_error_ratio_3_over_4=b3["quant_error_rel"] / b4["quant_error_rel"],
        E_mlp_ratio_3_over_4=b3["mean_E_mlp_4_perturbations"] / b4["mean_E_mlp_4_perturbations"],
        R_func_ratio_3_over_4=b3["rms_R_func_4_perturbations"] / b4["rms_R_func_4_perturbations"],
        consistency_4bit_G20261111=dict(E_mlp_here=b4["configs"]["G20261111"]["E_mlp_rel_bf16"], E_mlp_historical=hist["E_mlp_here"]),
        note="code disagreement is compared within a bit-width only (3- and 4-bit code labels are not comparable)")
    out["seconds"] = time.time() - t0
    H.jdump(out, H.p("dense3_functional_stability.json"))
    print(json.dumps(out["comparison"], indent=1))


if __name__ == "__main__":
    main()
