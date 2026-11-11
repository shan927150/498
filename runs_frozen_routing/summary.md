# Frozen-routing 2×2 factorial ablation (FR s0 / FC s0, four layers, ORIGINAL vs G20261111)

**Decision (rule frozen in protocol.json, sha256 `2353231ba957503f`): ROUTING-AND-ACTIVATION-INTERACT**

| strategy | condition | activation | routing | PPL | mean NLL | recon | h (median Hessian diff vs OO) | w (weight rel. L2 vs OO) |
|---|---|---|---|---|---|---|---|---|
| FR | OO | O | O | 9.267851 | 2.2265516 | 0.0046095 | 0.0000 | 0.0000 |
| FR | AO | A | O | 9.265861 | 2.2263368 | 0.0046091 | 0.0039 | 0.0694 |
| FR | OA | O | A | 9.268983 | 2.2266737 | 0.0046091 | 0.0123 | 0.0875 |
| FR | AA | A | A | 9.278182 | 2.2276656 | 0.0046093 | 0.0141 | 0.0889 |
| FC | OO | O | O | 9.277908 | 2.2276361 | 0.0047270 | 0.0000 | 0.0000 |
| FC | AO | A | O | 9.277881 | 2.2276331 | 0.0047266 | 0.0049 | 0.0711 |
| FC | OA | O | A | 9.272497 | 2.2270527 | 0.0047269 | 0.0123 | 0.0888 |
| FC | AA | A | A | 9.274113 | 2.2272270 | 0.0047266 | 0.0158 | 0.0904 |

| strategy | AO−OO | OA−OO | AA−OA | AA−AO | interaction | AA−OO |
|---|---|---|---|---|---|---|
| FR | -0.001990 | +0.001132 | +0.009199 | +0.012321 | +0.011189 | +0.010331 |
| FC | -0.000028 | -0.005411 | +0.001616 | -0.003768 | +0.001644 | -0.003795 |

| condition | Δ = FC − FR | window-bootstrap 95% CI (mean NLL) |
|---|---|---|
| OO | +0.010057 | [+0.00020, +0.00200] |
| AO | +0.012020 | [+0.00043, +0.00220] |
| OA | +0.003514 | [-0.00046, +0.00125] |
| AA | -0.004069 | [-0.00129, +0.00047] |

- ranking preserved (|Δ_AO − Δ_OO| ≤ 0.5 |Δ_AA − Δ_OO|): True
- flags: FR: local_suppression=False, local_routing_dominant=True, ppl_suppression=True, activation_dominant=False, interaction_large=True; FC: local_suppression=False, local_routing_dominant=True, ppl_suppression=True, activation_dominant=False, interaction_large=False
- natural regression: FR_OO passed (Hessians 480/480), FR_AA passed (Hessians 480/480), FC_OO passed (Hessians 480/480), FC_AA passed (Hessians 480/480)

