# Parallel 8-hour plan (one ICRN session, A100 20GB slice)

Five follow-up experiments after P2c, run side by side in ONE session
(Environment: PyTorch, Resource: **A100 20GB VRAM GPU slice, 12CPU/72GB**).
Processes that only replay single layers load just those layers
(`MOE_LOAD_LAYERS`), ~5 GB of RAM instead of ~30 GB, which is what makes
running them together possible.

| exp | question | GPU? | output |
|---|---|---|---|
| 1 | Is the P2c ranking gain actionable within a layer (the only reallocation sequence selection can make)? | no (seconds) | `runs/within_layer_ranking_b4_d1.md` |
| 2 | Headroom: how much do recon and PPL move with a larger random budget? | yes, full model | `runs_headroom/headroom_report.md` |
| 4 | Token count vs source diversity at fixed count | yes, 4 layers | `runs_diag/exposure_diversity_report.md` |
| 5 | Damping grid extended to 3.0/5.0 + adaptive-damping ablation | yes, 4 layers | `runs_diag/adaptive_damping.md`, `runs_diag/damping_threshold_report.md` |
| 6 | Does the layer x regime structure transfer to uncharacterized layers 4/12/20? | yes (cache: full model, then 3 layers) | `runs_newlayers/depth_generalization.md` |

## Run

```bash
cd ~/498 && git pull
mkdir -p logs
nohup bash scripts/parallel/launch.sh > logs/launch.log 2>&1 &
bash scripts/parallel/status.sh          # any time
```

Stages (keeps GPU < 20 GB, RAM < 72 GB):
1. Exp1 (seconds), then Exp4 + Exp5 + Exp6a (new-layer cache, ~40 min) together.
2. When the cache is done, Exp2 starts.
3. When Exp5 is done, Exp6b (E1 on new layers + analysis) starts.

Expected wall time 4.5–6 h on the slice (rough: from smoke-test speeds and
GPU sharing). Everything resumes: if a process dies or the session ends, run
`launch.sh` again and finished parts are skipped. Keep the browser tab open
(ICRN may end processes after > 1 h disconnected).

Disk: the new cache adds ~14 GB (`~/moe_cache_new`).
