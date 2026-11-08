import os, sys, torch
sys.path.insert(0, "/home/kaixuan6/498")
os.environ.pop("MOE_LOAD_LAYERS", None)
from moe_calib import model_utils as MU
C = os.path.expanduser("~/moe_cache")
ids = torch.load(f"{C}/ids.pt"); tok = torch.cat([ids["pool"], ids["holdout"]])
model, _ = MU.load_model()
olds = {L: torch.load(f"{C}/L{L}.pt", mmap=True) for L in (1, 8, 16, 23)}
for start in (0, 2128):                     # original batch-of-8 groups
    hs = MU.embed(model, tok[start:start + 8]).cpu()
    for L in range(24):
        if L in olds:
            o = olds[L][start:start + 8]
            print(start, L, "bitwise", torch.equal(o, hs), "max abs", float((o.float() - hs.float()).abs().max()), flush=True)
        if L == 23: break
        with MU.layer_on_gpu(model, L):
            hs = MU.run_layer(model, L, hs.to(MU.GPU)).cpu()
