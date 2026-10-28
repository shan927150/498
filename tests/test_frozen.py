"""configs/frozen_b4.json is internally consistent and check_e1_fp rejects drift."""
import copy
import os

import pytest

from moe_calib.frozen import check_e1_fp, load_frozen

CFG = load_frozen(os.path.join(os.path.dirname(__file__), "..", "configs", "frozen_b4.json"))


def frozen_fp(seed=0):
    return dict(model=CFG["model"], bits=4, group_size=128, percdamp=1.0, char_seed=seed,
                layers=[1, 8, 16, 23], n0=[16, 32, 64, 128], reps=3, K=4,
                split=dict(CFG["split"]))


def test_frozen_values():
    assert CFG["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    assert CFG["e1"]["char_seeds"] == [0, 1, 2]
    check_e1_fp(CFG, frozen_fp())


@pytest.mark.parametrize("key,val", [("percdamp", 0.01), ("bits", 3), ("n0", [32, 64, 128]),
                                     ("layers", [1]), ("char_seed", 5), ("reps", 2)])
def test_drift_rejected(key, val):
    fp = frozen_fp()
    fp[key] = val
    with pytest.raises(ValueError, match=key):
        check_e1_fp(CFG, fp)


def test_split_drift_rejected():
    fp = copy.deepcopy(frozen_fp())
    fp["split"]["tune"] = [2000, 2100]
    with pytest.raises(ValueError, match="split"):
        check_e1_fp(CFG, fp)
