import os

import numpy as np
import pandas as pd
import pytest

from seabass.core.models.mms import estimated_mean_mushra_score_file

SIGNAL_ROOT = "tests/assets/sassec/signals"
df = pd.read_csv("tests/assets/sassec/reference_mms_est.csv")


# @pytest.mark.parametrize("index", range(182))
@pytest.mark.parametrize("index", range(int(os.environ.get("SEBASS_N", 182))))
def test_sebass(index):
    row = df.iloc[index]

    ref = row["ref"]
    test = row["test"]

    ref_path = os.path.join(SIGNAL_ROOT, ref)
    test_path = os.path.join(SIGNAL_ROOT, test)

    assert os.path.exists(ref_path)
    assert os.path.exists(test_path)

    mms, movs = estimated_mean_mushra_score_file(test_path, ref_path, _return_movs=True)

    assert np.abs(mms - row["mms"]) < 0.75  # 0.7214317178189873
    assert np.abs(movs["adb"] - row["adb"]) < 0.025  # 0.020658357730113153
    assert np.abs(movs["avg_mod_diff1"] - row["amd1b"]) < 2.5  # 2.347835851056203
