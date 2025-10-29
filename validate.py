import os

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from seabass.core.models.mms import estimated_mean_mushra_score_file

df = pd.read_csv("./tests/assets/sassec/reference_mms_est.csv")

SIGNAL_ROOT = "./tests/assets/sassec/signals"

expected_mms = []
computed_mms = []
computed_amd1b = []
computed_adb = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    ref = row["ref"]
    test = row["test"]

    if ref == test:
        continue

    ref_path = os.path.join(SIGNAL_ROOT, ref)
    test_path = os.path.join(SIGNAL_ROOT, test)

    print(ref)
    print(test)

    mms, movs = estimated_mean_mushra_score_file(test_path, ref_path, _return_movs=True)
    true_mms = row["mms"]
    expected_mms.append(true_mms)
    computed_mms.append(mms)
    computed_amd1b.append(movs["avg_mod_diff1"])
    computed_adb.append(movs["adb"])

    print("mms", mms, true_mms, mms - true_mms)

expected_mms = np.array(expected_mms)
computed_mms = np.array(computed_mms)
computed_amd1b = np.array(computed_amd1b)
computed_adb = np.array(computed_adb)

print("mean", np.mean(np.abs(computed_mms - expected_mms)))
print("std", np.std(computed_mms - expected_mms))
print("max", np.max(computed_mms - expected_mms))
print("min", np.min(computed_mms - expected_mms))

print(stats.pearsonr(expected_mms, computed_mms))
print(stats.spearmanr(expected_mms, computed_mms))  # type: ignore

df["computed_mms"] = computed_mms
df["computed_amd1b"] = computed_amd1b
df["computed_adb"] = computed_adb

df.to_csv(
    "./tests/assets/sassec/reference_mms_est_profile.csv",
    index=False,
)
