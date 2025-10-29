from numbers import Real
from typing import Dict, Tuple, Union

import numpy as np
import resampy as rp
import soundfile as sf
from numba import njit
from numpy import typing as npt

from seabass.core.features.constants import _FS
from seabass.core.model_output_variables import PeaqModelOutputVariables
from seabass.core.models.constants import (
    _ADB_MULT,
    _AMD1_DENOM_CONST,
    _AMD1_DENOM_MULT,
    _AMD1_NUM_MULT,
    _MMS_CONST,
)


@njit
def _compute_mms(
    avg_mod_diff1: Union[Real, np.ndarray],
    adb: Union[Real, np.ndarray],
):
    amd_term = _AMD1_NUM_MULT / (
        1 + np.square(_AMD1_DENOM_MULT * avg_mod_diff1 + _AMD1_DENOM_CONST)
    )
    adb_term = _ADB_MULT * adb
    score = amd_term + adb_term + _MMS_CONST

    return score


def estimated_mean_mushra_score(
    test_signal: npt.ArrayLike,
    ref_signal: npt.ArrayLike,
    fs: int = _FS,
    clip: bool = True,
    signal_dtype: str = "double",
    _return_movs: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    assert fs == _FS, f"Only fs={_FS} is supported"

    movs = PeaqModelOutputVariables(
        fs=fs, test_signal=test_signal, ref_signal=ref_signal, signal_dtype=signal_dtype
    ).compute()

    score = _compute_mms(
        avg_mod_diff1=movs["avg_mod_diff1"],
        adb=movs["adb"],
    )
    if clip:
        score = np.clip(score, 0, 100)

    if _return_movs:
        return score, movs

    return score


def estimated_mean_mushra_score_file(
    test_path: str,
    ref_path: str,
    resampy_filter: str = "kaiser_best",
    _return_movs: bool = False,
):
    test_signal, fs_t = sf.read(test_path, always_2d=True)
    ref_signal, fs_r = sf.read(ref_path, always_2d=True)

    if fs_t != _FS:
        test_signal = rp.resample(test_signal, fs_t, _FS, axis=0, filter=resampy_filter)

    if fs_r != _FS:
        ref_signal = rp.resample(ref_signal, fs_r, _FS, axis=0, filter=resampy_filter)

    return estimated_mean_mushra_score(
        test_signal.T, ref_signal.T, fs=_FS, _return_movs=_return_movs
    )
