import numpy as np
from numba import njit

from seabass.core.features.constants import (
    _FSS,
    _LOUDNESS_POWER,
    _MODULATION_TAU_100,
    _MODULATION_TAU_MIN,
)
from seabass.core.features.utils import time_constants


@njit
def modulation_alphas(
    center_freqs_hz: np.ndarray,
    fss: float = _FSS,
    tau_min: float = _MODULATION_TAU_MIN,
    tau_100: float = _MODULATION_TAU_100,
) -> np.ndarray:
    return time_constants(center_freqs_hz, fss, tau_min, tau_100)


@njit
def envelope_modulation(
    avg_loudness: np.ndarray, avg_loudness_difference: np.ndarray
) -> np.ndarray:
    """
    Equation (67) of Kabal (2003).
    :param avg_loudness:
    :return: envelope_modulation
        np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    """

    env_mod = avg_loudness_difference / (1 + avg_loudness / _LOUDNESS_POWER)
    return env_mod


@njit
def instantaneous_modulation_difference1_basic(
    ref_envelope_modulation: np.ndarray,
    test_envelope_modulation: np.ndarray,
) -> np.ndarray:
    """
    Equation (73) of Kabal (2003).

    :return: instantaneous_modulation_difference_1
        np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    """

    md1 = np.abs(test_envelope_modulation - ref_envelope_modulation) / (
        1 + ref_envelope_modulation
    )

    return md1


def scaled_instantaneous_modulation_difference1_basic(
    instantaneous_modulation_difference1_basic: np.ndarray,
):
    """
    Equation (74) of Kabal (2003).

    :return: scaled_instantaneous_modulation_difference_1
        np.ndarray of shape (n_chan, n_frames,)
    """
    simd1 = 100 * np.mean(
        instantaneous_modulation_difference1_basic, axis=1, keepdims=False
    )

    return simd1


def average_modulation_difference1_basic_temporal_weights(
    ref_average_loudness: np.ndarray,
    internal_noise_: np.ndarray,
):
    """
    Equation (78) of Kabal (2003).

    :return: temporal_weights
        np.ndarray of shape (n_chan, n_frames,)
    """

    contrib = ref_average_loudness / (
        ref_average_loudness
        + 100 * np.power(internal_noise_[None, :, None], _LOUDNESS_POWER)
    )

    w1b = np.sum(contrib, axis=1)

    return w1b


def average_modulation_difference1_basic(
    scaled_instantaneous_modulation_difference: np.ndarray,
    temporal_weights: np.ndarray,
    delayed_avg_frames: int,
    keep_channels: bool = True,
):
    """
    Equation (77) of Kabal (2003).
    :param scaled_instantaneous_modulation_difference:
    :param temporal_weights:
    :param delayed_avg_frames:
    :return:
    """

    amd1 = np.sum(
        scaled_instantaneous_modulation_difference[:, delayed_avg_frames:]
        * temporal_weights[:, delayed_avg_frames:],
        axis=-1,
    ) / np.sum(temporal_weights[:, delayed_avg_frames:], axis=-1)

    if not keep_channels:
        amd1 = np.mean(amd1, axis=0)

    return amd1
