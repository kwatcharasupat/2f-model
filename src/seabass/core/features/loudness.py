import numpy as np
from numba import njit

from seabass.core.features.constants import _LOUDNESS_POWER
from seabass.core.features.utils import optimized_exponential_moving_average


@njit
def approximate_loudness(
    unsmeared_excitation: np.ndarray,
):
    approx_loudness = np.power(unsmeared_excitation, _LOUDNESS_POWER)
    return approx_loudness


@njit
def average_loudness(
    approx_loudness: np.ndarray, frequency_wise_alphas: np.ndarray
) -> np.ndarray:
    """
    Equation (65) of Kabal (2003).
    :return:
    :param approx_loudness: np.ndarray: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :param frequency_wise_alphas: np.ndarray of shape (n_center_freqs,)
    :return: average_loudness: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    """

    avg_loudness = optimized_exponential_moving_average(
        approx_loudness,
        frequency_wise_alphas,
    )

    return avg_loudness


def average_loudness_difference(
    approx_loudness: np.ndarray, frequency_wise_alphas: np.ndarray, fss: float
) -> np.ndarray:
    """
    Equation (66) of Kabal (2003).
    :param approx_loudness:
    :param frequency_wise_alphas:
    :param fss:
    :return:
    """
    approx_loudness_diff = fss * np.abs(np.diff(approx_loudness, axis=-1, prepend=0))
    # shape (n_center_freqs, n_frames)
    average_loudness_diff = optimized_exponential_moving_average(
        approx_loudness_diff,
        frequency_wise_alphas,
    )

    return average_loudness_diff
