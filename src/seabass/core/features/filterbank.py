import numpy as np
from scipy import signal as sps

from seabass.core.features.utils import (
    inverse_bark,
    optimized_exponential_moving_average,
)

raise ImportError("This module is not yet ready to be used.")

_SUBSAMPLING_FACTOR = 6
_N_TIME_SMEAR_FIR_TAPS = 12
_TIME_SMEAR_CONSTANT = 0.9761 / 6
_SPREADING_TIME_CONSTANT = 0.1
_SPREADING_FS_SCALE = 32


def instantaneous_spreading_function(
    center_freqs: np.ndarray,
    center_freq_energies: np.ndarray,
):
    """
    Equation (44) of Kabal (2003).

    :param center_freqs:  np.ndarray of shape (n_center_freqs,)
    :param center_freq_energies:  np.ndarray of shape (n_center_freqs, n_frames)
    :return: spreading: np.ndarray of shape (n_center_freqs, n_center_freqs, n_frames)
    """
    z = center_freqs[:, None]
    zc = center_freqs[None, :]

    spread_db = np.where(
        z <= zc,
        31,
        np.minimum(
            -4,
            -24
            - 230 / inverse_bark(zc)
            + 2 * np.log10(center_freq_energies[None, :, :]),
        ),
    ) * (z - zc)

    spread = np.power(10, spread_db / 20)

    return spread


def smoothed_spreading_function(
    instantaneous_spreading_function_: np.ndarray,
    fs: int,
    fs_scale: int = _SPREADING_FS_SCALE,
    tau: float = _SPREADING_TIME_CONSTANT,
):
    fss = fs / fs_scale
    alpha = np.exp(-1 / (fss * tau))

    return optimized_exponential_moving_average(
        instantaneous_spreading_function_, alpha, smooth_axis=-1
    )


def time_smear_filter(
    n_taps: int = _N_TIME_SMEAR_FIR_TAPS,
):
    """
    Equation (52) of Kabal (2003).
    :param n_taps:
    :return:
    """
    return np.square(
        np.cos((np.pi / n_taps) * (np.arange(n_taps) - ((n_taps - 1) / 2)))
    )


def time_smeared_excitation(
    freq_spread_energies_: np.ndarray,
    n_taps: int = _N_TIME_SMEAR_FIR_TAPS,
    subsampling_factor: int = _SUBSAMPLING_FACTOR,
):
    """
    Equation (51) of Kabal (2003).
    :param freq_spread_energies_: np.ndarray of shape (n_freqs, n_frames)
    :param n_taps: int, number of taps of the FIR filter used for time smearing
    :param subsampling_factor: int, subsampling factor used for time smearing
    :return: time_smeared_excitation_: np.ndarray of shape (n_freqs, n_subsampled_frames)

    n_subsampled_frames = (n_frames + n_taps - 1) // subsampling_factor
    """
    filter = time_smear_filter(n_taps=n_taps)

    conved = sps.oaconvolve(
        freq_spread_energies_, filter[None, :], mode="full", axes=-1
    )

    return conved[:, ::subsampling_factor]


def unsmeared_excitation(
    time_smeared_excitation_: np.ndarray,
    internal_noise_: np.ndarray,
):
    """
    Equation (53) of Kabal (2003).
    :param time_smeared_excitation_:
    :param internal_noise_:
    :return:
    """
    return time_smeared_excitation_ + internal_noise_
