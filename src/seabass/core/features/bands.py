from typing import Tuple

import numpy as np
from numba import njit

from seabass.core.features.constants import (
    _CRIT_MAXIMUM_FREQ_HZ,
    _CRIT_MINIMUM_FREQ_HZ,
    _DZ_BASIC,
    _FREQ_CENTERS_HZ,
    _FREQ_EDGES_HIGH_HZ,
    _FREQ_EDGES_LOW_HZ,
    _MINIMUM_ENERGY,
    _N_CRIT_BANDS_BASIC,
)
from seabass.core.features.utils import bark, inverse_bark


def critical_bands(
    dz: float = _DZ_BASIC,
    minimum_freq_hz: float = _CRIT_MINIMUM_FREQ_HZ,
    maximum_freq_hz: float = _CRIT_MAXIMUM_FREQ_HZ,
    use_kabal_constants: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param peaq_version:
    :return: freq_edge_low, freq_edge_high, freq_center
        np.ndarray, np.ndarray, np.ndarray
    """

    if use_kabal_constants:
        freq_edge_low = _FREQ_EDGES_LOW_HZ
        freq_center = _FREQ_CENTERS_HZ
        freq_edge_high = _FREQ_EDGES_HIGH_HZ

        return freq_edge_low, freq_edge_high, freq_center

    z_min = bark(minimum_freq_hz)
    z_max = bark(maximum_freq_hz)

    n_crit_bands = int(np.ceil((z_max - z_min) / dz))
    assert n_crit_bands == _N_CRIT_BANDS_BASIC, "Number of critical bands is not 109."

    band_edge_low = z_min + np.arange(n_crit_bands) * dz
    band_edge_high = np.minimum(z_min + (np.arange(n_crit_bands) + 1) * dz, z_max)
    band_center = (band_edge_low + band_edge_high) / 2

    freq_edge_low = inverse_bark(band_edge_low)
    freq_edge_high = inverse_bark(band_edge_high)
    freq_center = inverse_bark(band_center)

    return freq_edge_low, freq_edge_high, freq_center


@njit
def critical_band_matrix(
    n_fft: int,
    fs: int,
    freq_edge_low_hz: np.ndarray,
    freq_edge_high_hz: np.ndarray,
) -> np.ndarray:
    """
    Group frequencies into critical bands.

    :param freqs: Frequencies to group into critical bands.
    :param freq_edge_low_hz: Lower edge of critical bands.
    :param freq_edge_high_hz: Upper edge of critical bands.
    :return: band_weights
        np.ndarray (n_bands, n_freqs_stft)
    """

    n_freqs_stft = n_fft // 2 + 1
    df = fs / n_fft
    stft_upper = (2 * np.arange(0, n_freqs_stft) + 1) / 2 * df
    stft_lower = (2 * np.arange(0, n_freqs_stft) - 1) / 2 * df

    upper = np.minimum(
        stft_upper[None, :],
        freq_edge_high_hz[:, None],
    )

    lower = np.maximum(
        stft_lower[None, :],
        freq_edge_low_hz[:, None],
    )

    band_weights = (upper - lower) / df
    band_weights = np.where(
        band_weights < 0,
        0,
        band_weights,
    )

    return band_weights


# @njit(parallel=True)
def critical_band_energies(
    weighted_magnitude_squared: np.ndarray,
    band_weights: np.ndarray,
    minimum_energy: float = _MINIMUM_ENERGY,
):
    """
    Calculate critical band energies.

    :param weighted_magnitude_squared: Magnitude squared of STFT.
        np.ndarray (n_chan, n_freqs_stft, n_frames)
    :param band_weights: Band weights.
        np.ndarray (n_bands, n_freqs_stft)
    :return: band_energies
        np.ndarray (n_chan, n_bands, n_frames)
    """
    # using this instead of np.matmul for numba compatibility
    # n_chan, n_freqs_stft, n_frames = weighted_magnitude_squared.shape
    # n_bands = band_weights.shape[0]
    # band_energies = np.zeros((n_chan, n_bands, n_frames))
    # for c in prange(n_chan):
    #     band_energies[c] = np.dot(
    #         band_weights,
    #         np.ascontiguousarray(weighted_magnitude_squared[c]),
    #     )

    band_energies = np.matmul(
        band_weights,
        weighted_magnitude_squared,
    )

    band_energies = np.where(
        band_energies < minimum_energy,
        minimum_energy,
        band_energies,
    )

    return band_energies


@njit
def internal_noise(
    center_freqs_hz: np.ndarray,
):
    """

    :param center_freqs_hz: np.ndarray (n_bands,)
    :return: int_noise
        np.ndarray (n_bands,)
    """
    int_noise_db = 1.456 * np.power(center_freqs_hz / 1000, -0.8)
    int_noise = np.power(10, int_noise_db / 10)

    return int_noise


@njit
def pitch_patterns(
    critical_band_energies: np.ndarray,
    internal_noise: np.ndarray,
) -> np.ndarray:
    """

    :param center_freqs_hz: np.ndarray (n_bands,)
    :param critical_band_energies: np.ndarray (n_chan, n_bands, n_frames)
    :return: pitch_pattern
        np.ndarray (n_chan, n_bands, n_frames)
    """

    pitch_pattern = critical_band_energies + internal_noise[None, :, None]

    return pitch_pattern
