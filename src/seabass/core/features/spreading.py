from numbers import Real
from typing import Union

import numpy as np
from numba import njit

from seabass.core.features.constants import (
    _DZ_BASIC,
    _FREQ_SPREADING_POWER,
    _FSS,
    _TIME_SPREADING_TAU_100,
    _TIME_SPREADING_TAU_MIN,
)
from seabass.core.features.utils import (
    optimized_exponential_moving_average,
    time_constants,
)


@njit
def _kabal_frequency_spreading_func(
    center_freqs_hz: np.ndarray,
    pitch_pattern: np.ndarray,
    normalizing_terms: Union[Real, np.ndarray],
    dz: float = _DZ_BASIC,
    spread_pow: float = _FREQ_SPREADING_POWER,
):
    n_center_freqs = center_freqs_hz.shape[0]
    freq_arange = np.arange(n_center_freqs)

    aL = np.power(10.0, -2.7 * dz)  # (1,)
    aUC = np.power(10.0, (-2.4 - 23 / center_freqs_hz) * dz)  # (n_center_freqs,)

    aUCE = aUC[None, :, None] * np.power(pitch_pattern, 0.2 * dz)
    # (n_chan, n_center_freqs, n_frames)

    gIL = (1 - np.power(aL, freq_arange + 1)) / (1 - aL)
    # (n_center_freqs,)

    gIU = (1 - np.power(aUCE, n_center_freqs - freq_arange[None, :, None])) / (1 - aUCE)

    En = pitch_pattern / (gIL[None, :, None] + gIU - 1)
    aUCEe = np.power(aUCE, spread_pow)
    Ene = np.power(En, spread_pow)
    aLe = np.power(aL, spread_pow)

    Es = np.zeros_like(pitch_pattern)

    Es[:, -1, :] = Ene[:, -1, :]

    for f in np.arange(n_center_freqs - 1)[::-1]:
        Es[:, f, :] = aLe * Es[:, f + 1, :] + Ene[:, f, :]

    for f in np.arange(n_center_freqs - 1):
        r = Ene[:, f, :]
        a = aUCEe[:, f, :]
        apows = np.power(
            a[:, None, :], 1 + np.arange(n_center_freqs - f - 1)[None, :, None]
        )
        rapows = r[:, None, :] * apows
        Es[:, f + 1 :, :] += rapows

    Es = np.power(Es, 1.0 / spread_pow) / normalizing_terms
    return Es


@njit
def _kabal_unsmeared_excitation_pattern(
    center_freqs_hz: np.ndarray,
    center_freq_energies: np.ndarray,
    dz: float = _DZ_BASIC,
):
    n_chan, n_center_freqs, n_frames = center_freq_energies.shape

    normalizing_terms = _kabal_frequency_spreading_func(
        center_freqs_hz,
        np.ones((1, n_center_freqs, 1)),
        1.0,
        dz=dz,
    )

    spread = _kabal_frequency_spreading_func(
        center_freqs_hz,
        center_freq_energies,
        normalizing_terms,
        dz=dz,
    )

    return spread


@njit
def unsmeared_excitation_pattern(
    center_freqs_hz: np.ndarray,
    pitch_pattern: np.ndarray,
    dz: float = _DZ_BASIC,
):
    """
    Equation (19) of Kabal (2003).

    :param center_freqs_hz:  np.ndarray of shape (n_center_freqs,)
    :param pitch_pattern:  np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :param dz:
    :return: spreading: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    """

    return _kabal_unsmeared_excitation_pattern(
        center_freqs_hz,
        pitch_pattern,
        dz=dz,
    )


@njit
def time_spreading_alphas(
    center_freqs_hz: np.ndarray,
    fss: float = _FSS,
    tau_min: float = _TIME_SPREADING_TAU_MIN,
    tau_100: float = _TIME_SPREADING_TAU_100,
) -> np.ndarray:
    return time_constants(center_freqs_hz, fss, tau_min, tau_100)


@njit
def smeared_excitation_pattern(
    unsmeared_excitation_pattern: np.ndarray,
    time_spread_alphas: np.ndarray,
):
    """
    Equation (30) of Kabal (2003).

    :param time_spread_alphas: np.ndarray of shape (n_center_freqs,)
    :param unsmeared_excitation_pattern: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :return: spreading: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    """

    smoothed_excitation_pattern = optimized_exponential_moving_average(
        unsmeared_excitation_pattern,
        time_spread_alphas,
    )

    smeared_pattern = np.maximum(
        smoothed_excitation_pattern, unsmeared_excitation_pattern
    )

    return smeared_pattern
