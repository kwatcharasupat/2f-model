import numpy as np
from numba import njit
from scipy import signal as sps

from seabass.core.features.constants import (
    _FS,
    _FULL_SCALE,
    _HOP_SIZE,
    _LP_DBSPL,
    _NFFT,
    _NORMALIZED_TEST_FREQ,
)


@njit
def peak_factor(
    normalized_freq: float,
    n_fft: int = _NFFT,
    window_length: int = _NFFT - 1,
):
    df = 1 / n_fft
    k = np.floor(normalized_freq / df)
    df_norm = min((k + 1) * df - normalized_freq, normalized_freq - k * df)

    df_win = df_norm * window_length
    gp = np.sin(np.pi * df_win) / (np.pi * df_win * (1 - np.power(df_win, 2.0)))

    return gp


@njit
def window_scaler(
    n_fft: int = _NFFT,
    max_amplitude: float = _FULL_SCALE,
    normalized_freq: float = _NORMALIZED_TEST_FREQ,
    spl_level: float = _LP_DBSPL,
):
    window_length = n_fft - 1
    gp = peak_factor(normalized_freq, n_fft, window_length)
    gain = np.power(10.0, spl_level / 20.0) / (
        0.25 * gp * max_amplitude * window_length
    )

    return gain


@njit
def hann_window(
    n_fft: int = _NFFT,
):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1)))


def stft(
    signal: np.ndarray,
    n_fft: int = _NFFT,
    n_hop: int = _HOP_SIZE,
    fs: int = _FS,
    max_amplitude: float = _FULL_SCALE,
    normalized_freq: float = _NORMALIZED_TEST_FREQ,
    spl_level: float = _LP_DBSPL,
):
    """

    :param signal:
    :param n_fft:
    :param n_hop:
    :param fs:
    :param max_amplitude:
    :param normalized_freq:
    :param spl_level:
    :return: stft
        np.ndarray of shape (n_chan, n_freqs, n_frames)
    """
    window = hann_window(n_fft) * window_scaler(
        n_fft=n_fft,
        max_amplitude=max_amplitude,
        normalized_freq=normalized_freq,
        spl_level=spl_level,
    )
    win_energy = np.sqrt(np.square(np.sum(window)))
    f, t, stft = sps.stft(
        signal,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - n_hop,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        padded=False,
        boundary=None,
        scaling="spectrum",
        axis=-1,
    )

    stft = stft * win_energy
    return stft


@njit
def stft_magnitude_squared(
    stft_: np.ndarray,
) -> np.ndarray:
    """

    :param stft_: np.ndarray of shape (n_chan, n_freqs, n_frames)
    :return: stft_magnitude_squared
        np.ndarray of shape (n_chan, n_freqs, n_frames)
    """
    mag_ = np.square(np.abs(stft_))

    return mag_


@njit
def weighted_magnitude_squared(
    magnitude_squared: np.ndarray, n_fft: int, fs: int
) -> np.ndarray:
    """

    :param magnitude_squared:
    :param n_fft:
    :param fs:
    :return: weighted_magnitude_squared
        np.ndarray of shape (n_chan, n_freqs, n_frames)
    """
    weights = ear_model_filter(np.arange(0, n_fft // 2 + 1) * fs / n_fft)

    wms = magnitude_squared * weights[None, :, None]
    return wms


@njit
def ear_model_filter(
    center_freqs: np.ndarray,
):
    filter = np.zeros_like(center_freqs)

    center_freqs = center_freqs[1:]
    center_freqs_khz = center_freqs / 1000

    filter_db = (
        -2.184 * np.power(center_freqs_khz, -0.8)
        + 6.5 * np.exp(-0.6 * np.square((center_freqs_khz - 3.3)))
        - 0.001 * np.power(center_freqs_khz, 3.6)
    )

    filter[1:] = np.power(10, filter_db / 10)

    return filter
