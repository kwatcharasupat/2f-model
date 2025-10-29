from numbers import Real
from typing import Union

import numpy as np
from numba import njit
from numpy import typing as npt

from seabass.core.features.constants import _AMPLITUDE_THRESHOLD, _BOUND_WINDOW, _NFFT


@njit
def linear_pow_to_db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(x)


@njit
def signal_bounds(
    signal,
    threshold=_AMPLITUDE_THRESHOLD,
    win_size: int = _BOUND_WINDOW,
    hop_size: int = _NFFT // 2,
):
    n_samples = signal.shape[-1]

    bound_start = None
    for i in np.arange(n_samples - win_size):
        win = signal[:, i : i + win_size]
        if np.min(np.sum(np.abs(win), axis=-1)) > threshold:
            bound_start = i
            break

    rev_signal = signal[:, ::-1]
    bound_end = None
    for i in np.arange(n_samples - win_size):
        win = rev_signal[:, i : i + win_size]
        if np.min(np.sum(np.abs(win), axis=-1)) > threshold:
            bound_end = i
            break
    if bound_end is not None:
        bound_end = n_samples - bound_end + win_size

    if bound_start is None or bound_end is None:
        raise ValueError("Could not find a bound for the reference signal.")

    frame_start = np.floor(bound_start / hop_size)
    frame_end = np.floor((bound_end + 1 - hop_size) / hop_size)

    return int(frame_start), int(frame_end)


@njit
def optimized_exponential_moving_average(
    x: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    alpha_ = alpha[None, :]
    beta_ = 1 - alpha

    x_smooth = np.zeros_like(x)

    x_smooth[:, :, 0] = beta_ * x[:, :, 0]

    n_steps = x.shape[-1]

    for i in np.arange(1, n_steps):
        x_smooth[:, :, i] = alpha_ * x_smooth[:, :, i - 1] + beta_ * x[:, :, i]

    return x_smooth


def exponential_moving_average(
    x: np.ndarray,
    alpha: Union[float, npt.ArrayLike],
    alpha_axis: int = -2,
    smooth_axis: int = -1,
) -> np.ndarray:
    assert alpha_axis == -2

    n_steps = x.shape[smooth_axis]

    if isinstance(alpha, Real):
        alpha = np.ones(n_steps) * alpha

    assert alpha.ndim == 1

    if alpha.size != x.shape[alpha_axis]:
        raise ValueError(
            f"Expected `alpha` to be a scalar or an array of size {n_steps}, "
            f"got {alpha.size} instead."
        )

    if smooth_axis < 0:
        smooth_axis = x.ndim + smooth_axis

    if alpha_axis < 0:
        alpha_axis = x.ndim + alpha_axis

    if x.ndim == 3 and smooth_axis == 2 and alpha_axis == 1:
        return optimized_exponential_moving_average(x, alpha)

    def slicer(index):
        return tuple(slice(None) if i != smooth_axis else index for i in range(x.ndim))

    alpha_shape = [1 if i != alpha_axis else alpha.size for i in range(x.ndim)]

    balpha = np.broadcast_to(alpha.reshape(alpha_shape), x.shape)
    bbeta = 1.0 - balpha

    x_smooth = np.zeros_like(x)

    prev_slice_ = slicer(0)
    x_smooth[prev_slice_] = bbeta[prev_slice_] * x[prev_slice_]

    for i in np.arange(1, n_steps):
        slice_ = slicer(i)
        x_smooth[slice_] = (
            balpha[prev_slice_] * x_smooth[prev_slice_] + bbeta[slice_] * x[slice_]
        )
        prev_slice_ = slice_

    return x_smooth


@njit
def bark(freqs: Union[np.ndarray, Real]):
    return 7 * np.arcsinh(freqs / 650)


@njit
def inverse_bark(z: Union[np.ndarray, Real]):
    return 650 * np.sinh(z / 7)


@njit
def time_constants(
    center_freqs_hz: np.ndarray, fss: float, tau_min: float, tau_100: float
) -> np.ndarray:
    tau = tau_min + (tau_100 - tau_min) * 100 / center_freqs_hz

    alphs = np.exp(-1 / (fss * tau))

    return alphs
