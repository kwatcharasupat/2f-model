import numpy as np
from numba import njit

from seabass.core.features.constants import (
    _ASSYMMETRIC_EXCITATION_COEFF,
    _DETECTION_PROBABILITY_STEEPNESS_NEG,
    _DETECTION_PROBABILITY_STEEPNESS_POS,
    _EFFECTIVE_DETECTION_COEFF_C,
    _EFFECTIVE_DETECTION_COEFF_CONST,
    _EFFECTIVE_DETECTION_COEFF_D1,
    _EFFECTIVE_DETECTION_COEFF_D2,
    _EFFECTIVE_DETECTION_COEFF_GAMMA,
)


@njit
def asymmetric_excitation_db(
    ref_smeared_excitation_db: np.ndarray,
    test_smeared_excitation_db: np.ndarray,
) -> np.ndarray:
    """
    Equation (120) of Kabal (2003).
    :param ref_smeared_excitation_db: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :param test_smeared_excitation_db: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :return: asym_ex: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    """

    asym_ex = np.where(
        ref_smeared_excitation_db > test_smeared_excitation_db,
        _ASSYMMETRIC_EXCITATION_COEFF * ref_smeared_excitation_db
        + (1 - _ASSYMMETRIC_EXCITATION_COEFF) * test_smeared_excitation_db,
        test_smeared_excitation_db,
    )

    return asym_ex


@njit
def _inner_effective_detection_step_size(
    coeff_part: np.ndarray, asymmetric_excitation_db_: np.ndarray
):
    power_part = _EFFECTIVE_DETECTION_COEFF_D1 * np.power(
        _EFFECTIVE_DETECTION_COEFF_D2 / asymmetric_excitation_db_,
        _EFFECTIVE_DETECTION_COEFF_GAMMA,
    )

    positive_part = coeff_part + power_part

    eff_det_step_size = np.where(
        asymmetric_excitation_db_ > 0, positive_part, _EFFECTIVE_DETECTION_COEFF_CONST
    )

    return eff_det_step_size


def effective_detection_step_size(
    asymmetric_excitation_db_: np.ndarray,
):
    coeff_part = np.polynomial.polynomial.polyval(
        asymmetric_excitation_db_,
        _EFFECTIVE_DETECTION_COEFF_C,
    )

    return _inner_effective_detection_step_size(coeff_part, asymmetric_excitation_db_)


@njit
def detection_probability(
    smeared_excitation_db_diff: np.ndarray,
    eff_det_step_size: np.ndarray,
) -> np.ndarray:
    steepness = np.where(
        smeared_excitation_db_diff > 0,
        _DETECTION_PROBABILITY_STEEPNESS_POS,
        _DETECTION_PROBABILITY_STEEPNESS_NEG,
    )

    det_prob = 1 - np.power(
        0.5, np.power(smeared_excitation_db_diff / eff_det_step_size, steepness)
    )

    return det_prob


def steps_above_threshold(
    smeared_excitation_db_diff: np.ndarray,
    eff_det_step_size: np.ndarray,
) -> np.ndarray:
    steps_above_thresh = np.abs(np.fix(smeared_excitation_db_diff)) / eff_det_step_size

    return steps_above_thresh


def total_detection_probability(
    channelwise_detection_prob: np.ndarray,
) -> np.ndarray:
    """
    Equation (126) of Kabal (2003).

    :param channelwise_detection_prob: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :return: total_detection_prob: np.ndarray of shape (n_frames,)
    """
    max_channelwise_detection_prob = np.max(channelwise_detection_prob, axis=0)
    not_detected_prob = 1 - max_channelwise_detection_prob
    total_detection_prob = 1 - np.prod(not_detected_prob, axis=0)

    return total_detection_prob


def total_steps_above_threshold(
    channelwise_steps_above_thresh: np.ndarray,
) -> np.ndarray:
    """
    Equation (126) of Kabal (2003).
    :param channelwise_steps_above_thresh: np.ndarray of shape (n_chan, n_center_freqs, n_frames)
    :return: total_steps_above_thresh: np.ndarray of shape (n_frames,)
    """
    total_steps_above_thresh = np.sum(
        np.max(channelwise_steps_above_thresh, axis=0), axis=0
    )

    return total_steps_above_thresh


# @njit
def average_block_distortion_basic(
    total_detection_prob, total_steps_above_thresh, detection_prob_thresh=0.5
):
    qs = np.sum(
        total_steps_above_thresh[total_detection_prob > detection_prob_thresh], axis=-1
    )
    n = np.sum(total_detection_prob > detection_prob_thresh, axis=-1)

    safe_n = np.where(n > 0, n, 1)
    safe_qs = np.where(qs > 0, qs, 1)
    safe_log_qs_n = np.log10(safe_qs / safe_n)
    adb = np.where(n > 0, np.where(qs > 0, safe_log_qs_n, -0.5), 0)

    return adb
