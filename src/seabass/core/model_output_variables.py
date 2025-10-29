import warnings
from typing import Tuple, TypedDict, Union

import numpy as np
from numpy import typing as npt

from seabass.core.features.bands import (
    critical_band_energies,
    critical_band_matrix,
    critical_bands,
    internal_noise,
    pitch_patterns,
)
from seabass.core.features.constants import (
    _AMPLITUDE_THRESHOLD,
    _BOUND_WINDOW,
    _DETECTION_PROBABILITY_THRESHOLD,
    _DZ_BASIC,
    _FS,
    _FULL_SCALE,
    _NFFT,
)
from seabass.core.features.detection import (
    asymmetric_excitation_db,
    average_block_distortion_basic,
    detection_probability,
    effective_detection_step_size,
    steps_above_threshold,
    total_detection_probability,
    total_steps_above_threshold,
)
from seabass.core.features.loudness import (
    approximate_loudness,
    average_loudness,
    average_loudness_difference,
)
from seabass.core.features.modulation import (
    average_modulation_difference1_basic,
    average_modulation_difference1_basic_temporal_weights,
    envelope_modulation,
    instantaneous_modulation_difference1_basic,
    modulation_alphas,
    scaled_instantaneous_modulation_difference1_basic,
)
from seabass.core.features.spectral import (
    stft,
    stft_magnitude_squared,
    weighted_magnitude_squared,
)
from seabass.core.features.spreading import (
    smeared_excitation_pattern,
    time_spreading_alphas,
    unsmeared_excitation_pattern,
)
from seabass.core.features.utils import linear_pow_to_db, signal_bounds

MovsOutputDict = TypedDict(
    "MovsOutputDict",
    {
        "avg_mod_diff1": Union[np.ndarray, float],
        "adb": Union[np.ndarray, float],
    },
)


class PeaqModelOutputVariables:
    _ALLOWED_PEAQ_VERSIONS = ["basic"]  # ["basic", "advanced"]
    _N_CENTER_FREQS_BASIC = 55
    _N_CENTER_FREQS_ADVANCED = 109
    _FSS_DIVISOR = 1024
    _DELAY_AVG_SECONDS = 0.5

    def __init__(
        self,
        test_signal: npt.ArrayLike,
        ref_signal: npt.ArrayLike,
        signal_dtype: str = "double",
        fs: int = _FS,
        n_fft: int = _NFFT,
        peaq_version: str = "basic",
    ):
        self.fs = fs
        self.n_fft = n_fft

        if peaq_version not in self._ALLOWED_PEAQ_VERSIONS:
            raise ValueError(
                f"`peaq_version` must be one of {self._ALLOWED_PEAQ_VERSIONS}"
            )
        self.peaq_version = peaq_version

        if self.peaq_version == "basic":
            self.n_center_freqs = self._N_CENTER_FREQS_BASIC
        elif self.peaq_version == "advanced":  # pragma: no cover
            self.n_center_freqs = self._N_CENTER_FREQS_ADVANCED
        else:  # pragma: no cover
            raise ValueError(f"Invalid peaq_version: {self.peaq_version}")

        self.fss = self.fs / self._FSS_DIVISOR

        self.dz = _DZ_BASIC

        (
            self.freq_edge_low_hz,
            self.freq_edge_high_hz,
            self.freq_centers_hz,
        ) = critical_bands(self.dz)

        self.detection_probability_threshold = _DETECTION_PROBABILITY_THRESHOLD

        self.test_signal, self.ref_signal = self._validate_inputs(
            test_signal, ref_signal, signal_dtype
        )
        self.frame_start, self.frame_end = self.bound_signals()

    def _validate_inputs(
        self, test_signal: npt.ArrayLike, ref_signal: npt.ArrayLike, signal_dtype
    ) -> Tuple[np.ndarray, np.ndarray]:
        test_signal = np.asarray(test_signal)
        ref_signal = np.asarray(ref_signal)

        assert test_signal.ndim in [1, 2]
        assert ref_signal.ndim in [1, 2]

        if test_signal.ndim == 1:
            test_signal = test_signal[None, :]
        if ref_signal.ndim == 1:
            ref_signal = ref_signal[None, :]

        n_chan_e, n_sampl_t = test_signal.shape
        n_chan_r, n_sampl_r = ref_signal.shape

        assert n_chan_e == n_chan_r
        if n_sampl_t != n_sampl_r:
            warnings.warn(
                f"test_signal and ref_signal have different numbers of samples: "
                f"{n_sampl_t} and {n_sampl_r}, respectively. "
                f"Using the minimum number of samples ({min(n_sampl_t, n_sampl_r)})."
            )

            n_sampl = min(n_sampl_t, n_sampl_r)

            test_signal = test_signal[:, :n_sampl]
            ref_signal = ref_signal[:, :n_sampl]

        if signal_dtype != "int16":
            test_signal *= _FULL_SCALE + 1
            ref_signal *= _FULL_SCALE + 1

        return test_signal, ref_signal

    def bound_signals(
        self, threshold=_AMPLITUDE_THRESHOLD, win_size: int = _BOUND_WINDOW
    ) -> Tuple[int, int]:
        return signal_bounds(self.ref_signal, threshold, win_size, self.n_fft // 2)

    @property
    def ref_stft(self) -> np.ndarray:
        # print('ref_signal', self.ref_signal.shape)
        return stft(self.ref_signal)[..., self.frame_start : self.frame_end]

    @property
    def test_stft(self) -> np.ndarray:
        # print('test_signal', self.test_signal.shape)
        return stft(self.test_signal)[..., self.frame_start : self.frame_end]

    @property
    def stft_freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, 1 / self.fs)

    @property
    def ref_magnitude_squared(self) -> np.ndarray:
        return stft_magnitude_squared(self.ref_stft)

    @property
    def test_magnitude_squared(self) -> np.ndarray:
        return stft_magnitude_squared(self.test_stft)

    @property
    def test_weighted_magnitude_squared(self) -> np.ndarray:
        """
        Weighted magnitude squared of the test signal.
        :return: np.ndarray, shape=(n_chan, n_freq, n_time)
        """
        return weighted_magnitude_squared(
            self.test_magnitude_squared, fs=self.fs, n_fft=self.n_fft
        )

    @property
    def ref_weighted_magnitude_squared(self) -> np.ndarray:
        """
        Weighted magnitude squared of the reference signal.
        :return: np.ndarray, shape=(n_chan, n_freq, n_time)
        """
        return weighted_magnitude_squared(
            self.ref_magnitude_squared, fs=self.fs, n_fft=self.n_fft
        )

    @property
    def bark_filter_bank(self):
        return critical_band_matrix(
            self.n_fft, self.fs, self.freq_edge_low_hz, self.freq_edge_high_hz
        )

    @property
    def ref_center_freq_energies(self):
        return critical_band_energies(
            self.ref_weighted_magnitude_squared, self.bark_filter_bank
        )

    @property
    def test_center_freq_energies(self):
        return critical_band_energies(
            self.test_weighted_magnitude_squared, self.bark_filter_bank
        )

    @property
    def internal_noise(self) -> np.ndarray:
        return internal_noise(self.freq_centers_hz)

    @property
    def ref_pitch_pattern(self):
        return pitch_patterns(self.ref_center_freq_energies, self.internal_noise)

    @property
    def test_pitch_pattern(self):
        return pitch_patterns(self.test_center_freq_energies, self.internal_noise)

    @property
    def ref_unsmeared_excitation_pattern(self):
        return unsmeared_excitation_pattern(
            self.freq_centers_hz,
            self.ref_pitch_pattern,
            dz=self.dz,
        )

    @property
    def test_unsmeared_excitation_pattern(self):
        return unsmeared_excitation_pattern(
            self.freq_centers_hz,
            self.test_pitch_pattern,
            dz=self.dz,
        )

    @property
    def ref_approx_loudness(self):
        return approximate_loudness(self.ref_unsmeared_excitation_pattern)

    @property
    def test_approx_loudness(self):
        return approximate_loudness(self.test_unsmeared_excitation_pattern)

    @property
    def modulation_alphas(self):
        return modulation_alphas(self.freq_centers_hz, fss=self.fss)

    @property
    def ref_average_loudness(self):
        return average_loudness(self.ref_approx_loudness, self.modulation_alphas)

    @property
    def test_average_loudness(self):
        return average_loudness(self.test_approx_loudness, self.modulation_alphas)

    @property
    def ref_average_loudness_difference(self):
        return average_loudness_difference(
            self.ref_approx_loudness, self.modulation_alphas, self.fss
        )

    @property
    def test_average_loudness_difference(self):
        return average_loudness_difference(
            self.test_approx_loudness, self.modulation_alphas, self.fss
        )

    @property
    def ref_envelope_modulation(self) -> np.ndarray:
        return envelope_modulation(
            self.ref_average_loudness, self.ref_average_loudness_difference
        )

    @property
    def test_envelope_modulation(self) -> np.ndarray:
        return envelope_modulation(
            self.test_average_loudness, self.test_average_loudness_difference
        )

    @property
    def instantaneous_modulation_difference1_basic(
        self,
    ) -> np.ndarray:
        return instantaneous_modulation_difference1_basic(
            self.ref_envelope_modulation,
            self.test_envelope_modulation,
        )

    @property
    def scaled_instantaneous_modulation_difference1_basic(
        self,
    ):
        return scaled_instantaneous_modulation_difference1_basic(
            self.instantaneous_modulation_difference1_basic,
        )

    @property
    def delayed_avg_frames(self):
        return max(
            0, int(np.ceil(self._DELAY_AVG_SECONDS * self.fss)) - self.frame_start
        )

    @property
    def average_modulation_difference1_basic(
        self,
    ) -> np.ndarray:
        return average_modulation_difference1_basic(
            self.scaled_instantaneous_modulation_difference1_basic,
            self.average_modulation_difference1_basic_temporal_weights,
            self.delayed_avg_frames,
            keep_channels=False,
        )

    @property
    def average_modulation_difference1_basic_temporal_weights(self):
        return average_modulation_difference1_basic_temporal_weights(
            self.ref_average_loudness,
            self.internal_noise,
        )

    @property
    def time_spread_alphas(self):
        return time_spreading_alphas(self.freq_centers_hz, self.fss)

    @property
    def ref_smeared_excitation_pattern(self):
        return smeared_excitation_pattern(
            self.ref_unsmeared_excitation_pattern, self.time_spread_alphas
        )

    @property
    def test_smeared_excitation_pattern(self):
        return smeared_excitation_pattern(
            self.test_unsmeared_excitation_pattern, self.time_spread_alphas
        )

    @property
    def ref_smeared_excitation_pattern_db(self):
        return linear_pow_to_db(self.ref_smeared_excitation_pattern)

    @property
    def test_smeared_excitation_pattern_db(self):
        return linear_pow_to_db(self.test_smeared_excitation_pattern)

    @property
    def asymmetric_excitation_db(self):
        return asymmetric_excitation_db(
            self.ref_smeared_excitation_pattern_db,
            self.test_smeared_excitation_pattern_db,
        )

    @property
    def effective_detection_step_size(self):
        return effective_detection_step_size(self.asymmetric_excitation_db)

    @property
    def smeared_excitation_pattern_db_difference(self):
        return (
            self.ref_smeared_excitation_pattern_db
            - self.test_smeared_excitation_pattern_db
        )

    @property
    def detection_probability(self):
        return detection_probability(
            self.smeared_excitation_pattern_db_difference,
            self.effective_detection_step_size,
        )

    @property
    def steps_above_threshold(self):
        return steps_above_threshold(
            self.smeared_excitation_pattern_db_difference,
            self.effective_detection_step_size,
        )

    @property
    def total_detection_probability(self):
        return total_detection_probability(self.detection_probability)

    @property
    def total_steps_above_threshold(self):
        return total_steps_above_threshold(self.steps_above_threshold)

    @property
    def average_block_distortion(self) -> float:
        return average_block_distortion_basic(
            self.total_detection_probability,
            self.total_steps_above_threshold,
            detection_prob_thresh=self.detection_probability_threshold,
        )

    def compute(self) -> MovsOutputDict:
        movs = {
            "avg_mod_diff1": self.average_modulation_difference1_basic,
            "adb": self.average_block_distortion,
        }
        return movs
