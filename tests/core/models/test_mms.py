import numpy as np
import pytest
import soundfile as sf

from seabass.core.models.mms import (
    _compute_mms,
    estimated_mean_mushra_score,
    estimated_mean_mushra_score_file,
)


class TestResampleFile:
    def test_resample_file1(self):
        ref = "tests/assets/sassec/signals44100/orig/female_inst_sim_1.wav"
        test = "tests/assets/sassec/signals44100/orig/female_inst_sim_1.wav"

        score = estimated_mean_mushra_score_file(test, ref)
        assert np.isclose(score, 100.0)

    def test_resample_file2(self):
        ref = "tests/assets/sassec/signals/orig/female_inst_sim_1.wav"
        test = "tests/assets/sassec/signals44100/orig/female_inst_sim_1.wav"

        score = estimated_mean_mushra_score_file(test, ref)
        assert np.isclose(score, 100.0, atol=1e-3)

    def test_resample_file3(self):
        ref = "tests/assets/sassec/signals44100/orig/female_inst_sim_1.wav"
        test = "tests/assets/sassec/signals44100/anchor/female_inst_sim_1.wav"

        score = estimated_mean_mushra_score_file(test, ref)
        # score = 10.176891729814045

        assert np.abs(score - 10.147) < 0.05


class TestSignal:
    @pytest.mark.parametrize(
        "test_path, ref_path, expected_score, atol",
        [
            (
                "tests/assets/sassec/signals/orig/female_inst_sim_1.wav",
                "tests/assets/sassec/signals/orig/female_inst_sim_1.wav",
                100.0,
                1e-8,
            ),
            (
                "tests/assets/sassec/signals/Algo1/male_inst_sim_1.wav",
                "tests/assets/sassec/signals/orig/male_inst_sim_1.wav",
                52.987,
                0.1,
            ),
        ],
    )
    def test_signal(self, test_path, ref_path, expected_score, atol):
        test_signal, fs_t = sf.read(test_path, always_2d=True)
        ref_signal, fs_r = sf.read(ref_path, always_2d=True)

        score = estimated_mean_mushra_score(test_signal.T, ref_signal.T, fs=fs_t)

        assert np.isclose(score, expected_score, atol=atol)


class TestCompute:
    def test_compute_mms1(self):
        score = _compute_mms(0.0, 0.0)
        assert score >= 100.0

    def test_compute_mms2(self):
        score = _compute_mms(23.6544585012445, 1.92560575700812)
        assert np.isclose(score, 50.885, atol=1e-3)

    def test_compute_mms3(self):
        score = _compute_mms(np.array(23.6544585012445), np.array(1.92560575700812))
        assert np.isclose(score, 50.885, atol=1e-3)


class test_return_movs_true:
    ret = estimated_mean_mushra_score(
        np.ones((48000,)), np.ones((48000,)), _return_movs=True
    )

    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert isinstance(ret[0], float)
    assert isinstance(ret[1], dict)
