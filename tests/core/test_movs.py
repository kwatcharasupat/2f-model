import numpy as np
import pytest

from seabass.core.model_output_variables import PeaqModelOutputVariables


@pytest.mark.parametrize(
    "peaq_version, assert_", [("basic", True), ("advanced", False), ("expert", False)]
)
def test_peaq_version(peaq_version, assert_):
    if not assert_:
        with pytest.raises(ValueError):
            PeaqModelOutputVariables(
                peaq_version=peaq_version,
                test_signal=np.zeros(48000),
                ref_signal=np.zeros(48000),
            )
    else:
        try:
            PeaqModelOutputVariables(
                peaq_version=peaq_version,
                test_signal=np.ones(48000),
                ref_signal=np.ones(48000),
            )
        except ValueError:
            pytest.fail(f"peaq_version={peaq_version} should be valid")


def test_unequal_sample():
    with pytest.warns(UserWarning):
        PeaqModelOutputVariables(
            test_signal=np.ones(48000), ref_signal=np.ones(48000 * 2)
        )


def test_freqs():
    movs = PeaqModelOutputVariables(
        test_signal=np.ones(48000), ref_signal=np.ones(48000)
    )
    freqs = movs.stft_freqs

    assert np.allclose(freqs, np.fft.rfftfreq(2048, 1 / 48000))
