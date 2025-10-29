import numpy as np
import pytest
from numpy import testing as nptest

from seabass.core.features.utils import linear_pow_to_db, signal_bounds


class TestDbConversion:
    def test_linear_pow_to_db(self):
        x = np.linspace(1e-8, 1, 100)
        db = 10 * np.log10(x)
        nptest.assert_allclose(linear_pow_to_db(x), db)


class TestSignalBounds:
    def test_signal_bounds1(self):
        with pytest.raises(ValueError):
            signal = np.zeros((1, 48000))
            signal_bounds(signal)

    def test_signal_bounds2(self):
        with pytest.raises(ValueError):
            signal = (199 / 5) * np.ones((1, 48000))
            signal_bounds(signal)

    def test_signal_bounds3(self):
        signal = 200 * np.ones((1, 48000))
        start, end = signal_bounds(signal)
        assert start == 0
        assert end == 45
