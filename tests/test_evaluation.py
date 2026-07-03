from __future__ import annotations

import numpy as np

from pinn_piezo.evaluation import relative_l2_error


class TestRelativeL2:
    def test_perfect_match(self):
        assert relative_l2_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

    def test_known_value(self):
        pred = np.array([2.0, 4.0])
        ref = np.array([1.0, 2.0])
        expected = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
        assert np.isclose(relative_l2_error(pred, ref), expected)
