from __future__ import annotations

import numpy as np

from pinn_piezo import materials as M


class TestMaterialConstants:
    def test_c11_c12(self):
        expected_c11 = M.E / (1 - M.NU ** 2)
        assert np.isclose(M.C11, expected_c11)
        assert np.isclose(M.C12, M.NU * M.C11)

    def test_ctop_length(self):
        assert len(M.ctop) == 36  # 9 groups of 4 coefficients each

    def test_cbot_length(self):
        assert len(M.cbot) == 36

    def test_ctop_cbot_differ(self):
        assert not np.allclose(M.ctop, M.cbot)

    def test_sign_flip_e31(self):
        assert np.isclose(M.e31_top, -M.e31_bottom)
        assert M.e31_top < 0
        assert M.e31_bottom > 0

    def test_sign_flip_e33(self):
        assert np.isclose(M.e33_top, -M.e33_bottom)

    def test_sign_flip_e11(self):
        assert np.isclose(M.e11_top, -M.e11_bottom)

    def test_epsilon_nonzero(self):
        assert abs(M.epsilon_1) > 0
        assert abs(M.epsilon_2) > 0

    def test_permittivity_free_space_positive(self):
        assert M.permittivity_free_space > 0
