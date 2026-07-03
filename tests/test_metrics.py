from __future__ import annotations

import numpy as np

from pinn_piezo.metrics import (
    mae,
    max_abs_error,
    normalized_rmse,
    relative_l2_error,
    rmse,
    field_metrics,
)


class TestScalarMetrics:
    def test_perfect_match(self):
        x = np.array([1.0, 2.0, 3.0])
        assert relative_l2_error(x, x) == 0.0
        assert rmse(x, x) == 0.0
        assert mae(x, x) == 0.0
        assert max_abs_error(x, x) == 0.0

    def test_relative_l2(self):
        pred = np.array([2.0, 3.0, 4.0])
        ref = np.array([1.0, 2.0, 3.0])
        expected = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
        assert np.isclose(relative_l2_error(pred, ref), expected)

    def test_rmse(self):
        pred = np.array([2.0, 2.0])
        ref = np.array([0.0, 0.0])
        assert np.isclose(rmse(pred, ref), 2.0)

    def test_mae(self):
        pred = np.array([3.0, -1.0])
        ref = np.array([0.0, 0.0])
        assert np.isclose(mae(pred, ref), 2.0)

    def test_max_abs(self):
        pred = np.array([10.0, -5.0, 3.0])
        ref = np.array([0.0, 0.0, 0.0])
        assert np.isclose(max_abs_error(pred, ref), 10.0)

    def test_normalized_rmse(self):
        pred = np.array([1.0, 1.0])
        ref = np.array([0.0, 2.0])
        nrmse = normalized_rmse(pred, ref)
        expected = np.sqrt(np.mean((pred - ref) ** 2)) / (2.0 - 0.0)
        assert np.isclose(nrmse, expected)

    def test_normalized_rmse_constant_ref(self):
        ref = np.array([5.0, 5.0, 5.0])
        pred = np.array([5.5, 4.5, 5.0])
        assert np.isnan(normalized_rmse(pred, ref))

    def test_zero_ref_nan(self):
        ref = np.array([0.0, 0.0])
        pred = np.array([1.0, 1.0])
        assert np.isnan(relative_l2_error(pred, ref))
        assert np.isnan(normalized_rmse(pred, ref))


class TestFieldMetrics:
    def test_field_metrics_keys(self):
        x = np.array([1.0, 2.0, 3.0])
        m = field_metrics(x, x)
        assert set(m.keys()) == {"rel_L2", "RMSE", "MAE", "max_abs", "nRMSE"}

    def test_metrics_table(self, capsys):
        from pinn_piezo.metrics import metrics_table
        preds = {"u": np.array([1.0, 2.0]), "v": np.array([0.5, 1.5])}
        refs = {"u": np.array([1.0, 2.0]), "v": np.array([1.0, 1.0])}
        df = metrics_table(preds, refs, fields=["u", "v"])
        assert list(df.index) == ["u", "v"]
        assert df.loc["u", "rel_L2"] == 0.0
