from __future__ import annotations

from pathlib import Path

import pytest

from pinn_piezo.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from pinn_piezo.experiment import ExperimentConfig


class TestPaths:
    def test_default_resolve(self):
        assert DATA_DIR.name == "data"
        assert MODELS_DIR.name == "models"
        assert OUTPUTS_DIR.name == "outputs"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PINN_PIEZO_DATA_DIR", "/tmp/custom_data")
        monkeypatch.setenv("PINN_PIEZO_OUTPUTS_DIR", "/tmp/custom_outputs")
        import importlib
        import pinn_piezo.config as cfg
        importlib.reload(cfg)
        assert str(cfg.DATA_DIR) == "/tmp/custom_data"
        assert str(cfg.OUTPUTS_DIR) == "/tmp/custom_outputs"


class TestExperimentConfig:
    def test_default_indirect(self):
        c = ExperimentConfig.default_indirect()
        assert c.formulation == "indirect"
        assert c.training.epochs_adam == 1000
        assert c.training.epochs_lbfgs == 200
        assert c.training.data_suffix == "_m1"
        assert c.training.data_fraction == 1.0

    def test_default_direct(self):
        c = ExperimentConfig.default_direct()
        assert c.formulation == "direct"
        assert c.training.epochs_adam == 3000
        assert c.training.epochs_lbfgs == 0
        assert c.training.data_suffix == "_m1_d"
        assert c.training.data_fraction == 0.75

    def test_to_dict(self):
        c = ExperimentConfig.default_indirect()
        d = c.to_dict()
        assert d["formulation"] == "indirect"
        assert d["beam"]["width"] == 0.1
        assert d["training"]["epochs_adam"] == 1000

    def test_from_dict(self):
        data = {
            "formulation": "direct",
            "beam": {"width": 0.2, "height": 0.002},
            "training": {"epochs_adam": 500},
        }
        c = ExperimentConfig.from_dict(data)
        assert c.formulation == "direct"
        assert c.beam.width == 0.2
        assert c.beam.height == 0.002  # default
        assert c.training.epochs_adam == 500
        assert c.training.data_suffix == "_m1_d"  # direct default applied

    def test_yaml_roundtrip(self, tmp_path):
        c = ExperimentConfig.default_indirect()
        p = tmp_path / "test_config.yaml"
        c.save(p)
        c2 = ExperimentConfig.load(p)
        assert c2.formulation == c.formulation
        assert c2.training.epochs_adam == c.training.epochs_adam
        assert c2.training.data_suffix == c.training.data_suffix

    def test_override(self):
        c = ExperimentConfig.default_indirect()
        c.override("training.lr_adam=0.005")
        assert c.training.lr_adam == 0.005

    def test_override_nested(self):
        c = ExperimentConfig.default_indirect()
        c.override("beam.width=0.2")
        assert c.beam.width == 0.2

    def test_clone(self):
        c = ExperimentConfig.default_indirect()
        c2 = c.clone()
        c2.training.epochs_adam = 999
        assert c.training.epochs_adam == 1000
        assert c2.training.epochs_adam == 999

    def test_load_sample_indirect(self):
        repo = Path(__file__).resolve().parents[1]
        cfg_path = repo / "experiments" / "base_indirect.yaml"
        if not cfg_path.exists():
            pytest.skip("sample config not found")
        c = ExperimentConfig.load(cfg_path)
        assert c.formulation == "indirect"
        assert c.training.epochs_adam == 1000

    def test_load_sample_direct(self):
        repo = Path(__file__).resolve().parents[1]
        cfg_path = repo / "experiments" / "base_direct.yaml"
        if not cfg_path.exists():
            pytest.skip("sample config not found")
        c = ExperimentConfig.load(cfg_path)
        assert c.formulation == "direct"
        assert c.training.epochs_adam == 3000
