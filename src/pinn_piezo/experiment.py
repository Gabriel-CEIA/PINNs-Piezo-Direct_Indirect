"""Experiment configuration: dataclass with YAML load/save and CLI overrides.

Usage:
    config = ExperimentConfig.load("config.yaml")
    config.override("training.lr_adam=0.005")
    config.save("output.yaml")
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BeamConfig:
    width: float = 100e-3
    height: float = 1e-3


@dataclass
class MaterialConfig:
    E: float = 2.0e9
    NU: float = 0.29
    G: float = 0.775e9
    d31: float = 2.2e-11
    d33: float = -3.0e-11
    rel_permittivity: float = 12


@dataclass
class ArchitectureConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [100, 250])
    activation: str = "tanh"
    model_type: str = "pyramid"


@dataclass
class TrainingConfig:
    epochs_adam: int = 1000
    epochs_lbfgs: int = 200
    lr_adam: float = 0.001
    lr_lbfgs: float = 0.01
    data_fraction: float = 1.0
    adaptive_weight_f: int = 500
    data_suffix: str = "_m1"
    gradient_clip: float = 0.5
    adaptive_alpha: float = 0.9


@dataclass
class GeometryConfig:
    n_points: int = 400
    n_collocation: int = 150
    n_collocation_test: int = 200


@dataclass
class LoadingConfig:
    voltage: float = 100.0
    applied_force_y: float = 0.1


@dataclass
class RunConfig:
    seed: int | None = None
    name: str | None = None


@dataclass
class ExperimentConfig:
    formulation: str = "indirect"

    beam: BeamConfig = field(default_factory=BeamConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    arch: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self):
        if self.formulation == "direct":
            self._apply_direct_defaults()

    def _apply_direct_defaults(self):
        self.training.epochs_adam = 3000
        self.training.epochs_lbfgs = 0
        self.training.lr_adam = 0.005
        self.training.lr_lbfgs = 0.0001
        self.training.data_fraction = 0.75
        self.training.adaptive_weight_f = 200
        self.training.data_suffix = "_m1_d"

    @staticmethod
    def _dict_to_dataclass(data: dict, cls: type) -> Any:
        from typing import get_type_hints
        hints = get_type_hints(cls)
        kwargs = {}
        for key, value in data.items():
            if key in hints:
                ft = hints[key]
                if hasattr(ft, "__dataclass_fields__"):
                    kwargs[key] = ExperimentConfig._dict_to_dataclass(value, ft)
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls._dict_to_dataclass(data, cls)
        config.__post_init__()
        return config

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        config = cls._dict_to_dataclass(data, cls)
        config.__post_init__()
        return config

    def override(self, key_value: str) -> None:
        key, value = key_value.split("=", 1)
        parts = key.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        raw = yaml.safe_load(value)
        setattr(obj, parts[-1], raw)

    def clone(self) -> ExperimentConfig:
        return copy.deepcopy(self)

    @classmethod
    def default_indirect(cls) -> ExperimentConfig:
        return cls(formulation="indirect")

    @classmethod
    def default_direct(cls) -> ExperimentConfig:
        config = cls(formulation="direct")
        config._apply_direct_defaults()
        return config
