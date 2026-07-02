"""Train the direct (force-driven) PINN.

Usage:
    python -m scripts.train_direct
    python -m scripts.train_direct --config config.yaml
    python -m scripts.train_direct --mlflow
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import torch
from torchsummary import summary

from pinn_piezo.config import DATA_DIR, RUNS_DIR, get_device
from pinn_piezo.direct import model as model_mod
from pinn_piezo.direct import train as train_mod
from pinn_piezo.experiment import ExperimentConfig


def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file.")
    p.add_argument("--mlflow", action="store_true", default=False,
                   help="Enable MLflow experiment tracking.")
    p.add_argument("--mlflow-experiment", type=str, default="pinn-piezo",
                   help="MLflow experiment name.")
    p.add_argument("override", nargs="*",
                   help="Override config values: training.lr_adam=0.005")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig.default_direct()

    for ov in args.override:
        config.override(ov)

    mlflow = _get_mlflow()
    if args.mlflow and mlflow is not None:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=config.run.name)
        mlflow.log_params(config.to_dict())

    torch.set_default_dtype(torch.float32)
    device = get_device()
    print(f"Using device: {device}")

    run_name = config.run.name or (
        f"train_direct_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    config.save(run_dir / "config.yaml")

    model = model_mod.build_default_model(
        device=device,
        hidden_sizes=tuple(config.arch.hidden_sizes),
    )
    try:
        summary(model, (2,))
    except Exception:
        pass

    arrays = train_mod.load_dataset(
        DATA_DIR,
        suffix=config.training.data_suffix,
        fraction=config.training.data_fraction,
    )
    tensors = train_mod.to_device(arrays, device, dtype=torch.float32)
    print("Collocation shapes:",
          tensors["x_collocation"].shape, tensors["y_collocation"].shape)

    result = train_mod.train(
        model, tensors,
        epochs_adam=config.training.epochs_adam,
        epochs_lbfgs=config.training.epochs_lbfgs,
        lr_adam=config.training.lr_adam,
        lr_lbfgs=config.training.lr_lbfgs,
        f=config.training.adaptive_weight_f,
        mlflow=mlflow,
    )

    save_path = models_dir / "model_PINN_direct.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")

    np.save(run_dir / "loss_direct.npy", np.array(result["loss_list"]))

    if args.mlflow and mlflow is not None:
        mlflow.log_artifact(str(save_path))
        mlflow.log_artifact(str(run_dir / "config.yaml"))
        mlflow.log_metric("best_loss", result["loss_list"][-1])
        mlflow.log_metric("total_time_seconds", result["total_time"])
        if result.get("component_history"):
            ch = result["component_history"]
            np.save(run_dir / "component_history.npy", ch)
        mlflow.end_run()


if __name__ == "__main__":
    main()
