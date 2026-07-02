"""Parameter sweep orchestrator for the PINN pipeline.

Generates a cartesian product of parameter values, runs each combination,
and collects results for comparison.

Usage:
    # Grid search over learning rates and architectures:
    python -m scripts.sweep \\
        --base-config experiments/base_indirect.yaml \\
        --grid training.lr_adam=0.001,0.0005,0.0001 \\
        --grid arch.hidden_sizes="[50,50],[100,100],[100,250]"

    # Linear sweep over a single parameter:
    python -m scripts.sweep \\
        --base-config experiments/base_direct.yaml \\
        --grid loading.applied_force_y=0.05,0.1,0.2

    # Named runs (each combination gets a descriptive run name):
    python -m scripts.sweep \\
        --base-config experiments/base_indirect.yaml \\
        --grid training.lr_adam=0.001,0.0005 \\
        --name-template "lr_{training.lr_adam}"
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pinn_piezo.config import OUTPUTS_DIR
from pinn_piezo.experiment import ExperimentConfig


def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=str, required=True,
                   help="Base YAML config file.")
    p.add_argument("--grid", type=str, action="append", default=[],
                   help="Parameter grid: key=val1,val2,val3 "
                        "(repeat for multiple params).")
    p.add_argument("--name-template", type=str, default=None,
                   help="Run name template, e.g. 'lr_{training.lr_adam}'")
    p.add_argument("--mlflow-experiment", type=str, default="pinn-piezo-sweep",
                   help="MLflow experiment name for sweep runs.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for sweep summary.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the parameter grid without running.")
    return p.parse_args()


def _parse_grid(grid_args):
    """Parse --grid args into {key: [values]} dict."""
    grid = {}
    for arg in grid_args:
        key, values_str = arg.split("=", 1)
        raw_values = values_str.split(",")
        parsed = []
        for v in raw_values:
            parsed.append(_parse_value(v))
        grid[key] = parsed
    return grid


def _parse_value(v: str):
    """Parse a string value into int, float, list, or keep as string."""
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return _parse_value(parts[0]) if len(parts) == 1 else [_parse_value(p) for p in parts]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none":
        return None
    return v


def _build_run_name(template: str | None, combination: dict) -> str:
    if template is None:
        parts = [f"{k}={v}" for k, v in combination.items()]
        return "_".join(parts)
    name = template
    for key, value in combination.items():
        placeholder = "{" + key + "}"
        name = name.replace(placeholder, str(value))
    return name


def _apply_combination(config: ExperimentConfig, combination: dict):
    for key, value in combination.items():
        obj = config
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def _train_and_evaluate(config: ExperimentConfig, run_name: str, mlflow):
    """Run a single training + evaluation for the given config."""
    from pinn_piezo import plotting, geometry
    import torch

    formulation = config.formulation
    run_dir = OUTPUTS_DIR / "runs" / run_name
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    if mlflow is not None:
        mlflow.log_params(config.to_dict())

    torch.set_default_dtype(
        torch.float64 if formulation == "indirect" else torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate geometry if data doesn't exist
    from pinn_piezo.config import DATA_DIR
    suffix = config.training.data_suffix
    data_exists = (DATA_DIR / f"x_collocation_non_normalized{suffix}.npy").exists()
    if not data_exists:
        geometry.generate_and_save(
            n_points=config.geometry.n_points,
            n_collocation=config.geometry.n_collocation,
            n_collocation_test=config.geometry.n_collocation_test,
            suffix=suffix,
            data_dir=DATA_DIR,
        )

    # Train
    if formulation == "indirect":
        from pinn_piezo.indirect import model as ind_model
        from pinn_piezo.indirect import train as ind_train

        model = ind_model.build_default_model(
            device=device,
            model_type=config.arch.model_type,
            hidden_sizes=tuple(config.arch.hidden_sizes),
        )
        arrays = ind_train.load_dataset(
            DATA_DIR, suffix=config.training.data_suffix,
            fraction=config.training.data_fraction,
        )
        tensors = ind_train.to_device(arrays, device)
        result = ind_train.train(
            model, tensors,
            epochs_adam=config.training.epochs_adam,
            epochs_lbfgs=config.training.epochs_lbfgs,
            lr_adam=config.training.lr_adam,
            lr_lbfgs=config.training.lr_lbfgs,
            f=config.training.adaptive_weight_f,
            mlflow=mlflow,
        )
    else:
        from pinn_piezo.direct import model as dir_model
        from pinn_piezo.direct import train as dir_train

        model = dir_model.build_default_model(
            device=device,
            hidden_sizes=tuple(config.arch.hidden_sizes),
        )
        arrays = dir_train.load_dataset(
            DATA_DIR, suffix=config.training.data_suffix,
            fraction=config.training.data_fraction,
        )
        tensors = dir_train.to_device(arrays, device)
        result = dir_train.train(
            model, tensors,
            epochs_adam=config.training.epochs_adam,
            epochs_lbfgs=config.training.epochs_lbfgs,
            lr_adam=config.training.lr_adam,
            lr_lbfgs=config.training.lr_lbfgs,
            f=config.training.adaptive_weight_f,
            mlflow=mlflow,
        )

    # Save model and loss
    weights_path = run_dir / "models" / f"model_PINN_{formulation}.pt"
    torch.save(model.state_dict(), weights_path)
    np.save(run_dir / f"loss_{formulation}.npy", np.array(result["loss_list"]))

    if result.get("component_history"):
        np.save(run_dir / "component_history.npy", result["component_history"])

    # Evaluate
    x_test = np.load(
        DATA_DIR / f"x_collocation_test_non_normalized{suffix}.npy"
    )
    x_test = x_test[:, :2]
    x_test_t = torch.tensor(
        x_test, dtype=(torch.float64 if formulation == "indirect"
                        else torch.float32),
        device=device,
    )
    preds = model(x_test_t).detach().cpu().numpy()

    if mlflow is not None:
        mlflow.log_metric("best_loss", float(np.min(result["loss_list"])))
        mlflow.log_metric("total_time_seconds", result["total_time"])
        mlflow.log_artifact(str(weights_path))

    figures_dir = run_dir / "figures"
    plotting.plot_loss_curve(
        result["loss_list"], save=True, save_dir=figures_dir,
        filename="loss_curve.png", show=False,
    )

    for field_data, name in [(preds[:, 0], "u"), (preds[:, 1], "v"),
                              (preds[:, 2], "phi")]:
        plot_filename = f"{name}_field.png"
        plotting.plot_results(
            x_test[:, 0], x_test[:, 1], field_data,
            title=f"{name.upper()} — {formulation}",
            filename=plot_filename,
            xlabel='x(m)', ylabel='y(m)',
            colorbar_label=f"{name}(m)" if name != "phi" else "phi(V)",
            save=True, save_dir=figures_dir, show=False,
        )

    ch = result.get("component_history")
    if ch:
        plotting.plot_component_losses(
            result["loss_list"], ch.get("pde", []), ch.get("bc", []),
            save_dir=figures_dir, filename="component_losses.png",
        )
        plotting.plot_training_dashboard(
            result["loss_list"], ch.get("pde", []), ch.get("bc", []),
            ch.get("lambda_pde", []), ch.get("lambda_bc", []),
            save_dir=figures_dir, filename="training_dashboard.png",
        )

    return {
        "run_name": run_name,
        "best_loss": float(np.min(result["loss_list"])),
        "total_time": result["total_time"],
        "run_dir": str(run_dir),
    }


def main():
    args = parse_args()

    base_config = ExperimentConfig.load(args.base_config)
    grid = _parse_grid(args.grid)

    if not grid:
        print("No grid parameters specified. Use --grid key=val1,val2")
        sys.exit(1)

    # Generate cartesian product
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    print(f"Parameter grid: {len(combinations)} combinations")
    for combo in combinations:
        combo_dict = dict(zip(keys, combo))
        print(f"  {combo_dict}")

    if args.dry_run:
        return

    sweep_start = time.time()
    results = []

    mlflow = _get_mlflow()
    if mlflow is not None:
        mlflow.set_experiment(args.mlflow_experiment)

    for idx, combo in enumerate(combinations, 1):
        combo_dict = dict(zip(keys, combo))
        config = copy.deepcopy(base_config)
        _apply_combination(config, combo_dict)

        run_name = _build_run_name(args.name_template, combo_dict)
        if args.name_template is None:
            run_name = (
                f"sweep_{config.formulation}_{idx:03d}_"
                f"{datetime.now().strftime('%H%M%S')}"
            )

        print(f"\n[{idx}/{len(combinations)}] Running: {run_name}")
        print(f"  Params: {combo_dict}")

        if mlflow is not None:
            mlflow.start_run(run_name=run_name)

        run_result = _train_and_evaluate(config, run_name, mlflow)
        run_result["params"] = combo_dict
        results.append(run_result)

        print(f"  Best loss: {run_result['best_loss']:.6e}")
        print(f"  Time: {run_result['total_time']:.1f}s")

        if mlflow is not None:
            mlflow.end_run()

    # Summarize
    total_time = time.time() - sweep_start
    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(results)} runs in {total_time:.1f}s")
    print(f"{'='*60}")

    table = []
    for r in results:
        row = {"run": r["run_name"], "best_loss": r["best_loss"],
               "time_s": f"{r['total_time']:.1f}", **r["params"]}
        table.append(row)

    sweep_dir = Path(args.output_dir or OUTPUTS_DIR / "sweeps" /
                     datetime.now().strftime("%Y%m%d-%H%M%S"))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_path = sweep_dir / "sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump({"parameters": keys, "runs": results}, f, indent=2,
                  default=str)
    print(f"\nResults saved to {summary_path}")

    # Print comparison table
    print(f"\n{'Run':<30} {'Best Loss':<15} {'Time (s)':<10}  Params")
    print("-" * 80)
    for r in results:
        param_str = " ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"{r['run_name']:<30} {r['best_loss']:<15.6e} "
              f"{r['total_time']:<10.1f} {param_str}")

    # Generate comparison plot if we have loss data
    if len(results) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from pinn_piezo import plotting as plt_mod

            loss_curves = []
            for r in results:
                loss_path = Path(r["run_dir"]) / f"loss_{base_config.formulation}.npy"
                if loss_path.exists():
                    loss_curves.append(np.load(loss_path))

            if len(loss_curves) == len(results):
                labels = [r["run_name"][:20] for r in results]
                plt_mod.plot_run_comparison(
                    loss_curves, labels,
                    save_dir=sweep_dir, filename="sweep_comparison.png",
                )
        except Exception as e:
            print(f"  Comparison plot skipped: {e}")


if __name__ == "__main__":
    main()
