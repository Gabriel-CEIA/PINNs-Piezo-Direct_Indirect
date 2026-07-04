"""Training driver for the indirect PINN."""

from __future__ import annotations

import time
from math import nan
from pathlib import Path

import numpy as np
import torch

from ..config import HEIGHT, WIDTH
from .losses import loss_func


def tensorize(x, device, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=True)


def load_dataset(data_dir: Path, suffix: str = "_m1", fraction: float = 1.0):
    data_dir = Path(data_dir)
    xy_top = np.load(data_dir / f"xy_top_non_normalized{suffix}.npy")
    xy_bottom = np.load(data_dir / f"xy_bottom_non_normalized{suffix}.npy")
    xy_right = np.load(data_dir / f"xy_right_non_normalized{suffix}.npy")
    xy_left = np.load(data_dir / f"xy_left_non_normalized{suffix}.npy")
    x_collocation_orig = np.load(data_dir / f"x_collocation_non_normalized{suffix}.npy")

    num_samples = int(fraction * len(x_collocation_orig))
    indices = np.random.choice(len(x_collocation_orig), num_samples, replace=False)
    x_collocation = x_collocation_orig[indices]

    x_collocation, coefficients = np.split(x_collocation, [2], axis=1)
    x_collocation, y_collocation = np.split(x_collocation, [1], axis=1)
    coefficients[:, 7] = -coefficients[:, 7]

    return {
        "xy_top": xy_top,
        "xy_bottom": xy_bottom,
        "xy_right": xy_right,
        "xy_left": xy_left,
        "x_collocation": x_collocation,
        "y_collocation": y_collocation,
        "coefficients": coefficients,
    }


def to_device(arrays, device, dtype=torch.float64, normalize=False):
    result = {k: tensorize(v, device, dtype=dtype).to(device)
              for k, v in arrays.items()}
    if normalize:
        result["x_collocation"] = result["x_collocation"] / WIDTH
        result["y_collocation"] = result["y_collocation"] / HEIGHT
        for key in ["xy_top", "xy_bottom", "xy_right", "xy_left"]:
            result[key][:, 0:1] = result[key][:, 0:1] / WIDTH
            result[key][:, 1:2] = result[key][:, 1:2] / HEIGHT
    return result


def run_adam(model, tensors, *,
             epochs: int = 1000,
             lr: float = 0.001,
             loss_weights=None,
             f: int = 500,
             checkpoints_dir: Path | None = None,
             mlflow=None,
             log_every: int = 10,
             lr_step_size: int = 5000,
             lr_gamma: float = 0.95,
             early_stop_patience: int = 200,
             early_stop_min_delta: float = 1e-8,
             normalize=False):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 1.0}

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    best_loss = float('inf')
    loss_list = []
    pde_list = []
    bc_list = []
    lambda_pde_list = []
    lambda_bc_list = []
    no_improve_count = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_result = loss_func(
            tensors["xy_top"], tensors["xy_bottom"],
            tensors["xy_right"], tensors["xy_left"],
            tensors["x_collocation"], tensors["y_collocation"],
            model, tensors["coefficients"], loss_weights, epoch, f,
            normalize=normalize,
        )
        loss, loss_weights = loss_result[0], loss_result[1]
        physics_total, bc_term = loss_result[2], loss_result[3]

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_list.append(loss_val)
        pde_list.append(physics_total.item())
        bc_list.append(bc_term.item())
        lambda_pde_list.append(loss_weights['pde'].item())
        lambda_bc_list.append(loss_weights['bc'].item())

        # Early stopping check
        if loss_val < best_loss - early_stop_min_delta:
            best_loss = loss_val
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}/{epochs}. Loss: {loss_val:.6e}.")
            if mlflow is not None:
                mlflow.log_metric("train_loss", loss_val, step=epoch)
                mlflow.log_metric("pde_loss", physics_total.item(), step=epoch)
                mlflow.log_metric("bc_loss", bc_term.item(), step=epoch)
                mlflow.log_metric("lambda_pde", loss_weights['pde'].item(), step=epoch)
                mlflow.log_metric("lambda_bc", loss_weights['bc'].item(), step=epoch)

            if loss_val < best_loss and checkpoints_dir is not None:
                best_loss = loss_val
                ckpt_path = Path(checkpoints_dir) / (
                    f"model_epoch_{epoch}_loss_{best_loss:.4f}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                    'loss_weights': loss_weights,
                }, ckpt_path)
                print(f"Checkpoint saved at epoch {epoch} with loss "
                      f"{best_loss:.4f}")

        if epoch % f == 0:
            print(f"Lambda_1: {loss_weights}.")

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {early_stop_patience} epochs).")
            break

    return loss_list, loss_weights, best_loss, {
        "pde": pde_list, "bc": bc_list,
        "lambda_pde": lambda_pde_list, "lambda_bc": lambda_bc_list,
    }


def run_lbfgs(model, tensors, loss_weights, *,
              epochs: int = 200,
              lr: float = 0.01,
              f: int = 500,
              checkpoints_dir: Path | None = None,
               epochs_adam_offset: int = 0,
               mlflow=None,
               normalize=False):
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr)
    best_loss = float('inf')
    loss_list = []
    total_epochs = epochs_adam_offset + epochs

    for epoch in range(epochs):

        def closure():
            nonlocal loss_weights
            optimizer.zero_grad()
            loss_result = loss_func(
                tensors["xy_top"], tensors["xy_bottom"],
                tensors["xy_right"], tensors["xy_left"],
                tensors["x_collocation"], tensors["y_collocation"],
                model, tensors["coefficients"], loss_weights, epoch, f,
                normalize=normalize,
            )
            loss_result[0].backward()
            return loss_result[0]

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step(closure)
        loss_result = loss_func(
            tensors["xy_top"], tensors["xy_bottom"],
            tensors["xy_right"], tensors["xy_left"],
            tensors["x_collocation"], tensors["y_collocation"],
            model, tensors["coefficients"], loss_weights, epoch, f,
            normalize=normalize,
        )
        loss, loss_weights = loss_result[0], loss_result[1]
        loss_list.append(loss.item())

        if loss.item() == nan:
            print('nan')
            break

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch: {epochs_adam_offset + epoch}/{total_epochs}. "
                  f"Loss: {loss.item():.6e}.")
            if mlflow is not None:
                mlflow.log_metric("train_loss", loss.item(),
                                  step=epochs_adam_offset + epoch)
                mlflow.log_metric("pde_loss", loss_result[2].item(),
                                  step=epochs_adam_offset + epoch)
                mlflow.log_metric("bc_loss", loss_result[3].item(),
                                  step=epochs_adam_offset + epoch)
                mlflow.log_metric("lambda_pde", loss_weights['pde'].item(),
                                  step=epochs_adam_offset + epoch)
                mlflow.log_metric("lambda_bc", loss_weights['bc'].item(),
                                  step=epochs_adam_offset + epoch)

        if loss.item() < best_loss and checkpoints_dir is not None:
            best_loss = loss.item()
            ckpt_path = Path(checkpoints_dir) / (
                f"model_epoch_{epoch}_loss_{best_loss:.4f}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'loss_weights': loss_weights,
            }, ckpt_path)

    return loss_list, loss_weights, best_loss


def train(model, tensors, *,
          epochs_adam: int = 1000,
          epochs_lbfgs: int = 200,
          lr_adam: float = 0.001,
          lr_lbfgs: float = 0.01,
          loss_weights=None,
          f: int = 500,
          checkpoints_adam_dir: Path | None = None,
          checkpoints_lbfgs_dir: Path | None = None,
          mlflow=None,
          lr_step_size: int = 5000,
          lr_gamma: float = 0.95,
           early_stop_patience: int = 200,
           early_stop_min_delta: float = 1e-8,
           normalize=False):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 1.0}

    start_time = time.time()

    loss_list_adam, loss_weights, best_loss_adam, component_history = run_adam(
        model, tensors,
        epochs=epochs_adam, lr=lr_adam,
        loss_weights=loss_weights, f=f,
        checkpoints_dir=checkpoints_adam_dir,
        mlflow=mlflow,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        normalize=normalize,
    )

    loss_list_lbfgs, loss_weights, best_loss_lbfgs = run_lbfgs(
        model, tensors, loss_weights,
        epochs=epochs_lbfgs, lr=lr_lbfgs, f=f,
        checkpoints_dir=checkpoints_lbfgs_dir,
        epochs_adam_offset=epochs_adam,
        mlflow=mlflow,
        normalize=normalize,
    )

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    return {
        "loss_list": loss_list_adam + loss_list_lbfgs,
        "best_loss_adam": best_loss_adam,
        "best_loss_lbfgs": best_loss_lbfgs,
        "loss_weights": loss_weights,
        "component_history": component_history,
        "total_time": total_time,
    }
