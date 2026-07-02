"""Plotting helpers shared across the two PINN formulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PAPER_RC = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
}


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)  # type: ignore[arg-type]


def _ensure_dir(save_dir):
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)


def _save_or_show(fig, save_dir, filename, show, dpi=300):
    if save_dir and filename:
        out = Path(save_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format='png', dpi=dpi, bbox_inches='tight')
        print(f"Figure saved at {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_results(x, y, value, title, filename=None,
                 xlabel='x', ylabel='y',
                 colorbar_label='Value', save=False, save_dir=None,
                 show=True):
    """Scatter colormap used to visualise the PINN/FEM fields."""
    fig = plt.figure(figsize=(7, 2))
    ax = fig.add_subplot(111)

    ax.set_title(title, fontsize=11, fontfamily='serif')
    ax.set_xlabel(xlabel, fontsize=10, fontfamily='serif')
    ax.set_ylabel(ylabel, fontsize=10, fontfamily='serif')

    scatter = ax.scatter(x, y, c=value, cmap='jet', s=30, edgecolors='none')

    ax.tick_params(axis='both', which='major', labelsize=8,
                   direction='in', length=4, width=0.8)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(0.8)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(colorbar_label, fontsize=10, fontfamily='serif')
    cbar.ax.tick_params(labelsize=8)

    _save_or_show(fig, save_dir if save else None,
                  filename if save else None, show)


def plot_loss_curve(loss_list, save=False, save_dir=None,
                    filename="loss_curve.png", show=True):
    apply_paper_style()
    loss_array = np.array(loss_list)
    log_loss = np.log(np.clip(loss_array, 1e-30, None))

    fig = plt.figure(figsize=(6, 4))
    plt.plot(log_loss, color='navy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss (log scale)')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    _save_or_show(fig, save_dir if save else None,
                  filename if save else None, show)


def plot_component_losses(total_loss, pde_loss, bc_loss, *,
                          save_dir=None, filename="component_losses.png",
                          show=False):
    """Overlaid PDE, BC, and total loss curves (log-scale)."""
    apply_paper_style()
    fig = plt.figure(figsize=(8, 5))

    total_arr = np.log(np.clip(np.array(total_loss), 1e-30, None))
    pde_arr = np.log(np.clip(np.array(pde_loss), 1e-30, None))
    bc_arr = np.log(np.clip(np.array(bc_loss), 1e-30, None))

    plt.plot(total_arr, color='navy', linewidth=2, label='Total')
    plt.plot(pde_arr, color='crimson', linewidth=1.5, alpha=0.8, label='PDE')
    plt.plot(bc_arr, color='forestgreen', linewidth=1.5, alpha=0.8, label='BC')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss Components')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    _save_or_show(fig, save_dir, filename, show)


def plot_weight_evolution(lambda_pde, lambda_bc, *,
                          save_dir=None, filename="weight_evolution.png",
                          show=False):
    """Adaptive loss weight values over epochs."""
    apply_paper_style()
    fig = plt.figure(figsize=(8, 4))

    plt.plot(lambda_pde, color='crimson', linewidth=1.5, label=r'$\lambda_{PDE}$')
    plt.plot(lambda_bc, color='forestgreen', linewidth=1.5, label=r'$\lambda_{BC}$')
    plt.xlabel('Epoch')
    plt.ylabel('Weight value')
    plt.title('Adaptive Loss Weights')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    _save_or_show(fig, save_dir, filename, show)


def plot_error_histogram(errors, field_name="u", *,
                         bins=50, save_dir=None,
                         filename=None, show=False):
    """Distribution of pointwise prediction errors for a field."""
    apply_paper_style()
    fig = plt.figure(figsize=(7, 4))

    errors_flat = np.asarray(errors).ravel()
    plt.hist(errors_flat, bins=bins, color='steelblue', edgecolor='white',
             alpha=0.8, density=True)
    plt.axvline(np.median(errors_flat), color='red', linestyle='--',
                linewidth=1.5, label=f'Median: {np.median(errors_flat):.2e}')
    plt.axvline(np.mean(errors_flat), color='orange', linestyle=':',
                linewidth=1.5, label=f'Mean: {np.mean(errors_flat):.2e}')
    plt.xlabel(f'{field_name} error')
    plt.ylabel('Density')
    plt.title(f'{field_name.upper()} Error Distribution')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    _save_or_show(fig, save_dir,
                  filename or f"error_hist_{field_name}.png", show)


def plot_training_dashboard(total_loss, pde_loss=None, bc_loss=None,
                            lambda_pde=None, lambda_bc=None, *,
                            save_dir=None, filename="training_dashboard.png",
                            show=False):
    """2x2 dashboard: total loss, component losses, weight evolution,
    and loss ratio."""
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    total_arr = np.log(np.clip(np.array(total_loss), 1e-30, None))
    ax = axes[0, 0]
    ax.plot(total_arr, color='navy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('log(Total Loss)')
    ax.set_title('Total Training Loss')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    if pde_loss is not None and bc_loss is not None:
        ax = axes[0, 1]
        pde_arr = np.log(np.clip(np.array(pde_loss), 1e-30, None))
        bc_arr = np.log(np.clip(np.array(bc_loss), 1e-30, None))
        ax.plot(pde_arr, color='crimson', linewidth=1.5, alpha=0.8, label='PDE')
        ax.plot(bc_arr, color='forestgreen', linewidth=1.5, alpha=0.8, label='BC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('log(Loss)')
        ax.set_title('PDE vs BC Loss')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    if lambda_pde is not None and lambda_bc is not None:
        ax = axes[1, 0]
        ax.plot(lambda_pde, color='crimson', linewidth=1.5,
                label=r'$\lambda_{PDE}$')
        ax.plot(lambda_bc, color='forestgreen', linewidth=1.5,
                label=r'$\lambda_{BC}$')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Loss Weights')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        ax = axes[1, 1]
        ratio = np.array(lambda_pde) / (np.array(lambda_bc) + 1e-12)
        ax.plot(ratio, color='purple', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\lambda_{PDE} / \lambda_{BC}$')
        ax.set_title('Weight Ratio')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_yscale('log')

    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, show)


def plot_run_comparison(run_data_list, run_labels, metric="loss", *,
                        save_dir=None, filename="run_comparison.png",
                        show=False):
    """Overlay loss curves from multiple runs for side-by-side comparison.

    Args:
        run_data_list: list of loss arrays (one per run).
        run_labels: list of str labels for the legend.
        metric: "loss" or "pde" or "bc".
    """
    apply_paper_style()
    fig = plt.figure(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_data_list)))

    for data, label, color in zip(run_data_list, run_labels, colors):
        arr = np.log(np.clip(np.array(data), 1e-30, None))
        plt.plot(arr, color=color, linewidth=1.5, label=label, alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.capitalize()} Loss (log scale)')
    plt.title(f'Run Comparison — {metric.capitalize()} Loss')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    _save_or_show(fig, save_dir, filename, show)


def plot_beam_deformation(x_coords, y_coords, u_pred, v_pred,
                          scale_factor=50,
                          ylim=(-0.03, 0.03),
                          save=False, save_dir=None,
                          filename="beam_deformation.png", show=True):
    apply_paper_style()

    x_deformed = x_coords + scale_factor * u_pred
    y_deformed = y_coords + scale_factor * v_pred

    fig = plt.figure(figsize=(6, 4))
    original_color = "#1f77b4"
    deformed_color = "#ff7f0e"

    plt.plot(x_coords, y_coords, color=original_color, linestyle='-',
             linewidth=1.2, marker='o', markersize=1.5, label='Original beam')
    plt.plot(x_deformed, y_deformed, color=deformed_color, linestyle='-',
             linewidth=1.2, marker='o', markersize=1.5,
             label=f'Deformed beam (x{scale_factor} scale)')

    if ylim is not None:
        plt.ylim(*ylim)
    plt.title('2D Beam Deformation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    _save_or_show(fig, save_dir if save else None,
                  filename if save else None, show)
