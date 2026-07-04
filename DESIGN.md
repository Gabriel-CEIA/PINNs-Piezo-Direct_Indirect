# PINNs-Piezo-Direct_Indirect — Architecture Design

## Overview

Physics-Informed Neural Networks (PINNs) for a 2D piezoelectric bimorph beam.
Two complementary formulations share the same network backbone but differ in
their boundary conditions, loss terms, and input/output semantics.

## Components

### Source tree (`src/pinn_piezo/`)

```
pinn_piezo/
├── __init__.py          # Package exports
├── config.py            # Geometry constants, filesystem paths, env overrides
├── experiment.py        # ExperimentConfig: dataclass + YAML load/save + CLI overrides
├── materials.py         # Piezoelectric coefficients (bimorph), coupled matrix assembly
├── geometry.py          # Latin Hypercube sampling, .npy persistence
├── evaluation.py        # FEM ground-truth CSV loading, relative L2 error
├── metrics.py           # RMSE, MAE, max-abs, nRMSE, field_metrics, metrics_table
├── fem.py               # scikit-fem coupled piezo FEM solver (reference)
├── plotting.py          # Matplotlib helpers: loss curves, fields, deformation, dashboard
│
├── indirect/            # Voltage-driven formulation (100 V → deformation)
│   ├── model.py         # FCNUniform, FCNPyramid, hard constraints for clamp + electrodes
│   ├── losses.py        # Physics loss (constitutive + divergence), BC losses, adaptive weights
│   ├── train.py         # load_dataset, to_device, run_adam, run_lbfgs, train
│   └── standard.py      # Ablation baseline: conventional (u,v,phi)-only, 2nd-order PDE
│
└── direct/              # Force-driven formulation (0.1 N → electric potential)
    ├── model.py         # FCN with SinActivation (SIREN init), hard constraints
    ├── losses.py        # Physics loss (column-wise Jacobian), BC losses, adaptive weights
    └── train.py         # load_dataset, to_device, run_adam, run_lbfgs, train
```

### Scripts (`scripts/`)

```
scripts/
├── generate_geometry.py   # Generate .npy datasets for both formulations
├── train_indirect.py      # Train indirect PINN (config/CLI/MLflow)
├── train_direct.py        # Train direct PINN (config/CLI/MLflow)
├── evaluate.py            # Evaluate trained model vs FEM ground truth
├── run_all.py             # End-to-end pipeline (data → train → eval, both formulations)
└── sweep.py               # Grid search over hyperparameters
```

## Data Flow

```
YAML config ──→ ExperimentConfig ──→ Script entry point
                                          │
                                      load_dataset()  ──→ .npy files
                                          │
                                      to_device()
                                          │
                                      model.forward(x, y)
                                          │
                                     ┌────┴────┐
                                     │         │
                              physics_loss   BC_losses
                              (PDE residual)  (traction, electric, displacement)
                                     │         │
                                     └────┬────┘
                                          │
                                    weighted sum
                                          │
                                    optimizer.step()
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                         model.pt              evaluation.py
                         loss.npy                 │
                         figures/            FEM comparison
                                              metrics table
```

## Formulation Comparison

| Aspect | Indirect | Direct |
|---|---|---|
| **Input** | 100 V across electrodes | 0.1 N tip traction |
| **Output** | Beam deformation (u, v, phi) | Electric potential (phi) |
| **Default dtype** | `torch.float64` | `torch.float32` |
| **Data suffix** | `_m1` | `_m1_d` |
| **Data fraction** | 100% | 75% |
| **Adam epochs** | 1000 | 3000 |
| **L-BFGS epochs** | 200 | 0 |
| **Model type** | Pyramid or Uniform | FCN (Tanh or Sin) |
| **Hard φ constraint** | `y*(y-H)*φ + V/H*y` | `(y/H) * φ` |

## Key Design Decisions

### Hard boundary constraints
Displacement clamp at x=0 and electric potential BCs are embedded in the
forward pass (not loss-penalized). This reduces PDE order from 2nd to 1st,
requiring only first-order autograd for the physics loss. The network predicts
all 8 primal fields directly: `(u, v, phi, σ_xx, σ_zz, τ_xz, D_x, D_y)`.

### Adaptive loss weighting
Gradient-norm balancing with EMA (α=0.9). The weight for each loss component
is proportional to the ratio of total gradient norm to that component's
gradient norm. Updated every `f` epochs.

### Config system
Python dataclasses (`ExperimentConfig`) with nested sub-configs (beam,
material, arch, training, geometry, loading, run). YAML serialization for
persistence and sharing. CLI dot-notation overrides for ad-hoc changes.

### Experiment tracking
MLflow (optional) logs per-epoch loss components, adaptive weights, final L2
errors, and model artifacts. CSV/npy fallback when MLflow is not installed.

### Mixed formulation
The indirect formulation (voltage-driven) and direct formulation (force-driven)
share the same 8-field network architecture but differ in boundary conditions,
loss function structure, and default hyperparameters. Each lives in its own
subpackage with isomorphic interfaces (`load_dataset`, `to_device`, `train`).

## Dependencies

- **Required**: numpy, scipy, pandas, matplotlib, scikit-learn, torch,
  torchopt, torchsummary, pyDOE, pyyaml, scikit-fem, openpyxl
- **Optional**: mlflow (experiment tracking), pytest/ruff/mypy (dev),
  pre-commit (hooks)
