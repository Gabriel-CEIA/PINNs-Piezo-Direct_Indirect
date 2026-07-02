# AGENTS.md — PINNs-Piezo-Direct_Indirect

## Project

Physics-Informed Neural Networks (PINNs) for a 2D piezoelectric bimorph beam.
Two complementary formulations:

| Formulation | Input | Output | Key files |
|---|---|---|---|
| **Indirect** | Voltage (100 V) across electrodes | Beam deformation | `src/pinn_piezo/indirect/` |
| **Direct** | Tip traction (0.1 N) | Electric potential | `src/pinn_piezo/direct/` |

The beam is a *bimorph* — two oppositely poled PZT layers — so the
piezoelectric coefficients ``e31`` / ``e33`` flip sign across the mid-plane
(``materials.py:70-73``).

## Critical gotchas

- **dtype differs by formulation.** Indirect uses `torch.float64`, direct uses `torch.float32`. The training script sets `torch.set_default_dtype` accordingly. Using the wrong dtype silently degrades accuracy or breaks.
- **Data suffixes differ.** Indirect loads `*_m1.npy`, direct loads `*_m1_d.npy`.
- **Default data fraction differs.** Indirect uses 100% of collocation points, direct uses 75%.
- **L-BFGS is disabled for direct by default** (0 epochs). Indirect defaults to 200 L-BFGS epochs after Adam.
- **No `tests/` directory exists.** `pytest` discovers nothing. There are no CI workflows.
- **Sample configs** at `experiments/base_indirect.yaml` and `experiments/base_direct.yaml` — ready to use with `--config`.

## Architecture

The network predicts all 8 primal fields directly: `(u, v, phi, sigmax, sigmaz, tauxz, Dx, Dy)`.

**Hard boundary constraints** are embedded in the forward pass (not loss-penalized):
- Displacement clamp at x=0: `u_modified = x * u`, `v_modified = x * v`
- Electric potential (indirect): `phi_modified = y*(y-HEIGHT)*phi + V/HEIGHT*y`
- Electric potential (direct): `phi_modified = (y/HEIGHT) * phi`

This reduces PDE order from 2nd to 1st, requiring only first-order
autograd for the physics loss.

**Adaptive loss weighting** via gradient-norm balancing (moving average
alpha = 0.9) in `indirect/losses.py:150-170` and `direct/losses.py:205-228`.

## Ablation baseline

`src/pinn_piezo/indirect/standard.py` implements the conventional ("Case A")
PINN — only `(u, v, phi)` as outputs with 2nd-order derivatives. Used for
the reviewer-requested architecture comparison.

## Commands

### Setup
```bash
pip install -e .
# optional: pip install -e ".[mlflow,dev]"
```

### Training
```bash
python -m scripts.run_all                                    # full pipeline
python -m scripts.run_all --formulations indirect             # one formulation
python -m scripts.run_all --use-pretrained --skip-data        # eval only
python -m scripts.train_indirect                             # train indirect only
python -m scripts.train_direct                               # train direct only
```

Console scripts also work: `pinn-train-indirect`, `pinn-train-direct`.

### Evaluation
```bash
python -m scripts.evaluate --formulation indirect --state models/indirect/model_PINN_indirect_paper_3.pt
python -m scripts.evaluate --formulation direct --state models/direct/model_PINN_direct_paper_3.pt
```

Output goes to `outputs/runs/<run_name>/` — one self-contained directory
with figures, models, loss history, and `summary.json`.

### Config-driven usage
```bash
# Create a YAML config file, then pass it to any script:
python -m scripts.train_indirect --config experiments/base_indirect.yaml
python -m scripts.train_direct --config experiments/base_direct.yaml
python -m scripts.run_all --config-indirect experiments/base_indirect.yaml --config-direct experiments/base_direct.yaml

# Override individual params from CLI:
python -m scripts.train_indirect --config experiments/base_indirect.yaml training.lr_adam=0.005
```

Config files use nested YAML sections: `beam`, `material`, `arch`, `training`,
`geometry`, `loading`, `run`. See `src/pinn_piezo/experiment.py` for all fields.

### Environment variables (optional path overrides)
| Variable | Default |
|---|---|
| `PINN_PIEZO_DATA_DIR` | `<repo>/data` |
| `PINN_PIEZO_MODELS_DIR` | `<repo>/models` |
| `PINN_PIEZO_OUTPUTS_DIR` | `<repo>/outputs` |

All dirs are created on import and committed pre-trained weights are at
`models/indirect/model_PINN_indirect_paper_3.pt` and
`models/direct/model_PINN_direct_paper_3.pt`.

### Experiment tracking (MLflow)
```bash
pip install mlflow                # optional dependency
python -m scripts.train_indirect --mlflow    # log to MLflow
python -m scripts.train_direct --mlflow
mlflow ui                         # browse runs at localhost:5000
```
MLflow logs per-epoch loss components, adaptive weights, final L2 errors,
and model artifacts. Without `--mlflow`, everything still works (CSV/npy).

### Parameter sweeps
```bash
python -m scripts.sweep \
    --base-config experiments/base_indirect.yaml \
    --grid training.lr_adam=0.001,0.0005,0.0001 \
    --grid arch.hidden_sizes="[50,50],[100,100],[100,250]"
```
Saves per-run results under `outputs/runs/` and a sweep summary + comparison
plot under `outputs/sweeps/`.

### Code quality
```bash
ruff check .      # lint
ruff format .     # format
mypy src/         # type check
```

## Pre-installed workspace tooling

- `opencode` CLI, `get-shit-done-cc`, `uv`, `specify-cli`
- `review-code` skill (`.opencode/skills/review-code/`) — review current git diff
- `design-md` skill (`.opencode/skills/design-md/`) — scaffold DESIGN.md

## Style conventions

- Ruff for linting/formatting; configured in pyproject.toml dev deps.
- No `__init__.py` re-exports — import directly from the module.
- Prefer `from __future__ import annotations` at the top of every module.
- Matplotlib uses `Agg` backend in non-interactive scripts (`matplotlib.use("Agg")`).
