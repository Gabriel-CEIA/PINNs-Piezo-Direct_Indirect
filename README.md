# PINNs-Piezo-Direct_Indirect

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-F77F00?style=flat-square&logo=mozilla)](https://opensource.org/licenses/MPL-2.0)
[![OpenCode](https://img.shields.io/badge/OpenCode-CLI-8B5CF6?style=flat-square&logo=openai&logoColor=white)](https://opencode.ai/)

---

Physics-Informed Neural Networks (PINNs) applied to a 2D piezoelectric beam problem. Two complementary formulations are implemented: an **indirect (voltage-driven)** formulation where a potential difference produces beam deformation, and a **direct (force-driven)** formulation where a mechanical load generates an electric potential.

---

## Why

### Problem

Simulating piezoelectric beams requires solving coupled mechanical and electrostatic PDEs. Traditional FEM methods are accurate but computationally expensive for design-space exploration and inverse problems.

### Motivation

PINNs embed the governing physics directly into the neural network loss function, eliminating mesh generation and providing a differentiable surrogate that can be evaluated anywhere in the domain at negligible cost.

### Intended Impact

Provide a reproducible, mesh-free framework for piezoelectric beam simulation that achieves competitive accuracy with FEM while enabling rapid prototyping and gradient-based optimization.

---

## How

### Architecture

The network predicts all primal fields — displacement *(u, v)*, electric potential *(phi)*, stress *(sigmax, sigmaz, tauxz)*, and electric displacement *(Dx, Dy)* — as direct outputs. Hard boundary constraints are enforced via **distance functions** in the network's forward pass, reducing the PDE order from 2nd to 1st.

### Two Formulations

| Formulation | Input | Output | Key File |
|---|---|---|---|
| Indirect | Voltage across electrodes | Beam deformation | `src/pinn_piezo/indirect/` |
| Direct | Traction at right end | Electric potential | `src/pinn_piezo/direct/` |

### Training

A two-stage optimizer pipeline:
1. **Adam** — adaptive momentum for stable initial convergence
2. **L-BFGS** — quasi-Newton refinement to polish the solution

Adaptive loss weighting via gradient norm balancing (moving average alpha = 0.9).

---

## What

### Repository Layout

```
src/pinn_piezo/
    config.py            # geometric constants and configurable paths
    materials.py         # piezoelectric material coefficients
    geometry.py          # boundary / collocation point sampling
    plotting.py          # shared matplotlib helpers
    evaluation.py        # FEM ground-truth comparison
    indirect/
        model.py         # FCNPyramid / FCNUniform with hard constraints
        losses.py        # physics + boundary losses (voltage-driven)
        train.py         # Adam + L-BFGS training driver
    direct/
        model.py         # FCN with hard constraints
        losses.py        # physics + boundary losses (force-driven)
        train.py         # Adam (+ optional L-BFGS) training driver

scripts/
    generate_geometry.py # create .npy data files
    train_indirect.py    # train the indirect formulation
    train_direct.py      # train the direct formulation
    evaluate.py          # field plots and FEM comparison
    run_all.py           # full pipeline: data -> train -> evaluate

notebooks/               # original Colab notebooks (reference only)
models/                  # paper-quality trained weights (committed)
data/                    # .npy / FEM.csv files (generated)
outputs/
    runs/<run_name>/     # one self-contained directory per invocation
```

All scripts write artefacts into `outputs/runs/<run_name>/`.

### Pre-trained Models

Two checkpoints are committed and correspond to the paper results:
- `models/indirect/model_PINN_indirect_paper_3.pt`
- `models/direct/model_PINN_direct_paper_3.pt`

---

## Installation

### With `uv` (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### With `pip` and `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Full Pipeline

```bash
python -m scripts.run_all
```

Train specific formulations:

```bash
python -m scripts.run_all --formulations indirect
python -m scripts.run_all --formulations direct
```

Use pre-trained weights (skip training, regenerate figures):

```bash
python -m scripts.run_all --use-pretrained --skip-data
```

### Individual Steps

```bash
# 1. Generate geometry datasets
python -m scripts.generate_geometry

# 2. Train
python -m scripts.train_indirect
python -m scripts.train_direct

# 3. Evaluate
python -m scripts.evaluate \
    --formulation indirect \
    --state models/indirect/model_PINN_indirect_paper_3.pt
```

### Configurable Paths

Set these environment variables to override default paths:

| Variable | Default |
|---|---|
| `PINN_PIEZO_DATA_DIR` | `<repo>/data` |
| `PINN_PIEZO_MODELS_DIR` | `<repo>/models` |
| `PINN_PIEZO_OUTPUTS_DIR` | `<repo>/outputs` |

---

## License

This project is licensed under the Mozilla Public License 2.0.  
SPDX-License-Identifier: MPL-2.0

See the [LICENSE](LICENSE) file for full details.

---

## Acknowledgments

This repository builds on the original work by Daniel Gonzalez ([PINNs_piezoelectricity](https://github.com/Daniel14gonc/PINNs_piezoelectricity)) and the `py-opencode-scaffold` development environment template.
