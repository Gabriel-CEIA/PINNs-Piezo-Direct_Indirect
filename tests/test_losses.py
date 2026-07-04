from __future__ import annotations

import torch

from pinn_piezo.direct.model import FCN as DirectFCN
from pinn_piezo.direct.losses import physics_loss as direct_physics
from pinn_piezo.direct.losses import loss_func as direct_loss_func
from pinn_piezo.direct.losses import (
    displacement_BC_loss,
    electric_potential_BC_loss,
    stress_BC_loss as direct_stress_BC,
    electric_BC_loss as direct_electric_BC,
)
from pinn_piezo.indirect.model import FCNPyramid
from pinn_piezo.indirect.losses import loss_func as indirect_loss_func
from pinn_piezo.indirect.losses import physics_loss as indirect_physics
from pinn_piezo.indirect.losses import stress_BC_loss as indirect_stress_BC
from pinn_piezo.indirect.losses import electric_BC_loss as indirect_electric_BC


def _coeff(batch_size):
    c = torch.randn(batch_size, 8)
    c[:, 6] = c[:, 6].abs() + 0.1  # e31
    c[:, 7] = c[:, 7].abs() + 0.1  # e33
    return c


class TestIndirectLosses:
    def test_physics_loss_runs(self, device, batch_size):
        model = FCNPyramid(2, [10, 20], 8).to(device)
        x = torch.randn(batch_size, 1, device=device, requires_grad=True)
        y = torch.randn(batch_size, 1, device=device, requires_grad=True)
        coeff = _coeff(batch_size).to(device)
        total, mech, elec, div = indirect_physics(x, y, model, coeff)
        assert total.ndim == 0
        assert total > 0
        assert mech > 0
        assert elec > 0
        assert div > 0

    def test_stress_bc_loss_runs(self, device, batch_size):
        model = FCNPyramid(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        loss = indirect_stress_BC(xy, xy, xy, xy, model)
        assert loss.ndim == 0
        assert loss > 0

    def test_electric_bc_loss_runs(self, device, batch_size):
        model = FCNPyramid(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        loss = indirect_electric_BC(xy, xy, model)
        assert loss.ndim == 0
        assert loss > 0

    def test_loss_func_runs(self, device, batch_size):
        model = FCNPyramid(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        x = torch.randn(batch_size, 1, device=device, requires_grad=True)
        y = torch.randn(batch_size, 1, device=device, requires_grad=True)
        coeff = _coeff(batch_size).to(device)
        weights = {"pde": torch.tensor(1.0), "bc": torch.tensor(1.0)}
        loss, w2, pt, bc, *_ = indirect_loss_func(xy, xy, xy, xy, x, y, model, coeff, weights, 0, 10)
        assert loss.ndim == 0
        assert loss > 0


class TestDirectLosses:
    def test_physics_loss_runs(self, device, batch_size):
        model = DirectFCN(2, [10, 20], 8).to(device)
        x = torch.randn(batch_size, 1, device=device, requires_grad=True)
        y = torch.randn(batch_size, 1, device=device, requires_grad=True)
        coeff = _coeff(batch_size).to(device)
        result = direct_physics(x, y, model, coeff)
        total = result[0]
        assert total.ndim == 0
        assert total > 0

    def test_stress_bc_loss_runs(self, device, batch_size):
        model = DirectFCN(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        loss = direct_stress_BC(xy, xy, xy, xy, model)
        assert loss.ndim == 0
        assert loss > 0

    def test_electric_bc_loss_runs(self, device, batch_size):
        model = DirectFCN(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        loss = direct_electric_BC(xy, xy, xy, model)
        assert loss.ndim == 0
        assert loss > 0

    def test_displacement_bc_loss_runs(self, device, batch_size):
        model = DirectFCN(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        loss = displacement_BC_loss(xy, model)
        assert loss.ndim == 0

    def test_potential_bc_loss_runs(self, device, batch_size):
        model = DirectFCN(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        loss = electric_potential_BC_loss(xy, model)
        assert loss.ndim == 0

    def test_loss_func_runs(self, device, batch_size):
        model = DirectFCN(2, [10, 20], 8).to(device)
        xy = torch.randn(batch_size, 2, device=device)
        x = torch.randn(batch_size, 1, device=device, requires_grad=True)
        y = torch.randn(batch_size, 1, device=device, requires_grad=True)
        coeff = _coeff(batch_size).to(device)
        weights = {"pde": torch.tensor(1.0), "bc": torch.tensor(1.0)}
        result = direct_loss_func(xy, xy, xy, xy, x, y, model, coeff, weights, 0, 10)
        loss = result[0]
        assert loss.ndim == 0
        assert loss > 0
