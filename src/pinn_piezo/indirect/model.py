"""Network architectures for the indirect PINN."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.init as init
from torch import nn

from .. import config
from ..config import HEIGHT, WIDTH


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def u_constraint(x, y):
    return 0


def v_constraint(x, y):
    return 0


def phi_constraint(x, y):
    return config.VOLTAGE / HEIGHT * y


class FCNUniform(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 normalize=False):
        super().__init__()
        self.normalize = normalize
        activation = nn.Tanh

        layers = [
            ('input', nn.Linear(input_size, hidden_size)),
            ('act0', activation()),
        ]
        for i in range(num_layers):
            layers.append((f'hidden_{i}', nn.Linear(hidden_size, hidden_size)))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_size, output_size)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        outputs = self.net(x)
        u, v, phi = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

        x_phys = x[:, 0:1] * (WIDTH if self.normalize else 1.0)
        y_phys = x[:, 1:2] * (HEIGHT if self.normalize else 1.0)

        u_modified = x_phys * u
        v_modified = x_phys * v
        phi_modified = (y_phys * (y_phys - HEIGHT) * phi
                        + phi_constraint(x_phys, y_phys))

        return torch.cat([u_modified, v_modified, phi_modified, outputs[:, 3:]],
                         dim=1)


class FCNPyramid(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation=nn.Tanh, normalize=False):
        super().__init__()
        self.normalize = normalize

        layers = [
            ('input', nn.Linear(input_size, hidden_sizes[0])),
            ('act0', activation()),
        ]
        for i in range(1, len(hidden_sizes)):
            layers.append((f'hidden_{i - 1}',
                           nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_sizes[-1], output_size)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        outputs = self.net(x)
        u, v, phi = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

        x_phys = x[:, 0:1] * (WIDTH if self.normalize else 1.0)
        y_phys = x[:, 1:2] * (HEIGHT if self.normalize else 1.0)

        u_modified = x_phys * u
        v_modified = x_phys * v
        phi_modified = (y_phys * (y_phys - HEIGHT) * phi
                        + phi_constraint(x_phys, y_phys))

        return torch.cat([u_modified, v_modified, phi_modified, outputs[:, 3:]],
                         dim=1)


def get_model(input_size, hidden_sizes, output_size, type='uniform',
              normalize=False):
    if type == 'uniform':
        return FCNUniform(input_size, hidden_sizes, output_size,
                          normalize=normalize)
    return FCNPyramid(input_size, hidden_sizes, output_size,
                      normalize=normalize)


def build_default_model(device=None,
                        model_type: str = 'pyramid',
                        input_size: int = 2,
                        output_size: int = 8,
                        hidden_sizes=(100, 250),
                        normalize: bool = False):
    if model_type == 'pyramid':
        model = get_model(input_size, list(hidden_sizes), output_size,
                          type=model_type, normalize=normalize)
    else:
        model = get_model((2, 300, 3, 8), type=model_type,
                          normalize=normalize)
    model.apply(init_weights)
    if device is not None:
        model.to(device)
    return model
