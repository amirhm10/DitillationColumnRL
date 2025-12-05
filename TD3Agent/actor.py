import torch
import torch.nn as nn
from typing import List
from utils.helpers_net import build_network


class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int],
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
            max_action: float = 1.0,
            squash: str = "tanh",
    ):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.squash = squash.lower()

        self.model = build_network(
            in_dim=state_dim,
            hidden_dims=hidden_dims,
            out_dim=action_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="pi"
        )

        # The following for avoiding early saturation in the output layer
        out = getattr(self.model, "output_layer")
        nn.init.uniform_(out.weight, -1e-3, 1e-3)
        nn.init.uniform_(out.bias, -1e-3, 1e-3)

        self.model.add_module(squash, nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = self.model(state)
        # scale to max action
        return a * self.max_action
