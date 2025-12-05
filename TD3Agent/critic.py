import torch
import torch.nn as nn
from typing import List
from utils.helpers_net import build_network


class Critic(nn.Module):
    """
    Double Q Network for DQN
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int],
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
    ):
        super(Critic, self).__init__()
        in_dim = state_dim + action_dim
        out_dim = 1

        self.q1_network = build_network(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="q1"
        )

        self.q2_network = build_network(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="q2"
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=1)
        q1_out = self.q1_network(x)
        q2_out = self.q2_network(x)
        return q1_out, q2_out

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=1)
        q1_out = self.q1_network(x)
        return q1_out

    def combined_forward(self, state: torch.Tensor, action: torch.Tensor, mode: str = "min"):
        q1, q2 = self.forward(state, action)
        if mode == "q1": return q1
        if mode == "min": return torch.min(q1, q2)
        if mode == "max": return torch.max(q1, q2)
        if mode == "mean": return 0.5 * (q1 + q2)
        raise ValueError("mode must be min/max/mean/q1")
