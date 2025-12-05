import torch
import torch.nn as nn
from typing import List
from utils.helpers_net import build_network


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian Policy with Tanh Squash for SAC

    - Given s, output a distribution over actions:
        a ~ tanh(N(mu(s), sigma(s))) * max_action
    - Provides:
        - sample(s): action, log_prob, mean_action
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int],
            activation: str="relu",
            use_layer_norm: bool = False,
            dropout: float = 0.0,
            max_action: float = 1.0,
            log_std_min: float = -20.0,
            log_std_max: float = 2.0,
    ):
        super(GaussianActor, self).__init__()
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        last_hidden = hidden_dims[-1] if hidden_dims else state_dim

        self.backbone = build_network(
            in_dim=state_dim,
            hidden_dims=hidden_dims,
            out_dim=last_hidden,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
            prefix="pi"
        )

        # separate linear heads for mean and log_std
        self.mean_layer = nn.Linear(last_hidden, action_dim)
        self.log_std_layer = nn.Linear(last_hidden, action_dim)

        # small initialization on the heads -> this will avoid huge initial log_std/mean_layer
        nn.init.uniform_(self.mean_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mean_layer.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_layer.bias, -1e-3, 1e-3)

    def _forward_stats(self, s: torch.Tensor):
        """
        Given state s, returns (mu(s), log_std(s))
        """
        h = self.backbone(s)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, std, log_std

    def sample(self, s: torch.Tensor):
        """
        Sample a stochastic action and its log probability.

        Returns:
             action: [B, A] in [-max_action, max_action]
             log_prob: [B, 1]
             mean_action: [B, A] deterministic mean, useful for eval
        """

        mean, std, log_std = self._forward_stats(s)

        # Re parameterization: u = mu +sigma * eps
        normal = torch.distributions.Normal(mean, std)
        eps = normal.rsample() # resample -> allows gradient wrt mean/std
        pre_tanh = eps
        #squash
        y = torch.tanh(pre_tanh)
        action = y * self.max_action

        # log_prob with tanh correction
        # log pi(a|s) = log N(pre_tanh; mu, std) - log(1 - tanh(pre_tanh)^2)
        log_prob = normal.log_prob(pre_tanh)
        # sum over action dims
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # tanh jacobian correction
        log_prob -= torch.log(1.0 - y.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # deterministic action for eval (without sampling)
        mean_action = torch.tanh(mean) * self.max_action

        return action, log_prob, mean_action

    def deterministic_action(self, s: torch.Tensor) -> torch.Tensor:
        """
        Mean action for evaluation (no randomness)
        """

        mean, std, log_std = self._forward_stats(s)
        return torch.tanh(mean) * self.max_action