from typing import List
import torch.nn as nn


# ------------------ helpers ----------------------
def get_activation(name: str) -> nn.Module:
    # Lower case of the name:
    name = name.lower()

    # Different Activation Functions
    if name == "relu":  return nn.ReLU(inplace=True)
    if name == "silu":  return nn.SiLU(inplace=True)
    if name == "gelu":  return nn.GELU()
    if name == "elu":   return nn.ELU(inplace=True)
    if name == "tanh":  return nn.Tanh()
    raise ValueError(f"Unknown activation '{name}'")


def init_layer(layer: nn.Linear, non_linearity: str = "relu") -> None:
    # Initializing the layer with different strategy
    # He/Kaiming for RELU, Xavier otherwise
    if non_linearity.lower() in {"relu", "silu", "elu", "gelu"}:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0.0)


def build_network(
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: str = "relu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        prefix: str = "q"
) -> nn.Sequential:
    """
    Builds: Linear -> [Layer Norm] -> Activation -> [Dropout] ...
    """

    layers = nn.Sequential()

    prev_dim = in_dim

    for i, h in enumerate(hidden_dims):
        lin = nn.Linear(prev_dim, h)
        init_layer(lin, non_linearity=activation)

        layers.add_module(f"layer{i}", lin)

        if use_layernorm:
            layers.add_module(f"{prefix}_norm{i}", nn.LayerNorm(h))

        layers.add_module(f"activation{i}", get_activation(activation))

        if dropout > 0.0:
            layers.add_module(f"{prefix}_dropout{i}", nn.Dropout(dropout))

        prev_dim = h

    lin_out = nn.Linear(prev_dim, out_dim)
    init_layer(lin_out, non_linearity="linear")
    layers.add_module(f"output_layer", lin_out)
    return layers
