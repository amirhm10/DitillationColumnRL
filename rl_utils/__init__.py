"""
Shared RL utilities extracted from the training notebooks.

Modules:
    rewards: reward function builders.
    disturbances: disturbance generation helpers.
    training: RL training loop utilities.
    plotting: plotting/compare helpers for RL vs MPC.
"""

from rl_utils.rewards import make_reward_fn_relative_QR, make_reward_fn_quadratic
from rl_utils.disturbances import generate_disturbance_sequence, preprocess_inputs
from rl_utils.training import run_rl_train, map_to_bounds
from rl_utils.plotting import (
    plot_rl_results_disturbance,
    compare_mpc_rl,
    compare_mpc_rl_disturbance,
    compare_mpc_rl_rl1,
)

__all__ = [
    "make_reward_fn_relative_QR",
    "make_reward_fn_quadratic",
    "generate_disturbance_sequence",
    "preprocess_inputs",
    "run_rl_train",
    "map_to_bounds",
    "plot_rl_results_disturbance",
    "compare_mpc_rl",
    "compare_mpc_rl_disturbance",
    "compare_mpc_rl_rl1",
]
