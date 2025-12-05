"""
Shared RL training loop utilities (extracted from notebooks).
"""

import os
import numpy as np
from utils.helpers import apply_min_max, reverse_min_max, apply_rl_scaled, generate_setpoints_training_rl


def map_to_bounds(a, low, high):
    return low + ((a + 1.0) / 2.0) * (high - low)


def run_rl_train(
    system,
    A_aug,
    B_aug,
    C_aug,
    y_sp_scenario,
    n_tests,
    set_points_len,
    steady_states,
    min_max_dict,
    agent,
    MPC_obj,
    L,
    data_min,
    data_max,
    warm_start,
    test_cycle,
    reward_fn,
    actor_freeze=0,
    alt_reward_fn=None,
    alt_log_path=None,
    disturbance_fn=None,
):
    """
    RL training loop used by OnlineTrainingNoDist* notebooks.
    Returns trajectory data and diagnostics.

    Args:
        disturbance_fn: optional callable that takes the step index (int) and
            returns a disturbance array to pass to `system.step(disturbances=...)`.
            If None, no disturbance is applied and `system.step()` is called
            without arguments.
    """
    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, WARM_START = generate_setpoints_training_rl(
        y_sp_scenario, n_tests, set_points_len, warm_start, test_cycle
    )

    n_inputs = B_aug.shape[1]
    n_outputs = C_aug.shape[0]
    n_states = A_aug.shape[0]

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = system.current_output
    u_rl = np.zeros((nFE, n_inputs))
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    rewards = np.zeros(nFE)
    avg_rewards = []
    alt_rewards = [] if alt_reward_fn is not None else None
    delta_y_storage = []

    boundary = time_in_sub_episodes / 2
    test = False

    for i in range(nFE):
        if i in test_train_dict:
            test = test_train_dict[i]

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs

        current_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i], y_sp[i, :], scaled_current_input_dev)

        if not test:
            action = agent.take_action(current_rl_state, explore=not test)
        else:
            action = agent.act_eval(current_rl_state)

        u_scaled = map_to_bounds(action, u_min, u_max)
        u_rl[i, :] = u_scaled + ss_scaled_inputs
        u_plant = reverse_min_max(u_rl[i, :], data_min[:n_inputs], data_max[:n_inputs])

        delta_u = u_rl[i, :] - scaled_current_input

        system.current_input = u_plant

        disturbance = disturbance_fn(i) if disturbance_fn is not None else None
        if disturbance is not None:
            system.step(disturbances=np.atleast_1d(disturbance))
        else:
            system.step()
        y_system[i + 1, :] = system.current_output

        y_current_scaled = apply_min_max(y_system[i + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled

        delta_y = y_current_scaled - y_sp[i, :]

        yhat[:, i] = np.dot(MPC_obj.C, xhatdhat[:, i])
        xhatdhat[:, i + 1] = (
            np.dot(MPC_obj.A, xhatdhat[:, i])
            + np.dot(MPC_obj.B, (u_rl[i, :] - ss_scaled_inputs))
            + np.dot(L, (y_prev_scaled - yhat[:, i])).T
        )

        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        try:
            reward = reward_fn(delta_y, delta_u, y_sp_phys)
        except TypeError:
            reward = reward_fn(delta_y, delta_u)

        if alt_reward_fn is not None:
            try:
                alt_r = alt_reward_fn(delta_y, delta_u, y_sp_phys)
            except TypeError:
                alt_r = alt_reward_fn(delta_y, delta_u)
            alt_rewards.append(alt_r)

        rewards[i] = reward
        delta_y_storage.append(np.abs(delta_y))

        next_u_dev = u_rl[i, :] - ss_scaled_inputs
        next_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i + 1], y_sp[i, :], next_u_dev)
        done = 0.0

        if not test:
            agent.push(
                current_rl_state,
                action.astype(np.float32),
                float(reward),
                next_rl_state,
                float(done),
            )
            if i >= WARM_START and i >= actor_freeze:
                _ = agent.train_step()

        if i in sub_episodes_changes_dict:
            avg_rewards.append(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1]))
            print("Sub_Episode:", sub_episodes_changes_dict[i], "| avg. reward:", avg_rewards[-1])
            if hasattr(agent, "_expl_sigma"):
                print("Exploration noise:", agent._expl_sigma)

    u_rl = reverse_min_max(u_rl, data_min[:n_inputs], data_max[:n_inputs])

    if alt_reward_fn is not None and alt_log_path is not None:
        os.makedirs(os.path.dirname(alt_log_path), exist_ok=True)
        np.save(alt_log_path, np.array(alt_rewards, dtype=float))

    return (
        y_system,
        u_rl,
        avg_rewards,
        rewards,
        xhatdhat,
        nFE,
        time_in_sub_episodes,
        y_sp,
        yhat,
        delta_y_storage,
        alt_rewards,
    )
