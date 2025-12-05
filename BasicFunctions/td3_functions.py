import scipy.optimize as spo
import torch
import numpy as np
import pickle
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import control as ct

import os
import pickle
from Simulation.mpc import augment_state_space, apply_min_max

import matplotlib.pyplot as plt
# from BasicFunctions.bs_fns import reverse_min_max
from Simulation.mpc import generate_setpoints


def load_and_prepare_system_data(steady_states, setpoint_y, u_min, u_max, data_dir='Data', n_inputs=2):
    """
    Loads system matrices, scaling factors, and min-max state info from files,
    augments the state space, and applies min-max scaling to the steady states
    and setpoint. Returns a dictionary with the processed data.

    Parameters:
        data_dir (str): Directory where the data files are stored. Defaults to 'Data'.
        steady_states (dict): Dictionary containing:
            - 'y_ss': steady-state outputs.
            - 'ss_inputs': steady-state inputs.
            This is required.
        u_min, u_max (float): Min-max inputs respectively.
        setpoint_y (np.ndarray): 2D array for the output setpoint.
        n_inputs (int): Number of input channels. Defaults to 2.

    Returns:
        dict: A dictionary containing:
            - 'A', 'B', 'C': original system matrices.
            - 'A_aug', 'B_aug', 'C_aug': augmented system matrices.
            - 'data_min', 'data_max': scaling factor arrays.
            - 'min_max_states': dictionary loaded from the min-max states file.
            - 'y_ss_scaled': scaled steady-state outputs.
            - 'y_sp_scaled': scaled setpoint outputs.
            - 'y_sp_scaled_deviation': deviation of setpoint from steady-state.
            - 'u_ss_scaled': scaled steady-state inputs.
            - 'b_min', 'b_max': scaled control bounds (in deviation form).
            - 'min_max_dict': a dictionary combining state bounds and setpoint/input bounds.
    """

    # Ensure the full data directory exists
    full_data_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.exists(full_data_dir):
        os.makedirs(full_data_dir)

    # Load the system matrices dictionary (A, B, C)
    system_dict_path = os.path.join(full_data_dir, "system_dict.pickle")
    with open(system_dict_path, 'rb') as file:
        system_dict = pickle.load(file)

    A = system_dict['A']
    B = system_dict['B']
    C = system_dict['C']

    # Augment the state space
    A_aug, B_aug, C_aug = augment_state_space(A, B, C)

    # Load scaling factors (min and max)
    scaling_factor_path = os.path.join(full_data_dir, "scaling_factor.pickle")
    with open(scaling_factor_path, 'rb') as file:
        scaling_factor = pickle.load(file)
    data_min = scaling_factor["min"]
    data_max = scaling_factor["max"]

    # Load the min-max states dictionary
    min_max_states_path = os.path.join(full_data_dir, "min_max_states.pickle")
    with open(min_max_states_path, 'rb') as file:
        min_max_states = pickle.load(file)

    # Scale the steady-state outputs and setpoint outputs (using apply_min_max)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled = apply_min_max(setpoint_y, data_min[n_inputs:], data_max[n_inputs:])

    # Compute the deviation (setpoint - steady-state)
    y_sp_scaled_deviation = y_sp_scaled - y_ss_scaled

    # Scale the steady-state inputs
    u_ss_scaled = apply_min_max(steady_states['ss_inputs'], data_min[:n_inputs], data_max[:n_inputs])

    # Apply scaling to the bounds and subtract the steady-state inputs to get deviations
    b_min = apply_min_max(u_min, data_min[:n_inputs], data_max[:n_inputs]) - u_ss_scaled
    b_max = apply_min_max(u_max, data_min[:n_inputs], data_max[:n_inputs]) - u_ss_scaled

    # Create a dictionary combining the scaled state and control bounds
    min_max_dict = {
        "x_max": min_max_states["max_s"],
        "x_min": min_max_states["min_s"],
        "y_sp_min": y_sp_scaled_deviation[0],
        "y_sp_max": y_sp_scaled_deviation[1],
        "u_max": b_max,
        "u_min": b_min
    }

    return {
        "A": A,
        "B": B,
        "C": C,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "data_min": data_min,
        "data_max": data_max,
        "min_max_states": min_max_states,
        "y_ss_scaled": y_ss_scaled,
        "y_sp_scaled": y_sp_scaled,
        "y_sp_scaled_deviation": y_sp_scaled_deviation,
        "u_ss_scaled": u_ss_scaled,
        "b_min": b_min,
        "b_max": b_max,
        "min_max_dict": min_max_dict
    }


def print_accuracy(replay_buffer, agent, n_samples=1000, device="cpu"):
    entire_states, entire_actions = replay_buffer.complete_states_and_actions(n_samples, device)
    entire_accuracy = r2_score(entire_actions.detach().cpu().numpy(),
                               agent.actor(entire_states).detach().cpu().numpy())
    accuracy_input1 = r2_score(entire_actions.detach().cpu().numpy()[:, 0],
                               agent.actor(entire_states).detach().cpu().numpy()[:, 0])
    accuracy_input2 = r2_score(entire_actions.detach().cpu().numpy()[:, 1],
                               agent.actor(entire_states).detach().cpu().numpy()[:, 1])
    print(f"Agent r2 score for the predicted inputs compare to MPC inputs: {entire_accuracy:6f}")
    print(f"Agent r2 score for the predicted input 1 compare to MPC input 1: {accuracy_input1:6f}")
    print(f"Agent r2 score for the predicted input 1 compare to MPC input 2: {accuracy_input2:6f}")


def optimize_sample(i, MPC_obj, y_sp, u, x0_model, IC_opt, bnds, cons):
    sol = spo.minimize(
        lambda x: MPC_obj.mpc_opt_fun(x, y_sp, u, x0_model),
        IC_opt,
        bounds=bnds, constraints=cons
    )

    return sol.x[:MPC_obj.B.shape[1]]


def filling_the_buffer(
        min_max_dict,
        A, B, C,
        MPC_obj,
        mpc_pretrain_samples_numbers,
        Q_penalty, R_penalty,
        agent,
        IC_opt, bnds, cons,
        chunk_size=10000):
    """
    Fill the replay buffer in batches to optimize the performance and manage memory

    Parameters:
        - min_max_dict: Dictionary containing min and max values for states, actions, and setpoints.
        - A, B, C: System matrices.
        - MPC_obj: Instance of MpcSolver.
        - mpc_pretrain_samples_numbers: Total number of pretraining samples to generate.
        - Q_penalty, R_penalty: Penalty matrices for the objective function.
        - agent: Agent instance containing the replay buffer.
        - u_ss: Steady-state input.
        - IC_opt: Initial guess for the optimizer.
        - bnds: Bounds for the optimizer.
        - cons: Constraints for the optimizer.
        - chunk_size: Number of samples to process in each chunk.
    """
    num_full_chunks = mpc_pretrain_samples_numbers // chunk_size
    remaining_samples = mpc_pretrain_samples_numbers % chunk_size

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    # mu = 0
    # sigma = (x_max - x_min)/10.0

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    for chunk in range(num_full_chunks):
        print(f"Processing chunk {chunk + 1}/{num_full_chunks}")

        # x_d_states = np.random.normal(
        #     mu, sigma, size=(chunk_size, A.shape[0])
        # )

        x_d_states = np.random.uniform(
            low=x_min,
            high=x_max,
            size=(chunk_size, A.shape[0])
        )

        x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

        y_sp = np.random.uniform(
            low=y_sp_min,
            high=y_sp_max,
            size=(chunk_size, C.shape[0])
        )

        y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

        # u is in deviation form because u_min and u_max is in deviation form
        u = np.random.uniform(
            low=u_min,
            high=u_max,
            size=(chunk_size, B.shape[1])
        )

        u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

        # Perform parallel optimization for the current chunk
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(optimize_sample)(
                i + chunk * chunk_size,
                MPC_obj,
                y_sp[i, :],
                u[i, :],
                x_d_states[i, :],
                IC_opt,
                bnds,
                cons
            )
            for i in range(chunk_size)
        )

        u_mpc = np.array(results)

        next_x_d_states = np.dot(A, x_d_states.T) + np.dot(B, u_mpc.T)
        y_pred = np.dot(C, next_x_d_states)

        next_x_d_states_scaled = 2 * ((next_x_d_states.T - x_min) / (x_max - x_min)) - 1
        u_mpc_scaled = 2 * ((u_mpc - u_min) / (u_max - u_min)) - 1

        rewards = np.zeros(chunk_size)
        for k in range(chunk_size):
            rewards[k] = (-1.0 * (
                    (y_pred[:, k] - y_sp[k, :]).T @ Q_penalty @ (y_pred[:, k] - y_sp[k, :]) +
                    (u[k, :] - u_mpc[k, :]).T @ R_penalty @ (u[k, :] - u_mpc[k, :])
            ))

        actions = u_mpc_scaled.copy()

        states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
        next_states = np.hstack((next_x_d_states_scaled, y_sp_scaled, u_mpc_scaled))

        agent.pretrain_push(states, actions, rewards, next_states)

    print("Replay buffer has been filled with generated samples.")


def add_steady_state_samples(
        min_max_dict,
        A, B, C,
        MPC_obj,
        steady_state_samples_numbers,
        Q_penalty, R_penalty,
        agent,
        IC_opt, bnds, cons,
        chunk_size=10000):
    """
    Fill the replay buffer in batches to optimize the performance and manage memory

    Parameters:
        - min_max_dict: Dictionary containing min and max values for states, actions, and setpoints.
        - A, B, C: System matrices.
        - MPC_obj: Instance of MpcSolver.
        - mpc_pretrain_samples_numbers: Total number of pretraining samples to generate.
        - Q_penalty, R_penalty: Penalty matrices for the objective function.
        - agent: Agent instance containing the replay buffer.
        - u_ss: Steady-state input.
        - IC_opt: Initial guess for the optimizer.
        - bnds: Bounds for the optimizer.
        - cons: Constraints for the optimizer.
        - chunk_size: Number of samples to process in each chunk.
    """
    num_full_chunks = steady_state_samples_numbers // chunk_size

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    mu = 0
    sigma = (x_max - x_min) / 10.0e12

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    for chunk in range(num_full_chunks):
        print(f"Processing chunk {chunk + 1}/{num_full_chunks}")

        x_d_states = np.random.normal(
            mu, sigma, size=(chunk_size, A.shape[0])
        )

        x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

        y_sp = np.random.uniform(
            low=0,
            high=0,
            size=(chunk_size, C.shape[0])
        )

        y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

        # u is in deviation form because u_min and u_max is in deviation form
        u = np.random.uniform(
            low=0,
            high=1e-08,
            size=(chunk_size, B.shape[1])
        )

        u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

        # Perform parallel optimization for the current chunk
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(optimize_sample)(
                i + chunk * chunk_size,
                MPC_obj,
                y_sp[i, :],
                u[i, :],
                x_d_states[i, :],
                IC_opt,
                bnds,
                cons
            )
            for i in range(chunk_size)
        )

        u_mpc = np.array(results)

        next_x_d_states = np.dot(A, x_d_states.T) + np.dot(B, u_mpc.T)
        y_pred = np.dot(C, next_x_d_states)

        next_x_d_states_scaled = 2 * ((next_x_d_states.T - x_min) / (x_max - x_min)) - 1
        u_mpc_scaled = 2 * ((u_mpc - u_min) / (u_max - u_min)) - 1

        rewards = np.zeros(chunk_size)
        for k in range(chunk_size):
            rewards[k] = (-1.0 * (
                    (y_pred[:, k] - y_sp[k, :]).T @ Q_penalty @ (y_pred[:, k] - y_sp[k, :]) +
                    (u[k, :] - u_mpc[k, :]).T @ R_penalty @ (u[k, :] - u_mpc[k, :])
            ))

        actions = u_mpc_scaled.copy()

        states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
        next_states = np.hstack((next_x_d_states_scaled, y_sp_scaled, u_mpc_scaled))

        agent.pretrain_push(states, actions, rewards, next_states)

    print("Replay buffer has been filled up with the steady_state values.")


def apply_rl_scaled(min_max_dict, x_d_states, y_sp, u):
    """
    This function will apply RL scaling for the neural networks
    :return: rl scaled of the state
    """

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

    y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

    u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

    states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))

    return states


def run_rl(system, y_sp_scenario, n_tests, set_points_len,
           steady_states, min_max_dict, agent, MPC_obj,
           Q1_penalty, Q2_penalty, R1_penalty, R2_penalty, L, data_min, data_max, n_inputs):
    # defining setpoints
    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes = generate_setpoints(y_sp_scenario, n_tests,
                                                                                    set_points_len)

    # Set exploration to False
    agent.warm_start = False

    # Output of the system
    y_system = np.zeros((nFE + 1, MPC_obj.C.shape[0]))
    y_system[0, :] = system.current_output

    # RL inputs
    u_rl = np.zeros((nFE, MPC_obj.B.shape[1]))

    # Record states of the state space model
    xhatdhat = np.zeros((MPC_obj.A.shape[0], nFE + 1))
    yhat = np.zeros((MPC_obj.C.shape[0], nFE))

    # Reward recording
    rewards = np.zeros(nFE)
    avg_rewards = []

    # Scaled steady states inputs and outputs
    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    # Minimum and Maximum of the rl action
    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    for i in range(nFE):
        # So we need to apply scaling for rl because the formulation of the MPC was in scaled deviation
        # current input needs to be scaled and then deviation form
        # y_sp is already in scaled and deviation form
        # States from state space model is scaled deviation from as well
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs

        # Set the current state
        current_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i], y_sp[i, :], scaled_current_input_dev)

        # Taking the action of the TD3 Agent
        action = agent.take_action(current_rl_state)

        # First converting the action into the scaled mpc from rl scaled
        u = ((action + 1.0) / 2.0) * (u_max - u_min) + u_min

        # take the control action (this is in scaled deviation form)
        u_rl[i, :] = u + ss_scaled_inputs

        # u (reverse scaling of the mpc)
        u_plant = reverse_min_max(u_rl[i, :], data_min[:n_inputs], data_max[:n_inputs])

        # Calculate Delta U in scaled deviation form
        delta_u = (u_rl[i, :] - ss_scaled_inputs) - (scaled_current_input - ss_scaled_inputs)

        # Change the current input
        system.current_input = u_plant

        # Apply the action on the system
        system.step()

        # Record the system output
        y_system[i + 1, :] = system.current_output

        # Since the state space calculation is in scaled will transform it
        y_current_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled

        # Calculate Delta y in deviation form
        delta_y = y_current_scaled - y_sp[i, :]

        # Calculate the next state in deviation form
        yhat[:, i] = np.dot(MPC_obj.C, xhatdhat[:, i])
        xhatdhat[:, i + 1] = np.dot(MPC_obj.A, xhatdhat[:, i]) + np.dot(MPC_obj.B,
                                                                        (u_rl[i, :] - ss_scaled_inputs)) + \
                             np.dot(L, (y_current_scaled - yhat[:, i])).T

        # Reward Calculation
        reward = - (Q1_penalty * delta_y[0] ** 2 + Q2_penalty * delta_y[1] ** 2 +
                    R1_penalty * delta_u[0] ** 2 + R2_penalty * delta_u[1] ** 2)

        # Record rewards
        rewards[i] = reward

        # Calculate average reward and printing
        if i in sub_episodes_changes_dict.keys():
            # Averaging the rewards from the last setpoint change till current
            avg_rewards.append(np.mean(rewards[i - time_in_sub_episodes + 1: i]))

            # printing
            print('Sub_Episode : ', sub_episodes_changes_dict[i], ' | avg. reward :', avg_rewards[-1])

    u_rl = reverse_min_max(u_rl, data_min[:n_inputs], data_max[:n_inputs])

    return y_system, u_rl, avg_rewards, rewards, xhatdhat, nFE, time_in_sub_episodes, y_sp, yhat


def plot_rl_mpc(y_sp, steady_states, nFE, delta_t, y_rl, u_rl, mpc_results, data_min,
                data_max, xhatdhat):
    # Canceling the deviation form
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = (reverse_min_max(y_sp, data_min[2:], data_max[2:])).T

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)

    y_mpc = mpc_results["y_mpc"]
    u_mpc = mpc_results["u_mpc"]
    xhatdhat_mpc = mpc_results["xhatdhat"]

    ####### Plot 1  ###############

    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, y_rl[:, 0], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.plot(time_plot, y_mpc[:, 0], 'y--', lw=2, label=r'$\mathbf{MPC}$')
    plt.step(time_plot[:-1], y_sp[0, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{Tray 24 Ethane Comp.}$', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_rl[:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.plot(time_plot, y_mpc[:, 1], 'y--', lw=2, label=r'$\mathbf{MPC}$')
    plt.step(time_plot[:-1], y_sp[1, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{Tray 85 Temp.}$ (C)', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()

    ####### Plot 2  ###############
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot[:-1], u_rl[:, 0], 'r-', lw=2, label=r'$\mathbf{RL}$')
    plt.plot(time_plot[:-1], u_mpc[:, 0], 'b--', lw=2, label=r'$\mathbf{MPC}$')
    plt.ylabel(r'$\mathbf{R}$ (kg/h)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot[:-1], u_rl[:, 1], 'r-', lw=2, label=r'$\mathbf{RL}$')
    plt.plot(time_plot[:-1], u_mpc[:, 1], 'b--', lw=2, label=r'$\mathbf{MPC}$')
    plt.ylabel(r'$\mathbf{Q}_R$ (Gj/h)', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()

    ###### Plot 3 ########
    fig, axes = plt.subplots(nrows=xhatdhat.shape[0], ncols=1,
                             figsize=(10, 3 * xhatdhat.shape[0]),
                             sharex=True)

    for i in range(xhatdhat.shape[0]):
        # Plot RL (xhatdhat)
        axes[i].plot(time_plot, xhatdhat[i, :], 'r-', lw=2, label='RL')
        # Plot MPC (xhatdhat_mpc)
        axes[i].plot(time_plot, xhatdhat_mpc[i, :], 'y--', lw=2, label='MPC', alpha=0.6)

        # Labeling, grids, etc.
        axes[i].grid(True)
        axes[i].set_ylabel(f'State {i}', fontsize=14)
        axes[i].legend(loc='best', fontsize=12)

    # Label the bottom (shared) X-axis:
    axes[-1].set_xlabel('Time (h)', fontsize=14)

    fig.suptitle('Comparison of RL vs. MPC States', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()
