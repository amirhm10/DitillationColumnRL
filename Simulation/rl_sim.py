import numpy as np
from BasicFunctions.bs_fns import reverse_min_max, apply_min_max
import matplotlib.pyplot as plt


# Defining Test and Training scenario
def generate_setpoints_training_rl(y_sp_scenario, n_tests, set_points_len, warm_start, test_cycle):
    # For each scenario, create a block of size (set_points_len, n_outputs)
    blocks = [np.full((set_points_len, y_sp_scenario.shape[1]), scenario)
              for scenario in y_sp_scenario]

    # Concatenate the blocks to form one cycle
    cycle = np.concatenate(blocks, axis=0)
    # Repeat the cycle 'repetitions' times
    y_sp = np.concatenate([cycle] * n_tests, axis=0)

    # Test train scenario
    test_cycle = test_cycle * int(n_tests / len(test_cycle))
    # Try making everything trainable but te end cycle should be only for testing
    test_cycle[-1] = True

    time_in_sub_episodes = set_points_len * len(y_sp_scenario)

    nFE = int(y_sp.shape[0])
    idxs_setpoints = np.arange(time_in_sub_episodes - 1, nFE, time_in_sub_episodes)
    idxs_tests = np.arange(0, nFE, time_in_sub_episodes)
    sub_episodes_changes = np.arange(1, len(idxs_setpoints) + 1)
    sub_episodes_changes_dict = {}
    test_train_dict = {}
    for i in range(len(idxs_setpoints)):
        sub_episodes_changes_dict[idxs_setpoints[i]] = sub_episodes_changes[i]
    for i in range(len(idxs_tests)):
        test_train_dict[idxs_tests[i]] = test_cycle[i]
    warm_start = list(test_train_dict.keys())[warm_start]

    return y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, warm_start


def apply_rl_scaled(min_max_dict, x_d_states, y_sp, u):
    """
    This function will apply RL scaling for the neural networks
    :param min_max_dict:
    :param state:
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


def run_rl_train(system, y_sp_scenario, n_tests, set_points_len,
                 steady_states, min_max_dict, agent, MPC_obj, batch_size,
                 Q1_penalty, Q2_penalty, R1_penalty, R2_penalty, L, data_min, data_max, n_inputs, warm_start,
                 test_cycle):
    # First state of the system is always false in testing
    test = False
    explore = True

    # defining setpoints
    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, WARM_START = generate_setpoints_training_rl(
        y_sp_scenario, n_tests, set_points_len, warm_start, test_cycle)

    # Output of the system
    y_system = np.zeros((nFE + 1, MPC_obj.C.shape[0]))
    y_system[0, :] = system.current_output

    # RL inputs
    u_rl = np.zeros((nFE, MPC_obj.B.shape[1]))

    # Record states of the state space model
    xhatdhat = np.zeros((MPC_obj.A.shape[0], nFE + 1))
    xhatdhat[:, 0] = np.random.uniform(low=min_max_dict["x_min"], high=min_max_dict["x_max"])
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
        # Set the condition of the training if it is in test or training
        if i in test_train_dict.keys():
            test = test_train_dict[i]

        # So we need to apply scaling for rl because the formulation of the MPC was in scaled deviation
        # current input needs to be scaled and then deviation form
        # y_sp is already in scaled and deviation form
        # States from state space model is scaled deviation from as well
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs

        # Set the current state
        current_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i], y_sp[i, :], scaled_current_input_dev)

        # Taking the action of the TD3 Agent, and also will check if it needs to explore or not
        if test:
            action = agent.take_action(current_rl_state)
        else:
            action = agent.take_action(current_rl_state, explore)

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

        # Next input in deviation form
        next_u_dev = u_rl[i, :] - ss_scaled_inputs

        # Calculate the next state in deviation form
        yhat[:, i] = np.dot(MPC_obj.C, xhatdhat[:, i])
        xhatdhat[:, i + 1] = np.dot(MPC_obj.A, xhatdhat[:, i]) + np.dot(MPC_obj.B, next_u_dev) + \
                             np.dot(L, (y_current_scaled - yhat[:, i])).T

        # Reward Calculation
        reward = - (Q1_penalty * delta_y[0] ** 2 + Q2_penalty * delta_y[1] ** 2 +
                    R1_penalty * delta_u[0] ** 2 + R2_penalty * delta_u[1] ** 2)

        # Record rewards
        rewards[i] = reward

        # Set the current state
        next_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i + 1], y_sp[i, :], next_u_dev)

        if not test:
            agent.replay_buffer.add(current_rl_state, action, reward, next_rl_state)

        if i >= WARM_START and not test:
            agent.train(batch_size)

        # Calculate average reward and printing
        if i in sub_episodes_changes_dict.keys():
            # Averaging the rewards from the last setpoint change till current
            avg_rewards.append(np.mean(rewards[i - time_in_sub_episodes + 1: i]))

            # printing
            idx_test = 0.0 if i - time_in_sub_episodes + 1 < 0 else i - time_in_sub_episodes + 1
            print(test_train_dict[idx_test])
            print('Sub_Episode : ', sub_episodes_changes_dict[i], ' | avg. reward :', avg_rewards[-1])
            print('Exploration: ', agent.exploration_noise_std)

    u_rl = reverse_min_max(u_rl, data_min[:n_inputs], data_max[:n_inputs])

    return y_system, u_rl, avg_rewards, rewards, xhatdhat, nFE, time_in_sub_episodes, y_sp, yhat

def plot_rl_results(y_sp, steady_states, nFE, delta_t, time_in_sub_episodes, y_mpc, u_mpc, avg_rewards, data_min,
                     data_max, yhat=None):
    # Canceling the deviation form
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = (reverse_min_max(y_sp, data_min[2:], data_max[2:])).T

    ####### Plot 1  ###############
    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)

    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, y_mpc[:, 0], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot[:-1], y_sp[0, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{\eta}$ (L/g)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_mpc[:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot[:-1], y_sp[1, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{T}$ (K)', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()

    ########### last 400 ##########
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 0], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot_hour[:-1], y_sp[0, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{\eta}$ (L/g)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{T}$ (K)', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hr)', fontsize=18)
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
    plt.step(time_plot[:-1], u_mpc[:, 0], 'k-', lw=2, label=r'$\mathbf{Q}_c$')
    plt.ylabel(r'$\mathbf{Q}_c$ (L/h)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.step(time_plot[:-1], u_mpc[:, 1], 'k-', lw=2, label=r'$\mathbf{Q}_m$')
    plt.ylabel(r'$\mathbf{Q}_m$ (L/h)', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()

    ############# Plot 3 (Reward) #######################

    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(1, len(avg_rewards) + 1), avg_rewards, 'ko-', lw=2, label='Reward per Episode')
    plt.ylabel(r'Avg. Reward', fontsize=16, fontweight='bold')
    plt.xlabel(r'Episode #', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    # plt.hlines(-3.0233827500429884)
    # plt.xticks(np.arange(1, len(avg_rewards) + 1), fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=16)

    plt.show()

    if yhat is not None:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

        # For Output 1:
        # Convert real y_mpc[:,0] into scaled deviation: (y - min)/(max-min) - (y_ss in scaled)
        # data_min[2] and data_max[2] assume your first output uses that index
        y_mpc_scaled_1 = ((y_mpc[:, 0] - steady_states["y_ss"][0]) -
                          0.0)  # real domain deviation
        y_mpc_scaled_1 = (y_mpc_scaled_1 - (data_min[2] - data_min[2])) / (data_max[2] - data_min[2])

        axs[0].plot(yhat[0, :], 'b-', linewidth=2, label=r'$\mathbf{T}$ (Observer)')
        axs[0].plot(y_mpc_scaled_1, 'r--', linewidth=2, label=r'$\mathbf{T}$ (Measurement)')
        axs[0].set_ylabel('Scaled Deviation')
        axs[0].set_title('Observer vs. Real (Output 1)')
        axs[0].legend()
        axs[0].grid(True)

        # For Output 2:
        y_mpc_scaled_2 = ((y_mpc[:, 1] - steady_states["y_ss"][1]) -
                          0.0)  # real domain deviation
        y_mpc_scaled_2 = (y_mpc_scaled_2 - (data_min[3] - data_min[3])) / (data_max[3] - data_min[3])

        axs[1].plot(yhat[1, :], 'b-', linewidth=2, label=r'$\mathbf{\eta}$ (Observer)')
        axs[1].plot(y_mpc_scaled_2, 'r--', linewidth=2, label=r'$\mathbf{\eta}$ (Measurement)')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Scaled Deviation')
        axs[1].set_title('Observer vs. Real (Output 2)')
        axs[1].legend()
        axs[1].grid(True)

        fig.tight_layout()
        plt.show()