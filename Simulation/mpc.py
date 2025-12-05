import numpy as np
import control
from scipy import signal
import scipy.optimize as spo
import matplotlib.pyplot as plt
from Simulation.sys_ids import apply_min_max, reverse_min_max

plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})


class MpcSolver(object):
    def __init__(self, A: np.array, B: np.array, C: np.array,
                 Q1_penalty, Q2_penalty, R1_penalty, R2_penalty,
                 NP, NC,
                 D=None):
        """
        Note that since this is an offset free MPC, the system matrices should be Augmented
        Also the system matrices are in numpy array format
        """

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.NP = NP
        self.NC = NC

        self.Q1_penalty = Q1_penalty * np.eye(NP)
        self.Q2_penalty = Q2_penalty * np.eye(NP)
        self.R1_penalty = R1_penalty * np.eye(NC)
        self.R2_penalty = R2_penalty * np.eye(NC)

    def mpc_opt_fun(self, x, y_sp, u_pts, x0_model: np.array):
        n_input = self.B.shape[1]

        U_Matrix = x[:n_input * self.NC].copy().reshape(self.NC, n_input)

        x_init = np.zeros((self.A.shape[0], self.NP + 1))
        x_init[:, 0] = x0_model

        for j in range(self.NP):
            control_index = j if j < self.NC else self.NC - 1
            x_init[:, j + 1] = self.A @ x_init[:, j] + self.B @ U_Matrix[control_index]

        yy = self.C @ x_init

        y_dev = yy[:, 1:] - y_sp[:, np.newaxis]
        u_dev = U_Matrix - np.vstack((u_pts.reshape(1, -1), U_Matrix[:-1]))

        obj_val = (np.sum(self.Q1_penalty * y_dev[0] ** 2 + self.Q2_penalty * y_dev[1] ** 2) +
                   np.sum(self.R1_penalty * u_dev[:, 0] ** 2 + self.R2_penalty * u_dev[:, 1] ** 2))

        return obj_val


def augment_state_space(A, B, C):
    """
    Augments a state-space model for offset-free MPC

    Parameters
    ----------
    A : np.ndarray
        The state matrix of size (n_states, n_states).
    B : np.ndarray
        The input matrix of size (n_states, n_inputs).
    C : np.ndarray
        The output matrix of size (n_outputs, n_states).

    Returns
    -------
    A_aug : np.ndarray
        The augmented state matrix of size ((n_states+n_outputs), (n_states+n_outputs)).
    B_aug : np.ndarray
        The augmented input matrix of size ((n_states+n_outputs), n_inputs).
    C_aug : np.ndarray
        The augmented output matrix of size (n_outputs, (n_states+n_outputs)).
    """
    n_states = A.shape[0]
    n_outputs = C.shape[0]

    # Construct integrator part for offset-free formulation
    # Bd: zeros for the integrator dynamics (n_states x n_outputs)
    Bd = np.zeros((n_states, n_outputs))
    # Augment A: Top block is [A, Bd], bottom block is [zeros, I]
    zeros_A = np.zeros((n_outputs, n_states))
    ident_A = np.eye(n_outputs)
    A_aug = np.vstack((np.hstack((A, Bd)),
                       np.hstack((zeros_A, ident_A))))

    # Augment B: Append zeros for the integrator states
    zeros_B = np.zeros((n_outputs, B.shape[1]))
    B_aug = np.vstack((B, zeros_B))

    # Augment C: Append identity so that the integrator states appear in the output
    Cd = np.eye(n_outputs)
    C_aug = np.hstack((C, Cd))

    return A_aug, B_aug, C_aug


def compute_observer_gain(A, C, desired_poles):
    """
    Compute an observer gain L for the given MPC system using the desired poles.
    Also performs an observability check.

    Parameters:
    -----------
    A, C : np.ndarray
        System Matrices
    desired_poles : np.ndarray
        A vector of desired observer poles.

    Returns:
    --------
    L : np.ndarray
        The observer gain matrix.
    """
    # Compute the observer gain using pole placement
    obs_gain_calc = signal.place_poles(A.T, C.T, desired_poles, method='KNV0')
    L = np.squeeze(obs_gain_calc.gain_matrix).T

    # Check observability
    observability_matrix = control.obsv(A, C)
    rank = np.linalg.matrix_rank(observability_matrix)
    if rank == A.shape[0]:
        print("The system is observable.")
    else:
        print("The system is not observable.")
    return L


def generate_setpoints(y_sp_scenario, n_tests, set_points_len):
    # For each scenario, create a block of size (set_points_len, n_outputs)
    blocks = [np.full((set_points_len, y_sp_scenario.shape[1]), scenario)
              for scenario in y_sp_scenario]

    # Concatenate the blocks to form one cycle
    cycle = np.concatenate(blocks, axis=0)
    # Repeat the cycle 'repetitions' times
    y_sp = np.concatenate([cycle] * n_tests, axis=0)

    time_in_sub_episodes = set_points_len * len(y_sp_scenario)

    nFE = int(y_sp.shape[0])
    idxs_setpoints = np.arange(time_in_sub_episodes - 1, nFE, time_in_sub_episodes)
    sub_episodes_changes = np.arange(1, len(idxs_setpoints) + 1)
    sub_episodes_changes_dict = {}
    for i in range(len(idxs_setpoints)):
        sub_episodes_changes_dict[idxs_setpoints[i]] = sub_episodes_changes[i]

    return y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes


def run_mpc(system, MPC_obj, y_sp_scenario, n_tests, set_points_len,
            steady_states, IC_opt, bnds, cons,
            Q1_penalty, Q2_penalty, R1_penalty, R2_penalty, L, data_min, data_max, n_inputs):
    # defining setpoints
    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes = generate_setpoints(y_sp_scenario, n_tests,
                                                                                    set_points_len)

    # Output of the system
    y_mpc = np.zeros((nFE + 1, MPC_obj.C.shape[0]))
    y_mpc[0, :] = system.current_output

    # MPC inputs
    u_mpc = np.zeros((nFE, MPC_obj.B.shape[1]))

    # Record states of the state space model
    x0_model = np.zeros(MPC_obj.A.shape[0])
    xhatdhat = np.zeros((MPC_obj.A.shape[0], nFE + 1))
    yhat = np.zeros((MPC_obj.C.shape[0], nFE))

    # Reward recording
    rewards = np.zeros(nFE)
    avg_rewards = []

    for i in range(nFE):
        # So we need to apply scaling for MPC because the formulation was in scaled deviation

        ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])

        # Solving MPC optimization problem
        sol = spo.minimize(
            lambda x: MPC_obj.mpc_opt_fun(x, y_sp[i, :], (scaled_current_input - ss_scaled_inputs),
                                          x0_model), IC_opt, bounds=bnds, constraints=cons)

        # take the first control action (this is in scaled deviation form)
        u_mpc[i, :] = sol.x[:MPC_obj.B.shape[1]] + ss_scaled_inputs

        # u (reverse scaling of the mpc)
        u_plant = reverse_min_max(u_mpc[i, :], data_min[:n_inputs], data_max[:n_inputs])

        # Calculate Delta U in scaled deviation form
        delta_u = (u_mpc[i, :] - ss_scaled_inputs) - (scaled_current_input - ss_scaled_inputs)

        # Change the current input
        system.current_input = u_plant

        # Apply the action on the system
        system.step()

        # Record the system output
        y_mpc[i + 1, :] = system.current_output

        # Since the state space calculation is in scaled will transform it
        y_current_scaled = apply_min_max(y_mpc[i, :], data_min[n_inputs:], data_max[n_inputs:]) - apply_min_max(
            steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

        # Calculate Delta y in deviation form
        delta_y = y_current_scaled - y_sp[i, :]

        # Calculate the next state in deviation form
        yhat[:, i] = np.dot(MPC_obj.C, xhatdhat[:, i])
        xhatdhat[:, i + 1] = np.dot(MPC_obj.A, xhatdhat[:, i]) + np.dot(MPC_obj.B,
                                                                        (u_mpc[i, :] - ss_scaled_inputs)) + \
                             np.dot(L, (y_current_scaled - yhat[:, i])).T
        x0_model = xhatdhat[:, i + 1]

        # Reward Calculation
        reward = - (Q1_penalty * delta_y[0] ** 2 + Q2_penalty * delta_y[1] ** 2 +
                    R1_penalty * delta_u[0] ** 2 + R2_penalty * delta_u[1] ** 2)

        # Record rewards
        rewards[i] = reward

        # Calculate average reward and printing
        if i in sub_episodes_changes_dict.keys():
            # Averaging the rewards from the last setpoint change till curtrent
            avg_rewards.append(np.mean(rewards[i - time_in_sub_episodes + 1: i]))

            # printing
            print('Sub_Episode : ', sub_episodes_changes_dict[i], ' | avg. reward :', avg_rewards[-1])

    u_mpc = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])

    return y_mpc, u_mpc, avg_rewards, rewards, xhatdhat, nFE, time_in_sub_episodes, y_sp, yhat


def plot_mpc_results(y_sp, steady_states, nFE, delta_t, time_in_sub_episodes, y_mpc, u_mpc, avg_rewards, data_min,
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
    plt.plot(time_plot, y_mpc[:, 0], 'b-', lw=2, label=r'$\mathbf{MPC}$')
    plt.step(time_plot[:-1], y_sp[0, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{Tray 24 Ethane Comp.}$', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_mpc[:, 1], 'b-', lw=2, label=r'$\mathbf{MPC}$')
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

    ########### last 400 ##########
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 0], 'b-', lw=2, label=r'$\mathbf{MPC}$')
    plt.step(time_plot_hour[:-1], y_sp[0, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{Tray 24 Ethane Comp.}$', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], 'b-', lw=2, label=r'$\mathbf{MPC}$')
    plt.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel(r'$\mathbf{Tray 85 Temp.}$ (C)', fontsize=18)
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
    plt.ylabel(r'$\mathbf{R}$ (kg/h)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.step(time_plot[:-1], u_mpc[:, 1], 'k-', lw=2, label=r'$\mathbf{Q}_m$')
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

    ############# Plot 3 (Reward) #######################

    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(1, len(avg_rewards) + 1), avg_rewards, 'ko-', lw=2, label='Reward per Episode')
    plt.ylabel(r'Avg. Reward', fontsize=16, fontweight='bold')
    plt.xlabel(r'Episode #', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.xticks(np.arange(1, len(avg_rewards) + 1), fontsize=14, fontweight='bold')
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

        axs[0].plot(yhat[0, :], 'b-', linewidth=2, label=r'$\mathbf{Tray 24 Ethane Comp.}$ (Observer)')
        axs[0].plot(y_mpc_scaled_1, 'r--', linewidth=2, label=r'$\mathbf{Tray 24 Ethane Comp.}$ (Measurement)')
        axs[0].set_ylabel('Scaled Deviation')
        axs[0].set_title('Observer vs. Real (Tray 24 Ethane Comp.)')
        axs[0].legend()
        axs[0].grid(True)

        # For Output 2:
        y_mpc_scaled_2 = ((y_mpc[:, 1] - steady_states["y_ss"][1]) -
                          0.0)  # real domain deviation
        y_mpc_scaled_2 = (y_mpc_scaled_2 - (data_min[3] - data_min[3])) / (data_max[3] - data_min[3])

        axs[1].plot(yhat[1, :], 'b-', linewidth=2, label=r'$\mathbf{Tray 85 Temp.}$ (Observer)')
        axs[1].plot(y_mpc_scaled_2, 'r--', linewidth=2, label=r'$\mathbf{Tray 85 Temp.}$ (Measurement)')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Scaled Deviation')
        axs[1].set_title('Observer vs. Real (Tray 85 Temp.)')
        axs[1].legend()
        axs[1].grid(True)

        fig.tight_layout()
        plt.show()
