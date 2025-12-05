"""
Converted version of OnlineTrainingNoDist1.ipynb.
Uses shared rl_utils modules so other notebooks/scripts can import the same pieces.
"""

import os
import numpy as np
import torch

from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.systemFunctions import DistillationColumnAspen
from TD3Agent.agent import TD3Agent
from rl_utils.rewards import make_reward_fn_relative_QR, make_reward_fn_quadratic
from rl_utils.training import run_rl_train
from rl_utils.plotting import plot_rl_results_disturbance
from utils.helpers import load_and_prepare_system_data, apply_min_max


# ---- Paths / files (edit as needed) ----
ASPEN_PATH = r"C:/Users/HAMEDI/Desktop/FinalDocuments/FinalDocuments/C2SplitterControlFiles/AspenFiles/dynsim/Plant/C2S_SS_simulation2.dynf"
ASPEN_SNAPS = r"C:/Users/HAMEDI/Desktop/FinalDocuments/FinalDocuments/C2SplitterControlFiles/AspenFiles/dynsim/Plant/AM_C2S_SS_simulation2"
AGENT_PATH = r"C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/distillationColumnRL/Data/models/agent_2507131207.pkl"
DATA_DIR = "Data"


def build_agent(state_dim, action_dim, actor_freeze):
    ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
    CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]
    BUFFER_CAPACITY = 200000
    ACTOR_LR = 5e-5
    CRITIC_LR = 5e-4
    SMOOTHING_STD = 0.02
    NOISE_CLIP = 0.05
    GAMMA = 0.995
    TAU = 0.005
    MAX_ACTION = 1
    POLICY_DELAY = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    STD_START = 0.01
    STD_END = 0.0
    STD_DECAY_RATE = 0.99992
    STD_DECAY_MODE = "exp"

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden=ACTOR_LAYER_SIZES,
        critic_hidden=CRITIC_LAYER_SIZES,
        gamma=GAMMA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        batch_size=BATCH_SIZE,
        policy_delay=POLICY_DELAY,
        target_policy_smoothing_noise_std=SMOOTHING_STD,
        noise_clip=NOISE_CLIP,
        max_action=MAX_ACTION,
        tau=TAU,
        std_start=STD_START,
        std_end=STD_END,
        std_decay_rate=STD_DECAY_RATE,
        std_decay_mode=STD_DECAY_MODE,
        buffer_size=BUFFER_CAPACITY,
        device=DEVICE,
        actor_freeze=actor_freeze,
    )
    return agent


def main():
    # ---- System setup ----
    nominal_conditions = np.array([1.50032484e05, -2.10309105e01, 2.08083248e01, 6.30485237e-01, 3.69514734e-01, -2.4e01])
    ss_inputs = np.array([320000.0, 110.0])
    delta_t = 1 / 6  # 10 mins

    dl = DistillationColumnAspen(ASPEN_PATH, ss_inputs, nominal_conditions)
    steady_states = {"ss_inputs": dl.ss_inputs, "y_ss": dl.y_ss}

    # ---- Data prep ----
    setpoint_y = np.array([[0.002, -26.0], [0.05, -16.0]])
    u_min = np.array([300000.0, 100.0])
    u_max = np.array([460000.0, 150.0])
    system_data = load_and_prepare_system_data(steady_states=steady_states, setpoint_y=setpoint_y, u_min=u_min, u_max=u_max, data_dir=DATA_DIR)

    A_aug = system_data["A_aug"]
    B_aug = system_data["B_aug"]
    C_aug = system_data["C_aug"]
    data_min = system_data["data_min"]
    data_max = system_data["data_max"]
    min_max_states = system_data["min_max_states"]
    y_sp_scaled_deviation = system_data["y_sp_scaled_deviation"]
    b_min = system_data["b_min"]
    b_max = system_data["b_max"]
    min_max_dict = system_data["min_max_dict"]

    # ---- Setpoints / scheduling ----
    inputs_number = int(B_aug.shape[1])
    y_sp_scenario = np.array([[0.013, -23.0], [0.028, -21.0]])
    y_sp_scenario = apply_deviation(y_sp_scenario, steady_states["y_ss"], data_min, data_max, inputs_number)
    n_tests = 200
    set_points_len = 200
    TEST_CYCLE = [False, False, False, False, False]
    warm_start = 10
    ACTOR_FREEZE = 10 * set_points_len
    warm_start_plot = warm_start * 2 * set_points_len + ACTOR_FREEZE

    # ---- Observer ----
    poles = np.array([0.032, 0.03501095, 0.04099389, 0.04190188, 0.07477281, 0.01153274, 0.41036367])
    L = compute_observer_gain(A_aug, C_aug, poles)

    # ---- Agent ----
    set_points_number = int(C_aug.shape[0])
    STATE_DIM = int(A_aug.shape[0]) + set_points_number + inputs_number
    ACTION_DIM = int(B_aug.shape[1])
    td3_agent = build_agent(STATE_DIM, ACTION_DIM, ACTOR_FREEZE)
    td3_agent.load(AGENT_PATH)

    # ---- MPC ----
    predict_h = 6
    cont_h = 3
    b1 = (b_min[0], b_max[0])
    b2 = (b_min[1], b_max[1])
    bnds = (b1, b2) * cont_h
    cons = []
    IC_opt = np.zeros(inputs_number * cont_h)
    Q1_penalty = 1.0
    Q2_penalty = 1.0
    R1_penalty = 1.0
    R2_penalty = 1.0
    MPC_obj = MpcSolver(A_aug, B_aug, C_aug, Q1_penalty, Q2_penalty, R1_penalty, R2_penalty, predict_h, cont_h)

    # ---- Reward ----
    k_rel = np.array([0.3, 0.02])
    band_floor_phys = np.array([0.003, 0.3])
    Q_diag = np.array([3.7e4, 1.5e3])
    R_diag = np.array([2.5e3, 2.5e3])
    params, reward_fn = make_reward_fn_relative_QR(
        data_min,
        data_max,
        inputs_number,
        k_rel,
        band_floor_phys,
        Q_diag,
        R_diag,
        tau_frac=0.7,
        gamma_out=0.5,
        gamma_in=0.5,
        beta=7.0,
        gate="geom",
        lam_in=1.0,
        bonus_kind="exp",
        bonus_k=12.0,
        bonus_p=0.6,
        bonus_c=20.0,
    )
    print("Reward params (relative):", params)

    # Alternate simple quadratic reward for logging
    quad_reward_fn = make_reward_fn_quadratic(
        Q1_penalty=1.0, Q2_penalty=1.0, R1_penalty=1.0, R2_penalty=1.0
    )
    print("Alt reward: quadratic (classic Q/R penalties)")

    # ---- Train ----
    alt_log_path = os.path.join(DATA_DIR, "alt_reward_quadratic.npy")

    y_system, u_rl, avg_rewards, rewards, xhatdhat, nFE, time_in_sub_episodes, y_sp, yhat, delta_y_storage, alt_rewards = run_rl_train(
        dl,
        A_aug,
        B_aug,
        C_aug,
        y_sp_scenario,
        n_tests,
        set_points_len,
        steady_states,
        min_max_dict,
        td3_agent,
        MPC_obj,
        L,
        data_min,
        data_max,
        warm_start,
        TEST_CYCLE,
        reward_fn,
        actor_freeze=ACTOR_FREEZE,
        alt_reward_fn=quad_reward_fn,
        alt_log_path=alt_log_path,
    )

    out_dir = plot_rl_results_disturbance(
        y_sp,
        steady_states,
        nFE,
        delta_t,
        time_in_sub_episodes,
        y_system,
        u_rl,
        avg_rewards,
        data_min,
        data_max,
        warm_start_plot,
        directory=os.path.join(os.getcwd(), DATA_DIR),
        prefix_name="dl_no_disturb",
        agent=td3_agent,
        delta_y_storage=delta_y_storage,
        rewards=rewards,
        alt_rewards=alt_rewards,
    )
    print("Saved results to:", out_dir)
    dl.close(ASPEN_SNAPS)


def apply_deviation(y_sp_scenario, y_ss, data_min, data_max, inputs_number):
    """
    Convert setpoints to scaled deviation space (helper for main).
    """
    y_sp_scaled = (apply_min_max(y_sp_scenario, data_min[inputs_number:], data_max[inputs_number:]) -
                   apply_min_max(y_ss, data_min[inputs_number:], data_max[inputs_number:]))
    return y_sp_scaled


if __name__ == "__main__":
    main()
