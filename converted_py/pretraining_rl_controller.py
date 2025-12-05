# Converted from pretraining_rl_controller.ipynb
from Simulation.mpc import *
from Simulation.systemFunctions import DistillationColumnAspen
from utils.helpers import *

# System and Snapshot paths
path = r"C:/Users\HAMEDI\Desktop\FinalDocuments\FinalDocuments\C2SplitterControlFiles\AspenFiles\dynsim\Plant\C2S_SS_simulation9.dynf"
path_snaps = r"C:/Users\HAMEDI\Desktop\FinalDocuments\FinalDocuments\C2SplitterControlFiles\AspenFiles\dynsim\Plant\AM_C2S_SS_simulation9"

# First initiate the system
# Nominal Conditions
nominal_conditions =  np.array([1.50032484e+05, -2.10309105e+01, 2.08083248e+01, 6.30485237e-01, 3.69514734e-01, -2.40000000e+01])

# Steady State inputs
ss_inputs = np.array([320000.0, 110.0])

# Sampling time of the system
delta_t = 1 / 6 # 10 mins

# steady state values
dl = DistillationColumnAspen(path, ss_inputs, nominal_conditions)
steady_states={"ss_inputs":dl.ss_inputs,
               "y_ss":dl.y_ss}
print(steady_states)
dl.close(path_snaps)

dir_path = os.path.join(os.getcwd(), "Data/models")

# Defining the range of setpoints for data generation
setpoint_y = np.array([[0.002, -26.0],
                       [0.05, -16.0]])
u_min = np.array([300000.0, 100.0])
u_max = np.array([460000, 150.0])

system_data = load_and_prepare_system_data(steady_states=steady_states, setpoint_y=setpoint_y, u_min=u_min, u_max=u_max)

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

# Setpoints in deviation form
inputs_number = int(B_aug.shape[1])
y_sp_scenario = np.array([[0.013, -23.],
                         [0.018, -22.]])

y_sp_scenario = (apply_min_max(y_sp_scenario, data_min[inputs_number:], data_max[inputs_number:])
                 - apply_min_max(steady_states["y_ss"], data_min[inputs_number:], data_max[inputs_number:]))

n_tests = 200
set_points_len = 200
TEST_CYCLE = [False, False, False, False, False]
warm_start = 5
ACTOR_FREEZE = 4 * set_points_len
warm_start_plot = warm_start * 2 * set_points_len + ACTOR_FREEZE

# # # Observer Gain
# poles = np.array([0.032, 0.03501095, 0.04099389, 0.04190188, 0.07477281,
#                   0.01153274, 0.41036367])
# L = compute_observer_gain(A_aug, C_aug, poles)
# L
# # Observer Gain
poles = np.array([0.032, 0.03501095, 0.04099389, 0.04190188, 0.07477281,
                  0.5153274, 0.61036367])
# poles = np.array([0.6, 0.6, 0.55, 0.5, 0.5, 0.98, 0.95])
L = compute_observer_gain(A_aug, C_aug, poles)
L

from TD3Agent.agent import TD3Agent
import torch

set_points_number = int(C_aug.shape[0])
STATE_DIM = int(A_aug.shape[0]) + set_points_number + inputs_number
ACTION_DIM = int(B_aug.shape[1])
n_outputs = C_aug.shape[0]
ACTOR_LAYER_SIZES = [256, 256, 256]
CRITIC_LAYER_SIZES = [256, 256, 256]
BUFFER_CAPACITY = 5_000_000
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
SMOOTHING_STD = 0.0001
NOISE_CLIP = 0.2
EXPLORATION_NOISE_STD = 0.05
GAMMA = 0.995
TAU = 0.005
MAX_ACTION = 1
POLICY_DELAY = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
STD_START = 0.005
STD_END = 0.0
STD_DECAY_RATE = 0.99995
STD_DECAY_MODE = "exp"

td3_agent = TD3Agent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
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
    actor_freeze=ACTOR_FREEZE,
    mode="mpc"
    )

from BasicFunctions.td3_functions import filling_the_buffer, add_steady_state_samples

# MPC parameters
predict_h = 6
cont_h = 3
b1 = (b_min[0], b_max[0])
b2 = (b_min[1], b_max[1])
bnds = (b1, b2)*cont_h
cons = []
IC_opt = np.zeros(inputs_number*cont_h)
Q1_penalty = 1.
Q2_penalty = 1.
R1_penalty = 1.
R2_penalty = 1.
Q_penalty = np.array([[Q1_penalty, 0], [0, Q2_penalty]])
R_penalty = np.array([[R1_penalty, 0], [0, R2_penalty]])

MPC_obj = MpcSolver(A_aug, B_aug, C_aug,
                    Q1_penalty, Q2_penalty, R1_penalty, R2_penalty,
                    predict_h, cont_h)

steady_states_samples_number = 100000
mpc_pretrain_samples_numbers = BUFFER_CAPACITY - steady_states_samples_number

filling_the_buffer(
        min_max_dict,
        A_aug, B_aug, C_aug,
        MPC_obj,
        mpc_pretrain_samples_numbers,
        Q_penalty, R_penalty,
        td3_agent,
        IC_opt, bnds, cons, chunk_size= 100000)

add_steady_state_samples(
        min_max_dict,
        A_aug, B_aug, C_aug,
        MPC_obj,
        steady_states_samples_number,
        Q_penalty, R_penalty,
        td3_agent,
        IC_opt, bnds, cons, chunk_size= 100000)

for g in td3_agent.actor_optimizer.param_groups:
    g['lr'] = 1e-7
    print(g["lr"])
# for g in td3_agent.critic_optimizer.param_groups:
#     g['lr'] = 1e-6
#     print(g["lr"])

td3_agent.pretrain_from_buffer(
    num_updates=2000000,
    use_target_noise=False,
    log_interval=2000,
)

filename_agent = td3_agent.save(dir_path)

import torch
import numpy as np

@torch.no_grad()
def print_accuracy(agent, n_samples: int = 10_000):
    """
    Simple BC accuracy: sample from agent.buffer, compare actor(s) to dataset actions.
    Prints overall R^2 and per-action-dimension R^2.
    """
    device = getattr(agent, "device", torch.device("cpu"))
    if len(agent.buffer) == 0:
        print("Buffer is empty; cannot evaluate.")
        return

    B = min(n_samples, len(agent.buffer))
    sample = agent.buffer.sample(B, device=device)

    # Handle ReplayBuffer (5-tuple) or PER buffer (7-tuple)
    if len(sample) == 5:
        s, a, _, _, _ = sample
    else:
        s, a, _, _, _, _, _ = sample

    s = s.to(device).float()
    a = a.to(device).float()
    if a.ndim == 1:
        a = a.unsqueeze(-1)  # [B] -> [B,1]

    # Actor predictions
    pred = agent.actor(s).clamp(-agent.max_action, agent.max_action)

    # To numpy
    y_true = a.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()

    # Basic sanity
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true {y_true.shape} vs pred {y_pred.shape}. "
                         "Check ACTION_DIM and how actions are stored.")

    # R^2 per dimension and mean
    eps = 1e-12
    num = np.sum((y_true - y_pred) ** 2, axis=0)                              # SSE
    den = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)  # SST
    r2_per_dim = 1.0 - num / np.maximum(den, eps)
    r2_mean = float(np.mean(r2_per_dim))

    print(f"Agent R^2 vs MPC (mean over {y_true.shape[1]} actions) : {r2_mean:.6f}")
    for j, r2j in enumerate(r2_per_dim):
        print(f"  - R^2(action[{j}]): {r2j:.6f}")

print_accuracy(td3_agent, n_samples=5)



