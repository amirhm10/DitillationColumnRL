# Converted from pretraining_rl_controller_SAC.ipynb
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

from SACAgent.sac_agent import SACAgent
import torch

set_points_number = int(C_aug.shape[0])
STATE_DIM = int(A_aug.shape[0]) + set_points_number + inputs_number
ACTION_DIM = int(B_aug.shape[1])
n_outputs = C_aug.shape[0]
ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]
BUFFER_CAPACITY = 5_000_000
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ALPHA_LR = 1e-4
GAMMA = 0.99
TAU = 0.005
MAX_ACTION = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

sac_agent = SACAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    actor_hidden=ACTOR_LAYER_SIZES,
    critic_hidden=CRITIC_LAYER_SIZES,
    gamma=GAMMA,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    alpha_lr=ALPHA_LR,
    batch_size=BATCH_SIZE,
    grad_clip_norm=10.0,
    init_alpha=0.2,
    learn_alpha=True,
    target_entropy=None,
    target_update="soft",
    tau=TAU,
    hard_update_interval=10_000,
    activation="relu",
    use_layernorm=False,
    dropout=0.0,
    max_action=MAX_ACTION,
    buffer_size=BUFFER_CAPACITY,
    use_per=False,
    device=DEVICE,
    use_adamw=True,
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
        sac_agent,
        IC_opt, bnds, cons, chunk_size= 100000)

add_steady_state_samples(
        min_max_dict,
        A_aug, B_aug, C_aug,
        MPC_obj,
        steady_states_samples_number,
        Q_penalty, R_penalty,
        sac_agent,
        IC_opt, bnds, cons, chunk_size= 100000)

for g in sac_agent.actor_optimizer.param_groups:
    g['lr'] = 1e-10
    print(g["lr"])
# for g in td3_agent.critic_optimizer.param_groups:
#     g['lr'] = 1e-6
#     print(g["lr"])

logs = sac_agent.pretrain_from_buffer(
    num_updates=1_500_000,
    log_interval=2000,
)

filename_agent = sac_agent.save(dir_path, prefix="sac")

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
        a = a.unsqueeze(-1)

    # Actor predictions: deterministic SAC policy
    pred = agent.actor.deterministic_action(s)
    pred = pred.clamp(-agent.max_action, agent.max_action)

    y_true = a.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true {y_true.shape} vs pred {y_pred.shape}.")

    eps = 1e-12
    num = np.sum((y_true - y_pred) ** 2, axis=0)
    den = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_dim = 1.0 - num / np.maximum(den, eps)
    r2_mean = float(np.mean(r2_per_dim))

    print(f"Agent R^2 vs MPC (mean over {y_true.shape[1]} actions) : {r2_mean:.6f}")
    for j, r2j in enumerate(r2_per_dim):
        print(f"  - R^2(action[{j}]): {r2j:.6f}")

print_accuracy(sac_agent, n_samples=2)



