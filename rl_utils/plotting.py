"""
Plotting helpers shared across RL notebooks and scripts.
"""

import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils.helpers import apply_min_max, reverse_min_max


def _ensure_dir(base_dir, prefix_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_rl_results_disturbance(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_mpc,
    u_mpc,
    avg_rewards,
    data_min,
    data_max,
    warm_start_plot,
    directory=None,
    prefix_name="agent_result",
    agent=None,
    delta_y_storage=None,
    rewards=None,
    dist=None,
    alt_rewards=None,
):
    """
    Save RL results and plots (mirrors the notebook function).
    Returns output directory path.
    """
    if directory is None:
        directory = os.getcwd()
    out_dir = _ensure_dir(directory, prefix_name)

    def _savefig(name):
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), bbox_inches="tight", dpi=300)
        plt.close()

    y_sp_original = np.array(y_sp, copy=True)
    actor_losses = getattr(agent, "actor_losses", None) if agent is not None else None
    critic_losses = getattr(agent, "critic_losses", None) if agent is not None else None
    dy_arr = np.array(delta_y_storage) if delta_y_storage is not None else None
    rewards_arr = np.array(rewards) if rewards is not None else None

    input_data = {
        "y_sp": y_sp_original,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_mpc": y_mpc,
        "u_mpc": u_mpc,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "delta_y_storage": delta_y_storage,
        "rewards": rewards,
        "dist": dist,
    }
    input_data["alt_rewards"] = alt_rewards

    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as f:
        pickle.dump(input_data, f)

    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp_plot = reverse_min_max(y_sp_original + y_ss, data_min[2:], data_max[2:]).T
    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    time_plot_short = np.linspace(0, warm_start_plot * delta_t, warm_start_plot + 1)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, y_mpc[:, 0], "b-", lw=2, label="RL")
    plt.step(time_plot[:-1], y_sp_plot[0, :], "r--", lw=2, label="Setpoint")
    plt.ylabel("Tray 24 Ethan Comp.")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_mpc[:, 1], "b-", lw=2, label="RL")
    plt.step(time_plot[:-1], y_sp_plot[1, :], "r--", lw=2, label="Setpoint")
    plt.ylabel("Tray 85 Temp")
    plt.xlabel("Time (hour)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    _savefig("outputs_all.png")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_short, y_mpc[: warm_start_plot + 1, 0], "b-", lw=2, label="RL")
    plt.step(time_plot_short[:-1], y_sp_plot[0, :warm_start_plot], "r--", lw=2, label="Setpoint")
    plt.ylabel("Tray 24 Ethan Comp.")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_plot_short, y_mpc[: warm_start_plot + 1, 1], "b-", lw=2, label="RL")
    plt.step(time_plot_short[:-1], y_sp_plot[1, :warm_start_plot], "r--", lw=2, label="Setpoint")
    plt.ylabel("Tray 85 Temp")
    plt.xlabel("Time (hour)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    _savefig("outputs_warm_start.png")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.step(time_plot[:-1], u_mpc[:, 0], "k-", lw=2, label="R")
    plt.ylabel("R (kg/h)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.step(time_plot[:-1], u_mpc[:, 1], "k-", lw=2, label="RQ")
    plt.ylabel("Q_R (Gj/h)")
    plt.xlabel("Time (hour)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    _savefig("inputs_all.png")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(avg_rewards) + 1), avg_rewards, "ko-", lw=2, label="Reward per Episode")
    plt.ylabel("Avg. Reward")
    plt.xlabel("Episode #")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    _savefig("reward.png")

    if actor_losses is not None and len(actor_losses) > 0:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(actor_losses, lw=1.8, color="tab:green")
        plt.ylabel("Actor Loss")
        plt.xlabel("Update Step")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("loss_actor.png")

    if critic_losses is not None and len(critic_losses) > 0:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(critic_losses, lw=1.8, color="tab:orange")
        plt.ylabel("Critic Loss")
        plt.xlabel("Update Step")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("loss_critic.png")

    if dy_arr is not None and dy_arr.ndim == 2 and dy_arr.shape[1] >= 2:
        n = dy_arr.shape[0]
        i0 = max(0, n - 300)
        w = dy_arr[i0:n]
        if len(w) > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.plot(w[:, 0], c="r", label=r"$\Delta y_1$")
            plt.plot(w[:, 1], c="b", label=r"$\Delta y_2$")
            plt.ylabel(r"$\Delta y$")
            plt.xlabel("Step")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("delta_y_last300.png")

        j0 = max(0, n - 700)
        j1 = max(0, n - 400)
        w2 = dy_arr[j0:j1]
        if len(w2) > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.plot(w2[:, 0], c="r", label=r"$\Delta y_1$")
            plt.plot(w2[:, 1], c="b", label=r"$\Delta y_2$")
            plt.ylabel(r"$\Delta y$")
            plt.xlabel("Step")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("delta_y_700_400.png")

    if rewards_arr is not None and rewards_arr.ndim == 1 and rewards_arr.size > 0:
        n = rewards_arr.size
        j0 = max(0, n - 700)
        j1 = max(0, n - 400)
        w = rewards_arr[j0:j1]
        if w.size > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.scatter(range(w.size), w, s=10)
            plt.ylabel("Reward")
            plt.xlabel("Step")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("rewards_700_400.png")

        i0 = max(0, n - 300)
        w2 = rewards_arr[i0:n]
        if w2.size > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.scatter(range(w2.size), w2, s=10)
            plt.ylabel("Reward")
            plt.xlabel("Step")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("rewards_last300.png")

        plt.figure(figsize=(7.6, 4.2))
        plt.scatter(range(rewards_arr.size), rewards_arr, s=10)
        plt.ylabel("Reward")
        plt.xlabel("Step")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("rewards_all.png")

    if dist is not None:
        time_plot_dist = time_plot[10:]
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(time_plot_dist, dist[9:], lw=1.8, color="tab:blue")
        plt.ylabel("Reflux (Kg/h)")
        plt.xlabel("Time (hour)")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("feed_dist.png")

    return out_dir


def compare_mpc_rl(y_rl, y_mpc, y_sp, avg_rewards_rl, avg_rewards_mpc, delta_t, directory, prefix_name):
    """
    Compare RL vs MPC trajectories and reward curves.
    """
    out_dir = _ensure_dir(directory, prefix_name)

    def _savefig(name_png):
        plt.tight_layout()
        full_png = os.path.join(out_dir, name_png)
        plt.savefig(full_png, bbox_inches="tight", dpi=300)
        base, _ = os.path.splitext(full_png)
        plt.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()

    start_idx = -800
    y_rl_tail = np.asarray(y_rl[start_idx:, :], float)
    y_mpc_tail = np.asarray(y_mpc[start_idx:, :], float)
    sp_tail = np.asarray(y_sp[start_idx:, :], float).T

    L_line = min(y_rl_tail.shape[0], y_mpc_tail.shape[0])
    W_step = sp_tail.shape[1]
    W = min(W_step, L_line - 1)

    y_rl_tail = y_rl_tail[-(W + 1) :, :]
    y_mpc_tail = y_mpc_tail[-(W + 1) :, :]
    sp_tail = sp_tail[:, -W:]

    t_line = np.linspace(0, W * delta_t, W + 1)
    t_step = t_line[:-1]

    fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
    axs[0].plot(t_line, y_rl_tail[:, 0], "-", lw=2.2, color="tab:blue", label="RL", zorder=2)
    axs[0].plot(t_line, y_mpc_tail[:, 0], "--", lw=2.2, color="black", label="MPC", zorder=2)
    axs[0].step(t_step, sp_tail[0, :], where="post", linestyle="--", lw=2.2, color="tab:red", alpha=0.95, label="Setpoint", zorder=3)
    axs[0].set_ylabel(r"$x_{23,\mathrm{C_2H_6}}$ (–)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=1.0, facecolor="white")

    axs[1].plot(t_line, y_rl_tail[:, 1], "-", lw=2.2, color="tab:blue", label="RL", zorder=2)
    axs[1].plot(t_line, y_mpc_tail[:, 1], "--", lw=2.2, color="black", label="MPC", zorder=2)
    axs[1].step(t_step, sp_tail[1, :], where="post", linestyle="--", lw=2.2, color="tab:red", alpha=0.95, label="Setpoint", zorder=3)
    axs[1].set_ylabel(r"$T_{85}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=1.0, facecolor="white")

    plt.gcf().subplots_adjust(right=0.82)
    fig.tight_layout()
    _savefig("outputs_compare.png")

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x_main = np.arange(1, len(avg_rewards_rl))
    ax.plot(x_main, avg_rewards_rl[1:], "o-", lw=2, ms=4, color="tab:blue", label="RL")
    ax.hlines(avg_rewards_mpc[-1], xmin=x_main[0], xmax=x_main[-1], color="tab:orange", linestyle="--", lw=2, label="MPC")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")

    axins = zoomed_inset_axes(ax, zoom=2.0, loc="center right", bbox_to_anchor=(1.02, 0.5), bbox_transform=ax.transAxes, borderpad=0.0)
    axins.plot(x_main, avg_rewards_rl[1:], "o-", lw=1.8, ms=3.5, color="tab:blue")
    axins.hlines(avg_rewards_mpc[-1], xmin=x_main[0], xmax=x_main[-1], color="tab:orange", linestyle="--", lw=1.8)
    x1, x2, y1, y2 = 70, 85, 30, 60
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, linestyle="--", alpha=0.35)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.6", lw=1.1)

    _savefig("reward_compare.png")

    return out_dir


def compare_mpc_rl_disturbance(y_rl, y_mpc, y_sp, u_mpc, u_rl, avg_rewards_rl, avg_rewards_mpc, time_in_sub_episodes, delta_t):
    """
    Matplotlib overlay for disturbance scenarios (keeps original notebook layout).
    """
    y_sp = y_sp.T
    time_plot_sub = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes)
    start_idx = -400

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_sub, y_rl[start_idx:, 0], "b-", lw=2)
    plt.plot(time_plot_sub, y_mpc[start_idx:, 0], "k--", lw=2)
    plt.step(time_plot_sub, y_sp[0, start_idx:], "r--", lw=2)
    plt.ylabel(r"$x_{24,\mathrm{C_2H_6}}$")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_plot_sub, y_rl[start_idx:, 1], "b-", lw=2)
    plt.plot(time_plot_sub, y_mpc[start_idx:, 1], "k--", lw=2)
    plt.step(time_plot_sub, y_sp[1, start_idx:], "r--", lw=2)
    plt.ylabel(r"$T_{85}$ (K)")
    plt.xlabel("Time (hour)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    avg_rewards_rl = avg_rewards_rl[50:]
    plt.plot(np.arange(1, len(avg_rewards_rl) + 1), avg_rewards_rl, "bo-", lw=2)
    plt.hlines(avg_rewards_mpc[-1], xmin=0, xmax=np.arange(1, len(avg_rewards_rl) + 1)[-1], linestyles="--", lw=2)
    plt.ylabel(r"Avg. Reward")
    plt.xlabel(r"Episode #")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_sub, u_rl[start_idx:, 0], "b-", lw=2, label=r"RL")
    plt.plot(time_plot_sub, u_mpc[start_idx:, 0], "k--", lw=2, label=r"MPC")
    plt.ylabel(r"$\mathbf{R}$ (kg/h)")
    plt.grid(True)
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(time_plot_sub, u_rl[start_idx:, 1], "b-", lw=2, label=r"RL")
    plt.plot(time_plot_sub, u_mpc[start_idx:, 1], "k--", lw=2, label=r"MPC")
    plt.ylabel(r"$\mathbf{Q}_R$ (Gj/h)")
    plt.xlabel("Time (hour)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def compare_mpc_rl_rl1(y_rl, y_mpc, y_sp, time_in_sub_episodes, delta_t, y_rl2):
    """
    Overlay two RL runs vs MPC for a window (as in notebooks).
    """
    y_sp = y_sp.T
    time_plot_sub = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes)
    start_idx = -400

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_sub, y_rl[start_idx:, 0], "b-", lw=2)
    plt.plot(time_plot_sub, y_mpc[start_idx:, 0], "k--", lw=2)
    plt.plot(time_plot_sub, y_rl2[start_idx:, 0], "g-", lw=2)
    plt.step(time_plot_sub, y_sp[0, start_idx:], "r--", lw=2)
    plt.ylabel(r"$x_{24,\mathrm{C_2H_6}}$")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_plot_sub, y_rl[start_idx:, 1], "b-", lw=2)
    plt.plot(time_plot_sub, y_mpc[start_idx:, 1], "k--", lw=2)
    plt.plot(time_plot_sub, y_rl2[start_idx:, 1], "g-", lw=2)
    plt.step(time_plot_sub, y_sp[1, start_idx:], "r--", lw=2)
    plt.ylabel(r"$T_{85}$ (K)")
    plt.xlabel("Time (hour)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
