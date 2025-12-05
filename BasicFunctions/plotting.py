import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from BasicFunctions.bs_fns import apply_min_max, reverse_min_max


def save_and_plot_rl_results(y_sp, steady_states, nFE, delta_t, time_in_sub_episodes,
                             y_mpc, u_mpc, avg_rewards, data_min, data_max, directory, agent_path="agent_result",
                             yhat=None):
    # Create a timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = os.path.join(directory, agent_path, timestamp)
    os.makedirs(directory, exist_ok=True)

    # Saving inputs in a pickle file
    input_data = {
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_mpc": y_mpc,
        "u_mpc": u_mpc,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "yhat": yhat
    }

    with open(os.path.join(directory, 'input_data.pkl'), 'wb') as f:
        pickle.dump(input_data, f)

    # Canceling the deviation form
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = (reverse_min_max(y_sp, data_min[2:], data_max[2:])).T

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    # Plotting functions
    def save_plot(name):
        plt.tight_layout()
        plt.savefig(os.path.join(directory, name), dpi=300)
        plt.close()

    # Plot 1
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, y_mpc[:, 0], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot[:-1], y_sp[0, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel('Tray 24 Ethan Comp.', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_mpc[:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot[:-1], y_sp[1, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel('Tray 85 Temp', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    save_plot('Outputs_all_episodes.png')

    ########### last 400 ##########
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 0], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot_hour[:-1], y_sp[0, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel('Tray 24 Ethan Comp.', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel('Tray 85 Temp', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hr)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    save_plot('plot_last_400.png')

    ####### Plot 2  ###############
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.step(time_plot[:-1], u_mpc[:, 0], 'k-', lw=2, label=r'R')
    plt.ylabel(r'$\mathbf{R}$ (kg/h)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.step(time_plot[:-1], u_mpc[:, 1], 'k-', lw=2, label=r'RQ')
    plt.ylabel(r'$\mathbf{Q}_R$ (Gj/h)', fontsize=18)
    plt.xlabel(r'$\mathbf{Time}$ (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='both', labelsize=16)

    save_plot('Inputs_all_the_episodes.png')

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

    save_plot('plot_reward.png')

    # Optional yhat plot
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
        fig.savefig(os.path.join(directory, 'plot_observer.png'), dpi=300)
        plt.close(fig)

    print(f"File has been saved in {directory}")
    return directory


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
    plt.ylabel('Tray 24 Ethan Comp.', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_mpc[:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot[:-1], y_sp[1, :], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel('Tray 85 Temp', fontsize=18)
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
    plt.ylabel('Tray 24 Ethan Comp.', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], 'b-', lw=2, label=r'$\mathbf{RL}$')
    plt.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], 'r--', lw=2, label=r'$\mathbf{Setpoint}$')
    plt.ylabel('Tray 85 Temp', fontsize=18)
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
    plt.step(time_plot[:-1], u_mpc[:, 0], 'k-', lw=2, label=r'R')
    plt.ylabel(r'$\mathbf{R}$ (kg/h)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.step(time_plot[:-1], u_mpc[:, 1], 'k-', lw=2, label=r'RQ')
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


def load_and_save_plotly_results(directory, downsample_max=50000, write_png=False):
    """
    Loads 'input_data.pkl' from the given `directory`, transforms the data,
    creates Plotly figures (including setpoints), and saves them as interactive HTML plus optional PNG.

    Parameters
    ----------
    directory : str
        Path to folder containing 'input_data.pkl'.
    downsample_max : int
        Maximum number of points to plot for long time series.
    write_png : bool
        If True, attempts to save PNG (requires 'kaleido').

    Prints progress at each major step.
    """
    # 1) Load data
    input_path = os.path.join(directory, 'input_data.pkl')
    print(f"Loading data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded.")

    # 2) Unpack
    y_sp = data["y_sp"]
    steady_states = data["steady_states"]
    nFE = data["nFE"]
    delta_t = data["delta_t"]
    time_in_sub = data["time_in_sub_episodes"]
    y_mpc = data["y_mpc"]
    u_mpc = data["u_mpc"]
    avg_rewards = data["avg_rewards"]
    data_min = data["data_min"]
    data_max = data["data_max"]
    yhat = data.get("yhat", None)
    print("Variables unpacked.")

    # 3) Reverse scaling of setpoints
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = reverse_min_max(y_sp, data_min[2:], data_max[2:]).T  # shape: (outputs, nFE)
    print("Setpoints scaled back.")

    # 4) Time arrays
    time_total = np.linspace(0, nFE * delta_t, nFE + 1)
    time_seg = np.linspace(0, time_in_sub * delta_t, time_in_sub + 1)
    print("Time arrays prepared.")

    # 5) Create output folder
    out_folder = os.path.join(directory, 'plotly_results')
    os.makedirs(out_folder, exist_ok=True)
    print(f"Results folder: {out_folder}")

    # Utility to save figures
    def save_fig(fig, name):
        html_path = os.path.join(out_folder, f"{name}.html")
        fig.write_html(html_path)
        print(f"Wrote {name}.html")
        if write_png:
            try:
                png_path = html_path.replace('.html', '.png')
                fig.write_image(png_path)
                print(f"Wrote {name}.png")
            except Exception:
                print("Skipped PNG (kaleido not installed)")

    # Utility to downsample long traces
    def downsample(x, y, max_pts=downsample_max):
        if len(x) > max_pts:
            idx = np.linspace(0, len(x) - 1, max_pts, dtype=int)
            return x[idx], y[idx]
        return x, y

    # FIGURE 1: Full trajectory (with setpoints)
    print("Building Plot 1: Full Trajectory…")
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Comp. vs Time", "Temp. vs Time"))
    # η (RL)
    xt_rl1, yt_rl1 = downsample(time_total, y_mpc[:, 0])
    fig1.add_trace(go.Scatter(x=xt_rl1, y=yt_rl1, mode='lines', name='RL Comp.'), row=1, col=1)
    # η (Setpoint)
    xt_sp1, yt_sp1 = downsample(time_total[:-1], y_sp[0])
    fig1.add_trace(go.Scatter(x=xt_sp1, y=yt_sp1, mode='lines', name='Setpoint Comp.', line=dict(dash='dash')), row=1, col=1)

    # T (RL)
    xt_rl2, yt_rl2 = downsample(time_total, y_mpc[:, 1])
    fig1.add_trace(go.Scatter(x=xt_rl2, y=yt_rl2, mode='lines', name='RL T'), row=2, col=1)
    # T (Setpoint)
    xt_sp2, yt_sp2 = downsample(time_total[:-1], y_sp[1])
    fig1.add_trace(go.Scatter(x=xt_sp2, y=yt_sp2, mode='lines', name='Setpoint T', line=dict(dash='dash')), row=2, col=1)

    fig1.update_layout(height=600, title_text="Full Trajectory")
    save_fig(fig1, 'plot_1_full_trajectory')

    # FIGURE 2: Last segment (with setpoints)
    print("→ Building Plot 2: Last Segment…")
    start = nFE - time_in_sub
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Comp. Last Segment", "T Last Segment"))
    # η (RL)
    xs_rl1, ys_rl1 = downsample(time_seg, y_mpc[start:, 0])
    fig2.add_trace(go.Scatter(x=xs_rl1, y=ys_rl1, mode='lines', name='RL Comp.'), row=1, col=1)
    # η (Setpoint)
    xs_sp1, ys_sp1 = downsample(time_seg[:-1], y_sp[0, start:])
    fig2.add_trace(go.Scatter(x=xs_sp1, y=ys_sp1, mode='lines', name='Setpoint Comp.', line=dict(dash='dash')), row=1, col=1)

    # T (RL)
    xs_rl2, ys_rl2 = downsample(time_seg, y_mpc[start:, 1])
    fig2.add_trace(go.Scatter(x=xs_rl2, y=ys_rl2, mode='lines', name='RL T'), row=2, col=1)
    # T (Setpoint)
    xs_sp2, ys_sp2 = downsample(time_seg[:-1], y_sp[1, start:])
    fig2.add_trace(go.Scatter(x=xs_sp2, y=ys_sp2, mode='lines', name='Setpoint T', line=dict(dash='dash')), row=2, col=1)

    fig2.update_layout(height=600, title_text="Last Sub-Episode")
    save_fig(fig2, 'plot_2_last_segment')

    # FIGURE 3: Manipulated variables
    print("→ Building Plot 3: Manipulated Variables…")
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Reflux", "Reboiler"))
    xt3, yt3 = downsample(time_total[:-1], u_mpc[:, 0])
    fig3.add_trace(go.Scatter(x=xt3, y=yt3, mode='lines', name='R'), row=1, col=1)
    xt4, yt4 = downsample(time_total[:-1], u_mpc[:, 1])
    fig3.add_trace(go.Scatter(x=xt4, y=yt4, mode='lines', name='Qr'), row=2, col=1)
    fig3.update_layout(height=600, title_text="Manipulated Variables")
    save_fig(fig3, 'plot_3_manipulated_vars')

    # FIGURE 4: Avg Reward
    print("→ Building Plot 4: Avg Rewards…")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=np.arange(1, len(avg_rewards)+1),
                              y=avg_rewards,
                              mode='lines+markers',
                              name='Avg Reward'))
    fig4.update_layout(height=400, title_text="Average Reward per Episode",
                       xaxis_title="Episode", yaxis_title="Reward")
    save_fig(fig4, 'plot_4_avg_rewards')

    # FIGURE 5: Observer vs Measurement
    if yhat is not None:
        print("→ Building Plot 5: Observer vs Measurement…")
        fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Observer vs Meas Comp.", "Observer vs Meas T"))
        # η (Observer)
        xt5, yt5 = downsample(np.arange(len(yhat[0])), yhat[0])
        fig5.add_trace(go.Scatter(x=xt5, y=yt5, mode='lines', name='η (obs)'), row=1, col=1)
        # η (Meas)
        xt6, yt6 = downsample(np.arange(len(y_mpc[:,0])),
                              (y_mpc[:,0] - steady_states['y_ss'][0])/(data_max[2]-data_min[2]))
        fig5.add_trace(go.Scatter(x=xt6, y=yt6, mode='lines', name='η (meas)', line=dict(dash='dash')), row=1, col=1)
        # T (Observer)
        xt7, yt7 = downsample(np.arange(len(yhat[1])), yhat[1])
        fig5.add_trace(go.Scatter(x=xt7, y=yt7, mode='lines', name='T (obs)'), row=2, col=1)
        # T (Meas)
        xt8, yt8 = downsample(np.arange(len(y_mpc[:,1])),
                              (y_mpc[:,1] - steady_states['y_ss'][1])/(data_max[3]-data_min[3]))
        fig5.add_trace(go.Scatter(x=xt8, y=yt8, mode='lines', name='T (meas)', line=dict(dash='dash')), row=2, col=1)
        fig5.update_layout(height=600, title_text="Observer vs Measurement")
        save_fig(fig5, 'plot_5_observer')

    print("→ All plots saved.")