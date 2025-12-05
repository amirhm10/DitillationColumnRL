import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# import matlab.engine
# import matlab
from Simulation.systemFunctions import DistillationColumnAspen


def generate_step_test_data(u_start, step_value,
                            initial_duration=40,
                            step_duration=200,
                            step_index=0):
    """
    Generate test data with a step change.

    Parameters:
    -----------
    u_start : array-like of shape (n_inputs,)
        The starting input values.
    step_value : float
        The value to add to u_start on the specified input.
    initial_duration : int, optional
        Number of time steps to hold the initial value (default is 40).
    step_duration : int, optional
        Number of time steps to hold the stepped value (default is 200).
    step_index : int, optional
        The index of the inputs to which the step is applied (default is 0).

    Returns:
    --------
    test_data : np.ndarray
        An array of shape ((initial_duration + step_duration), n_inputs)
        containing the test data with the step change applied.
    """
    # Create an array for the initial duration using the starting values
    initial_array = np.full((initial_duration, len(u_start)), u_start)

    # Copy the starting value and apply the step change to the specified channel
    stepped_input = np.array(u_start, copy=True)
    stepped_input[step_index] += step_value

    # Create an array for the stepped duration with the modified input
    step_array = np.full((step_duration, len(u_start)), stepped_input)

    # Concatenate the two arrays to form the complete test data
    test_data = np.concatenate((initial_array, step_array), axis=0)

    return test_data


def simulate_system(system, input_sequence):
    """
    Simulate the system with the given input sequence.

    Parameters:
    -----------
    system : object
        The system to be simulated. It must have a method `step()` and
        an attribute `current_input` that can be set.
    input_sequence : array-like
        A sequence (e.g. numpy array) of inputs to apply at each time step.

    Returns:
    --------
    results : dict
        A dictionary with keys:
          - 'inputs': all applied inputs (including the initial condition)
          - 'outputs': the outputs recorded after each step
   """

    # Initialize lists to store simulation data.
    # Record the initial output (and state, if available)
    outputs = [system.current_output]

    # Loop over each input step.
    for inp in input_sequence:
        system.current_input = inp
        system.step()  # Advance the simulation one time step.

        outputs.append(system.current_output)

    results = {
        'inputs': np.array(input_sequence),
        'outputs': np.array(outputs)
    }

    return results


def plot_results(time, outputs, inputs, output_labels=None, input_labels=None):
    """
    Plot system outputs and inputs.

    Parameters:
    -----------
    time : array-like
        Time points corresponding to the simulation data.
    outputs : numpy.ndarray
        Array of outputs with shape (n_points, n_outputs).
    inputs : numpy.ndarray
        Array of inputs with shape (n_points-1, n_inputs) (if the first output was recorded before applying any input).
    output_labels : list of str
        Labels for the output channels.
    input_labels : list of str
        Labels for the input channels.
    """
    if input_labels is None:
        input_labels = ['Input 1', 'Input 2']
    if output_labels is None:
        output_labels = ['Output 1', 'Output 2']
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

    # Plot outputs
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, outputs[:, 0], 'b-', lw=2, label=output_labels[0])
    plt.ylabel(output_labels[0])
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(time, outputs[:, 1], 'b-', lw=2, label=output_labels[1])
    plt.ylabel(output_labels[1])
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Plot inputs (using step plots)
    plt.figure(figsize=(10, 8))
    time_input = time[:-1]  # assuming one input per step (applied before each step)
    plt.subplot(2, 1, 1)
    plt.step(time_input, inputs[:, 0], 'k-', lw=2, label=input_labels[0])
    plt.ylabel(input_labels[0])
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.step(time_input, inputs[:, 1], 'k-', lw=2, label=input_labels[1])
    plt.ylabel(input_labels[1])
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def save_simulation_data(data, filename, column_names):
    """
    Save simulation data (e.g. concatenated inputs and outputs) to a CSV file.

    Parameters:
    -----------
    data : numpy.ndarray
        Data array to save.
    filename : str
        Path to the CSV file.
    column_names : list of str
        Column headers for the CSV.
    """
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(filename, index=False)


def run_dl_experiment(step_value, step_channel, save_filename,
                      path, ss_inputs, nominal_conditions, delta_t, path_snaps):
    # Instantiate the reactor
    dl = DistillationColumnAspen(path, ss_inputs, nominal_conditions)

    # Retrieve initial input from the reactor.
    u_start = dl.current_input

    # Generate step test input data
    step_data = generate_step_test_data(u_start, step_value, step_index=step_channel)

    # Run the simulation
    results = simulate_system(dl, step_data)

    # Create a time vector
    n_points = results['outputs'].shape[0]
    time = np.linspace(0, n_points * delta_t, n_points)

    # Save the combined data (here we concatenate the input and the outputs excluding the initial output)
    # Adjust the column names as appropriate.
    data_to_save = np.concatenate((results['inputs'], results['outputs'][1:]), axis=1)
    data_dir = os.path.join(os.getcwd(), 'Data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    save_path = os.path.join(data_dir, save_filename)
    column_names = ["Reflux Flow", "Reboiler Duty", "Tray 24 Ethane Comp.", "Tray 85 Temp."]
    save_simulation_data(data_to_save, save_path, column_names)

    # Plot the results
    plot_results(time, results['outputs'], results['inputs'],
                 output_labels=["Tray 24 Ethane Comp.", "Tray 85 Temp."],
                 input_labels=["Reflux Flow", "Reboiler Duty"])

    dl.close(path_snaps)

    return results


def scaling_min_max_factors(file_paths):
    """
    Reads CSV data files, applies the scaling transformation, and returns min max values
    :param file_paths:
    :return: min_max_factors
    """
    data_min = []
    data_max = []
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        # Find the maximum and minimum and append to the lists
        data_min.append(df.min())
        data_max.append(df.max())

    return np.min(data_min, axis=0), np.max(data_max, axis=0)


def apply_deviation_form_scaled(steady_states, file_paths, data_min, data_max):
    """
    Reads CSV data files, applies the deviation transformation, and returns
    the resulting DataFrames.

    Parameters
    ----------
    steady_states : dict
        A dictionary that provides steady state data.
        'ss_inputs' and 'y_ss' which are used to form the steady state vector.
    file_paths : dict
        A dictionary mapping a key (e.g., "Qc", "Qm") to a file path.

    Returns
    -------
    deviations : dict
        A dictionary mapping each key to its deviation-form DataFrame.
    """
    # Construct the full steady state vector
    u_ss = steady_states['ss_inputs']  # e.g., steady inputs
    y_ss = steady_states['y_ss']  # e.g., steady outputs
    ss = np.concatenate((u_ss, y_ss), axis=0)

    ss_scaled = apply_min_max(ss, data_min, data_max)

    deviations = {}
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        # Subtract the steady state vector from all columns
        deviations[key] = apply_min_max(df, data_min, data_max) - ss_scaled
    return deviations


def apply_min_max(data, min_val, max_val):
    """
    Applies min and max values to data.
    :param data:
    :param min_val:
    :param max_val:
    :return: scaled data
    """
    data = (data - min_val) / (max_val - min_val)
    return data


def reverse_min_max(scaled_data, min_val, max_val):
    """
    Reverses min-max scaling to recover the original data.

    Parameters:
    -----------
    scaled_data : array-like
        Data that has been scaled to the [0, 1] range.
    min_val : float or array-like
        The minimum value(s) used in the original scaling.
    max_val : float or array-like
        The maximum value(s) used in the original scaling.

    Returns:
    --------
    original_data : array-like
        The data mapped back to its original scale.
    """
    original_data = scaled_data * (max_val - min_val) + min_val
    return original_data


def data_time28_63_dict(df, mode=0, sampling_period=0.5, interactive=True, limit=25):
    """
    Processes a deviation-form DataFrame to extract transfer function parameters.
    The function assumes that the input column is at the position indicated by 'mode'
    and that the last two columns correspond to the system outputs.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame in deviation form (steady state subtracted).
    mode : int, optional
        The index of the input column on which the test was performed (default is 0).
    sampling_period : float, optional
        The sampling period (in hours) used to create the time vector (default is 0.5).
    interactive : bool, optional
        If True, the function will prompt the user for time values (default is True).

    Returns
    -------
    transfer_functions_dict : dict
        A dictionary where each key is an output channel and the value is another
        dictionary containing the transfer function parameters: Time 63, Time 28, kp,
        taup, and theta.
    data : pd.DataFrame
        The subset of the original data starting just before the step change.
    """
    # Determine the input column to use (for detecting the step change)
    input_col = df.columns[mode]
    constant_value = df[input_col].iloc[0]

    # Find the first row where the input deviates from the constant value.
    mask = df[input_col] != constant_value
    if mask.any():
        index_change = df.index[mask][0] - 1
    else:
        index_change = 0

    # Extract the data starting at (or just before) the step change
    data = df.iloc[index_change:].reset_index(drop=True)
    point_numbers = len(data)
    # Create a time vector
    time_plot = np.arange(point_numbers) * sampling_period

    transfer_functions_dict = {}

    # Process each output column (assumed to be the last two columns of the DataFrame)
    for output in df.columns[-2:]:
        while True:
            if interactive:
                user_input = input(
                    f"Enter 'time_28' and 'time_63' (in hours) for {output} separated by a comma (or type 'Done' to finish): "
                )
            else:
                raise ValueError("Non-interactive mode not implemented. Set interactive=True.")

            if user_input.lower() == 'done':
                break

            try:
                time_28, time_63 = map(float, user_input.split(','))
            except ValueError:
                print("Invalid input. Please enter two numeric values separated by a comma.")
                continue

            # Calculate the change in output and input over the test interval
            delta_y = data[output].iloc[-1] - data[output].iloc[0]
            delta_u = data[input_col].iloc[-1] - data[input_col].iloc[0]
            kp = delta_y / delta_u if delta_u != 0 else np.nan

            # Determine the 28% and 63% response levels
            y_28 = data[output].iloc[0] + 0.28 * delta_y
            y_63 = data[output].iloc[0] + 0.63 * delta_y

            # Plot the response along with horizontal lines at y_28 and y_63
            plt.figure(figsize=(8, 6))
            plt.plot(time_plot, data[output], label=f'Response of {output}')
            plt.hlines(y=y_28, xmin=time_plot[0], xmax=time_plot[-1],
                       colors="red", label="y_28 (28% response)")
            plt.hlines(y=y_63, xmin=time_plot[0], xmax=time_plot[-1],
                       colors="yellow", label="y_63 (63% response)")
            plt.scatter([time_28], [y_28], color="red", zorder=5)
            plt.scatter([time_63], [y_63], color="yellow", zorder=5)
            plt.xlabel("Time (hour)")
            plt.ylabel(output)
            plt.title(f"Time Response for {output}")
            plt.legend()
            plt.xlim([0, time_plot[limit]])
            plt.show()

            # Confirm with the user if the chosen times are correct.
            correct_input = input("Are the times correct? Type 'yes' to confirm, or 'no' to re-enter values: ")
            if correct_input.lower() == 'yes':
                # Optionally, convert days to hours:
                time_63_hours = time_63
                time_28_hours = time_28
                delta_t = time_63_hours - time_28_hours
                taup = 1.5 * delta_t
                theta = time_28_hours - delta_t  # adjusted formula for theta
                transfer_functions_dict[output] = {
                    "Time 63 (hrs)": time_63_hours,
                    "Time 28 (hrs)": time_28_hours,
                    "kp": kp,
                    "taup": taup,
                    "theta": theta
                }
                break

    print(f'\nTransfer Function details for input mode "{input_col}":')
    for key, value in transfer_functions_dict.items():
        print(f'{key}: {value}')

    return transfer_functions_dict, data


def state_space_form_using_matlab(u1_dict, u2_dict, delay_list, data_u1, data_u2, sampling_time=0.5):
    # Start MATLAB engine
    input_name1, input_name2 = data_u1.columns[0], data_u1.columns[1]
    output_name1, output_name2 = data_u1.columns[2], data_u1.columns[3]
    delta_u1 = data_u1[input_name1].iloc[1] - data_u1[input_name1].iloc[0]
    delta_u2 = data_u2[input_name2].iloc[1] - data_u2[input_name2].iloc[0]
    delta_u = [delta_u1, delta_u2]
    end_time = (data_u1.shape[0]-1) * sampling_time
    eng = matlab.engine.start_matlab()
    # Create num, den, and delay variables
    num = (
        f'num = {{{u1_dict[output_name1]["kp"]}, {u2_dict[output_name1]["kp"]}; '
        f'{u1_dict[output_name2]["kp"]}, {u2_dict[output_name2]["kp"]}}};')
    den = (
        f'den = {{[{u1_dict[output_name1]["taup"]}, 1], [{u2_dict[output_name1]["taup"]}, 1]; '
        f'[{u1_dict[output_name2]["taup"]}, 1], [{u2_dict[output_name2]["taup"]}, 1]}};')
    delay = f'delay = [{delay_list[0]}, {delay_list[1]}; {delay_list[2]}, {delay_list[3]}];'

    eng.eval(num, nargout=0)
    eng.eval(den, nargout=0)
    eng.eval(delay, nargout=0)
    eng.eval("tf_system = tf(num, den, 'IODelay', delay, 'TimeUnit', 'hours');", nargout=0)

    # Convert the transfer function to a state-space model
    eng.eval("ss_system = ss(tf_system);", nargout=0)

    # Discretize the state-space model with the specified sampling time
    Ts = sampling_time  # Sampling time in hours
    eng.workspace['end_time'] = end_time
    eng.workspace['Ts'] = Ts
    eng.eval("mimo_ss_dis = c2d(ss_system, Ts);", nargout=0)

    # Set up options for the step response
    eng.eval("opt = stepDataOptions('InputOffset', 0, 'StepAmplitude', " + str(delta_u) + ");", nargout=0)

    # Define the time vector for simulation
    eng.eval("t = 0:Ts:end_time;", nargout=0)

    # Absorb the delay into the discretized state-space model
    eng.eval("mimo_ss_dis_ab_delay = absorbDelay(mimo_ss_dis);", nargout=0)

    # Optionally, simulate or visualize the response
    eng.eval("step(mimo_ss_dis_ab_delay, t, opt);", nargout=0)

    # Run step response and fetch results
    eng.eval("t = 0:Ts:end_time;", nargout=0)
    eng.eval("opt = stepDataOptions('InputOffset', 0, 'StepAmplitude', " + str(delta_u) + ");", nargout=0)
    eng.eval("mimo_ss_dis_ab_delay = absorbDelay(mimo_ss_dis);", nargout=0)
    eng.eval("[y_dis_model_ss_ab_delay, tOut] = step(mimo_ss_dis_ab_delay, t, opt);", nargout=0)

    # Retrieve data
    y_dis_model_ss_ab_delay = eng.workspace['y_dis_model_ss_ab_delay']
    tOut = eng.workspace['tOut']

    # Convert MATLAB matrix to numpy array
    y_dis_model_ss_ab_delay = np.array(y_dis_model_ss_ab_delay)
    tOut = np.array(tOut)

    A_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.A", nargout=1))
    B_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.B", nargout=1))
    C_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.C", nargout=1))
    D_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.D", nargout=1))

    return A_matrix, B_matrix, C_matrix, D_matrix, y_dis_model_ss_ab_delay, tOut


def plot_results_statespace(tOut, y_dis_model_ss_ab_delay, data_u1, data_u2):
    input_name1, input_name2 = data_u1.columns[0], data_u1.columns[1]
    output_name1, output_name2 = data_u1.columns[2], data_u1.columns[3]

    y1_actual_ref_dev = data_u1[output_name1]
    y2_actual_ref_dev = data_u1[output_name2]

    y1_actual_reb_dev = data_u2[output_name1]
    y2_actual_reb_dev = data_u2[output_name2]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a figure with a 2x2 grid of axes

    # Plot for Tray 24 Composition from Reflux
    axs[0, 0].plot(tOut, y1_actual_ref_dev, 'r-', linewidth=2, label="Actual system response")
    axs[0, 0].plot(tOut, y_dis_model_ss_ab_delay[:, :, 0][:, 0], 'b--', linewidth=2, label="Step response")
    axs[0, 0].set(xlabel='Time (hr)', ylabel=output_name1, title=f'Step in {input_name1}')
    axs[0, 0].tick_params(direction='in', length=6, width=1)

    # Plot for Tray 85 Temperature from Reflux
    axs[1, 0].plot(tOut, y2_actual_ref_dev, 'r-', linewidth=2, label="Actual system response")
    axs[1, 0].plot(tOut, y_dis_model_ss_ab_delay[:, :, 0][:, 1], 'b--', linewidth=2, label="Step response")
    axs[1, 0].set(xlabel='Time (hr)', ylabel=output_name2)

    # Plot for Tray 24 Composition from Reboiler
    axs[0, 1].plot(tOut, y1_actual_reb_dev, 'r-', linewidth=2, label="Actual system response")
    axs[0, 1].plot(tOut, y_dis_model_ss_ab_delay[:, :, 1][:, 0], 'b--', linewidth=2, label="Step response")
    axs[0, 1].set(xlabel='Time (hr)', ylabel=output_name1, title=f'Step in {input_name2}')

    # Plot for Tray 85 Temperature from Reboiler
    axs[1, 1].plot(tOut, y2_actual_reb_dev, 'r-', linewidth=2, label="Actual system response")
    axs[1, 1].plot(tOut, y_dis_model_ss_ab_delay[:, :, 1][:, 1], 'b--', linewidth=2, label="Step response")
    axs[1, 1].set(xlabel='Time (hr)', ylabel=output_name2)

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    plt.show()
