"""
Disturbance-related helpers used by disturbance-training notebooks.
"""

import numpy as np
import pandas as pd


def generate_disturbance_sequence(total_steps: int, nominal: float) -> np.ndarray:
    """
    Simple ramp disturbance from nominal to 154000 over total_steps.
    """
    return np.linspace(nominal, 154000, total_steps)


def preprocess_inputs(path, eth_setpoints_bnds, mode="Cascade"):
    """
    Clean and augment raw CSV inputs; assign random ETH setpoints in blocks.
    """
    input_df = pd.read_csv(path, low_memory=False)
    columns_to_drop = [
        "Time step ",
        "time",
        "Unnamed: 13",
        "Unnamed: 14",
        "Propylene(C3H6)",
        "Propane (C3H8)",
        "Methane (CH4)",
    ]
    input_df.drop(columns_to_drop, axis=1, inplace=True)
    input_df.replace("#DIV/0!", np.nan, inplace=True)

    column_names = list(input_df.columns)
    column_names[-1] = 'STREAMS("HXOUT").T'
    column_names[-2] = "TC.SPRemote"
    column_names[-3] = "EAC.SPRemote"
    input_df = pd.DataFrame(input_df.values[1:, :].astype(np.float32), columns=column_names)
    input_df.dropna(inplace=True)
    column_to_move = "TC.SPRemote"
    new_order = [col for col in input_df.columns if col != column_to_move] + [column_to_move]
    input_df = input_df[new_order]
    input_df.iloc[:, 4] = 1.0 - input_df.iloc[:, 3]

    if "ETH.SPRemote" not in input_df.columns:
        input_df["ETH.SPRemote"] = np.nan

    group_size = 400
    for start in range(0, input_df.shape[0], group_size):
        end = start + group_size
        random_value = np.random.uniform(eth_setpoints_bnds[0], eth_setpoints_bnds[1])
        input_df.loc[start:end, "ETH.SPRemote"] = random_value

    if mode == "Cascade":
        input_df.drop(["TC.SPRemote"], axis=1, inplace=True)
    else:
        input_df.drop(["ETH.SPRemote"], axis=1, inplace=True)
    return input_df

