"""
Reward function builders shared across training scripts.
"""

import numpy as np


def make_reward_fn_quadratic(Q1_penalty, Q2_penalty, R1_penalty, R2_penalty):
    """
    Classic quadratic reward: -(Q1*dy1^2 + Q2*dy2^2 + R1*du1^2 + R2*du2^2).
    """

    def reward_fn(delta_y, delta_u, *_args, **_kwargs):
        return -(
            Q1_penalty * float(delta_y[0]) ** 2
            + Q2_penalty * float(delta_y[1]) ** 2
            + R1_penalty * float(delta_u[0]) ** 2
            + R2_penalty * float(delta_u[1]) ** 2
        )

    return reward_fn


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _phi(z, kind="exp", k=8.0, p=0.6, c=20.0):
    z = np.clip(z, 0.0, 1.0)
    if kind == "linear":
        return 1.0 - z
    if kind == "quadratic":
        return (1.0 - z) ** 2
    if kind == "exp":
        return (np.exp(-k * z) - np.exp(-k)) / (1.0 - np.exp(-k))
    if kind == "power":
        return 1.0 - np.power(z, p)
    if kind == "log":
        return np.log1p(c * (1.0 - z)) / np.log1p(c)
    raise ValueError("unknown bonus kind")


def make_reward_fn_relative_QR(
    data_min,
    data_max,
    n_inputs,
    k_rel,
    band_floor_phys,
    Q_diag,
    R_diag,
    tau_frac=0.7,
    gamma_out=0.5,
    gamma_in=0.5,
    beta=5.0,
    gate="geom",
    lam_in=1.0,
    bonus_kind="exp",
    bonus_k=12.0,
    bonus_p=0.6,
    bonus_c=20.0,
):
    """
    Relative-band reward (from OnlineTrainingNoDist1* notebooks).
    Band scales with |setpoint| with a floor in physical units.
    """
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)
    band_floor_scaled = band_floor_phys / np.maximum(dy, 1e-12)

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e_scaled = np.asarray(e_scaled, float)
        du_scaled = np.asarray(du_scaled, float)

        if y_sp_phys is None:
            band_scaled = band_floor_scaled
        else:
            y_sp_phys_arr = np.asarray(y_sp_phys, float)
            band_phys = np.maximum(k_rel * np.abs(y_sp_phys_arr), band_floor_phys)
            band_scaled = band_phys / np.maximum(dy, 1e-12)

        tau_scaled = tau_frac * band_scaled

        abs_e = np.abs(e_scaled)
        s_i = _sigmoid((band_scaled - abs_e) / np.maximum(tau_scaled, 1e-12))
        if gate == "prod":
            w_in = float(np.prod(s_i, dtype=np.float64))
        elif gate == "mean":
            w_in = float(np.mean(s_i))
        elif gate == "geom":
            w_in = float(np.prod(s_i, dtype=np.float64) ** (1.0 / len(s_i)))
        else:
            raise ValueError("gate must be 'prod'|'mean'|'geom'")

        err_quad = np.sum(Q_diag * (e_scaled ** 2))
        err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)
        move = np.sum(R_diag * (du_scaled ** 2))

        slope_at_edge = 2.0 * Q_diag * band_scaled
        overflow = np.maximum(abs_e - band_scaled, 0.0)
        lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow)
        inside_mag = np.minimum(abs_e, band_scaled)
        lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag)

        qb2 = Q_diag * (band_scaled ** 2)
        z = abs_e / np.maximum(band_scaled, 1e-12)
        phi_val = _phi(z, kind=bonus_kind, k=bonus_k, p=bonus_p, c=bonus_c)
        bonus = w_in * beta * np.sum(qb2 * phi_val)

        return -(err_eff + move + lin_out + lin_in) + bonus

    params = dict(
        k_rel=k_rel,
        band_floor_phys=band_floor_phys,
        band_floor_scaled=band_floor_scaled,
        Q_diag=Q_diag,
        R_diag=R_diag,
        tau_frac=tau_frac,
        gamma_out=gamma_out,
        gamma_in=gamma_in,
        beta=beta,
        gate=gate,
        lam_in=lam_in,
        bonus_kind=bonus_kind,
        bonus_k=bonus_k,
        bonus_p=bonus_p,
        bonus_c=bonus_c,
    )
    return params, reward_fn
