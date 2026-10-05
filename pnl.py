from __future__ import annotations

import math
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

from factors import FACTOR_NAMES
from history import build_factor_history
from sensitivities import SENSITIVITY_VECTOR


def black_scholes_price(spot: float, strike: float, rate: float, vol: float, tau: float) -> float:
    if tau <= 0.0 or vol <= 0.0:
        return max(spot - strike, 0.0)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    normal = NormalDist()
    return spot * normal.cdf(d1) - strike * math.exp(-rate * tau) * normal.cdf(d2)


def compute_rtpl(shocks: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    return (shocks.T @ sensitivity).astype(float)


def compute_hpl(levels: np.ndarray, shocks: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    rtpl = compute_rtpl(shocks, sensitivity)
    eq_idx = FACTOR_NAMES.index("EQ_SPX")
    vol_90_idx = FACTOR_NAMES.index("VOL_SPX_90")
    vol_100_idx = FACTOR_NAMES.index("VOL_SPX_100")

    spot_level = levels[eq_idx, :]
    vol_level = levels[vol_100_idx, :]
    risk_free = 0.02
    tau = 0.25

    shocked_spot = spot_level * (1.0 + shocks[eq_idx, :])
    shocked_vol = vol_level * (1.0 + shocks[vol_100_idx, :])
    base_option = np.array(
        [black_scholes_price(s, s, risk_free, v, tau) for s, v in zip(spot_level, vol_level)],
        dtype=float,
    )
    shocked_option = np.array(
        [black_scholes_price(s, s, risk_free, v, tau) for s, v in zip(shocked_spot, shocked_vol)],
        dtype=float,
    )
    option_reval = shocked_option - base_option

    linear_option = (
        sensitivity[eq_idx] * shocks[eq_idx, :] +
        sensitivity[vol_90_idx] * shocks[vol_90_idx, :] +
        sensitivity[vol_100_idx] * shocks[vol_100_idx, :]
    )

    return rtpl + option_reval - linear_option


def plot_overlay(rtpl: np.ndarray, hpl: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(len(rtpl)), rtpl, label="RTPL", linewidth=1.6, alpha=0.9)
    ax.plot(np.arange(len(hpl)), hpl, label="HPL", linewidth=1.6, alpha=0.9)
    ax.set_title("RTPL vs HPL overlay")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("PnL")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("rtpl_hpl_overlay.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    levels, shocks, _ = build_factor_history()
    rtpl = compute_rtpl(shocks, SENSITIVITY_VECTOR)
    hpl = compute_hpl(levels, shocks, SENSITIVITY_VECTOR)

    assert rtpl.shape == hpl.shape, f"PnL vectors differ in length: {rtpl.shape} vs {hpl.shape}"
    assert np.isfinite(rtpl).all(), "RTPL contains non-finite values"
    assert np.isfinite(hpl).all(), "HPL contains non-finite values"

    plot_overlay(rtpl, hpl)
    print(f"RTPL length: {len(rtpl)}")
    print(f"HPL length: {len(hpl)}")
    print(f"RTPL mean: {rtpl.mean():,.2f}")
    print(f"HPL mean: {hpl.mean():,.2f}")
    print(f"Max abs gap: {np.max(np.abs(hpl - rtpl)):,.2f}")
    print("Plot saved to rtpl_hpl_overlay.png")
