from __future__ import annotations

import math
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors import FACTOR_NAMES
from history import build_factor_history
from sensitivities import SENSITIVITY_VECTOR

SPX_CALL_NOTIONAL = 75_000_000.0
SPX_CALL_TAU = 1.0 / 12.0


def black_scholes_price(spot: float, strike: float, rate: float, vol: float, tau: float) -> float:
    if tau <= 0.0 or vol <= 0.0:
        return max(spot - strike, 0.0)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    normal = NormalDist()
    return spot * normal.cdf(d1) - strike * math.exp(-rate * tau) * normal.cdf(d2)


def _sensitivity_unit_shocks(shocks: np.ndarray, levels: np.ndarray) -> np.ndarray:
    unit_shocks = shocks.copy()
    for factor_name in ("EQ_SPX", "FX_GBPUSD"):
        factor_idx = FACTOR_NAMES.index(factor_name)
        unit_shocks[factor_idx] *= 100.0

    for factor_name in ("VOL_SPX_90", "VOL_SPX_100"):
        factor_idx = FACTOR_NAMES.index(factor_name)
        current_vol = levels[factor_idx]
        relative_shock = shocks[factor_idx]
        if np.any(1.0 + relative_shock <= 0.0):
            raise ValueError(f"{factor_name} relative shock implies a non-positive volatility level.")
        previous_vol = current_vol / (1.0 + relative_shock)
        unit_shocks[factor_idx] = (current_vol - previous_vol) * 100.0
    return unit_shocks


def compute_rtpl(
    shocks: np.ndarray,
    sensitivity: np.ndarray,
    levels: np.ndarray | None = None,
) -> np.ndarray:
    if levels is None:
        return (shocks.T @ sensitivity).astype(float)
    unit_shocks = _sensitivity_unit_shocks(shocks, levels)
    return (unit_shocks.T @ sensitivity).astype(float)


def compute_hpl(levels: np.ndarray, shocks: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    rtpl = compute_rtpl(shocks, sensitivity, levels)
    eq_idx = FACTOR_NAMES.index("EQ_SPX")
    vol_100_idx = FACTOR_NAMES.index("VOL_SPX_100")

    spot_level = levels[eq_idx]
    vol_level = levels[vol_100_idx]
    previous_spot = spot_level / (1.0 + shocks[eq_idx])
    previous_vol = vol_level / (1.0 + shocks[vol_100_idx])
    risk_free = 0.02
    tau = SPX_CALL_TAU
    shocked_tau = max(tau - 1.0 / 252.0, 0.0)
    if np.any(previous_spot <= 0.0) or np.any(previous_vol <= 0.0):
        raise ValueError("Option revaluation requires positive prior spot and volatility levels.")

    shocked_spot = spot_level
    shocked_vol = vol_level
    # The one-month ATM option is rolled daily: each day's starting strike equals
    # the previous close, then stays fixed during that day's shocked revaluation.
    base_option = np.array(
        [black_scholes_price(s, s, risk_free, v, tau) for s, v in zip(previous_spot, previous_vol)],
        dtype=float,
    )
    shocked_option = np.array(
        [
            black_scholes_price(s, k, risk_free, v, shocked_tau)
            for s, k, v in zip(shocked_spot, previous_spot, shocked_vol)
        ],
        dtype=float,
    )
    option_reval = SPX_CALL_NOTIONAL / previous_spot * (shocked_option - base_option)

    unit_shocks = _sensitivity_unit_shocks(shocks, levels)
    linear_option = (
        sensitivity[eq_idx] * unit_shocks[eq_idx]
        + sensitivity[vol_100_idx] * unit_shocks[vol_100_idx]
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
    levels, shocks, dates = build_factor_history(print_diagnostics=False)
    rtpl = compute_rtpl(shocks, SENSITIVITY_VECTOR, levels)
    hpl = compute_hpl(levels, shocks, SENSITIVITY_VECTOR)

    assert rtpl.shape == hpl.shape, f"PnL vectors differ in length: {rtpl.shape} vs {hpl.shape}"
    assert np.isfinite(rtpl).all(), "RTPL contains non-finite values"
    assert np.isfinite(hpl).all(), "HPL contains non-finite values"
    np.savez(
        "pnl_output.npz",
        rtpl=rtpl,
        hpl=hpl,
        shocks=shocks,
        levels=levels,
        dates=dates.values,
    )

    plot_overlay(rtpl, hpl)
    print(f"RTPL length: {len(rtpl)}")
    print(f"HPL length: {len(hpl)}")
    print(f"RTPL mean: {rtpl.mean():,.2f}")
    print(f"HPL mean: {hpl.mean():,.2f}")
    rtpl_std = float(np.std(rtpl, ddof=1))
    max_abs_gap = float(np.max(np.abs(hpl - rtpl)))
    if rtpl_std == 0.0:
        raise ValueError("Cannot scale the maximum absolute PnL gap by a zero RTPL standard deviation.")
    for name, values in (("RTPL", rtpl), ("HPL", hpl)):
        print(
            f"{name} stats: std={np.std(values, ddof=1):,.2f}, "
            f"min={np.min(values):,.2f}, max={np.max(values):,.2f}, "
            f"p01={np.percentile(values, 1):,.2f}, p99={np.percentile(values, 99):,.2f}"
        )
    print(f"RTPL/HPL Pearson correlation: {np.corrcoef(rtpl, hpl)[0, 1]:.4f}")
    print(
        "RTPL/HPL Spearman correlation: "
        f"{pd.Series(rtpl).rank().corr(pd.Series(hpl).rank()):.4f}"
    )
    print(f"Max abs gap: {max_abs_gap:,.2f}")
    print(f"Max abs gap / RTPL std: {max_abs_gap / rtpl_std:.4f}")
    print("Plot saved to rtpl_hpl_overlay.png")
