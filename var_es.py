from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

from factors import FACTOR_NAMES
from history import build_factor_history
from pnl import _sensitivity_unit_shocks, compute_hpl, compute_rtpl
from sensitivities import SENSITIVITY_VECTOR

OUTPUT_PATH = Path("pnl_output.npz")
PLOT_PATH = Path("var_es_backtest.png")
LAMBDA = 0.94
SEED_OBSERVATIONS = 100
ROLLING_WINDOW = 250
EXPECTED_EXCEPTION_RATE = 0.01
MONTE_CARLO_DRAWS = 10_000
MONTE_CARLO_SEED = 42


def _assert_positive_loss(value: float, label: str) -> float:
    assert math.isfinite(value) and value > 0.0, (
        f"{label} must be a positive loss number; got {value}."
    )
    return value


def historical_var_es(rtpl: np.ndarray) -> tuple[float, float]:
    # np.percentile's default linear interpolation is intentional: for 779
    # observations, the 1% point lies at rank 7.79, so this is a methodology choice.
    var_99 = -float(np.percentile(rtpl, 1))
    es_threshold = float(np.percentile(rtpl, 2.5))
    tail = rtpl[rtpl < es_threshold]
    if tail.size == 0:
        raise ValueError("No observations fall below the 2.5th-percentile ES threshold.")
    es_97_5 = -float(np.mean(tail))
    return (
        _assert_positive_loss(var_99, "Historical VaR_99"),
        _assert_positive_loss(es_97_5, "Historical ES_97.5"),
    )


def ewma_covariance(
    unit_shocks: np.ndarray,
    decay: float = LAMBDA,
    seed_observations: int = SEED_OBSERVATIONS,
) -> np.ndarray:
    if unit_shocks.ndim != 2:
        raise ValueError("Unit shocks must have shape (factors, observations).")
    if unit_shocks.shape[1] < seed_observations:
        raise ValueError(
            f"EWMA requires at least {seed_observations} observations; "
            f"got {unit_shocks.shape[1]}."
        )
    if not 0.0 < decay < 1.0:
        raise ValueError(f"EWMA decay must be between 0 and 1; got {decay}.")
    if not np.isfinite(unit_shocks).all():
        raise ValueError("Unit shocks contain non-finite values.")

    sigma = np.asarray(np.cov(unit_shocks[:, :seed_observations], ddof=1), dtype=float)
    for observation in unit_shocks[:, seed_observations:].T:
        sigma = decay * sigma + (1.0 - decay) * np.outer(observation, observation)
    return (sigma + sigma.T) / 2.0


def portfolio_sigma(covariance: np.ndarray, sensitivity: np.ndarray) -> float:
    variance = float(sensitivity @ covariance @ sensitivity)
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"Portfolio variance must be finite and positive; got {variance}.")
    return math.sqrt(variance)


def parametric_var_es(sigma_portfolio: float) -> tuple[float, float]:
    var_99 = 2.326 * sigma_portfolio
    es_97_5 = 2.338 * sigma_portfolio
    return (
        _assert_positive_loss(var_99, "Parametric VaR_99"),
        _assert_positive_loss(es_97_5, "Parametric ES_97.5"),
    )


def monte_carlo_var_es(
    covariance: np.ndarray,
    sensitivity: np.ndarray,
    draws: int = MONTE_CARLO_DRAWS,
    seed: int = MONTE_CARLO_SEED,
) -> tuple[float, float]:
    if draws <= 0:
        raise ValueError(f"Monte Carlo draw count must be positive; got {draws}.")
    cholesky = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)
    factor_draws = rng.standard_normal((draws, covariance.shape[0])) @ cholesky.T
    portfolio_draws = factor_draws @ sensitivity
    var_99 = -float(np.percentile(portfolio_draws, 1))
    es_threshold = float(np.percentile(portfolio_draws, 2.5))
    tail = portfolio_draws[portfolio_draws < es_threshold]
    if tail.size == 0:
        raise ValueError("No Monte Carlo observations fall below the ES threshold.")
    es_97_5 = -float(np.mean(tail))
    return (
        _assert_positive_loss(var_99, "Monte Carlo VaR_99"),
        _assert_positive_loss(es_97_5, "Monte Carlo ES_97.5"),
    )


def monte_carlo_standard_errors(sigma_portfolio: float) -> tuple[float, float]:
    normal = NormalDist()
    var_alpha = 0.01
    var_quantile = normal.inv_cdf(var_alpha)
    var_density = math.exp(-0.5 * var_quantile**2) / math.sqrt(2.0 * math.pi)
    var_se = sigma_portfolio * math.sqrt(
        var_alpha * (1.0 - var_alpha) / MONTE_CARLO_DRAWS
    ) / var_density

    es_alpha = 0.025
    es_quantile = normal.inv_cdf(es_alpha)
    es_density = math.exp(-0.5 * es_quantile**2) / math.sqrt(2.0 * math.pi)
    influence_second_moment = (
        es_alpha + es_quantile * es_density + es_quantile**2 * es_alpha
    )
    influence_mean = -es_density - es_quantile * es_alpha
    es_se = sigma_portfolio * math.sqrt(
        (influence_second_moment - influence_mean**2)
        / (es_alpha**2 * MONTE_CARLO_DRAWS)
    )
    return var_se, es_se


def kupiec_pof(exceptions: int, observations: int) -> float:
    if observations <= 0 or not 0 <= exceptions <= observations:
        raise ValueError(
            f"Invalid exception count {exceptions} for {observations} observations."
        )
    observed_rate = exceptions / observations
    null_log_likelihood = (
        (observations - exceptions) * math.log1p(-EXPECTED_EXCEPTION_RATE)
        + exceptions * math.log(EXPECTED_EXCEPTION_RATE)
    )
    if exceptions == 0:
        alternative_log_likelihood = 0.0
    elif exceptions == observations:
        alternative_log_likelihood = 0.0
    else:
        alternative_log_likelihood = (
            (observations - exceptions) * math.log1p(-observed_rate)
            + exceptions * math.log(observed_rate)
        )
    likelihood_ratio = max(
        0.0, -2.0 * (null_log_likelihood - alternative_log_likelihood)
    )
    return math.erfc(math.sqrt(likelihood_ratio / 2.0))


def rolling_backtest(
    rtpl: np.ndarray,
    unit_shocks: np.ndarray,
    sensitivity: np.ndarray,
    window: int = ROLLING_WINDOW,
) -> dict[str, dict[str, np.ndarray | int | float | str]]:
    if len(rtpl) != unit_shocks.shape[1]:
        raise ValueError("RTPL and factor shock observations must be aligned.")
    if len(rtpl) <= window:
        raise ValueError(f"Backtesting requires more than {window} observations.")

    hs_var = np.full(len(rtpl), np.nan, dtype=float)
    ewma_var = np.full(len(rtpl), np.nan, dtype=float)
    for t in range(window, len(rtpl)):
        hs_var[t] = _assert_positive_loss(
            -float(np.percentile(rtpl[t - window : t], 1)),
            f"Rolling HS VaR_99 at observation {t}",
        )
        covariance = ewma_covariance(unit_shocks[:, t - window : t])
        ewma_var[t] = _assert_positive_loss(
            2.326 * portfolio_sigma(covariance, sensitivity),
            f"Rolling EWMA VaR_99 at observation {t}",
        )

    test_start = window
    testable = len(rtpl) - test_start
    results: dict[str, dict[str, np.ndarray | int | float | str]] = {}
    for name, forecasts in (("HS", hs_var), ("EWMA Parametric", ewma_var)):
        test_forecasts = forecasts[test_start:]
        test_pnl = rtpl[test_start:]
        exception_mask = test_pnl < -test_forecasts
        exception_count = int(np.count_nonzero(exception_mask))
        recent_count = min(window, testable)
        recent_exceptions = int(np.count_nonzero(exception_mask[-recent_count:]))
        if recent_exceptions <= 4:
            zone = "Green"
        elif recent_exceptions <= 9:
            zone = "Amber"
        else:
            zone = "Red"
        results[name] = {
            "forecasts": forecasts,
            "exception_mask": exception_mask,
            "exception_count": exception_count,
            "exception_rate": exception_count / testable,
            "kupiec_p_value": kupiec_pof(exception_count, testable),
            "basel_zone": f"{zone} ({recent_exceptions}/{recent_count})",
            "testable_days": testable,
        }
    return results


def load_or_build_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if OUTPUT_PATH.exists():
        with np.load(OUTPUT_PATH, allow_pickle=False) as data:
            required = {"rtpl", "hpl", "shocks", "levels"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{OUTPUT_PATH} is missing required arrays: {sorted(missing)}")
            rtpl = np.asarray(data["rtpl"], dtype=float)
            hpl = np.asarray(data["hpl"], dtype=float)
            shocks = np.asarray(data["shocks"], dtype=float)
            levels = np.asarray(data["levels"], dtype=float)
            return rtpl, hpl, shocks, levels

    levels, shocks, _ = build_factor_history(print_diagnostics=False)
    rtpl = compute_rtpl(shocks, SENSITIVITY_VECTOR, levels)
    hpl = compute_hpl(levels, shocks, SENSITIVITY_VECTOR)
    if rtpl.shape != hpl.shape or rtpl.shape[0] != shocks.shape[1]:
        raise ValueError("Rebuilt P&L and factor-history arrays are not aligned.")
    np.savez(OUTPUT_PATH, rtpl=rtpl, hpl=hpl, shocks=shocks, levels=levels)
    return rtpl, hpl, shocks, levels


def plot_backtest(
    rtpl: np.ndarray,
    backtests: dict[str, dict[str, np.ndarray | int | float | str]],
) -> None:
    indices = np.arange(len(rtpl))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(indices, rtpl, label="RTPL", color="steelblue", linewidth=1.0, alpha=0.8)
    colors = {"HS": "darkorange", "EWMA Parametric": "seagreen"}
    markers = {"HS": "o", "EWMA Parametric": "x"}
    for name, result in backtests.items():
        forecasts = np.asarray(result["forecasts"])
        exception_mask = np.asarray(result["exception_mask"], dtype=bool)
        test_indices = indices[ROLLING_WINDOW:]
        ax.plot(
            test_indices,
            -forecasts[ROLLING_WINDOW:],
            label=f"{name} VaR_99 threshold",
            color=colors[name],
            linewidth=1.2,
        )
        exception_indices = test_indices[exception_mask]
        ax.scatter(
            exception_indices,
            rtpl[exception_indices],
            marker=markers[name],
            color=colors[name],
            s=24,
            label=f"{name} exceptions",
            zorder=3,
        )
    ax.set_title("RTPL with 250-day rolling 99% VaR backtest")
    ax.set_xlabel("Observation")
    ax.set_ylabel("P&L / VaR loss threshold")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def _print_summary(
    hs_metrics: tuple[float, float],
    parametric_metrics: tuple[float, float],
    mc_metrics: tuple[float, float],
    backtests: dict[str, dict[str, np.ndarray | int | float | str]],
) -> None:
    rows = []
    for name, metrics in (
        ("Historical Simulation", hs_metrics),
        ("EWMA Parametric", parametric_metrics),
        ("Monte Carlo control", mc_metrics),
    ):
        backtest = backtests.get("HS" if name == "Historical Simulation" else name)
        if backtest is None:
            backtest_columns = ("—", "—", "—", "Not backtested")
        else:
            backtest_columns = (
                f"{backtest['exception_count']}/{backtest['testable_days']}",
                f"{100 * float(backtest['exception_rate']):.2f}%",
                f"{float(backtest['kupiec_p_value']):.4f}",
                str(backtest["basel_zone"]),
            )
        rows.append(
            (
                name,
                f"{metrics[0]:,.2f}",
                f"{metrics[1]:,.2f}",
                *backtest_columns,
            )
        )

    headers = (
        "Method",
        "VaR_99",
        "ES_97.5",
        "Exceptions",
        "Exception rate",
        "Kupiec p-value",
        "Basel zone",
    )
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    testable_days = int(backtests["HS"]["testable_days"])
    print(
        f"Expected exceptions: {EXPECTED_EXCEPTION_RATE:.0%} of {testable_days} "
        f"testable days = {EXPECTED_EXCEPTION_RATE * testable_days:.2f}."
    )


def main() -> None:
    rtpl, _, shocks, levels = load_or_build_inputs()
    if rtpl.ndim != 1 or shocks.shape != levels.shape:
        raise ValueError("Cached RTPL, shocks, and levels have invalid dimensions.")
    if shocks.shape[0] != len(FACTOR_NAMES) or shocks.shape[1] != len(rtpl):
        raise ValueError(
            f"Expected aligned ({len(FACTOR_NAMES)}, {len(rtpl)}) factor matrices; "
            f"got shocks={shocks.shape}, levels={levels.shape}."
        )
    if not np.isfinite(rtpl).all():
        raise ValueError("RTPL contains non-finite values.")

    unit_shocks = _sensitivity_unit_shocks(shocks, levels)
    covariance = ewma_covariance(unit_shocks)
    sigma_portfolio = portfolio_sigma(covariance, SENSITIVITY_VECTOR)
    hs_metrics = historical_var_es(rtpl)
    parametric_metrics = parametric_var_es(sigma_portfolio)
    mc_metrics = monte_carlo_var_es(covariance, SENSITIVITY_VECTOR)
    mc_errors = monte_carlo_standard_errors(sigma_portfolio)
    backtests = rolling_backtest(rtpl, unit_shocks, SENSITIVITY_VECTOR)

    _print_summary(hs_metrics, parametric_metrics, mc_metrics, backtests)
    print(
        f"EWMA covariance shape: {covariance.shape}; lambda={LAMBDA:.2f}; "
        "seed=first 100 observations."
    )
    print(
        "Monte Carlo control deltas vs parametric: "
        f"VaR={mc_metrics[0] - parametric_metrics[0]:+,.2f}, "
        f"ES={mc_metrics[1] - parametric_metrics[1]:+,.2f}; "
        f"approx. standard errors: VaR={mc_errors[0]:,.2f}, ES={mc_errors[1]:,.2f}."
    )
    if any(
        abs(mc - parametric) > 3.0 * standard_error
        for mc, parametric, standard_error in zip(
            mc_metrics, parametric_metrics, mc_errors
        )
    ):
        print(
            "FLAG: Monte Carlo differs from the parametric result by more than "
            "three Monte Carlo standard errors; inspect the covariance/simulation."
        )

    hs_var_gap = hs_metrics[0] - parametric_metrics[0]
    hs_es_gap = hs_metrics[1] - parametric_metrics[1]
    if hs_var_gap <= 0.0:
        print(
            "FLAG: HS VaR_99 does not exceed EWMA parametric VaR_99; review EWMA "
            "seeding and the P&L/VaR sign convention before accepting the result."
        )
    if hs_var_gap > 0.0 and hs_es_gap / parametric_metrics[1] <= hs_var_gap / parametric_metrics[0]:
        print("FLAG: The proportional ES gap is not larger than the proportional VaR gap.")
    if int(backtests["EWMA Parametric"]["exception_count"]) <= int(
        backtests["HS"]["exception_count"]
    ):
        print("FLAG: EWMA parametric does not produce more exceptions than HS.")
    plot_backtest(rtpl, backtests)
    print(f"Plot saved to {PLOT_PATH}.")


if __name__ == "__main__":
    main()
