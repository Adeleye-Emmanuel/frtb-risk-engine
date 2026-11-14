from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from factors import FACTOR_NAMES, FACTORS
from ima import (
    CAPITAL_MULTIPLIER,
    NMRF_FACTORS,
    REDUCED_FACTORS,
    RHO,
    ROLLING_WINDOW,
    _positive_es,
    _rolling_stress_search,
    _variance_explained,
    liquidity_adjusted_es,
)
from pnl import _sensitivity_unit_shocks
from sa import (
    CORRELATION_SCENARIOS,
    RRAO,
    _curvature_charge,
    _drc_charge,
    _load_curvature_levels,
    _delta_charge,
    _vega_charge,
)
from sensitivities import SENSITIVITY_VECTOR
from var_es import historical_var_es, load_or_build_inputs

CACHE_PATH = Path("pnl_output.npz")
PLOT_PATH = Path("pla_diagnostics.png")
RECENT_OBSERVATIONS = 250


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 1 or right.shape != left.shape or left.size < 2:
        raise ValueError("Spearman correlation requires aligned 1D arrays of length >= 2.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite.")
    correlation = float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])
    if not math.isfinite(correlation):
        raise ValueError("Spearman correlation is undefined for constant input.")
    return correlation


def _empirical_ks(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float, float, float]:
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or right.size == 0:
        raise ValueError("KS statistic requires non-empty 1D samples.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("KS samples must be finite.")
    sorted_left = np.sort(left)
    sorted_right = np.sort(right)
    points = np.unique(np.concatenate((sorted_left, sorted_right)))
    left_cdf_values = np.stack(
        [
            np.searchsorted(sorted_left, points, side="left") / sorted_left.size,
            np.searchsorted(sorted_left, points, side="right") / sorted_left.size,
        ]
    )
    right_cdf_values = np.stack(
        [
            np.searchsorted(sorted_right, points, side="left") / sorted_right.size,
            np.searchsorted(sorted_right, points, side="right") / sorted_right.size,
        ]
    )
    distances = np.abs(left_cdf_values - right_cdf_values)
    side, index = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return (
        float(distances[side, index]),
        float(points[index]),
        float(left_cdf_values[side, index]),
        float(right_cdf_values[side, index]),
    )


def _pla_zone(spearman: float, ks_statistic: float) -> str:
    if spearman < 0.70 or ks_statistic > 0.12:
        return "Red"
    if spearman > 0.80 and ks_statistic < 0.09:
        return "Green"
    return "Amber"


def _binding_zone(*zones: str) -> str:
    rank = {"Green": 0, "Amber": 1, "Red": 2}
    return max(zones, key=rank.__getitem__)


def _run_ima_calculations(
    rtpl: np.ndarray,
    unit_shocks: np.ndarray,
) -> dict[str, float]:
    one_day_es = historical_var_es(rtpl)[1]
    full_es, _, _ = liquidity_adjusted_es(unit_shocks, SENSITIVITY_VECTOR)

    variance_explained = _variance_explained(
        unit_shocks, SENSITIVITY_VECTOR, REDUCED_FACTORS, rtpl
    )
    augmented_factors = (*REDUCED_FACTORS, "CSR_SINGLE_5Y")
    augmented_explained = _variance_explained(
        unit_shocks, SENSITIVITY_VECTOR, augmented_factors, rtpl
    )
    selected_factors = (
        REDUCED_FACTORS if variance_explained >= 0.75 else augmented_factors
    )

    _, stressed_start = _rolling_stress_search(
        unit_shocks,
        SENSITIVITY_VECTOR,
        selected_factors,
    )
    stressed_end = stressed_start + ROLLING_WINDOW
    stressed_global_imcc, _, _ = liquidity_adjusted_es(
        unit_shocks,
        SENSITIVITY_VECTOR,
        stressed_start,
        stressed_end,
    )
    class_imccs: dict[str, float] = {}
    for risk_class in dict.fromkeys(factor.risk_class for factor in FACTORS):
        class_factors = tuple(
            factor.name for factor in FACTORS if factor.risk_class == risk_class
        )
        class_imccs[risk_class], _, _ = liquidity_adjusted_es(
            unit_shocks,
            SENSITIVITY_VECTOR,
            stressed_start,
            stressed_end,
            class_factors,
        )
    imcc = RHO * stressed_global_imcc + (1.0 - RHO) * sum(class_imccs.values())

    ses_values = []
    for factor_name in NMRF_FACTORS:
        index = FACTOR_NAMES.index(factor_name)
        if SENSITIVITY_VECTOR[index] == 0.0:
            ses_values.append(0.0)
            continue
        factor_pnl = SENSITIVITY_VECTOR[index] * unit_shocks[
            index, stressed_start:stressed_end
        ]
        ses_values.append(_positive_es(factor_pnl, f"SES for {factor_name}"))
    ses = sum(ses_values)
    capital = max(imcc + ses, CAPITAL_MULTIPLIER * imcc + ses)
    if not math.isfinite(capital) or capital <= 0.0:
        raise ValueError(f"IMA capital must be finite and positive; got {capital}.")

    return {
        "one_day_es": one_day_es,
        "liquidity_adjusted_es": full_es,
        "imcc": imcc,
        "ses": ses,
        "ima_capital": capital,
    }


def _sa_capital() -> float:
    spot, volatility = _load_curvature_levels()
    drc, _, _ = _drc_charge()
    scenario_totals = []
    for scenario in CORRELATION_SCENARIOS:
        delta = _delta_charge(scenario, cdx_index_bucket=True)
        scenario_totals.append(
            sum(delta.values())
            + _vega_charge(scenario)
            + _curvature_charge(spot, volatility, scenario)
            + drc
            + RRAO
        )
    capital = max(scenario_totals)
    if not math.isfinite(capital) or capital <= 0.0:
        raise ValueError(f"SA capital must be finite and positive; got {capital}.")
    return capital


def _print_table(rows: list[tuple[str, str]]) -> None:
    component_width = max(len("Component"), *(len(name) for name, _ in rows))
    value_width = max(len("Value"), *(len(value) for _, value in rows))
    print(f"{'Component':<{component_width}} | {'Value':<{value_width}}")
    print(f"{'-' * component_width}-+-{'-' * value_width}")
    for name, value in rows:
        print(f"{name:<{component_width}} | {value:<{value_width}}")


def _display_date(date: np.datetime64) -> str:
    return str(date)[:10]


def _report_gap(
    rtpl: np.ndarray,
    hpl: np.ndarray,
    shocks: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gap = hpl - rtpl
    eq_shocks = shocks[FACTOR_NAMES.index("EQ_SPX")]
    correlation = float(np.corrcoef(gap, eq_shocks**2)[0, 1])
    print("HPL - RTPL attribution (full 779-observation sample):")
    print(
        f"  mean={np.mean(gap):,.2f}; std={np.std(gap, ddof=1):,.2f}; "
        f"min={np.min(gap):,.2f}; max={np.max(gap):,.2f}"
    )
    print(f"  Correlation with squared EQ_SPX shock: {correlation:.4f}")
    print("  Five largest positive-gap days:")
    print("  Date       | HPL - RTPL | EQ_SPX shock")
    order = np.argsort(gap)[-5:][::-1]
    for index in order:
        print(
            f"  {_display_date(dates[index])} | {gap[index]:>10,.2f} | "
            f"{eq_shocks[index]:>+.6f}"
        )
    return gap, eq_shocks


def _plot_pla(
    hpl: np.ndarray,
    rtpl: np.ndarray,
    dates: np.ndarray,
    recent_hpl: np.ndarray,
    recent_rtpl: np.ndarray,
) -> None:
    recent_gap = recent_hpl - recent_rtpl
    highlight_count = min(5, recent_gap.size)
    highlight = np.argsort(recent_gap)[-highlight_count:]
    fig, (scatter_ax, cdf_ax) = plt.subplots(1, 2, figsize=(14, 6))

    scatter_ax.scatter(
        recent_rtpl,
        recent_hpl,
        s=20,
        alpha=0.55,
        color="steelblue",
        label="Other days",
    )
    scatter_ax.scatter(
        recent_rtpl[highlight],
        recent_hpl[highlight],
        s=54,
        marker="D",
        color="darkorange",
        edgecolor="black",
        linewidth=0.4,
        label="Five largest positive gaps",
        zorder=3,
    )
    lower = float(min(np.min(recent_rtpl), np.min(recent_hpl)))
    upper = float(max(np.max(recent_rtpl), np.max(recent_hpl)))
    scatter_ax.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="45-degree line")
    for index in highlight:
        scatter_ax.annotate(
            _display_date(dates[-RECENT_OBSERVATIONS:][index]),
            (recent_rtpl[index], recent_hpl[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    scatter_ax.set_title("HPL vs RTPL (most recent 250 days)")
    scatter_ax.set_xlabel("RTPL")
    scatter_ax.set_ylabel("HPL")
    scatter_ax.grid(alpha=0.25)
    scatter_ax.legend(fontsize=8)

    sorted_hpl = np.sort(recent_hpl)
    sorted_rtpl = np.sort(recent_rtpl)
    hpl_cdf = np.arange(1, sorted_hpl.size + 1) / sorted_hpl.size
    rtpl_cdf = np.arange(1, sorted_rtpl.size + 1) / sorted_rtpl.size
    cdf_ax.step(sorted_hpl, hpl_cdf, where="post", label="HPL", color="darkorange")
    cdf_ax.step(sorted_rtpl, rtpl_cdf, where="post", label="RTPL", color="steelblue")
    ks, ks_x, hpl_at_ks, rtpl_at_ks = _empirical_ks(recent_hpl, recent_rtpl)
    cdf_ax.vlines(
        ks_x,
        min(hpl_at_ks, rtpl_at_ks),
        max(hpl_at_ks, rtpl_at_ks),
        color="black",
        linewidth=2,
        label=f"KS distance = {ks:.3f}",
    )
    cdf_ax.scatter(
        [ks_x, ks_x],
        [hpl_at_ks, rtpl_at_ks],
        color="black",
        s=20,
        zorder=4,
    )
    cdf_ax.set_title("Empirical CDFs (most recent 250 days)")
    cdf_ax.set_xlabel("P&L")
    cdf_ax.set_ylabel("Empirical CDF")
    cdf_ax.grid(alpha=0.25)
    cdf_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    rtpl, _, shocks, levels = load_or_build_inputs()
    with np.load(CACHE_PATH, allow_pickle=False) as data:
        if "hpl" not in data.files or "dates" not in data.files:
            raise ValueError(f"{CACHE_PATH} must contain hpl and dates arrays.")
        hpl = np.asarray(data["hpl"], dtype=float)
        dates = np.asarray(data["dates"])
    expected_shape = rtpl.shape
    if (
        rtpl.ndim != 1
        or hpl.shape != expected_shape
        or shocks.shape != (len(FACTOR_NAMES), expected_shape[0])
        or levels.shape != shocks.shape
        or dates.shape != expected_shape
    ):
        raise ValueError(
            "Cached RTPL, HPL, dates, shocks, and levels must be aligned; got "
            f"{rtpl.shape}, {hpl.shape}, {dates.shape}, {shocks.shape}, {levels.shape}."
        )
    if expected_shape[0] < RECENT_OBSERVATIONS:
        raise ValueError(
            f"PLA requires at least {RECENT_OBSERVATIONS} observations; "
            f"got {expected_shape[0]}."
        )
    if not all(
        np.isfinite(values).all()
        for values in (rtpl, hpl, shocks, levels)
    ):
        raise ValueError("Cached P&L, shock, and level arrays must be finite.")

    recent_rtpl = rtpl[-RECENT_OBSERVATIONS:]
    recent_hpl = hpl[-RECENT_OBSERVATIONS:]
    recent_spearman = _spearman_correlation(recent_hpl, recent_rtpl)
    recent_ks, _, _, _ = _empirical_ks(recent_hpl, recent_rtpl)
    recent_spearman_zone = _pla_zone(recent_spearman, 0.0)
    recent_ks_zone = _pla_zone(1.0, recent_ks)
    regulatory_zone = _binding_zone(recent_spearman_zone, recent_ks_zone)

    full_spearman = _spearman_correlation(hpl, rtpl)
    full_ks, _, _, _ = _empirical_ks(hpl, rtpl)
    full_spearman_zone = _pla_zone(full_spearman, 0.0)
    full_ks_zone = _pla_zone(1.0, full_ks)

    print("PLA statistics (MAR32.44):")
    print(
        f"  Regulatory window, most recent {RECENT_OBSERVATIONS} observations: "
        f"Spearman={recent_spearman:.4f} ({recent_spearman_zone}); "
        f"KS={recent_ks:.4f} ({recent_ks_zone}); "
        f"binding zone={regulatory_zone}."
    )
    print(
        "  Full-sample context, non-regulatory ("
        f"{len(rtpl)} observations): Spearman={full_spearman:.4f} "
        f"({full_spearman_zone}); KS={full_ks:.4f} ({full_ks_zone}); "
        f"binding zone={_binding_zone(full_spearman_zone, full_ks_zone)}."
    )

    unit_shocks = _sensitivity_unit_shocks(shocks, levels)
    _report_gap(rtpl, hpl, shocks, dates)

    hs_var, hs_es = historical_var_es(rtpl)
    ima_values = _run_ima_calculations(rtpl, unit_shocks)
    sa_capital = _sa_capital()
    output_floor = 0.725 * sa_capital
    binding_capital = max(ima_values["ima_capital"], output_floor)
    rows = [
        ("HS VaR 99 (1-day)", f"{hs_var:,.2f}"),
        ("HS ES 97.5 (1-day)", f"{hs_es:,.2f}"),
        ("Liquidity-adjusted ES", f"{ima_values['liquidity_adjusted_es']:,.2f}"),
        ("IMCC", f"{ima_values['imcc']:,.2f}"),
        ("SES", f"{ima_values['ses']:,.2f}"),
        ("IMA capital", f"{ima_values['ima_capital']:,.2f}"),
        ("SA capital (CDX bucket 17)", f"{sa_capital:,.2f}"),
        ("SA / IMA ratio", f"{sa_capital / ima_values['ima_capital']:.2f}"),
        ("Output floor (72.5% x SA)", f"{output_floor:,.2f}"),
        ("Binding capital", f"{binding_capital:,.2f}"),
    ]
    _print_table(rows)

    _plot_pla(hpl, rtpl, dates, recent_hpl, recent_rtpl)
    print(f"Plots saved to {PLOT_PATH} (scatter and empirical CDF panels).")


if __name__ == "__main__":
    main()
