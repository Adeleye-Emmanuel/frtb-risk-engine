from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from factors import FACTOR_NAMES, FACTORS
from pnl import _sensitivity_unit_shocks
from sensitivities import SENSITIVITY_VECTOR
from var_es import historical_var_es, load_or_build_inputs

INPUT_PATH = Path("pnl_output.npz")
PLOT_PATH = Path("ima_stressed_es.png")
BASE_HORIZON = 10
ROLLING_WINDOW = 250
REDUCED_FACTORS = ("GIRR_5Y", "GIRR_10Y", "CSR_CDX_5Y", "EQ_SPX", "FX_GBPUSD")
NMRF_FACTORS = ("CSR_SINGLE_5Y",)
RHO = 0.5
CAPITAL_MULTIPLIER = 1.5

# Basel Framework MAR21.8, Table 2: liquidity horizon assignments.
LIQUIDITY_HORIZON_BY_FACTOR = {
    "GIRR_1Y": 10,
    "GIRR_2Y": 10,
    "GIRR_5Y": 10,
    "GIRR_10Y": 10,
    "GIRR_30Y": 10,
    "EQ_SPX": 10,
    "FX_GBPUSD": 10,
    "VOL_SPX_90": 20,
    "VOL_SPX_100": 20,
    "CSR_CDX_1Y": 40,
    "CSR_CDX_3Y": 40,
    "CSR_CDX_5Y": 40,
    "CSR_CDX_10Y": 40,
    "CSR_SINGLE_1Y": 60,
    "CSR_SINGLE_3Y": 60,
    "CSR_SINGLE_5Y": 60,
    "CSR_SINGLE_10Y": 60,
}
LIQUIDITY_HORIZONS = tuple(sorted(set(LIQUIDITY_HORIZON_BY_FACTOR.values())))


def _validate_horizon_map() -> None:
    if set(LIQUIDITY_HORIZON_BY_FACTOR) != set(FACTOR_NAMES):
        raise ValueError("Liquidity horizon map does not match the 17-factor ordering.")
    if any(horizon < BASE_HORIZON for horizon in LIQUIDITY_HORIZON_BY_FACTOR.values()):
        raise ValueError(f"Every factor liquidity horizon must be at least {BASE_HORIZON} days.")


def _positive_es(pnl: np.ndarray, label: str) -> float:
    if pnl.ndim != 1 or pnl.size == 0 or not np.isfinite(pnl).all():
        raise ValueError(f"{label} requires a finite, non-empty P&L series.")
    threshold = float(np.percentile(pnl, 2.5))
    tail = pnl[pnl < threshold]
    if tail.size == 0:
        raise ValueError(f"{label} has no observations strictly below its 2.5th percentile.")
    es = -float(np.mean(tail))
    assert math.isfinite(es) and es > 0.0, (
        f"{label} must be a positive loss number; got {es}."
    )
    return es


def _portfolio_pnl(
    unit_shocks: np.ndarray,
    sensitivity: np.ndarray,
    factor_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    if unit_shocks.ndim != 2 or unit_shocks.shape[0] != len(FACTOR_NAMES):
        raise ValueError("Unit shocks must have shape (17 factors, observations).")
    if sensitivity.shape != (len(FACTOR_NAMES),):
        raise ValueError(f"Sensitivity vector must have shape ({len(FACTOR_NAMES)},).")
    masked = np.zeros_like(unit_shocks, dtype=float)
    if factor_names is None:
        selected = set(FACTOR_NAMES)
    else:
        selected = set(factor_names)
        unknown = selected.difference(FACTOR_NAMES)
        if unknown:
            raise ValueError(f"Unknown factor names: {sorted(unknown)}")
    for index, name in enumerate(FACTOR_NAMES):
        if name in selected:
            masked[index] = unit_shocks[index]
    return sensitivity @ masked


def liquidity_adjusted_es(
    unit_shocks: np.ndarray,
    sensitivity: np.ndarray,
    start: int = 0,
    end: int | None = None,
    factor_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[float, dict[int, float], dict[int, float]]:
    """Return adjusted ES, unscaled nested-subset ESs, and weighted contributions."""
    _validate_horizon_map()
    if end is None:
        end = unit_shocks.shape[1]
    if not 0 <= start < end <= unit_shocks.shape[1]:
        raise ValueError(f"Invalid observation interval [{start}, {end}).")

    selected = set(FACTOR_NAMES if factor_names is None else factor_names)
    unknown = selected.difference(FACTOR_NAMES)
    if unknown:
        raise ValueError(f"Unknown factor names: {sorted(unknown)}")

    subset_es: dict[int, float] = {}
    contributions: dict[int, float] = {}
    previous_horizon = BASE_HORIZON
    for index, horizon in enumerate(LIQUIDITY_HORIZONS):
        included = {
            name
            for name in selected
            if LIQUIDITY_HORIZON_BY_FACTOR[name] >= horizon
        }
        if not included:
            subset_es[horizon] = 0.0
            contributions[horizon] = 0.0
            previous_horizon = horizon
            continue
        pnl = _portfolio_pnl(
            unit_shocks[:, start:end],
            sensitivity,
            tuple(included),
        )
        subset_es[horizon] = _positive_es(
            pnl,
            f"ES(P, LH >= {horizon}d; interval [{start}, {end}))",
        )
        if index == 0:
            contributions[horizon] = subset_es[horizon]
        else:
            weight = math.sqrt((horizon - previous_horizon) / BASE_HORIZON)
            contributions[horizon] = subset_es[horizon] * weight
        previous_horizon = horizon

    adjusted_es = math.sqrt(sum(value**2 for value in contributions.values()))
    assert math.isfinite(adjusted_es) and adjusted_es > 0.0, (
        f"Liquidity-adjusted ES must be a positive loss number; got {adjusted_es}."
    )
    return adjusted_es, subset_es, contributions


def _load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rtpl, _, shocks, levels = load_or_build_inputs()
    dates = None
    with np.load(INPUT_PATH, allow_pickle=False) as data:
        if "dates" in data.files:
            dates = np.asarray(data["dates"])
    if (
        rtpl.ndim != 1
        or shocks.shape != (len(FACTOR_NAMES), len(rtpl))
        or levels.shape != shocks.shape
    ):
        raise ValueError(
            f"Expected RTPL length N and shock/level arrays of shape "
            f"({len(FACTOR_NAMES)}, N); got {rtpl.shape}, {shocks.shape}, {levels.shape}."
        )
    if dates is not None and dates.shape != rtpl.shape:
        raise ValueError(f"Cached dates must have shape {rtpl.shape}; got {dates.shape}.")
    if (
        not np.isfinite(rtpl).all()
        or not np.isfinite(shocks).all()
        or not np.isfinite(levels).all()
    ):
        raise ValueError("Cached RTPL, shocks, or levels contain non-finite values.")
    return rtpl, _sensitivity_unit_shocks(shocks, levels), dates


def _date_label(dates: np.ndarray | None, observation: int) -> str:
    if dates is None:
        return f"observation {observation}"
    value = dates[observation]
    return str(value)[:10]


def _variance_explained(
    unit_shocks: np.ndarray,
    sensitivity: np.ndarray,
    factor_names: tuple[str, ...],
    full_pnl: np.ndarray,
) -> float:
    reduced_pnl = _portfolio_pnl(unit_shocks, sensitivity, factor_names)
    full_variance = float(np.var(full_pnl, ddof=1))
    if full_variance <= 0.0:
        raise ValueError("Full-set P&L variance must be positive.")
    unexplained_variance = float(np.var(full_pnl - reduced_pnl, ddof=1))
    explained_share = 1.0 - unexplained_variance / full_variance
    tolerance = 1e-12
    if not -tolerance <= explained_share <= 1.0 + tolerance:
        raise ValueError(
            f"Reduced-set variance explained must be in [0, 1]; got {explained_share}."
        )
    return min(1.0, max(0.0, explained_share))


def _rolling_stress_search(
    unit_shocks: np.ndarray,
    sensitivity: np.ndarray,
    factor_names: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    observation_count = unit_shocks.shape[1]
    if observation_count < ROLLING_WINDOW:
        raise ValueError(
            f"Stressed-period search requires at least {ROLLING_WINDOW} observations."
        )
    # FRTB expects stressed search over history back to 2007. This project has only
    # three years of synthetic data, so its selected period is the worst benign window,
    # not a genuine historical crisis.
    es_by_start = np.empty(observation_count - ROLLING_WINDOW + 1, dtype=float)
    for start in range(es_by_start.size):
        es_by_start[start], _, _ = liquidity_adjusted_es(
            unit_shocks,
            sensitivity,
            start,
            start + ROLLING_WINDOW,
            factor_names,
        )
    return es_by_start, int(np.argmax(es_by_start))


def _format_table(rows: list[tuple[str, str]]) -> None:
    component_width = max(len("Component"), *(len(component) for component, _ in rows))
    value_width = max(len("Value"), *(len(value) for _, value in rows))
    print(f"{'Component':<{component_width}} | {'Value':<{value_width}}")
    print(f"{'-' * component_width}-+-{'-' * value_width}")
    for component, value in rows:
        print(f"{component:<{component_width}} | {value:<{value_width}}")


def _plot_rolling_es(
    es_values: np.ndarray,
    stressed_start: int,
    date_values: np.ndarray | None,
) -> None:
    starts = np.arange(es_values.size)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(starts, es_values, color="steelblue", linewidth=1.3, label="Reduced-set ES")
    ax.axvspan(
        stressed_start,
        stressed_start + ROLLING_WINDOW - 1,
        color="tomato",
        alpha=0.2,
        label="Selected stressed window",
    )
    ax.scatter(
        [stressed_start],
        [es_values[stressed_start]],
        color="darkred",
        zorder=3,
    )
    ax.set_title("Rolling 250-day reduced-set liquidity-adjusted ES")
    ax.set_xlabel("Window start observation")
    ax.set_ylabel("Liquidity-adjusted ES (positive loss)")
    ax.grid(alpha=0.25)
    ax.legend()
    if date_values is None:
        ax.text(
            0.01,
            0.02,
            "Calendar dates are not present in pnl_output.npz; x-axis uses observation indices.",
            transform=ax.transAxes,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    _validate_horizon_map()
    rtpl, unit_shocks, dates = _load_cache()
    one_day_es = historical_var_es(rtpl)[1]
    full_es, full_terms, full_contributions = liquidity_adjusted_es(
        unit_shocks,
        SENSITIVITY_VECTOR,
    )
    assert full_es > one_day_es, (
        f"Liquidity-adjusted ES ({full_es}) must exceed Block 2 one-day ES ({one_day_es})."
    )

    base_variance_share = _variance_explained(
        unit_shocks,
        SENSITIVITY_VECTOR,
        REDUCED_FACTORS,
        rtpl,
    )
    augmented_factors = (*REDUCED_FACTORS, "CSR_SINGLE_5Y")
    augmented_variance_share = _variance_explained(
        unit_shocks,
        SENSITIVITY_VECTOR,
        augmented_factors,
        rtpl,
    )
    if base_variance_share >= 0.75:
        selected_reduced_factors = REDUCED_FACTORS
        selected_variance_share = base_variance_share
    else:
        selected_reduced_factors = augmented_factors
        selected_variance_share = augmented_variance_share

    if selected_variance_share < 0.75:
        print(
            f"FLAG: Even after adding CSR_SINGLE_5Y, the reduced set explains only "
            f"{selected_variance_share:.1%} of full-set P&L variance."
        )

    rolling_es, stressed_start = _rolling_stress_search(
        unit_shocks,
        SENSITIVITY_VECTOR,
        selected_reduced_factors,
    )
    stressed_end = stressed_start + ROLLING_WINDOW - 1
    current_start = len(rtpl) - ROLLING_WINDOW
    current_end = len(rtpl)

    reduced_es_stress, _, _ = liquidity_adjusted_es(
        unit_shocks,
        SENSITIVITY_VECTOR,
        stressed_start,
        stressed_start + ROLLING_WINDOW,
        selected_reduced_factors,
    )
    full_es_current, _, _ = liquidity_adjusted_es(
        unit_shocks,
        SENSITIVITY_VECTOR,
        current_start,
        current_end,
    )
    reduced_es_current, _, _ = liquidity_adjusted_es(
        unit_shocks,
        SENSITIVITY_VECTOR,
        current_start,
        current_end,
        selected_reduced_factors,
    )
    scaling_ratio = full_es_current / reduced_es_current
    if not math.isfinite(scaling_ratio) or scaling_ratio <= 0.0:
        raise ValueError(f"Stressed ES scaling ratio must be positive; got {scaling_ratio}.")
    stressed_full_es = reduced_es_stress * scaling_ratio
    assert stressed_full_es > 0.0, "Scaled stressed ES must be a positive loss number."

    stressed_global_imcc, _, _ = liquidity_adjusted_es(
        unit_shocks,
        SENSITIVITY_VECTOR,
        stressed_start,
        stressed_start + ROLLING_WINDOW,
    )
    risk_class_names = tuple(dict.fromkeys(factor.risk_class for factor in FACTORS))
    class_imccs: dict[str, float] = {}
    for risk_class in risk_class_names:
        class_factors = tuple(
            factor.name for factor in FACTORS if factor.risk_class == risk_class
        )
        # A class IMCC may exceed global IMCC when other classes contain offsets.
        class_imccs[risk_class], _, _ = liquidity_adjusted_es(
            unit_shocks,
            SENSITIVITY_VECTOR,
            stressed_start,
            stressed_start + ROLLING_WINDOW,
            class_factors,
        )
    class_imcc_sum = sum(class_imccs.values())
    diversification_factor = class_imcc_sum / stressed_global_imcc
    imcc = RHO * stressed_global_imcc + (1.0 - RHO) * class_imcc_sum

    nmrf_ses_values = []
    for factor_name in NMRF_FACTORS:
        factor_index = FACTOR_NAMES.index(factor_name)
        if SENSITIVITY_VECTOR[factor_index] == 0.0:
            nmrf_ses_values.append(0.0)
            print(
                f"FLAG: {factor_name} is designated as an NMRF but has zero portfolio "
                "sensitivity; its portfolio SES is zero."
            )
            continue
        factor_pnl = _portfolio_pnl(
            unit_shocks[:, stressed_start : stressed_start + ROLLING_WINDOW],
            SENSITIVITY_VECTOR,
            (factor_name,),
        )
        nmrf_ses_values.append(_positive_es(factor_pnl, f"SES for {factor_name}"))
    # NMRF SES is summed rather than root-sum-squared: zero diversification is
    # the regulatory price of relying on factors that cannot be observed reliably.
    ses = sum(nmrf_ses_values)
    assert math.isfinite(ses) and ses >= 0.0, (
        f"Aggregated NMRF SES must be a non-negative loss number; got {ses}."
    )

    # With one point-in-time run, the 60-day averages equal current values. The
    # averaging and multiplier are time-series features not exercised by this run.
    imcc_avg60 = imcc
    ses_avg60 = ses
    imcc_prev = imcc
    ses_prev = ses
    capital = max(
        imcc_avg60 + ses_avg60,
        CAPITAL_MULTIPLIER * imcc_prev + ses_prev,
    )
    assert capital > 0.0, f"Final capital must be a positive loss number; got {capital}."
    print(
        "Point-in-time capital: the multiplier term always binds because "
        "IMCC_avg60 = IMCC_prev and SES_avg60 = SES_prev."
    )

    rows = [
        ("Block 2 historical 1-day ES_97.5", f"{one_day_es:,.2f}"),
        ("Liquidity-adjusted ES (full set)", f"{full_es:,.2f}"),
    ]
    for horizon in LIQUIDITY_HORIZONS:
        multiplier = (
            1.0
            if horizon == LIQUIDITY_HORIZONS[0]
            else math.sqrt(
                (
                    horizon
                    - LIQUIDITY_HORIZONS[LIQUIDITY_HORIZONS.index(horizon) - 1]
                )
                / BASE_HORIZON
            )
        )
        rows.append(
            (
                f"ES(P,j;j), LH >= {horizon}d",
                f"raw {full_terms[horizon]:,.2f}; weight {multiplier:.3f}; "
                f"term {full_contributions[horizon]:,.2f}",
            )
        )
    rows.extend(
        [
            ("Reduced-set variance explained (base)", f"{base_variance_share:.1%}"),
            (
                "Reduced-set variance explained (+ CSR_SINGLE_5Y)",
                f"{augmented_variance_share:.1%}",
            ),
            ("Reduced set used", ", ".join(selected_reduced_factors)),
            (
                "Stressed window",
                f"{_date_label(dates, stressed_start)} to {_date_label(dates, stressed_end)}",
            ),
            ("ES(R,S)", f"{reduced_es_stress:,.2f}"),
            ("ES(F,C)", f"{full_es_current:,.2f}"),
            ("ES(R,C)", f"{reduced_es_current:,.2f}"),
            ("Scaling ratio ES(F,C) / ES(R,C)", f"{scaling_ratio:.4f}"),
            ("ES_stressed_full", f"{stressed_full_es:,.2f}"),
            ("IMCC(global)", f"{stressed_global_imcc:,.2f}"),
        ]
    )
    rows.extend(
        (f"IMCC({risk_class})", f"{value:,.2f}")
        for risk_class, value in class_imccs.items()
    )
    rows.extend(
        [
            ("Sum of class IMCCs", f"{class_imcc_sum:,.2f}"),
            ("Diversification factor", f"{diversification_factor:.4f}"),
            ("IMCC", f"{imcc:,.2f}"),
            (
                f"SES ({', '.join(NMRF_FACTORS)} NMRF)",
                f"{ses:,.2f} (zero sensitivity)" if ses == 0.0 else f"{ses:,.2f}",
            ),
            ("Final capital", f"{capital:,.2f}"),
        ]
    )
    _format_table(rows)
    print(
        "Stressed search caveat: FRTB expects history back to 2007; this project "
        "has only 3 years of synthetic data, so the selected period is the worst "
        "window in a benign sample, not a true crisis."
    )

    if dates is None:
        print(
            "Calendar dates unavailable: pnl_output.npz contains no dates array; "
            f"stressed window observations are {stressed_start}-{stressed_end}."
        )
    j4 = LIQUIDITY_HORIZONS[-1]
    if full_contributions[j4] / full_es < 0.10:
        print(
            "FLAG: The 60-day single-name contribution is less than 10% of "
            "liquidity-adjusted ES."
        )
    _plot_rolling_es(rolling_es, stressed_start, dates)
    print(f"Plot saved to {PLOT_PATH}.")


if __name__ == "__main__":
    main()
