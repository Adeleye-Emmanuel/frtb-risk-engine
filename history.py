from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from factors import FACTOR_NAMES, RISK_CLASS_BY_NAME, SHOCK_CONVENTION_BY_NAME

IDIOSYNCRATIC_VARIANCE_SHARE = 0.30
GIRR_ADJACENT_CORRELATIONS = (0.87, 0.95, 0.95, 0.95)
CSR_ADJACENT_CORRELATIONS = (0.95, 0.95, 0.90)
GIRR_MEAN_REVERSION = 0.005
CSR_MEAN_REVERSION = 0.10


def fetch_market_data() -> pd.DataFrame:
    tickers = ["^GSPC", "^VIX", "TLT", "LQD", "GBPUSD=X"]
    raw = yf.download(
        tickers,
        period="3y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    if raw.empty:
        raise ValueError("No market data was returned from yfinance for the required 3y sample.")

    market = pd.DataFrame(index=raw.index)
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            series = raw[ticker]
            close = series["Close"] if "Close" in series.columns else series.iloc[:, 0]
        else:
            close = raw[ticker] if "Close" not in raw.columns else raw["Close"]
        market[ticker] = pd.to_numeric(close, errors="coerce")

    market = market.sort_index()
    for column in market.columns:
        market[column] = market[column].ffill(limit=2)
    market = market.dropna(how="any")
    return market


def _scale_driver(values: pd.Series, target_sigma: float) -> pd.Series:
    realised_sigma = float(values.std())
    if not np.isfinite(realised_sigma) or realised_sigma == 0.0:
        raise ValueError("Cannot calibrate a factor with zero or non-finite volatility.")
    return values * (target_sigma / realised_sigma)


def _independent_noise(
    rng: np.random.Generator,
    driver: pd.Series,
    standard_deviation: float,
) -> pd.Series:
    noise = pd.Series(rng.standard_normal(len(driver)), index=driver.index, dtype=float)
    centered_driver = driver - driver.mean()
    centered_noise = noise - noise.mean()
    variance = float(centered_driver.var())
    if variance == 0.0:
        raise ValueError("Cannot construct independent noise from a constant driver.")
    noise = centered_noise - centered_noise.cov(centered_driver) / variance * centered_driver
    return noise * (standard_deviation / float(noise.std()))


def _target_correlation_matrix(adjacent_correlations: tuple[float, ...]) -> np.ndarray:
    size = len(adjacent_correlations) + 1
    correlation = np.eye(size, dtype=float)
    for left in range(size):
        for right in range(left + 1, size):
            correlation[left, right] = correlation[right, left] = float(
                np.prod(adjacent_correlations[left:right])
            )
    return correlation


def _correlated_standard_normals(
    rng: np.random.Generator,
    count: int,
    correlation: np.ndarray,
) -> np.ndarray:
    try:
        cholesky = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Idiosyncratic correlation matrix is not positive definite.") from exc
    values = rng.standard_normal((count, correlation.shape[0]))
    values -= values.mean(axis=0)
    return values @ cholesky.T


def _residual_correlation(
    systematic: pd.DataFrame,
    target_correlation: np.ndarray,
) -> np.ndarray:
    systematic_correlation = systematic.corr().to_numpy()
    residual_correlation = (
        target_correlation
        - (1.0 - IDIOSYNCRATIC_VARIANCE_SHARE) * systematic_correlation
    ) / IDIOSYNCRATIC_VARIANCE_SHARE
    residual_correlation = (residual_correlation + residual_correlation.T) / 2.0
    np.fill_diagonal(residual_correlation, 1.0)
    return residual_correlation


def _residuals_with_30pct_variance(
    systematic: pd.DataFrame,
    standardized_residuals: np.ndarray,
) -> pd.DataFrame:
    residuals = pd.DataFrame(
        standardized_residuals,
        index=systematic.index,
        columns=systematic.columns,
    )
    residual_sigma = np.sqrt(
        IDIOSYNCRATIC_VARIANCE_SHARE
        / (1.0 - IDIOSYNCRATIC_VARIANCE_SHARE)
        * systematic.var()
    )
    residuals = residuals.div(residuals.std()).mul(residual_sigma, axis="columns")
    return systematic + residuals


def _add_correlated_residuals(
    rng: np.random.Generator,
    systematic: pd.DataFrame,
    target_correlation: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    residual_correlation = _residual_correlation(systematic, target_correlation)
    standardized = _correlated_standard_normals(
        rng,
        len(systematic),
        residual_correlation,
    )
    return _residuals_with_30pct_variance(systematic, standardized), standardized


def _build_shock_series(market: pd.DataFrame) -> pd.DataFrame:
    if len(market.index) < 2:
        raise ValueError("At least two market observations are required to build shocks.")

    rng = np.random.default_rng(42)
    index = market.index[1:]
    returns = market.pct_change().iloc[1:].astype(float)
    shocks = pd.DataFrame(index=index, columns=FACTOR_NAMES, dtype=float)
    count = len(index)

    # ETF duration converts daily price returns directly to basis-point rate moves.
    d_level = -returns["TLT"] / 17.0 * 10_000.0
    d_slope = pd.Series(rng.normal(0.0, 1.6, count), index=index)
    d_curve = pd.Series(rng.normal(0.0, 0.8, count), index=index)
    level_loadings = {
        "GIRR_1Y": 0.40,
        "GIRR_2Y": 0.70,
        "GIRR_5Y": 0.90,
        "GIRR_10Y": 1.00,
        "GIRR_30Y": 0.90,
    }
    slope_loadings = {
        "GIRR_1Y": -0.60,
        "GIRR_2Y": -0.30,
        "GIRR_5Y": 0.05,
        "GIRR_10Y": 0.35,
        "GIRR_30Y": 0.50,
    }
    curve_loadings = {
        "GIRR_1Y": 0.30,
        "GIRR_2Y": -0.20,
        "GIRR_5Y": -0.35,
        "GIRR_10Y": 0.10,
        "GIRR_30Y": 0.30,
    }
    girr_systematic = pd.DataFrame(index=index)
    for factor_name, level_loading in level_loadings.items():
        girr_systematic[factor_name] = (
            level_loading * d_level
            + slope_loadings[factor_name] * d_slope
            + curve_loadings[factor_name] * d_curve
        )
    girr, _ = _add_correlated_residuals(
        rng,
        girr_systematic,
        _target_correlation_matrix(GIRR_ADJACENT_CORRELATIONS),
    )
    shocks[girr.columns] = girr

    # LQD duration supplies a separate CDX spread driver; the single-name driver has
    # independent idiosyncratic risk and its own slope driver.
    d_spread = -returns["LQD"] / 8.5 * 10_000.0 * 0.55
    d_spread = d_spread - d_spread.mean()
    spread_sigma = float(d_spread.std())
    single_driver = 0.75 * d_spread + _independent_noise(
        rng,
        d_spread,
        float(np.sqrt(1.0 - 0.75**2)) * spread_sigma,
    )
    cdx_slope = _independent_noise(rng, d_spread, 0.30 * spread_sigma)
    single_slope = 0.75 * cdx_slope + _independent_noise(
        rng,
        d_spread,
        0.30 * spread_sigma * float(np.sqrt(1.0 - 0.75**2)),
    )
    single_loadings = {
        "CSR_SINGLE_1Y": 0.40,
        "CSR_SINGLE_3Y": 0.75,
        "CSR_SINGLE_5Y": 1.00,
        "CSR_SINGLE_10Y": 0.85,
    }
    cdx_loadings = {
        "CSR_CDX_1Y": 0.40,
        "CSR_CDX_3Y": 0.75,
        "CSR_CDX_5Y": 1.00,
        "CSR_CDX_10Y": 0.85,
    }
    credit_slope_loadings = (-0.5, -0.15, 0.15, 0.5)
    single_systematic = pd.DataFrame(
        {
            factor_name: loading * single_driver + slope * single_slope
            for (factor_name, loading), slope in zip(
                single_loadings.items(), credit_slope_loadings
            )
        },
        index=index,
    )
    cdx_systematic = pd.DataFrame(
        {
            factor_name: 0.65 * (loading * d_spread + slope * cdx_slope)
            for (factor_name, loading), slope in zip(
                cdx_loadings.items(), credit_slope_loadings
            )
        },
        index=index,
    )
    target_credit_correlation = _target_correlation_matrix(CSR_ADJACENT_CORRELATIONS)
    cdx, cdx_residual_standardized = _add_correlated_residuals(
        rng,
        cdx_systematic,
        target_credit_correlation,
    )
    residual_correlation = _residual_correlation(
        cdx_systematic, target_credit_correlation
    )
    independent_single_residual = _correlated_standard_normals(
        rng,
        count,
        residual_correlation,
    )
    single_residual_standardized = (
        0.75 * cdx_residual_standardized
        + np.sqrt(1.0 - 0.75**2) * independent_single_residual
    )
    single = _residuals_with_30pct_variance(
        single_systematic,
        single_residual_standardized,
    )
    shocks[single.columns] = single
    shocks[cdx.columns] = cdx

    shocks["EQ_SPX"] = returns["^GSPC"]

    # VIX is a 30-day proxy for 3-month volatility; the 0.6 factor adjusts its shocks.
    d_vol_100 = returns["^VIX"] * 0.6
    # The downside leg has extra negative spot sensitivity in addition to independent idio.
    d_vol_90 = (
        1.25 * d_vol_100
        - 2.0 * shocks["EQ_SPX"]
        + pd.Series(rng.normal(0.0, 0.02, count), index=index)
    )
    shocks["VOL_SPX_100"] = d_vol_100
    shocks["VOL_SPX_90"] = d_vol_90
    shocks["FX_GBPUSD"] = returns["GBPUSD=X"]

    if not np.isfinite(shocks.to_numpy(dtype=float)).all():
        raise ValueError("Generated factor shocks contain non-finite values.")
    return shocks


def build_factor_levels(market: pd.DataFrame) -> pd.DataFrame:
    return _build_factor_levels(market, _build_shock_series(market))


def _build_factor_levels(market: pd.DataFrame, factor_shocks: pd.DataFrame) -> pd.DataFrame:
    levels = pd.DataFrame(index=market.index, columns=FACTOR_NAMES, dtype=float)
    starts = {
        "GIRR_1Y": 0.033,
        "GIRR_2Y": 0.035,
        "GIRR_5Y": 0.038,
        "GIRR_10Y": 0.041,
        "GIRR_30Y": 0.043,
        "CSR_SINGLE_1Y": 0.015,
        "CSR_SINGLE_3Y": 0.0135,
        "CSR_SINGLE_5Y": 0.012,
        "CSR_SINGLE_10Y": 0.014,
        "CSR_CDX_1Y": 0.010,
        "CSR_CDX_3Y": 0.0085,
        "CSR_CDX_5Y": 0.0065,
        "CSR_CDX_10Y": 0.008,
    }
    for factor_name, start in starts.items():
        mean_reversion = (
            GIRR_MEAN_REVERSION if factor_name.startswith("GIRR_") else CSR_MEAN_REVERSION
        )
        path = np.empty(len(market.index), dtype=float)
        path[0] = start
        for step, shock_bp in enumerate(factor_shocks[factor_name].to_numpy(), start=1):
            previous_level = path[step - 1]
            path[step] = (
                previous_level
                + mean_reversion * (start - previous_level)
                + shock_bp / 10_000.0
            )
        levels[factor_name] = path
    for factor_name in starts:
        if (levels[factor_name] < 0.0).any():
            raise ValueError(f"Negative level in {factor_name} — check kappa calibration")

    levels["EQ_SPX"] = market["^GSPC"]
    levels["VOL_SPX_90"] = market["^VIX"].iloc[0] / 100.0 * 0.90
    levels.loc[market.index[1]:, "VOL_SPX_90"] = (
        levels["VOL_SPX_90"].iloc[0]
        * (1.0 + factor_shocks["VOL_SPX_90"]).cumprod().to_numpy()
    )
    levels["VOL_SPX_100"] = market["^VIX"].iloc[0] / 100.0
    levels.loc[market.index[1]:, "VOL_SPX_100"] = (
        levels["VOL_SPX_100"].iloc[0]
        * (1.0 + factor_shocks["VOL_SPX_100"]).cumprod().to_numpy()
    )
    levels["FX_GBPUSD"] = market["GBPUSD=X"]
    if (levels[["VOL_SPX_90", "VOL_SPX_100"]] <= 0.0).any().any():
        raise ValueError("Generated volatility level became non-positive.")
    return levels


def build_shock_matrix(levels: pd.DataFrame) -> pd.DataFrame:
    shocks = pd.DataFrame(index=levels.index, columns=levels.columns, dtype=float)
    for factor_name in levels.columns:
        series = levels[factor_name].astype(float)
        convention = SHOCK_CONVENTION_BY_NAME[factor_name]
        if convention == "ABSOLUTE":
            shocks[factor_name] = series.diff() * 10_000.0
        elif convention == "RELATIVE":
            shocks[factor_name] = series.pct_change()
        else:
            raise ValueError(f"Unsupported shock convention '{convention}' for {factor_name}.")
    return shocks.iloc[1:]


def print_shock_sd_table(shocks: pd.DataFrame) -> None:
    rows = []
    for factor_name in shocks.columns:
        values = shocks[factor_name].astype(float)
        convention = SHOCK_CONVENTION_BY_NAME[factor_name]
        std_dev = float(values.std())
        if convention == "ABSOLUTE":
            unit = "bp"
            display = std_dev
        else:
            unit = "%"
            display = std_dev * 100.0
        rows.append(
            {
                "factor": factor_name,
                "risk_class": RISK_CLASS_BY_NAME[factor_name],
                "convention": convention,
                "std_dev": display,
                "unit": unit,
            }
        )

    summary = pd.DataFrame(rows)
    print("Realised shock std dev by factor (ABSOLUTE in bp, RELATIVE in %):")
    print(summary[["factor", "risk_class", "convention", "std_dev", "unit"]].to_string(index=False))
    print("Full shock correlation matrix:")
    print(shocks.corr().to_string(float_format=lambda value: f"{value:.2f}"))
    print(
        "Adjacent GIRR correlations: "
        + ", ".join(
            f"{left}/{right}={shocks[left].corr(shocks[right]):.2f}"
            for left, right in zip(
                ("GIRR_1Y", "GIRR_2Y", "GIRR_5Y", "GIRR_10Y"),
                ("GIRR_2Y", "GIRR_5Y", "GIRR_10Y", "GIRR_30Y"),
            )
        )
    )
    print(
        "GIRR_1Y vs GIRR_10Y correlation: "
        f"{shocks['GIRR_1Y'].corr(shocks['GIRR_10Y']):.2f}"
    )
    print(
        "CSR_SINGLE_5Y vs CSR_CDX_5Y correlation: "
        f"{shocks['CSR_SINGLE_5Y'].corr(shocks['CSR_CDX_5Y']):.2f}"
    )
    print(
        "VOL vs EQ correlations (90%, 100%): "
        f"{shocks['VOL_SPX_90'].corr(shocks['EQ_SPX']):.2f}, "
        f"{shocks['VOL_SPX_100'].corr(shocks['EQ_SPX']):.2f}"
    )


def build_factor_history(print_diagnostics: bool = True) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    market = fetch_market_data()
    shocks = _build_shock_series(market)
    levels = _build_factor_levels(market, shocks)
    if print_diagnostics:
        print_shock_sd_table(shocks)
    aligned_levels = levels.loc[shocks.index]
    return aligned_levels.to_numpy().T.astype(float), shocks.to_numpy().T.astype(float), shocks.index


if __name__ == "__main__":
    levels, shocks, dates = build_factor_history()
    print(f"Levels matrix shape: {levels.shape}")
    print(f"Shocks matrix shape: {shocks.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
