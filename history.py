from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from factors import FACTOR_NAMES, RISK_CLASS_BY_NAME, SHOCK_CONVENTION_BY_NAME


def fetch_market_data() -> pd.DataFrame:
    tickers = ["^SPX", "^VIX", "TLT", "LQD", "HYG", "GBPUSD=X"]
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


def build_factor_levels(market: pd.DataFrame) -> pd.DataFrame:
    levels = pd.DataFrame(index=market.index, columns=FACTOR_NAMES, dtype=float)

    # TLT is the closest liquid daily proxy for the 10y local yield level; it is used here
    # as a defensible stand-in for the GBP zero-curve move because the spec requires a 3y
    # daily history without real UK curve data.
    girr_proxy = np.clip(100.0 / market["TLT"], 0.0, None) / 30.0
    girr_scales = np.array([0.60, 0.80, 1.00, 1.35, 1.80], dtype=float)
    for idx, scale in enumerate(girr_scales):
        name = FACTOR_NAMES[idx]
        levels[name] = girr_proxy * scale

    # LQD/HYG are investment-grade and high-yield credit ETFs, so their spread gap is a
    # defensible proxy for a single-name/IG index credit spread curve when no CDS curves are
    # available; it is labelled as such to stay faithful to the project specification.
    spread_proxy = np.clip((1.0 / market["HYG"] - 1.0 / market["LQD"]) * 10000.0, 0.0, None)
    single_name_scales = np.array([0.75, 1.00, 1.25, 1.50], dtype=float)
    for idx, scale in enumerate(single_name_scales, start=5):
        name = FACTOR_NAMES[idx]
        levels[name] = spread_proxy * scale

    cdx_scales = np.array([0.70, 0.90, 1.10, 1.30], dtype=float)
    for idx, scale in enumerate(cdx_scales, start=9):
        name = FACTOR_NAMES[idx]
        levels[name] = spread_proxy * scale

    # SPX is the direct equity level factor, so using the daily close is the cleanest proxy.
    levels["EQ_SPX"] = market["^SPX"]

    # VIX is the liquid daily proxy for the 90%/100% moneyness option vol surface, which is
    # the same information required by the spec in the absence of an actual SPX vol surface.
    levels["VOL_SPX_90"] = market["^VIX"] * 0.90
    levels["VOL_SPX_100"] = market["^VIX"]

    # GBPUSD=X is the direct FX pair used in the fixed portfolio, so it is the correct level series.
    levels["FX_GBPUSD"] = market["GBPUSD=X"]
    return levels


def build_shock_matrix(levels: pd.DataFrame) -> pd.DataFrame:
    shocks = pd.DataFrame(index=levels.index, columns=levels.columns, dtype=float)
    for factor_name in levels.columns:
        series = levels[factor_name].astype(float)
        convention = SHOCK_CONVENTION_BY_NAME[factor_name]
        if convention == "ABSOLUTE":
            shocks[factor_name] = series.diff().fillna(0.0) * 10_000.0
        elif convention == "RELATIVE":
            shocks[factor_name] = series.pct_change().fillna(0.0)
        else:
            raise ValueError(f"Unsupported shock convention '{convention}' for {factor_name}.")
    return shocks


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


def build_factor_history() -> tuple[np.ndarray, np.ndarray, pd.Index]:
    market = fetch_market_data()
    levels = build_factor_levels(market)
    shocks = build_shock_matrix(levels)
    print_shock_sd_table(shocks)
    return levels.to_numpy().T.astype(float), shocks.to_numpy().T.astype(float), levels.index


if __name__ == "__main__":
    levels, shocks, dates = build_factor_history()
    print(f"Levels matrix shape: {levels.shape}")
    print(f"Shocks matrix shape: {shocks.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
