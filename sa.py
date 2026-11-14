from __future__ import annotations

import math

import numpy as np

from factors import FACTOR_NAMES, RISK_CLASS_BY_NAME
from pnl import (
    SPX_CALL_NOTIONAL,
    SPX_CALL_TAU,
    black_scholes_price,
)


from sensitivities import SENSITIVITY_VECTOR

INPUT_PATH = "pnl_output.npz"
IMA_CAPITAL = 3_428_220.54
SPX_VEGA_LIQUIDITY_HORIZON = 20

# Block 4 source units: sensitivities are per 1bp for rates/credit and per 1%
# for EQ/FX; convert them to per-unit moves before multiplying decimal Basel RWs.
SENSITIVITY_UNIT_SCALE_BY_RISK_CLASS = {
    "GIRR": 10_000.0,
    "CSR": 10_000.0,
    "EQ": 100.0,
    "FX": 100.0,
}

# MAR21.41-21.42, Table 1: prescribed delta GIRR risk weights.
GIRR_RISK_WEIGHT_BY_TENOR = {
    1: 0.016,
    2: 0.013,
    5: 0.011,
    10: 0.011,
    30: 0.011,
}
GIRR_TENOR_BY_FACTOR = {
    "GIRR_1Y": 1,
    "GIRR_2Y": 2,
    "GIRR_5Y": 5,
    "GIRR_10Y": 10,
    "GIRR_30Y": 30,
}
# MAR21.46, Table 2 and its footnote: max(exp(-3% * |Tk-Tl|/min(Tk,Tl)), 40%).
GIRR_INTRA_BUCKET_RHO = {
    (left, right): (
        1.0
        if left == right
        else max(math.exp(-0.03 * abs(left - right) / min(left, right)), 0.40)
    )
    for left in GIRR_RISK_WEIGHT_BY_TENOR
    for right in GIRR_RISK_WEIGHT_BY_TENOR
}

# MAR21.51-21.54, Tables 3-4: IG industrials are bucket 4, RW 3%.
# CDX is normally IG index bucket 17 (RW 1.5%); placing it in bucket 4 here is
# the requested project simplification, with inter-name rho 35% per MAR21.54.
CSR_SECTOR_BUCKET = 4
CSR_CDX_INDEX_BUCKET = 17
CSR_RISK_WEIGHT_BY_BUCKET = {4: 0.03, 17: 0.015}
CSR_BUCKET_BY_FACTOR = {
    **{f"CSR_SINGLE_{tenor}Y": CSR_SECTOR_BUCKET for tenor in (1, 3, 5, 10)},
    **{f"CSR_CDX_{tenor}Y": CSR_SECTOR_BUCKET for tenor in (1, 3, 5, 10)},
}
CSR_NAME_BY_FACTOR = {
    **{f"CSR_SINGLE_{tenor}Y": "single-name" for tenor in (1, 3, 5, 10)},
    **{f"CSR_CDX_{tenor}Y": "CDX-IG" for tenor in (1, 3, 5, 10)},
}
FACTOR_BUCKET_BY_NAME = {
    **{f"GIRR_{tenor}Y": "GBP" for tenor in (1, 2, 5, 10, 30)},
    **{
        f"CSR_SINGLE_{tenor}Y": "CSR_IG_INDUSTRIAL"
        for tenor in (1, 3, 5, 10)
    },
    **{f"CSR_CDX_{tenor}Y": "CSR_IG_INDUSTRIAL" for tenor in (1, 3, 5, 10)},
    "EQ_SPX": "EQ_LARGE_CAP_ADVANCED_INDEX",
    "VOL_SPX_90": "EQ_LARGE_CAP_ADVANCED_INDEX",
    "VOL_SPX_100": "EQ_LARGE_CAP_ADVANCED_INDEX",
    "FX_GBPUSD": "FX_GBPUSD",
}
CSR_TENOR_BY_FACTOR = {
    **{f"CSR_SINGLE_{tenor}Y": tenor for tenor in (1, 3, 5, 10)},
    **{f"CSR_CDX_{tenor}Y": tenor for tenor in (1, 3, 5, 10)},
}
# MAR21.54(1)-(2): same-name different-tenor 65%; different-name 35%;
# identical risk factors 100%. The factor-specific curve term is 1 for both
# project curves, so no additional 99.90% basis adjustment is applied.
CSR_INTRA_BUCKET_RHO = {
    (left, right): (
        1.0
        if left == right
        else (
            (
                1.0
                if CSR_TENOR_BY_FACTOR[left] == CSR_TENOR_BY_FACTOR[right]
                else 0.65
            )
            if CSR_NAME_BY_FACTOR[left] == CSR_NAME_BY_FACTOR[right]
            else 0.35
        )
    )
    for left in CSR_BUCKET_BY_FACTOR
    for right in CSR_BUCKET_BY_FACTOR
}

# MAR21.72, Table 9 and MAR21.77, Table 10: SPX is index bucket 12, 15% RW.
EQUITY_BUCKET_BY_FACTOR = {
    "EQ_SPX": 12,
    "VOL_SPX_90": 12,
    "VOL_SPX_100": 12,
}
EQUITY_DELTA_RISK_WEIGHT_BY_BUCKET = {12: 0.15}
# MAR21.98: equity curvature uses the respective delta risk weight as shock size.
EQUITY_CURVATURE_RISK_WEIGHT_BY_BUCKET = {12: 0.15}
# MAR21.78(2)(e): delta correlation between equity index buckets is 80%.
EQUITY_INDEX_DELTA_RHO = 0.80

# MAR21.87 and MAR21.89: unique FX RW 15%, cross-bucket gamma 60%.
FX_RISK_WEIGHT_BY_FACTOR = {"FX_GBPUSD": 0.15}

# MAR21.50, MAR21.57, MAR21.80 and MAR21.89: applicable cross-bucket
# correlations. The project assigns one bucket per risk class, so these do not
# create cross-bucket terms in this portfolio.
CROSS_BUCKET_GAMMA_BY_CLASS = {
    "GIRR": 0.50,
    "CSR": 1.00,  # Same sector and rating category under MAR21.57.
    "EQ": 0.75,  # Equity index buckets 12/13 under MAR21.80(3).
    "FX": 0.60,
}

# MAR21.92, Table 13 and footnote 24: equity large-cap/index LH=20d,
# RW=min(55%*sqrt(LH/10),100%) = 77.78%.
EQUITY_VEGA_RISK_WEIGHT = min(
    0.55 * math.sqrt(SPX_VEGA_LIQUIDITY_HORIZON / 10.0),
    1.0,
)
# MAR21.93-21.94: same underlying at same option/underlying maturities implies
# unit maturity correlation; use 0.999 as the project assumption for distinct
# 90%/100% strikes when exact same-underlying vega rho is treated as ambiguous.
EQUITY_VEGA_INTRA_BUCKET_RHO = 0.999
CSR_INDEX_SECTOR_CROSS_BUCKET_GAMMA = 0.45  # MAR21.57, Table 5: bucket 4 vs IG index 17.

# MAR21.6: prescribed medium scenario, 1.25x capped high scenario, and
# max(2*rho-1, 75%*rho) low scenario.
CORRELATION_SCENARIOS = ("medium", "high", "low")

# MAR22.12(2) and MAR22.24 Table 2: senior debt LGD 75%; official table
# assigns BBB 6% and A 3%.
LGD_SENIOR_UNSECURED = 0.75
DRC_RISK_WEIGHT_BY_RATING = {
    "AAA": 0.005,
    "AA": 0.02,
    "A": 0.03,
    "BBB": 0.06,
}
DRC_CREDIT_POSITIONS = (
    {
        "name": "single-name CDS",
        "direction": "short",
        "notional": 50_000_000.0,
        "mtm_gain_pct": 0.0086,
        "rating": "BBB",
        "risky_annuity": 4.3,
    },
    {
        "name": "CDX IG",
        "direction": "long",
        "notional": 75_000_000.0,
        "mtm_gain_pct": 0.0180,
        "rating": "A",
        "risky_annuity": 4.5,
    },
)

# The simplified portfolio has no exotic underlying or non-vanilla payoff.
RRAO = 0.0


def _scenario_correlation(value: float, scenario: str) -> float:
    if scenario == "medium":
        return value
    if scenario == "high":
        return min(1.25 * value, 1.0)
    if scenario == "low":
        return max(2.0 * value - 1.0, 0.75 * value)
    raise ValueError(f"Unknown correlation scenario: {scenario}")


def _factor_risk_weight(factor_name: str) -> float:
    if factor_name in GIRR_TENOR_BY_FACTOR:
        tenor = GIRR_TENOR_BY_FACTOR[factor_name]
        try:
            return GIRR_RISK_WEIGHT_BY_TENOR[tenor]
        except KeyError as error:
            raise ValueError(f"Missing GIRR risk weight for tenor {tenor}Y.") from error
    if factor_name in CSR_BUCKET_BY_FACTOR:
        bucket = CSR_BUCKET_BY_FACTOR[factor_name]
        try:
            return CSR_RISK_WEIGHT_BY_BUCKET[bucket]
        except KeyError as error:
            raise ValueError(f"Missing CSR risk weight for bucket {bucket}.") from error
    if factor_name == "EQ_SPX":
        return EQUITY_DELTA_RISK_WEIGHT_BY_BUCKET[EQUITY_BUCKET_BY_FACTOR[factor_name]]
    if factor_name == "FX_GBPUSD":
        return FX_RISK_WEIGHT_BY_FACTOR[factor_name]
    raise ValueError(f"No prescribed delta risk weight available for {factor_name}.")


def _factor_risk_weight_for_scenario(
    factor_name: str,
    cdx_index_bucket: bool,
) -> float:
    if cdx_index_bucket and factor_name in CSR_BUCKET_BY_FACTOR:
        return CSR_RISK_WEIGHT_BY_BUCKET[_csr_bucket(factor_name, True)]
    return _factor_risk_weight(factor_name)


def _csr_bucket(factor_name: str, cdx_index_bucket: bool = False) -> int:
    if cdx_index_bucket and CSR_NAME_BY_FACTOR[factor_name] == "CDX-IG":
        return CSR_CDX_INDEX_BUCKET
    return CSR_BUCKET_BY_FACTOR[factor_name]


def _intra_bucket_rho(
    risk_class: str,
    left_factor: str,
    right_factor: str,
    scenario: str,
) -> float:
    if risk_class == "GIRR":
        left = GIRR_TENOR_BY_FACTOR[left_factor]
        right = GIRR_TENOR_BY_FACTOR[right_factor]
        prescribed = GIRR_INTRA_BUCKET_RHO[(left, right)]
    elif risk_class == "CSR":
        prescribed = CSR_INTRA_BUCKET_RHO[(left_factor, right_factor)]
    elif risk_class == "EQ":
        prescribed = 1.0 if left_factor == right_factor else EQUITY_INDEX_DELTA_RHO
    else:
        prescribed = 1.0
    return _scenario_correlation(prescribed, scenario)


def _aggregate_bucket_charges(
    bucket_positions: dict[str, list[tuple[str, float]]],
    scenario: str,
    risk_class: str,
    curvature: bool = False,
    cross_bucket_gamma: float | None = None,
) -> float:
    bucket_k, bucket_s = _bucket_components(
        bucket_positions,
        scenario,
        risk_class,
        curvature,
    )
    total_variance = sum(value**2 for value in bucket_k.values())
    buckets = list(bucket_positions)
    gamma_base = (
        CROSS_BUCKET_GAMMA_BY_CLASS[risk_class]
        if cross_bucket_gamma is None
        else cross_bucket_gamma
    )
    for i, left_bucket in enumerate(buckets):
        for right_bucket in buckets[i + 1 :]:
            gamma = _scenario_correlation(gamma_base, scenario)
            if curvature:
                gamma = gamma**2
            total_variance += (
                2.0
                * gamma
                * bucket_s[left_bucket]
                * bucket_s[right_bucket]
            )

    if total_variance < 0.0 and len(buckets) > 1:
        total_variance = sum(value**2 for value in bucket_k.values())
        bounded_s = {
            bucket: max(min(bucket_s[bucket], bucket_k[bucket]), -bucket_k[bucket])
            for bucket in buckets
        }
        for i, left_bucket in enumerate(buckets):
            for right_bucket in buckets[i + 1 :]:
                gamma = _scenario_correlation(gamma_base, scenario)
                if curvature:
                    gamma = gamma**2
                total_variance += (
                    2.0 * gamma * bounded_s[left_bucket] * bounded_s[right_bucket]
                )
    return math.sqrt(max(total_variance, 0.0))


def _bucket_components(
    bucket_positions: dict[str, list[tuple[str, float]]],
    scenario: str,
    risk_class: str,
    curvature: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    bucket_k: dict[str, float] = {}
    bucket_s: dict[str, float] = {}
    for bucket, positions in bucket_positions.items():
        variance = sum(weighted**2 for _, weighted in positions)
        for i, (left_factor, left_weighted) in enumerate(positions):
            for right_factor, right_weighted in positions[i + 1 :]:
                rho = _intra_bucket_rho(
                    risk_class,
                    left_factor,
                    right_factor,
                    scenario,
                )
                if curvature:
                    rho = rho**2
                variance += 2.0 * rho * left_weighted * right_weighted
        bucket_k[bucket] = math.sqrt(max(variance, 0.0))
        bucket_s[bucket] = sum(weighted for _, weighted in positions)
    return bucket_k, bucket_s


def _delta_charge(
    scenario: str,
    cdx_index_bucket: bool = False,
) -> dict[str, float]:
    bucket_positions = _delta_bucket_positions(cdx_index_bucket)
    charges: dict[str, float] = {}
    for risk_class, buckets in bucket_positions.items():
        gamma = (
            CSR_INDEX_SECTOR_CROSS_BUCKET_GAMMA
            if risk_class == "CSR" and cdx_index_bucket
            else None
        )
        charges[risk_class] = _aggregate_bucket_charges(
            buckets,
            scenario,
            risk_class,
            cross_bucket_gamma=gamma,
        )
    return charges


def _delta_bucket_positions(
    cdx_index_bucket: bool = False,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    bucket_positions: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for factor_name, risk_class in RISK_CLASS_BY_NAME.items():
        if risk_class not in ("GIRR", "CSR", "EQ", "FX"):
            continue
        sensitivity = float(SENSITIVITY_VECTOR[FACTOR_NAMES.index(factor_name)])
        # GIRR/CSR: per-bp to per-unit-rate requires x10,000. EQ/FX: per-1%
        # to per-unit-relative move requires x100 before applying decimal RW.
        scaling = SENSITIVITY_UNIT_SCALE_BY_RISK_CLASS[risk_class]
        assert scaling == {
            "GIRR": 10_000.0,
            "CSR": 10_000.0,
            "EQ": 100.0,
            "FX": 100.0,
        }[risk_class]
        risk_weight = _factor_risk_weight_for_scenario(
            factor_name,
            cdx_index_bucket,
        )
        weighted = sensitivity * scaling * risk_weight
        bucket = FACTOR_BUCKET_BY_NAME[factor_name]
        if risk_class == "CSR":
            bucket = f"CSR-{_csr_bucket(factor_name, cdx_index_bucket)}"
        bucket_positions.setdefault(risk_class, {}).setdefault(bucket, []).append(
            (factor_name, weighted)
        )
    return bucket_positions


def _csr_weighted_sensitivities(
    cdx_index_bucket: bool = False,
) -> list[tuple[str, float, float, float]]:
    rows = []
    for factor_name in CSR_BUCKET_BY_FACTOR:
        sensitivity = float(SENSITIVITY_VECTOR[FACTOR_NAMES.index(factor_name)])
        bucket = _csr_bucket(factor_name, cdx_index_bucket)
        risk_weight = CSR_RISK_WEIGHT_BY_BUCKET[bucket]
        weighted = (
            sensitivity
            * SENSITIVITY_UNIT_SCALE_BY_RISK_CLASS["CSR"]
            * risk_weight
        )
        rows.append((factor_name, sensitivity, risk_weight, weighted))
    return rows


def _delta_bucket_diagnostics(
    scenario: str,
    cdx_index_bucket: bool = False,
) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    result = {}
    for risk_class, buckets in _delta_bucket_positions(cdx_index_bucket).items():
        # Delta bucket K/S diagnostics; MAR21 uses S only in cross-bucket aggregation.
        result[risk_class] = _bucket_components(buckets, scenario, risk_class)
    return result


def _vega_charge(scenario: str) -> float:
    # Vega is specified per one volatility point (1%), so it is already in the
    # conventional vega unit to which the Basel vega RW is applied.
    factors = ("VOL_SPX_90", "VOL_SPX_100")
    weighted = [
        float(SENSITIVITY_VECTOR[FACTOR_NAMES.index(name)])
        * EQUITY_VEGA_RISK_WEIGHT
        for name in factors
    ]
    rho = _scenario_correlation(EQUITY_VEGA_INTRA_BUCKET_RHO, scenario)
    variance = sum(value**2 for value in weighted)
    variance += 2.0 * rho * weighted[0] * weighted[1]
    return math.sqrt(max(variance, 0.0))


def _curvature_charge(spot: float, volatility: float, scenario: str) -> float:
    risk_weight = EQUITY_CURVATURE_RISK_WEIGHT_BY_BUCKET[12]
    strike = spot
    shares = SPX_CALL_NOTIONAL / spot
    base_value = shares * black_scholes_price(
        spot,
        strike,
        0.02,
        volatility,
        SPX_CALL_TAU,
    )
    delta_per_relative_move = (
        float(SENSITIVITY_VECTOR[FACTOR_NAMES.index("EQ_SPX")]) * 100.0
    )
    plus_value = shares * black_scholes_price(
        spot * (1.0 + risk_weight),
        strike,
        0.02,
        volatility,
        SPX_CALL_TAU,
    )
    minus_value = shares * black_scholes_price(
        spot * (1.0 - risk_weight),
        strike,
        0.02,
        volatility,
        SPX_CALL_TAU,
    )
    cvr_plus = -(plus_value - base_value - risk_weight * delta_per_relative_move)
    cvr_minus = -(minus_value - base_value + risk_weight * delta_per_relative_move)
    assert math.isfinite(cvr_plus) and math.isfinite(cvr_minus)
    cvr = max(cvr_plus, cvr_minus, 0.0)
    # Every other portfolio instrument is linear and has exactly zero curvature.
    linear_instrument_curvature = {
        "GBP 5y receiver swap": 0.0,
        "10y Gilt": 0.0,
        "CDX IG": 0.0,
        "single-name CDS": 0.0,
        "GBPUSD forward": 0.0,
    }
    assert all(value == 0.0 for value in linear_instrument_curvature.values())
    return _aggregate_bucket_charges(
        {"SPX-index": [("EQ_SPX", cvr)]},
        scenario,
        "EQ",
        curvature=True,
    )


def _curvature_diagnostics(spot: float, volatility: float) -> dict[str, float]:
    risk_weight = EQUITY_CURVATURE_RISK_WEIGHT_BY_BUCKET[12]
    shares = SPX_CALL_NOTIONAL / spot
    strike = spot
    value = lambda shocked_spot: shares * black_scholes_price(
        shocked_spot,
        strike,
        0.02,
        volatility,
        SPX_CALL_TAU,
    )
    v_x = value(spot)
    v_down = value(spot * (1.0 - risk_weight))
    v_up = value(spot * (1.0 + risk_weight))
    delta_sensitivity = (
        float(SENSITIVITY_VECTOR[FACTOR_NAMES.index("EQ_SPX")]) * 100.0
    )
    delta_term = risk_weight * delta_sensitivity
    cvr_minus = -(v_down - v_x + delta_term)
    cvr_plus = -(v_up - v_x - delta_term)
    return {
        "risk_weight": risk_weight,
        "shock_down": spot * (1.0 - risk_weight),
        "shock_up": spot * (1.0 + risk_weight),
        "delta_sensitivity": delta_sensitivity,
        "rw_delta": delta_term,
        "v_x": v_x,
        "v_down": v_down,
        "v_up": v_up,
        "cvr_minus": cvr_minus,
        "cvr_plus": cvr_plus,
        "convexity_down": v_down - v_x + delta_term,
        "convexity_up": v_up - v_x - delta_term,
    }


def _print_calculation_diagnostics(
    spot: float,
    volatility: float,
) -> None:
    print(
        "GIRR bucket diagnostics "
        "(K_b is single-bucket charge; S_b is signed WS sum):"
    )
    for scenario in CORRELATION_SCENARIOS:
        bucket_k, bucket_s = _delta_bucket_diagnostics(scenario)["GIRR"]
        print(
            f"  {scenario}: "
            + ", ".join(
                f"{bucket} K_b={bucket_k[bucket]:,.2f}, "
                f"S_b={bucket_s[bucket]:,.2f}"
                for bucket in bucket_k
            )
        )
    print(
        "CSR bucket-4 comparison weighted-sensitivity diagnostics "
        "(RW is decimal; rate move = RW x 10,000 bp):"
    )
    for factor_name, sensitivity, risk_weight, weighted in _csr_weighted_sensitivities():
        print(
            f"  {factor_name}: sensitivity={sensitivity:,.2f}/bp, "
            f"RW={risk_weight:.4f}, RW shock={risk_weight * 10_000:.1f}bp, "
            f"WS={weighted:,.2f}"
        )

    details = _curvature_diagnostics(spot, volatility)
    assert details["delta_sensitivity"] == (
        float(SENSITIVITY_VECTOR[FACTOR_NAMES.index("EQ_SPX")]) * 100.0
    ), "Curvature must use the Basel delta sensitivity in unit relative-move scale."
    assert details["risk_weight"] == EQUITY_CURVATURE_RISK_WEIGHT_BY_BUCKET[12]
    print(
        "SPX curvature verification (Basel delta sensitivity s_i, "
        "separate MAR21.98 curvature risk weight):"
    )
    print(
        f"  spot x={spot:,.4f}; RW={details['risk_weight']:.4f}; "
        f"delta s={details['delta_sensitivity']:,.2f}; "
        f"curvature RW*delta={details['rw_delta']:,.2f}"
    )
    print(
        f"  V(x)={details['v_x']:,.2f}; "
        f"V(x-RW) at {details['shock_down']:,.4f}={details['v_down']:,.2f}; "
        f"V(x+RW) at {details['shock_up']:,.4f}={details['v_up']:,.2f}"
    )
    print(
        f"  CVR_minus={details['cvr_minus']:,.2f}; "
        f"CVR_plus={details['cvr_plus']:,.2f}; "
        f"positive curvature charge="
        f"{max(details['cvr_minus'], details['cvr_plus'], 0.0):,.2f}"
    )
    print(
        "  Curvature uses the same Basel delta sensitivity used for delta SBM, "
        "not raw Black-Scholes delta. Its 15% curvature RW is separately sourced "
        "and happens to equal the equity delta RW numerically."
    )


def _drc_charge() -> tuple[float, dict[str, float], float]:
    long_jtd = 0.0
    short_jtd = 0.0
    weighted_long = 0.0
    weighted_short = 0.0
    details: dict[str, float] = {}
    for position in DRC_CREDIT_POSITIONS:
        rating = str(position["rating"])
        if rating not in DRC_RISK_WEIGHT_BY_RATING:
            raise ValueError(f"Missing MAR22.24 DRC risk weight for rating {rating}.")
        # Risky annuities 4.3 (single name, 120bp) and 4.5 (CDX, 60bp)
        # imply the current MtM gains supplied for these off-par contracts.
        risky_annuity = float(position["risky_annuity"])
        notional = float(position["notional"])
        mtm_gain_pct = float(position["mtm_gain_pct"])
        if risky_annuity <= 0.0 or notional <= 0.0:
            raise ValueError(f"Invalid DRC position inputs for {position['name']}.")
        jtd_magnitude = notional * (LGD_SENIOR_UNSECURED - mtm_gain_pct)
        if jtd_magnitude < 0.0:
            raise ValueError(f"MtM gain exceeds LGD for {position['name']}.")
        if position["direction"] == "long":
            jtd = jtd_magnitude
            long_jtd += jtd
            weighted_long += DRC_RISK_WEIGHT_BY_RATING[rating] * jtd
        elif position["direction"] == "short":
            jtd = -jtd_magnitude
            short_jtd += abs(jtd)
            weighted_short += DRC_RISK_WEIGHT_BY_RATING[rating] * abs(jtd)
        else:
            raise ValueError(f"Invalid DRC direction for {position['name']}.")
        details[str(position["name"])] = jtd

    denominator = long_jtd + short_jtd
    hbr = long_jtd / denominator if denominator else 0.0
    return max(weighted_long - hbr * weighted_short, 0.0), details, hbr


def _load_curvature_levels() -> tuple[float, float]:
    try:
        with np.load(INPUT_PATH, allow_pickle=False) as data:
            if "levels" not in data.files:
                raise ValueError(f"{INPUT_PATH} does not contain factor levels.")
            levels = np.asarray(data["levels"], dtype=float)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{INPUT_PATH} is required for current SPX spot and volatility only; "
            "run Block 2 to create the cache."
        ) from error
    if levels.ndim != 2 or levels.shape[0] != len(FACTOR_NAMES) or levels.shape[1] == 0:
        raise ValueError(
            f"Cached levels must have shape ({len(FACTOR_NAMES)}, observations); "
            f"got {levels.shape}."
        )
    spot = float(levels[FACTOR_NAMES.index("EQ_SPX"), -1])
    volatility = float(levels[FACTOR_NAMES.index("VOL_SPX_100"), -1])
    if not math.isfinite(spot) or not math.isfinite(volatility) or spot <= 0.0 or volatility <= 0.0:
        raise ValueError(f"Current SPX spot/volatility must be positive; got {spot}, {volatility}.")
    return spot, volatility


def _print_table(
    scenario_charges: dict[str, dict[str, dict[str, float]]],
    drc: float,
    jtd: dict[str, float],
    hbr: float,
) -> tuple[float, str]:
    risk_classes = ("GIRR", "CSR", "EQ", "FX")
    rows: list[tuple[str, str, str, str, str]] = []
    scenario_totals: dict[str, float] = {}
    for risk_class in risk_classes:
        for measure in ("delta", "vega", "curvature"):
            values = [
                scenario_charges[scenario][measure].get(risk_class, 0.0)
                for scenario in CORRELATION_SCENARIOS
            ]
            rows.append(
                (
                    risk_class,
                    measure,
                    *(f"{value:,.2f}" for value in values),
                )
            )
    for scenario in CORRELATION_SCENARIOS:
        scenario_totals[scenario] = sum(
            sum(scenario_charges[scenario][measure].values())
            for measure in ("delta", "vega", "curvature")
        )
        rows.append(
            (
                "SBM total",
                scenario,
                *(f"{scenario_totals[scenario]:,.2f}" if item == scenario else "—"
                  for item in CORRELATION_SCENARIOS),
            )
        )
    binding_scenario = max(scenario_totals, key=scenario_totals.get)
    sbm = scenario_totals[binding_scenario]
    total_sa = sbm + drc + RRAO
    rows.append(
        (
            "Linear instruments curvature",
            "asserted exactly zero",
            "0.00",
            "0.00",
            "0.00",
        )
    )
    rows.extend(
        [
            ("DRC", "total", f"{drc:,.2f}", "—", "—"),
            ("JTD", "single-name CDS", f"{jtd['single-name CDS']:,.2f}", "—", "—"),
            ("JTD", "CDX IG", f"{jtd['CDX IG']:,.2f}", "—", "—"),
            ("DRC hedge benefit ratio", "HBR", f"{hbr:.4f}", "—", "—"),
            ("RRAO", "no exotic/non-vanilla payoffs", f"{RRAO:,.2f}", "—", "—"),
            (
                "Primary SA capital (CDX bucket 17)",
                binding_scenario,
                f"{total_sa:,.2f}",
                "—",
                "—",
            ),
        ]
    )
    headers = ("Risk class / component", "Measure / scenario", "Medium", "High", "Low")
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print(f"Binding correlation scenario: {binding_scenario}.")
    return total_sa, binding_scenario


def main() -> None:
    if SENSITIVITY_VECTOR.shape != (len(FACTOR_NAMES),):
        raise ValueError(
            f"Sensitivity vector must have shape ({len(FACTOR_NAMES)},); "
            f"got {SENSITIVITY_VECTOR.shape}."
        )
    if set(FACTOR_BUCKET_BY_NAME) != set(FACTOR_NAMES):
        raise ValueError("Factor bucket assignments do not cover every risk factor.")
    if set(CSR_BUCKET_BY_FACTOR) != {
        factor for factor in FACTOR_NAMES if factor.startswith("CSR_")
    }:
        raise ValueError("CSR bucket assignments do not cover every CSR factor.")

    spot, volatility = _load_curvature_levels()
    _print_calculation_diagnostics(spot, volatility)
    scenario_charges: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in CORRELATION_SCENARIOS:
        # MAR21 prescribes the IG corporate index in its separate bucket 17.
        delta = _delta_charge(scenario, cdx_index_bucket=True)
        scenario_charges[scenario] = {
            "delta": delta,
            "vega": {"EQ": _vega_charge(scenario)},
            "curvature": {"EQ": _curvature_charge(spot, volatility, scenario)},
        }

    drc, jtd, hbr = _drc_charge()
    primary_sa_capital, primary_binding_scenario = _print_table(
        scenario_charges, drc, jtd, hbr
    )
    bucket4_totals = {}
    for scenario in CORRELATION_SCENARIOS:
        bucket4_delta = _delta_charge(scenario)
        bucket4_totals[scenario] = (
            sum(bucket4_delta.values())
            + _vega_charge(scenario)
            + _curvature_charge(spot, volatility, scenario)
            + drc
            + RRAO
        )
    print(
        "Bucket-4 simplification sensitivity check: CDX combined with IG "
        "industrials (RW 3%; inter-name rho 35%):"
    )
    for factor_name, sensitivity, risk_weight, weighted in _csr_weighted_sensitivities():
        print(
            f"  {factor_name}: sensitivity={sensitivity:,.2f}/bp, "
            f"RW={risk_weight:.4f}, WS={weighted:,.2f}"
        )
    print(
        "Bucket-4 alternative SA total by correlation scenario: "
        + ", ".join(
            f"{scenario}={bucket4_totals[scenario]:,.2f}"
            for scenario in CORRELATION_SCENARIOS
        )
    )
    bucket4_binding = max(bucket4_totals, key=bucket4_totals.get)
    bucket4_sa_capital = bucket4_totals[bucket4_binding]
    print(
        f"Bucket-4 simplification alternative SA capital: "
        f"{bucket4_sa_capital:,.2f} ({bucket4_binding} scenario binds)."
    )
    primary_ratio = primary_sa_capital / IMA_CAPITAL
    bucket4_ratio = bucket4_sa_capital / IMA_CAPITAL
    print("Final comparison:")
    print(f"  SA capital  : {primary_sa_capital:,.2f} (CDX bucket 17, primary)")
    print(f"  IMA capital : {IMA_CAPITAL:,.2f}")
    print(f"  SA / IMA    : {primary_ratio:.2f}")
    print(
        f"  Bucket-4 simplification alternative: "
        f"{bucket4_sa_capital:,.2f}, ratio {bucket4_ratio:.2f}"
    )
    print(
        "Basel DRC source check: MAR22.24 assigns BBB 6% and A 3%; "
        "the values labelled BBB 3% and A 2% in the supplied note do not match "
        "the official table."
    )
    print(
        "CDX CSR treatment: primary calculation uses prescribed IG-index "
        "bucket 17 (1.5% RW); the bucket-4 combined treatment (3% RW, 35% "
        "inter-name rho) is shown as a documented sensitivity check."
    )
    print(
        f"Curvature option inputs: spot={spot:,.2f}, vol={volatility:.4%}, "
        f"tau={SPX_CALL_TAU:.6f}, notional={SPX_CALL_NOTIONAL:,.0f}."
    )
    print(
        "Vega assumption: 0.999 rho for the same SPX underlying at distinct strikes."
    )
    print(
        "Vega RW source: MAR21.92 Table 13 assigns 20d to large-cap/index equity "
        "vega (77.78%); the 10d figure is the scaling base in its formula."
    )
    if all(
        scenario_charges[scenario]["curvature"].get("EQ", 0.0) == 0.0
        for scenario in CORRELATION_SCENARIOS
    ):
        print(
            "SPX curvature accepted at zero: the long call's positive gamma means "
            "both spot shocks gain beyond delta, so neither CVR is a positive loss. "
            "This is the same convexity that drives the HPL-RTPL gap in Block 5."
        )
    if primary_binding_scenario == "low":
        print(
            "Finding: the low-correlation scenario binds; this is consistent with "
            "the offsetting single-name/CDX credit positions."
        )


if __name__ == "__main__":
    main()
