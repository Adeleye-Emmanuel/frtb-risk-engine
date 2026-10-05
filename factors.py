from dataclasses import dataclass


@dataclass(frozen=True)
class RiskFactor:
    name: str
    risk_class: str
    shock_convention: str


# NOTE: PROJECT.md says "20 risk factors" in the summary, but the explicit factor list
# enumerates 17 buckets. This file follows the explicit list as the canonical ordering so
# obligations downstream remain consistent with the documented factor definitions.
FACTOR_ORDER = [
    RiskFactor("GIRR_1Y", "GIRR", "ABSOLUTE"),
    RiskFactor("GIRR_2Y", "GIRR", "ABSOLUTE"),
    RiskFactor("GIRR_5Y", "GIRR", "ABSOLUTE"),
    RiskFactor("GIRR_10Y", "GIRR", "ABSOLUTE"),
    RiskFactor("GIRR_30Y", "GIRR", "ABSOLUTE"),
    RiskFactor("CSR_SINGLE_1Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_SINGLE_3Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_SINGLE_5Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_SINGLE_10Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_CDX_1Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_CDX_3Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_CDX_5Y", "CSR", "ABSOLUTE"),
    RiskFactor("CSR_CDX_10Y", "CSR", "ABSOLUTE"),
    RiskFactor("EQ_SPX", "EQ", "RELATIVE"),
    RiskFactor("VOL_SPX_90", "VOL", "RELATIVE"),
    RiskFactor("VOL_SPX_100", "VOL", "RELATIVE"),
    RiskFactor("FX_GBPUSD", "FX", "RELATIVE"),
]

FACTORS = FACTOR_ORDER
FACTOR_NAMES = [factor.name for factor in FACTORS]
RISK_CLASS_BY_NAME = {factor.name: factor.risk_class for factor in FACTORS}
SHOCK_CONVENTION_BY_NAME = {factor.name: factor.shock_convention for factor in FACTORS}


def validate_factor_order() -> None:
    expected = [
        "GIRR_1Y",
        "GIRR_2Y",
        "GIRR_5Y",
        "GIRR_10Y",
        "GIRR_30Y",
        "CSR_SINGLE_1Y",
        "CSR_SINGLE_3Y",
        "CSR_SINGLE_5Y",
        "CSR_SINGLE_10Y",
        "CSR_CDX_1Y",
        "CSR_CDX_3Y",
        "CSR_CDX_5Y",
        "CSR_CDX_10Y",
        "EQ_SPX",
        "VOL_SPX_90",
        "VOL_SPX_100",
        "FX_GBPUSD",
    ]
    assert FACTOR_NAMES == expected, f"Factor ordering mismatch: expected {expected}, got {FACTOR_NAMES}"
    assert len(FACTORS) == len(expected)


validate_factor_order()
