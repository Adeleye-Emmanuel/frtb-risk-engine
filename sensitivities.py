from __future__ import annotations

import numpy as np

from factors import FACTOR_NAMES, FACTORS

EXPECTED_ORDER = [
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

actual_order = [factor.name for factor in FACTORS]
assert actual_order == EXPECTED_ORDER, f"Sensitivity ordering mismatch: expected {EXPECTED_ORDER}, got {actual_order}"

# Sensitivities are the hedged bucket deltas from the fixed portfolio in PROJECT.md.
# Values are in pounds per +1bp for absolute factors and per +1% for relative factors.
SENSITIVITY_VECTOR = np.array(
    [
        -370.0,
        -1833.0,
        -42314.0,
        0.0,
        0.0,
        -40.0,
        -110.0,
        21000.0,
        0.0,
        0.0,
        0.0,
        -31000.0,
        0.0,
        110000.0,
        9000.0,
        24000.0,
        198000.0,
    ],
    dtype=float,
)

assert SENSITIVITY_VECTOR.shape == (len(FACTORS),), (
    f"Sensitivity vector length mismatch: expected {len(FACTORS)}, got {SENSITIVITY_VECTOR.shape[0]}"
)
assert FACTOR_NAMES == EXPECTED_ORDER, f"Factor export mismatch: {FACTOR_NAMES}"

if __name__ == "__main__":
    print("Sensitivity vector length:", len(SENSITIVITY_VECTOR))
    print("First five sensitivities:", SENSITIVITY_VECTOR[:5])
