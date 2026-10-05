# frtb-risk-engine

## Purpose
Build a self-contained market risk engine over a small cross-asset portfolio,
producing: historical/parametric/MC VaR and ES, FRTB IMA capital (liquidity-
horizon ES, stressed calibration, IMCC), FRTB SA capital (SBM delta + curvature,
DRC), and the P&L Attribution test linking the two.


## Portfolio (fixed — do not change)
| Instrument | Position | Risk classes |
|---|---|---|
| GBP 5y receiver swap | £100m | GIRR |
| 10y Gilt | £50m long | GIRR |
| CDX IG 5y | £75m protection sold | CSR non-sec |
| Single-name CDS 5y | £50m protection bought | CSR non-sec, DRC |
| SPX 3m ATM call | $20m notional | Equity, Vega, Curvature |
| GBPUSD 3m forward | $25m | FX |

## Risk factors (20)
GIRR  — GBP OIS zero curve: 1y, 2y, 5y, 10y, 30y
CSR   — single-name spread curve: 1y, 3y, 5y, 10y
CSR   — CDX IG spread curve: 1y, 3y, 5y, 10y
EQ    — SPX spot
VOL   — SPX 3m implied vol at 90% and 100% moneyness
FX    — GBPUSD spot

## Locked conventions

**Shock convention (per factor class):**
- GIRR, CSR → ABSOLUTE (basis points).
  Relative shocks explode as rates approach zero (a 5bp move at 0.15% is a 33%
  return; applied to a 4% level it becomes 133bp) and are undefined/sign-flipping
  at negative rates.
- Equity, FX, Vol → RELATIVE.
  Equity/vol are scale-free with a floor at zero. FX relative shocks are
  invariant under inversion of the quote convention; absolute shocks are not,
  so VaR would depend on an arbitrary quoting choice.

**Curve interpolation: LINEAR ON ZERO RATES.**
Keeps bucketed deltas local and additive. Cubic splines leak sensitivity
non-locally and produce sign-flipping off-pillar deltas.

**Sensitivities are bucketed onto pillars, never a single total DV01.**
Off-pillar sensitivity for instruments with no cashflows in that region is
exactly zero under linear interpolation. This is a structural result, not an
approximation — do not smooth it away.

**CDS off-pillar CS01 arises only from the risky annuity.**
A par CDS has zero off-pillar CS01 by construction (the bootstrapper pins the
par spread, so protection and premium legs move together). Non-zero off-pillar
CS01 comes from the annuity moving against the (par spread − fixed coupon)
wedge on an off-par contract.

**Vega: sticky-strike sensitivities, relative shocks.**
Delta and vega treated as orthogonal in the linear P&L. This deliberately omits
the vanna cross-term, which is the intended dominant contributor to the
HPL−RTPL gap in the PLA test. DO NOT "fix" this.

## Sensitivity vector (hardcoded spec — per +1bp or per +1%)
GIRR per +1bp (£):
  swap:  1y −370, 2y −1,833, 5y −42,314, 10y 0, 30y 0   (total DV01 −44,517)
  gilt:  1y −60, 2y −310, 5y −3,100, 10y −37,500, 30y 0
CSR per +1bp (£):
  single-name (protection bought): 1y −40, 3y −110, 5y +21,000, 10y 0
  CDX (protection sold):           5y −31,000
EQ  per +1%: £110,000
VOL per vol point: 90% strike £9,000, 100% strike £24,000
FX  per +1%: £198,000

## Data
yfinance, 3y daily: ^SPX, ^VIX, TLT, LQD, HYG, GBPUSD=X.
Credit spread curve nodes are SYNTHESISED from LQD/HYG with a spread-vol
overlay. This is a documented proxy, not real CDS data — label it as such in
every output. Align dates across series, forward-fill gaps ≤2 days, drop the rest.

## Build order (one block at a time; test before advancing)
1. Factors, history, shock matrix, sensitivity vector, RTPL + HPL vectors
2. VaR/ES: historical sim, parametric (EWMA λ=0.94), Monte Carlo.
   99% VaR, 97.5% ES. Backtest: exceptions, Kupiec POF, Basel traffic light
3. FRTB IMA: liquidity-horizon ES (10/20/40/60/120d buckets), stressed period
   selection by sliding 12m window, full/reduced scaling ratio,
   IMCC = 0.5·IMCC_global + 0.5·Σ IMCC_i, NMRF/SES stub
4. FRTB SA: SBM delta for GIRR/CSR/EQ/FX with prescribed risk weights,
   intra-bucket ρ and cross-bucket γ, three correlation scenarios (take max).
   Curvature on the SPX option. DRC via JTD on the two credit names
5. PLA: Spearman correlation and KS test on HPL vs RTPL, traffic-light zone.
   Final comparison table: HS VaR, HS ES, IMA capital, SA capital, ratio

## Non-goals
No live market data feeds. No real CDS bootstrapping from quoted spreads
(sensitivities are spec-given). No performance optimisation. NMRF is a stub.

## Output discipline
Every capital number printed with its inputs. Every proxy labelled as a proxy.
If a result looks wrong, surface it rather than tuning parameters to hide it.