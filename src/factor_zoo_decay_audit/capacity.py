"""Frazzini-Israel-Moskowitz (2018) -- style simplified capacity model.

The point of this module is the SURVIVAL CURVE: how many factors retain a
meaningful Sharpe (>= 0.30 by default) as AUM grows from $100M to $100B.

Cost model (per side, per rebalance, in basis points)
-----------------------------------------------------
    cost_bps = HALF_SPREAD + IMPACT_COEF * participation_pct

    participation_pct = (trade_$ / universe_monthly_$_volume) * 100

Defaults (calibrated to FIM 2018 Table 7 / Novy-Marx-Velikov 2016 Table 5):
    HALF_SPREAD       = 5 bps   (effective half-spread, large-cap CRSP)
    IMPACT_COEF       = 10 bps  (impact per 1% of universe monthly volume)
    UNIVERSE_VOLUME   = $1T / month (representative CRSP common-stock $ volume,
                                     2010-2020 average; 1990s ~$200B/m so this
                                     is a conservative MODERN baseline)
    MONTHLY_TURNOVER  = 0.50    (50% of AUM traded per rebalance, both sides;
                                  approximates academic LS portfolios with
                                  monthly rebalance)
    REBALANCES_PER_YR = 12      (monthly rebalance)
    SIDES             = 2       (long + short legs trade together)

Capacity-adjusted Sharpe
------------------------
    annual_cost_bps    = REBALANCES_PER_YR * SIDES * cost_bps
    annual_cost_pct    = annual_cost_bps / 100        (bps -> percent)
    annualized_std_pct = monthly_std_pct * sqrt(12)
    sharpe_haircut     = annual_cost_pct / annualized_std_pct
    capacity_sharpe    = post_pub_sharpe - sharpe_haircut

The haircut subtracts cost from the *return numerator* of the Sharpe ratio;
costs are deterministic at the portfolio level (do not change std meaningfully).

Limitations
-----------
* Single representative universe ADV -- factors that trade smaller stocks
  (size, microcap value) face higher impact than this model implies.
* Linear impact (no concavity / sqrt component) -- for very large AUM, real
  costs grow more slowly than this. Therefore the high-AUM tail of our
  survival curve is conservative (pessimistic).
* Turnover is signal-agnostic -- in reality momentum is ~200% annual, value
  ~30%. We hold turnover constant as a first-order approximation; sensitivity
  is reported in `run_sensitivity`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default parameters -- documented above.
HALF_SPREAD_BPS = 5.0
IMPACT_COEF_BPS_PER_PCT = 10.0
UNIVERSE_MONTHLY_VOLUME_USD = 1_000_000_000_000.0   # $1T
MONTHLY_TURNOVER = 0.50
REBALANCES_PER_YEAR = 12
SIDES = 2

DEFAULT_AUM_SCENARIOS_USD = (1e8, 1e9, 1e10, 1e11)   # $100M, $1B, $10B, $100B

SHARPE_VIABILITY_THRESHOLD = 0.30


@dataclass(frozen=True)
class CostParams:
    half_spread_bps: float = HALF_SPREAD_BPS
    impact_coef_bps_per_pct: float = IMPACT_COEF_BPS_PER_PCT
    universe_monthly_volume_usd: float = UNIVERSE_MONTHLY_VOLUME_USD
    monthly_turnover: float = MONTHLY_TURNOVER
    rebalances_per_year: int = REBALANCES_PER_YEAR
    sides: int = SIDES


def per_signal_volatility(ls_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-signal monthly LS return std (for converting cost bps to Sharpe)."""
    rets = ls_returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    g = rets.groupby("signalname")["ret"]
    return pd.DataFrame({
        "Acronym": g.size().index,
        "monthly_std_pct": g.std(ddof=1).values,
        "n_months_total": g.size().values,
    })


def cost_per_side_per_rebalance(
    aum_usd: float, params: CostParams = CostParams()
) -> tuple[float, float]:
    """Returns (cost_bps_per_side, participation_pct)."""
    trade_dollars = aum_usd * params.monthly_turnover
    participation_pct = (trade_dollars / params.universe_monthly_volume_usd) * 100.0
    cost_bps = params.half_spread_bps + params.impact_coef_bps_per_pct * participation_pct
    return cost_bps, participation_pct


def annual_cost_bps(aum_usd: float, params: CostParams = CostParams()) -> float:
    cost_per_side, _ = cost_per_side_per_rebalance(aum_usd, params)
    return params.rebalances_per_year * params.sides * cost_per_side


def capacity_adjust(
    decay_per_factor: pd.DataFrame,
    volatility: pd.DataFrame,
    aum_scenarios_usd: tuple[float, ...] = DEFAULT_AUM_SCENARIOS_USD,
    params: CostParams = CostParams(),
) -> pd.DataFrame:
    """Long-form DataFrame with capacity-adjusted Sharpe at each AUM.

    Columns:
        Acronym, mechanism, is_sharpe, post_sharpe, monthly_std_pct,
        annualized_std_pct, aum_usd, annual_cost_bps, sharpe_haircut,
        capacity_sharpe, viable (capacity_sharpe >= 0.3)
    """
    base = decay_per_factor.merge(volatility, on="Acronym", how="left").copy()
    base["annualized_std_pct"] = base["monthly_std_pct"] * np.sqrt(12)

    rows: list[dict] = []
    for aum in aum_scenarios_usd:
        cost_bps_yr = annual_cost_bps(aum, params)
        cost_pct_yr = cost_bps_yr / 100.0
        haircut = cost_pct_yr / base["annualized_std_pct"]
        cap_sharpe = base["post_sharpe"] - haircut
        for r, h, cs in zip(base.itertuples(index=False), haircut, cap_sharpe):
            rows.append({
                "Acronym": r.Acronym,
                "mechanism": r.mechanism,
                "is_sharpe": r.is_sharpe,
                "post_sharpe": r.post_sharpe,
                "monthly_std_pct": r.monthly_std_pct,
                "annualized_std_pct": r.annualized_std_pct,
                "analysis_eligible": r.analysis_eligible,
                "aum_usd": aum,
                "annual_cost_bps": cost_bps_yr,
                "sharpe_haircut": float(h) if pd.notna(h) else float("nan"),
                "capacity_sharpe": float(cs) if pd.notna(cs) else float("nan"),
                "viable": bool(cs >= SHARPE_VIABILITY_THRESHOLD) if pd.notna(cs) else False,
            })
    return pd.DataFrame(rows)


def survival_curve(adjusted: pd.DataFrame, *, threshold: float = SHARPE_VIABILITY_THRESHOLD) -> pd.DataFrame:
    """Aggregate survival curve: count of viable factors by (AUM, mechanism)."""
    src = adjusted[adjusted["analysis_eligible"]].copy()
    src["viable"] = src["capacity_sharpe"] >= threshold
    grp = (
        src.groupby(["aum_usd", "mechanism"], dropna=False)
        .agg(
            n_factors=("Acronym", "count"),
            n_viable=("viable", "sum"),
            mean_capacity_sharpe=("capacity_sharpe", "mean"),
            median_capacity_sharpe=("capacity_sharpe", "median"),
        )
        .reset_index()
    )
    grp["viable_pct"] = grp["n_viable"] / grp["n_factors"]

    # Also overall (across mechanisms)
    overall = (
        src.groupby("aum_usd")
        .agg(
            n_factors=("Acronym", "count"),
            n_viable=("viable", "sum"),
            mean_capacity_sharpe=("capacity_sharpe", "mean"),
            median_capacity_sharpe=("capacity_sharpe", "median"),
        )
        .reset_index()
    )
    overall["mechanism"] = "ALL"
    overall["viable_pct"] = overall["n_viable"] / overall["n_factors"]

    return pd.concat([grp, overall[grp.columns]], ignore_index=True)


# =============================================================================
# v2: per-factor turnover + cap-tier ADV + sqrt-linear hybrid impact
# =============================================================================
#
# Calibration sources:
#   * Novy-Marx & Velikov (2016) Tables 4-5: per-factor turnover by category.
#   * Frazzini-Israel-Moskowitz (2018) Tables 6-7: linear + nonlinear impact.
#   * Almgren et al. (2005): square-root impact term.
#
# Universe daily $ volume by tier (CRSP common stocks, period-averaged):
#   large_cap     : NYSE-only signals             ~$30B/day
#   ex_microcap   : price>5 / ME>NYSE 20pct       ~$40B/day
#   full_universe : default                       ~$50B/day
#
# Half-spread by tier (effective, period-averaged):
#   large_cap   : 5 bps    ex_microcap : 10 bps    full_universe : 15 bps
#
# Cost function per side per rebalance:
#   cost_bps = half_spread + LINEAR_COEF * part_pct + SQRT_COEF * sqrt(part_pct)
# with LINEAR_COEF = 5, SQRT_COEF = 20 (FIM-2018 / NMV-2016 calibrated).
#
# Borrow cost on short leg, annualized bps by tier:
#   large_cap : 50    ex_microcap : 150    full_universe : 300

TIER_PARAMS_V2 = {
    "large_cap":     {"universe_daily_volume_usd": 30e9, "half_spread_bps": 5,
                      "borrow_cost_bps_yr": 50},
    "ex_microcap":   {"universe_daily_volume_usd": 40e9, "half_spread_bps": 10,
                      "borrow_cost_bps_yr": 150},
    "full_universe": {"universe_daily_volume_usd": 50e9, "half_spread_bps": 15,
                      "borrow_cost_bps_yr": 300},
}

LINEAR_COEF_BPS_PER_PCT = 5.0
SQRT_COEF_BPS_PER_PCT_SQRT = 20.0
EXECUTION_DAYS_PER_REBALANCE = 5

# Per-side monthly turnover (NMV-2016 Table 5 calibration). High = trades a lot.
TURNOVER_BY_CAT_ECONOMIC = {
    "momentum": 0.30,
    "short-term reversal": 0.45,
    "lead lag": 0.25,
    "long term reversal": 0.10,
    "accruals": 0.08,
    "external financing": 0.06,
    "investment": 0.06,
    "investment alt": 0.06,
    "asset composition": 0.06,
    "sales growth": 0.08,
    "valuation": 0.04,
    "profitability": 0.03,
    "profitability alt": 0.03,
    "earnings forecast": 0.10,
    "earnings growth": 0.10,
    "composite accounting": 0.05,
    "R&D": 0.04,
    "payout indicator": 0.04,
    "leverage": 0.05,
    "short sale constraints": 0.10,
    "volatility": 0.15,
    "risk": 0.10,
    "liquidity": 0.20,
    "volume": 0.20,
    "optionrisk": 0.20,
    "size": 0.04,
}
DEFAULT_TURNOVER = 0.10


def assign_cap_tier(filter_str: str | float | None) -> str:
    """Map OAP doc Filter string (NaN/'NYSEonly'/'me>nyse20pct'/...) to tier."""
    if filter_str is None or (isinstance(filter_str, float) and pd.isna(filter_str)):
        return "full_universe"
    s = str(filter_str).lower()
    if "nyse" in s and "20" not in s and "pct" not in s and ">" not in s:
        return "large_cap"
    if "20pct" in s or "me>" in s or "me_gt" in s or "price" in s or "abs(prc)" in s:
        return "ex_microcap"
    return "full_universe"


def assign_turnover(cat_economic: str | float | None) -> float:
    """Map OAP doc Cat.Economic to per-side monthly turnover."""
    if cat_economic is None or (isinstance(cat_economic, float) and pd.isna(cat_economic)):
        return DEFAULT_TURNOVER
    return TURNOVER_BY_CAT_ECONOMIC.get(str(cat_economic), DEFAULT_TURNOVER)


def build_factor_profiles(signal_doc: pd.DataFrame) -> pd.DataFrame:
    """Per-Acronym capacity profile: cap_tier + monthly turnover."""
    df = signal_doc[["Acronym", "Filter", "Cat.Economic"]].copy()
    df["cap_tier"] = df["Filter"].apply(assign_cap_tier)
    df["monthly_turnover_per_side"] = df["Cat.Economic"].apply(assign_turnover)
    df["universe_daily_volume_usd"] = df["cap_tier"].map(
        lambda t: TIER_PARAMS_V2[t]["universe_daily_volume_usd"]
    )
    df["half_spread_bps"] = df["cap_tier"].map(lambda t: TIER_PARAMS_V2[t]["half_spread_bps"])
    df["borrow_cost_bps_yr"] = df["cap_tier"].map(lambda t: TIER_PARAMS_V2[t]["borrow_cost_bps_yr"])
    return df


def cost_v2_per_side_per_rebalance(
    aum_usd: float, profile_row: pd.Series,
    *, linear_coef: float = LINEAR_COEF_BPS_PER_PCT,
    sqrt_coef: float = SQRT_COEF_BPS_PER_PCT_SQRT,
    execution_days: int = EXECUTION_DAYS_PER_REBALANCE,
) -> tuple[float, float]:
    """Return (cost_bps_per_side, daily_participation_pct).

    Uses sqrt-linear hybrid impact; participation in DAILY universe volume
    averaged over the execution window (default 5 trading days).
    """
    turnover = profile_row["monthly_turnover_per_side"]
    universe_daily = profile_row["universe_daily_volume_usd"]
    half_spread = profile_row["half_spread_bps"]
    trade_per_day = aum_usd * turnover / execution_days
    part_pct = trade_per_day / universe_daily * 100.0
    cost_bps = half_spread + linear_coef * part_pct + sqrt_coef * np.sqrt(max(part_pct, 0.0))
    return cost_bps, part_pct


def annual_cost_v2_bps(
    aum_usd: float, profile_row: pd.Series,
    *, rebalances_per_year: int = REBALANCES_PER_YEAR, sides: int = SIDES,
    include_borrow: bool = True,
) -> tuple[float, float]:
    """Return (annual_cost_bps_total, annual_borrow_bps).

    Total = trading cost (both sides x rebalances) + borrow cost on short side.
    """
    cps, _ = cost_v2_per_side_per_rebalance(aum_usd, profile_row)
    trading_cost = rebalances_per_year * sides * cps
    borrow = profile_row["borrow_cost_bps_yr"] if include_borrow else 0.0
    return trading_cost + borrow, borrow


def capacity_adjust_v2(
    decay_per_factor: pd.DataFrame,
    volatility: pd.DataFrame,
    profiles: pd.DataFrame,
    aum_scenarios_usd: tuple[float, ...] = DEFAULT_AUM_SCENARIOS_USD,
    *,
    include_borrow: bool = True,
) -> pd.DataFrame:
    """v2 capacity-adjusted Sharpe with per-factor turnover + tier ADV + sqrt impact + borrow."""
    base = decay_per_factor.merge(volatility, on="Acronym", how="left")
    base = base.merge(profiles, on="Acronym", how="left")
    base["annualized_std_pct"] = base["monthly_std_pct"] * np.sqrt(12)

    rows: list[dict] = []
    for _, r in base.iterrows():
        if pd.isna(r.get("monthly_turnover_per_side")):
            continue
        for aum in aum_scenarios_usd:
            cps, part_pct = cost_v2_per_side_per_rebalance(aum, r)
            ann_total, ann_borrow = annual_cost_v2_bps(aum, r, include_borrow=include_borrow)
            ann_trading = ann_total - ann_borrow
            haircut = (ann_total / 100.0) / r["annualized_std_pct"] if r["annualized_std_pct"] > 0 else float("nan")
            cap_sharpe = r["post_sharpe"] - haircut if pd.notna(r["post_sharpe"]) else float("nan")
            rows.append({
                "Acronym": r["Acronym"],
                "mechanism": r.get("mechanism"),
                "cap_tier": r["cap_tier"],
                "monthly_turnover_per_side": r["monthly_turnover_per_side"],
                "is_sharpe": r.get("is_sharpe"),
                "post_sharpe": r.get("post_sharpe"),
                "annualized_std_pct": r["annualized_std_pct"],
                "analysis_eligible": r.get("analysis_eligible", False),
                "aum_usd": aum,
                "daily_participation_pct": part_pct,
                "cost_per_side_per_rebal_bps": cps,
                "annual_trading_bps": ann_trading,
                "annual_borrow_bps": ann_borrow,
                "annual_total_cost_bps": ann_total,
                "sharpe_haircut": haircut,
                "capacity_sharpe": cap_sharpe,
                "viable": bool(cap_sharpe >= SHARPE_VIABILITY_THRESHOLD) if pd.notna(cap_sharpe) else False,
            })
    return pd.DataFrame(rows)


def survival_curve_v2(adjusted: pd.DataFrame, *, threshold: float = SHARPE_VIABILITY_THRESHOLD) -> pd.DataFrame:
    src = adjusted[adjusted["analysis_eligible"]].copy()
    src["viable"] = src["capacity_sharpe"] >= threshold
    grp = (
        src.groupby(["aum_usd", "mechanism"], dropna=False)
        .agg(n_factors=("Acronym", "count"),
             n_viable=("viable", "sum"),
             mean_capacity_sharpe=("capacity_sharpe", "mean"),
             median_capacity_sharpe=("capacity_sharpe", "median"),
             mean_total_cost_bps=("annual_total_cost_bps", "mean"))
        .reset_index()
    )
    grp["viable_pct"] = grp["n_viable"] / grp["n_factors"]
    overall = (
        src.groupby("aum_usd")
        .agg(n_factors=("Acronym", "count"),
             n_viable=("viable", "sum"),
             mean_capacity_sharpe=("capacity_sharpe", "mean"),
             median_capacity_sharpe=("capacity_sharpe", "median"),
             mean_total_cost_bps=("annual_total_cost_bps", "mean"))
        .reset_index()
    )
    overall["mechanism"] = "ALL"
    overall["viable_pct"] = overall["n_viable"] / overall["n_factors"]
    return pd.concat([grp, overall[grp.columns]], ignore_index=True)


def run_sensitivity(
    decay_per_factor: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    base_params: CostParams = CostParams(),
    aum_scenarios_usd: tuple[float, ...] = DEFAULT_AUM_SCENARIOS_USD,
) -> pd.DataFrame:
    """Vary key parameters one at a time and re-compute survival rates.

    Reports overall (ALL mechanisms) viable fraction at each AUM.
    """
    perturbations: list[tuple[str, CostParams]] = [
        ("base", base_params),
        ("kappa_x2",
         CostParams(impact_coef_bps_per_pct=base_params.impact_coef_bps_per_pct * 2,
                    half_spread_bps=base_params.half_spread_bps,
                    universe_monthly_volume_usd=base_params.universe_monthly_volume_usd,
                    monthly_turnover=base_params.monthly_turnover)),
        ("kappa_half",
         CostParams(impact_coef_bps_per_pct=base_params.impact_coef_bps_per_pct * 0.5,
                    half_spread_bps=base_params.half_spread_bps,
                    universe_monthly_volume_usd=base_params.universe_monthly_volume_usd,
                    monthly_turnover=base_params.monthly_turnover)),
        ("turnover_x2",
         CostParams(monthly_turnover=base_params.monthly_turnover * 2,
                    impact_coef_bps_per_pct=base_params.impact_coef_bps_per_pct,
                    half_spread_bps=base_params.half_spread_bps,
                    universe_monthly_volume_usd=base_params.universe_monthly_volume_usd)),
        ("turnover_half",
         CostParams(monthly_turnover=base_params.monthly_turnover * 0.5,
                    impact_coef_bps_per_pct=base_params.impact_coef_bps_per_pct,
                    half_spread_bps=base_params.half_spread_bps,
                    universe_monthly_volume_usd=base_params.universe_monthly_volume_usd)),
        ("universe_half",
         CostParams(universe_monthly_volume_usd=base_params.universe_monthly_volume_usd * 0.5,
                    impact_coef_bps_per_pct=base_params.impact_coef_bps_per_pct,
                    half_spread_bps=base_params.half_spread_bps,
                    monthly_turnover=base_params.monthly_turnover)),
    ]
    out_rows: list[dict] = []
    for name, p in perturbations:
        adj = capacity_adjust(decay_per_factor, volatility, aum_scenarios_usd, p)
        sc = survival_curve(adj)
        all_only = sc[sc["mechanism"] == "ALL"]
        for _, r in all_only.iterrows():
            out_rows.append({
                "perturbation": name,
                "aum_usd": r["aum_usd"],
                "n_factors": r["n_factors"],
                "n_viable": r["n_viable"],
                "viable_pct": r["viable_pct"],
                "annual_cost_bps": annual_cost_bps(r["aum_usd"], p),
            })
    return pd.DataFrame(out_rows)
