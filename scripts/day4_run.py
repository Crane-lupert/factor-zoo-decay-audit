"""Day 4-5: capacity-adjusted survival curve.

Inputs:
    results/day3/decay_per_factor.parquet      per-signal decay + mechanism
    cache/oap_ls_returns.parquet                monthly LS returns

Outputs (results/day4/):
    capacity_adjusted.parquet     long-form per (Acronym, AUM) capacity-adj. Sharpe
    survival_curve.csv            per-(AUM, mechanism) viable fraction
    sensitivity.csv               viable fraction under parameter perturbations
    capacity_report.txt           summary text
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.capacity import (  # noqa: E402
    DEFAULT_AUM_SCENARIOS_USD,
    SHARPE_VIABILITY_THRESHOLD,
    annual_cost_bps,
    capacity_adjust,
    cost_per_side_per_rebalance,
    per_signal_volatility,
    run_sensitivity,
    survival_curve,
)
from factor_zoo_decay_audit.load import load_ls_returns  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "day4"
DECAY_PARQUET = REPO_ROOT / "results" / "day3" / "decay_per_factor.parquet"


def fmt_aum(aum: float) -> str:
    if aum >= 1e9:
        return f"${aum/1e9:.0f}B"
    return f"${aum/1e6:.0f}M"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decay = pd.read_parquet(DECAY_PARQUET)
    ls = load_ls_returns()

    print(f"factors with decay metrics: {len(decay)}    eligible: {int(decay['analysis_eligible'].sum())}")

    vol = per_signal_volatility(ls)
    print(f"per-signal volatility rows: {len(vol)}    "
          f"median monthly std: {vol['monthly_std_pct'].median():.2f} pct")

    print()
    print("--- cost model parameters ---")
    print("  half-spread          : 5 bps (effective)")
    print("  impact coefficient   : 10 bps per 1% of universe monthly $ volume")
    print("  universe monthly $vol: $1T (CRSP common-stock baseline)")
    print("  monthly turnover     : 50% of AUM (both sides)")
    print("  rebalances per year  : 12, sides=2")
    print()
    print("  Per-AUM cost preview:")
    for aum in DEFAULT_AUM_SCENARIOS_USD:
        cps, part = cost_per_side_per_rebalance(aum)
        ann = annual_cost_bps(aum)
        print(f"    {fmt_aum(aum):<7}: participation={part:>7.4f}%  "
              f"cost/side/rebal={cps:>7.2f} bps  annual={ann:>8.0f} bps "
              f"({ann/100:.2f} pct/yr)")

    print()
    adj = capacity_adjust(decay, vol)
    adj.to_parquet(OUT_DIR / "capacity_adjusted.parquet", index=False)
    print(f"capacity-adjusted rows: {len(adj)}  "
          f"({len(decay)} factors x {len(DEFAULT_AUM_SCENARIOS_USD)} AUMs)")

    print()
    print("--- survival curve ---")
    sc = survival_curve(adj)
    sc.to_csv(OUT_DIR / "survival_curve.csv", index=False)
    pivot = (
        sc[sc["mechanism"].isin(["behavioral", "mispricing", "risk_premium", "ALL"])]
        .pivot_table(index="aum_usd", columns="mechanism",
                     values=["n_viable", "n_factors", "viable_pct"])
    )
    print()
    print("Viable factor count (capacity-adjusted Sharpe >= 0.30):")
    cnt = pivot["n_viable"].astype(int)
    cnt.index = [fmt_aum(x) for x in cnt.index]
    print(cnt.to_string())
    print()
    print("Viable factor pct:")
    pct = pivot["viable_pct"]
    pct.index = [fmt_aum(x) for x in pct.index]
    print((pct * 100).round(1).to_string())

    print()
    print("--- sensitivity (overall ALL viable_pct under perturbations) ---")
    sens = run_sensitivity(decay, vol)
    sens.to_csv(OUT_DIR / "sensitivity.csv", index=False)
    sens_pivot = sens.pivot_table(index="perturbation", columns="aum_usd", values="viable_pct")
    sens_pivot.columns = [fmt_aum(x) for x in sens_pivot.columns]
    print((sens_pivot * 100).round(1).to_string())

    # Save text report
    lines = ["Day-4/5 capacity-adjusted survival curve",
             "=========================================",
             f"factors eligible: {int(decay['analysis_eligible'].sum())}/{len(decay)}",
             "",
             "Annual cost (bps) by AUM:"]
    for aum in DEFAULT_AUM_SCENARIOS_USD:
        lines.append(f"  {fmt_aum(aum):<7}: {annual_cost_bps(aum):.0f} bps/yr")
    lines.extend(["",
                  f"Survival threshold: capacity_sharpe >= {SHARPE_VIABILITY_THRESHOLD}",
                  "",
                  "Viable counts:",
                  cnt.to_string(),
                  "",
                  "Viable %:",
                  (pct * 100).round(1).to_string(),
                  "",
                  "Sensitivity (overall viable_pct):",
                  (sens_pivot * 100).round(1).to_string()])
    (OUT_DIR / "capacity_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # Day-5 advance gate per CLAUDE.md: 212 factor x 4 AUM scenario completed
    n_factors = decay["Acronym"].nunique()
    n_complete = adj.dropna(subset=["capacity_sharpe"])["Acronym"].nunique()
    gate = n_complete == n_factors and len(DEFAULT_AUM_SCENARIOS_USD) == 4
    print()
    print(f"Day-5 advance gate (212 factor x 4 AUM scenario complete): "
          f"{'PASS' if gate else 'FAIL'} ({n_complete}/{n_factors} complete)")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
