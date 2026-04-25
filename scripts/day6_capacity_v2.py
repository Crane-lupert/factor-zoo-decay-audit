"""Capacity v2 runner: per-factor turnover + cap-tier ADV + sqrt impact + borrow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.capacity import (  # noqa: E402
    DEFAULT_AUM_SCENARIOS_USD,
    SHARPE_VIABILITY_THRESHOLD,
    annual_cost_v2_bps,
    build_factor_profiles,
    capacity_adjust_v2,
    cost_v2_per_side_per_rebalance,
    per_signal_volatility,
    survival_curve_v2,
)
from factor_zoo_decay_audit.load import load_ls_returns, load_signal_doc  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "day6"
DECAY_PARQUET = REPO_ROOT / "results" / "day3" / "decay_per_factor.parquet"


def fmt_aum(a: float) -> str:
    return f"${a/1e9:.0f}B" if a >= 1e9 else f"${a/1e6:.0f}M"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decay = pd.read_parquet(DECAY_PARQUET)
    ls = load_ls_returns()
    doc = load_signal_doc()

    profiles = build_factor_profiles(doc)
    print(f"factor profiles built: {len(profiles)} signals")
    print("  cap_tier distribution:", profiles["cap_tier"].value_counts().to_dict())
    print("  monthly_turnover_per_side mean:",
          round(profiles["monthly_turnover_per_side"].mean(), 3),
          "median:", round(profiles["monthly_turnover_per_side"].median(), 3))
    profiles.to_csv(OUT_DIR / "factor_profiles.csv", index=False)

    vol = per_signal_volatility(ls)
    print()
    print("--- v2 cost preview (Mom6m as exemplar momentum, BM as exemplar value) ---")
    for sig_a in ["Mom6m", "BM"]:
        prof_row = profiles[profiles["Acronym"] == sig_a]
        if not len(prof_row):
            continue
        prof_row = prof_row.iloc[0]
        print(f"  {sig_a}  tier={prof_row['cap_tier']:<14} "
              f"MTP={prof_row['monthly_turnover_per_side']:.2f}  "
              f"ADV/day=${prof_row['universe_daily_volume_usd']/1e9:.0f}B  "
              f"borrow={prof_row['borrow_cost_bps_yr']} bps/yr")
        for aum in DEFAULT_AUM_SCENARIOS_USD:
            cps, part = cost_v2_per_side_per_rebalance(aum, prof_row)
            ann_total, ann_borrow = annual_cost_v2_bps(aum, prof_row)
            print(f"    {fmt_aum(aum):<7}: part={part:>7.4f}%  "
                  f"cost/side/rebal={cps:>7.2f} bps  "
                  f"annual_trading={ann_total - ann_borrow:>6.0f}  "
                  f"borrow={ann_borrow:>4.0f}  total={ann_total:>6.0f} bps/yr")

    print()
    adj = capacity_adjust_v2(decay, vol, profiles, include_borrow=True)
    adj.to_parquet(OUT_DIR / "capacity_adjusted_v2.parquet", index=False)
    print(f"capacity-adjusted v2 rows: {len(adj)}  ({len(decay)} factors x {len(DEFAULT_AUM_SCENARIOS_USD)} AUMs)")

    sc = survival_curve_v2(adj)
    sc.to_csv(OUT_DIR / "survival_curve_v2.csv", index=False)
    pivot_n = sc[sc["mechanism"].isin(["behavioral", "mispricing", "risk_premium", "ALL"])].pivot_table(
        index="aum_usd", columns="mechanism", values="n_viable"
    ).astype(int)
    pivot_n.index = [fmt_aum(x) for x in pivot_n.index]
    print()
    print("Viable factor count v2 (capacity_sharpe >= 0.30, with borrow):")
    print(pivot_n.to_string())

    pivot_pct = sc[sc["mechanism"].isin(["behavioral", "mispricing", "risk_premium", "ALL"])].pivot_table(
        index="aum_usd", columns="mechanism", values="viable_pct"
    )
    pivot_pct.index = [fmt_aum(x) for x in pivot_pct.index]
    print()
    print("Viable factor pct v2:")
    print((pivot_pct * 100).round(1).to_string())

    # without borrow comparison
    adj_no_borrow = capacity_adjust_v2(decay, vol, profiles, include_borrow=False)
    sc_nb = survival_curve_v2(adj_no_borrow)
    nb_pct = sc_nb[sc_nb["mechanism"] == "ALL"].set_index("aum_usd")["viable_pct"]
    nb_pct.index = [fmt_aum(x) for x in nb_pct.index]
    print()
    print("ALL viable_pct -- borrow contribution:")
    yes_pct = pivot_pct["ALL"]
    print(pd.DataFrame({"with_borrow": yes_pct, "no_borrow": nb_pct,
                        "borrow_haircut_pp": (nb_pct - yes_pct) * 100}).round(3).to_string())

    (OUT_DIR / "capacity_v2_report.txt").write_text(
        "Capacity v2 (per-factor turnover + tier ADV + sqrt impact + borrow)\n"
        "===================================================================\n"
        + pivot_n.to_string() + "\n\nViable %:\n"
        + (pivot_pct * 100).round(1).to_string() + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
