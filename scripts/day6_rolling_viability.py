"""Compute per-factor rolling-viability statistics from walk-forward annual Sharpes.

Long-horizon post-pub Sharpe (used in headline) averages over 5-30+ years and
can mask factors that work in most years but were dragged down by a few bad
windows. This script produces a complementary view:

  per-factor:
    n_post_years            : number of post-pub years observed
    median_annual_sharpe    : median across post-pub years
    mean_annual_sharpe      : mean across post-pub years
    frac_positive           : share of post-pub years with annual Sharpe > 0
    frac_above_03           : share with annual Sharpe >= 0.30
    frac_above_05           : share with annual Sharpe >= 0.50
    decade_bin_means        : mean annual Sharpe by 0-5 / 5-10 / 10-15 / 15-20 / 20-30 / 30+ year bin

  decade-level aggregate (across mechanisms):
    {decade_bin x mechanism} -> mean / median / count
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = REPO_ROOT / "results" / "day6"
WF_PATH = REPO_ROOT / "results" / "day3" / "walk_forward_sharpes.parquet"
DECAY_PATH = REPO_ROOT / "results" / "day3" / "decay_per_factor.parquet"

DECADE_BINS = [0, 5, 10, 15, 20, 30, 50]
DECADE_LABELS = ["0-5y", "5-10y", "10-15y", "15-20y", "20-30y", "30+y"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf = pd.read_parquet(WF_PATH)
    decay = pd.read_parquet(DECAY_PATH)

    wf_post = wf[wf["years_since_pub"] > 0].copy()

    per_factor = (
        wf_post.groupby("Acronym")
        .agg(
            n_post_years=("sharpe_ann", "count"),
            median_annual_sharpe=("sharpe_ann", "median"),
            mean_annual_sharpe=("sharpe_ann", "mean"),
            frac_positive=("sharpe_ann", lambda s: (s > 0).mean()),
            frac_above_03=("sharpe_ann", lambda s: (s >= 0.30).mean()),
            frac_above_05=("sharpe_ann", lambda s: (s >= 0.50).mean()),
        )
        .reset_index()
    )
    per_factor.to_parquet(OUT_DIR / "rolling_viability.parquet", index=False)
    print(f"per-factor rolling viability rows: {len(per_factor)} -> rolling_viability.parquet")

    # Decade-level breakdown by mechanism
    wf_post = wf_post.merge(
        decay[["Acronym", "mechanism", "analysis_eligible"]],
        on="Acronym",
        how="left",
    )
    wf_post = wf_post[wf_post["analysis_eligible"]]
    wf_post["decade_bin"] = pd.cut(
        wf_post["years_since_pub"], bins=DECADE_BINS, labels=DECADE_LABELS
    )
    decade = (
        wf_post.groupby(["decade_bin", "mechanism"], observed=True)["sharpe_ann"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    decade.to_parquet(OUT_DIR / "decade_breakdown.parquet", index=False)
    print(f"decade x mechanism rows: {len(decade)} -> decade_breakdown.parquet")

    # Headline rollup vs eligibility
    elig = decay[decay["analysis_eligible"]]
    pf_elig = per_factor.merge(elig[["Acronym"]], on="Acronym", how="inner")
    n = len(pf_elig)
    long_horizon = (elig["post_sharpe"] >= 0.30).sum()
    pos_majority = (pf_elig["frac_positive"] >= 0.50).sum()
    above_03_majority = (pf_elig["frac_above_03"] >= 0.50).sum()
    print()
    print(f"eligible n={n}")
    print(f"  long-horizon Sharpe >= 0.30 viable    : {long_horizon}/{n} ({long_horizon/n:.1%})")
    print(f"  positive in majority of post-pub years: {pos_majority}/{n} ({pos_majority/n:.1%})")
    print(f"  Sharpe>=0.30 in majority of post-pub years: {above_03_majority}/{n} ({above_03_majority/n:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
