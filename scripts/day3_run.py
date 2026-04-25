"""Day-3: mechanism-conditional decay analysis.

Inputs:
    results/day1/sharpes.parquet           IS / post-pub annualized Sharpe
    results/day2/ensemble_majority.parquet ensemble mechanism label
    cache/oap_signal_doc.parquet           paper Year, Cat.Economic
    cache/oap_ls_returns.parquet           monthly LS returns

Outputs (results/day3/):
    decay_per_factor.parquet      per-signal decay metrics + mechanism
    decay_group_stats.csv         per-mechanism summary stats
    decay_bootstrap_ci.csv        per-mechanism mean(decay_diff) bootstrap CI
    decay_pairwise_tests.csv      Welch t / Mann-Whitney across mechanisms
    walk_forward_sharpes.parquet  (signal, year) -> annual Sharpe
    cohort_decay_curve.csv        mean / median annual Sharpe by years_since_pub x mechanism
    decay_report.txt              text summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.decay import (  # noqa: E402
    group_bootstrap_ci,
    group_decay_stats,
    join_sharpes_with_mechanism,
    pairwise_tests,
    walk_forward_annual_sharpes,
)
from factor_zoo_decay_audit.load import load_ls_returns, load_signal_doc  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "day3"
SHARPES_PARQUET = REPO_ROOT / "results" / "day1" / "sharpes.parquet"
ENSEMBLE_PARQUET = REPO_ROOT / "results" / "day2" / "ensemble_majority.parquet"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sharpes = pd.read_parquet(SHARPES_PARQUET)
    ensemble = pd.read_parquet(ENSEMBLE_PARQUET)
    doc = load_signal_doc()
    ls = load_ls_returns()

    print(f"sharpes rows : {len(sharpes)}    ensemble rows: {len(ensemble)}")

    decay = join_sharpes_with_mechanism(sharpes, ensemble, doc)
    n_eligible = int(decay["analysis_eligible"].sum())
    print(f"per-factor decay frame: {len(decay)} rows, {n_eligible} eligible "
          f"(IS>={36}m, post>={60}m, mechanism present)")
    decay.to_parquet(OUT_DIR / "decay_per_factor.parquet", index=False)

    print()
    print("--- per-mechanism decay summary (eligible only) ---")
    stats = group_decay_stats(decay)
    print(stats.round(3).to_string(index=False))
    stats.to_csv(OUT_DIR / "decay_group_stats.csv", index=False)

    print()
    print("--- bootstrap CI for mean(decay_diff) by mechanism (5000 reps, alpha=.05) ---")
    boot = group_bootstrap_ci(decay, col="decay_diff", n_boot=5000, alpha=0.05)
    print(boot.round(3).to_string(index=False))
    boot.to_csv(OUT_DIR / "decay_bootstrap_ci.csv", index=False)

    print()
    print("--- pairwise Welch t / Mann-Whitney on decay_diff ---")
    tests = pairwise_tests(decay, col="decay_diff")
    print(tests.round(4).to_string(index=False))
    tests.to_csv(OUT_DIR / "decay_pairwise_tests.csv", index=False)

    bvr = tests[
        ((tests["group_a"] == "behavioral") & (tests["group_b"] == "risk_premium"))
        | ((tests["group_a"] == "risk_premium") & (tests["group_b"] == "behavioral"))
    ]
    if len(bvr):
        row = bvr.iloc[0]
        # diff convention: positive => first group's mean is higher
        if row["group_a"] == "behavioral":
            beh_minus_rp = row["diff_a_minus_b"]
        else:
            beh_minus_rp = -row["diff_a_minus_b"]
        emp_direction = beh_minus_rp < 0  # EMP 2024: behavioral decays MORE => mean(decay_diff) MORE NEGATIVE
        print()
        print("--- EMP 2024 directional reproduction ---")
        print(f"  mean decay_diff (behavioral)   : {stats.loc[stats['mechanism']=='behavioral','mean_decay_diff'].iat[0]:.3f}")
        print(f"  mean decay_diff (risk_premium) : {stats.loc[stats['mechanism']=='risk_premium','mean_decay_diff'].iat[0]:.3f}")
        print(f"  behavioral - risk_premium      : {beh_minus_rp:+.3f}")
        print(f"  Welch p / MW p                 : {row['welch_p']:.4f}  /  {row['mw_p']:.4f}")
        print(f"  EMP direction (behavioral decays more): "
              f"{'YES' if emp_direction else 'NO (opposite direction)'}")

    print()
    print("--- walk-forward annual Sharpe ---")
    wf = walk_forward_annual_sharpes(ls, doc)
    print(f"  rows: {len(wf)}    unique signals: {wf['Acronym'].nunique()}")
    wf.to_parquet(OUT_DIR / "walk_forward_sharpes.parquet", index=False)

    wf_m = wf.merge(decay[["Acronym", "mechanism", "analysis_eligible"]], on="Acronym", how="left")
    wf_m = wf_m[wf_m["analysis_eligible"] & wf_m["years_since_pub"].notna()].copy()
    # Bucket years_since_pub into integer cohorts in [-25, +25]
    wf_m["yspub_bin"] = wf_m["years_since_pub"].clip(-25, 25).round().astype(int)
    cohort = (
        wf_m.groupby(["mechanism", "yspub_bin"])["sharpe_ann"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    cohort.to_csv(OUT_DIR / "cohort_decay_curve.csv", index=False)
    print(f"  cohort curve rows: {len(cohort)}")

    # Simple summary report
    lines = [
        "Day-3 mechanism-conditional decay report",
        "=========================================",
        f"factors total           : {len(decay)}",
        f"factors eligible        : {n_eligible}  (IS>=36m, post>=60m, mechanism present)",
        "",
        "Per-mechanism decay_diff summary:",
        stats.round(3).to_string(index=False),
        "",
        "Pairwise tests on decay_diff:",
        tests.round(4).to_string(index=False),
    ]
    if len(bvr):
        lines.extend([
            "",
            f"behavioral - risk_premium decay_diff: {beh_minus_rp:+.3f}  "
            f"(Welch p={row['welch_p']:.4f}, MW p={row['mw_p']:.4f})",
            f"EMP-2024 direction reproduced: {'YES' if emp_direction else 'NO'}",
        ])
    (OUT_DIR / "decay_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # Day-3 advance gate: behavioral decays more than risk_premium AND tests significant
    if len(bvr):
        gate = emp_direction and (row["welch_p"] < 0.10 or row["mw_p"] < 0.10)
        print()
        print(f"Day-3 advance gate (EMP-direction + p<0.10 on Welch or MW): "
              f"{'PASS' if gate else 'FAIL'}")
        return 0 if gate else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
