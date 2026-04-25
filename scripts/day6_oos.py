"""OOS post-2020 cohort decay runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.load import load_ls_returns  # noqa: E402
from factor_zoo_decay_audit.oos import (  # noqa: E402
    compute_oos_decay,
    oos_group_summary,
    oos_pairwise_tests,
)

OUT_DIR = REPO_ROOT / "results" / "day6"
SHARPES = REPO_ROOT / "results" / "day1" / "sharpes.parquet"
ENSEMBLE = REPO_ROOT / "results" / "day2" / "ensemble_majority.parquet"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sharpes = pd.read_parquet(SHARPES)
    ensemble = pd.read_parquet(ENSEMBLE)
    ls = load_ls_returns()

    oos = compute_oos_decay(ls, sharpes, ensemble)
    print(f"OOS decay rows: {len(oos)}  (post-2020 window, min 24 months)")
    oos.to_parquet(OUT_DIR / "oos_decay.parquet", index=False)

    summary = oos_group_summary(oos)
    print()
    print("--- per-mechanism OOS post-2020 summary ---")
    print(summary.round(3).to_string(index=False))
    summary.to_csv(OUT_DIR / "oos_group_summary.csv", index=False)

    tests = oos_pairwise_tests(oos)
    print()
    print("--- pairwise tests on OOS Sharpe ---")
    print(tests.round(4).to_string(index=False))
    tests.to_csv(OUT_DIR / "oos_pairwise_tests.csv", index=False)

    bvr = tests[
        ((tests["group_a"] == "behavioral") & (tests["group_b"] == "risk_premium"))
        | ((tests["group_a"] == "risk_premium") & (tests["group_b"] == "behavioral"))
    ]
    if len(bvr):
        r = bvr.iloc[0]
        beh_minus_rp = r["diff_a_minus_b"] if r["group_a"] == "behavioral" else -r["diff_a_minus_b"]
        emp_dir = beh_minus_rp < 0
        print()
        print(f"OOS behavioral - risk_premium Sharpe: {beh_minus_rp:+.3f}  "
              f"Welch p={r['welch_p']:.4f}  MW p={r['mw_p']:.4f}")
        print(f"EMP direction (behavioral fades more) survives in 2020-2024: "
              f"{'YES' if emp_dir else 'NO (opposite)'}")

    (OUT_DIR / "oos_report.txt").write_text(
        "OOS post-2020 cohort decay\n==========================\n"
        + summary.round(3).to_string(index=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
