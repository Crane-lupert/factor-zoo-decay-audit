"""Long/short leg decomposition runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.legs import (  # noqa: E402
    compute_leg_decay,
    compute_leg_sharpes,
    identify_legs,
    leg_group_summary,
)
from factor_zoo_decay_audit.load import load_op_returns, load_signal_doc  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "day6"
ENSEMBLE = REPO_ROOT / "results" / "day2" / "ensemble_majority.parquet"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    op = load_op_returns()
    doc = load_signal_doc()
    ens = pd.read_parquet(ENSEMBLE)

    legs = identify_legs(op, doc)
    print(f"identified legs for {len(legs)} signals")
    sh = compute_leg_sharpes(op, doc, legs)
    print(f"leg sharpes computed: {len(sh)} signals")
    decay = compute_leg_decay(sh, ens)
    eligible = int(decay["analysis_eligible"].sum())
    print(f"per-leg decay frame: {len(decay)} rows, {eligible} eligible")
    decay.to_parquet(OUT_DIR / "leg_decay.parquet", index=False)

    summary = leg_group_summary(decay)
    print()
    print("--- per-mechanism leg-decay summary ---")
    print(summary.round(3).to_string(index=False))
    summary.to_csv(OUT_DIR / "leg_decay_summary.csv", index=False)

    # EMP-style: short leg fades MORE than long leg => short_decay - long_decay < 0
    print()
    print("--- short_decay_diff - long_decay_diff (negative = short fades more) ---")
    for _, r in summary.iterrows():
        marker = "<-- SHORT FADES MORE" if r["mean_short_minus_long_decay"] < 0 else ""
        print(f"  {r['mechanism']:<20} : {r['mean_short_minus_long_decay']:+.3f}  n={int(r['n']):>3}  {marker}")

    (OUT_DIR / "leg_decay_report.txt").write_text(
        "Leg-level decay summary\n=======================\n"
        + summary.round(3).to_string(index=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
