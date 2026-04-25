"""Day-1 runner: load OAP data, compute IS/post-pub Sharpes, run sanity gate.

Usage:
    .venv/Scripts/python scripts/day1_run.py
    .venv/Scripts/python scripts/day1_run.py --force   # rebuild parquet caches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.load import (  # noqa: E402
    LS_RETURNS_PARQUET,
    OP_RETURNS_PARQUET,
    SIGNAL_DOC_PARQUET,
    load_all,
)
from factor_zoo_decay_audit.sanity import evaluate, format_report  # noqa: E402
from factor_zoo_decay_audit.sharpe import compute_period_sharpes  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "day1"
SHARPES_PARQUET = OUT_DIR / "sharpes.parquet"
SHARPES_CSV = OUT_DIR / "sharpes.csv"
DIAGNOSTICS_CSV = OUT_DIR / "sanity_diagnostics.csv"
REPORT_TXT = OUT_DIR / "sanity_report.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download OAP data and rebuild parquet caches")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] loading OAP signal doc + LS portfolio returns ...")
    ls_returns, signal_doc = load_all(force=args.force)
    print(f"      signal doc rows: {len(signal_doc):>5}   ({SIGNAL_DOC_PARQUET})")
    print(f"      op returns rows: {OP_RETURNS_PARQUET.exists() and 'cached' or 'n/a'}")
    print(f"      LS  returns rows: {len(ls_returns):>7}  signals={ls_returns['signalname'].nunique()}   ({LS_RETURNS_PARQUET})")

    print("[2/3] computing IS / post-publication Sharpe per signal ...")
    sharpes = compute_period_sharpes(ls_returns, signal_doc)
    sharpes.to_parquet(SHARPES_PARQUET, index=False)
    sharpes.to_csv(SHARPES_CSV, index=False)
    print(f"      sharpes rows: {len(sharpes)}   ({SHARPES_PARQUET})")

    print("[3/3] sanity check vs Chen-Zimmermann reported T-Stats ...")
    report, diagnostics = evaluate(sharpes, signal_doc)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    text = format_report(report)
    REPORT_TXT.write_text(text, encoding="utf-8")
    print()
    print(text)

    comp = diagnostics[diagnostics["raw_ls_test"] & diagnostics["abs_rel_err"].notna()]
    worst = comp.sort_values("abs_rel_err", ascending=False).head(10)[
        ["Acronym", "Year", "Test in OP", "is_sharpe", "reported_sharpe_annualized", "abs_rel_err", "is_n_months"]
    ]
    print("Top 10 worst |rel_err| within comparable subset:")
    print(worst.to_string(index=False))

    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
