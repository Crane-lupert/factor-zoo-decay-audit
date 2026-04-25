"""DSR (Deflated Sharpe Ratio) runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.dsr import deflated_sharpe_per_signal  # noqa: E402
from factor_zoo_decay_audit.load import load_ls_returns  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "day6"
SHARPES = REPO_ROOT / "results" / "day1" / "sharpes.parquet"
ENSEMBLE = REPO_ROOT / "results" / "day2" / "ensemble_majority.parquet"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sharpes = pd.read_parquet(SHARPES)
    ls = load_ls_returns()

    dsr = deflated_sharpe_per_signal(sharpes, ls)
    n = len(dsr)
    sigma = dsr.attrs.get("sigma_sr_cross_monthly", float("nan"))
    n_trials = dsr.attrs.get("n_trials", float("nan"))
    e_max_ann = dsr["expected_max_sr_annualized"].iloc[0] if n else float("nan")
    print(f"DSR rows: {n}    n_trials={n_trials}    sigma_SR_cross={sigma:.4f} (monthly)")
    print(f"Expected max annualized Sharpe under {n_trials} trials: {e_max_ann:.3f}")
    print()

    n_robust = int(dsr["robust_at_05"].sum())
    print(f"Factors with DSR > 0.95 (IS Sharpe robust to multiple-testing): {n_robust}/{n}")
    print(f"Mean DSR: {dsr['deflated_sharpe'].mean():.3f}    Median: {dsr['deflated_sharpe'].median():.3f}")

    print()
    print("DSR distribution buckets:")
    for low, high in [(0, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.001)]:
        m = (dsr["deflated_sharpe"] >= low) & (dsr["deflated_sharpe"] < high)
        print(f"  [{low:.2f}, {high:.2f}): {int(m.sum()):>3} ({m.mean():.1%})")

    # Join with ensemble mechanism for per-mechanism DSR summary
    ens = pd.read_parquet(ENSEMBLE).rename(columns={"acronym": "Acronym"})
    merged = dsr.merge(ens[["Acronym", "ensemble_label"]], on="Acronym", how="left")
    print()
    print("--- DSR by mechanism ---")
    by_mech = (
        merged.dropna(subset=["ensemble_label"])
        .groupby("ensemble_label")
        .agg(n=("Acronym", "count"),
             mean_dsr=("deflated_sharpe", "mean"),
             median_dsr=("deflated_sharpe", "median"),
             frac_robust=("robust_at_05", "mean"))
        .reset_index()
    )
    print(by_mech.round(3).to_string(index=False))

    dsr.to_parquet(OUT_DIR / "dsr.parquet", index=False)
    by_mech.to_csv(OUT_DIR / "dsr_by_mechanism.csv", index=False)

    (OUT_DIR / "dsr_report.txt").write_text(
        f"Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)\n"
        f"==================================================\n"
        f"n_trials: {n_trials}    sigma_SR_cross (monthly): {sigma:.4f}\n"
        f"Expected max annualized Sharpe: {e_max_ann:.3f}\n"
        f"DSR > 0.95: {n_robust}/{n}\n\n"
        f"By mechanism:\n{by_mech.round(3).to_string(index=False)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
