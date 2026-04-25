"""Day-2 post-classification analysis:
    1. Pairwise Cohen's kappa across the 3 models
    2. Per-model label distribution
    3. Ensemble majority + per-factor agreement
    4. Oracle vs ensemble kappa (gate: > 0.7)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.agreement import (  # noqa: E402
    label_distribution,
    majority_vote,
    oracle_kappa,
    pairwise_kappa_table,
)

CLASSIFICATIONS_PARQUET = REPO_ROOT / "results" / "day2" / "classifications.parquet"
ORACLE_CSV = REPO_ROOT / "data" / "oracle_mechanism_labels.csv"
OUT_DIR = REPO_ROOT / "results" / "day2"
ENSEMBLE_PARQUET = OUT_DIR / "ensemble_majority.parquet"
ORACLE_DIAG_CSV = OUT_DIR / "oracle_diagnostics.csv"
REPORT_TXT = OUT_DIR / "agreement_report.txt"


def main() -> int:
    cls = pd.read_parquet(CLASSIFICATIONS_PARQUET)
    oracle = pd.read_csv(ORACLE_CSV)
    print(f"classifications rows: {len(cls)}    factors: {cls['acronym'].nunique()}    models: {cls['model'].nunique()}")
    print(f"oracle rows         : {len(oracle)}")
    print()

    # 1. Per-model label distribution
    dist = label_distribution(cls)
    print("--- per-model label distribution ---")
    print(dist.to_string())
    print()

    # 2. Pairwise kappa
    kappa_table = pairwise_kappa_table(cls)
    print("--- pairwise Cohen's kappa ---")
    print(kappa_table.round(3).to_string())
    upper = kappa_table.where(~pd.isna(kappa_table.values))
    n_models = len(kappa_table)
    pairwise_vals = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairwise_vals.append(float(kappa_table.iat[i, j]))
    avg_pairwise = sum(pairwise_vals) / len(pairwise_vals) if pairwise_vals else float("nan")
    print(f"avg off-diagonal pairwise kappa: {avg_pairwise:.3f}")
    print()

    # 3. Ensemble majority
    maj = majority_vote(cls)
    maj.to_parquet(ENSEMBLE_PARQUET, index=False)
    print(f"--- ensemble majority distribution (n={len(maj)}) ---")
    print(maj["ensemble_label"].value_counts(dropna=False).to_string())
    print(f"unanimous (3/3 agreement): {int(maj['unanimous'].sum())}/{len(maj)}")
    parse_fail = (maj["ensemble_label"] == "PARSE_FAIL").sum()
    if parse_fail:
        print(f"PARSE_FAIL ensembles      : {parse_fail}")
    print()

    # 4. Oracle kappa
    k, n_overlap, joined = oracle_kappa(maj, oracle)
    joined.to_csv(ORACLE_DIAG_CSV, index=False)
    print(f"--- oracle vs ensemble (n_overlap={n_overlap}) ---")
    print(joined[["Acronym", "oracle_label", "ensemble_label", "votes_for", "total_votes", "match"]].to_string(index=False))
    print()
    print(f"oracle kappa: {k:.3f}    raw agreement: {joined['match'].mean():.1%}")
    gate = k > 0.70
    print(f"Day-2 gate (oracle kappa > 0.70): {'PASS' if gate else 'FAIL'}")

    # 5. Total cost
    cost = float(cls["cost_usd"].sum())
    print()
    print(f"total OpenRouter cost (this project): ${cost:.4f}")

    REPORT_TXT.write_text(
        f"Day-2 mechanism classification report\n"
        f"=====================================\n"
        f"factors classified : {cls['acronym'].nunique()}\n"
        f"models             : {cls['model'].nunique()}\n"
        f"avg pairwise kappa : {avg_pairwise:.3f}\n"
        f"unanimous          : {int(maj['unanimous'].sum())}/{len(maj)}\n"
        f"oracle kappa       : {k:.3f}    (gate >0.70: {'PASS' if gate else 'FAIL'})\n"
        f"oracle raw agreement: {joined['match'].mean():.1%}\n"
        f"total cost         : ${cost:.4f}\n",
        encoding="utf-8",
    )
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
