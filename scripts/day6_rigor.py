"""Day-6 rigor: FDR (Benjamini-Hochberg) + factor-level bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from factor_zoo_decay_audit.load import load_ls_returns  # noqa: E402
from factor_zoo_decay_audit.rigor import (  # noqa: E402
    apply_fdr,
    pairwise_bootstrap,
    post_pub_pvalues,
)

OUT_DIR = REPO_ROOT / "results" / "day6"
SHARPES = REPO_ROOT / "results" / "day1" / "sharpes.parquet"
DECAY = REPO_ROOT / "results" / "day3" / "decay_per_factor.parquet"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sharpes = pd.read_parquet(SHARPES)
    decay = pd.read_parquet(DECAY)
    ls = load_ls_returns()

    print("[1/2] BH-FDR on per-signal post-pub Sharpe t-stats ...")
    pvals = post_pub_pvalues(ls, sharpes)
    print(f"      computed p for {len(pvals)} signals")
    fdr = apply_fdr(pvals, alpha=0.05)
    n_naive = int((fdr["post_p"] < 0.05).sum())
    n_fdr = int(fdr["fdr_reject_at_0.05"].sum())
    print(f"      naive p<0.05 : {n_naive}/{len(fdr)} ({n_naive/len(fdr):.1%})")
    print(f"      BH-FDR q<0.05: {n_fdr}/{len(fdr)} ({n_fdr/len(fdr):.1%})")
    fdr.to_parquet(OUT_DIR / "fdr.parquet", index=False)
    fdr.to_csv(OUT_DIR / "fdr.csv", index=False)

    print()
    print("[2/2] factor-level bootstrap (5000 reps) on group-difference decay_diff ...")
    boot = pairwise_bootstrap(decay, col="decay_diff", n_boot=5000, seed=42)
    print(boot.round(4).to_string(index=False))
    boot.to_csv(OUT_DIR / "bootstrap_pairwise.csv", index=False)

    bvr = boot[
        ((boot["group_a"] == "behavioral") & (boot["group_b"] == "risk_premium"))
        | ((boot["group_a"] == "risk_premium") & (boot["group_b"] == "behavioral"))
    ]
    if len(bvr):
        r = bvr.iloc[0]
        print()
        print(f"behavioral vs risk_premium decay_diff (factor-bootstrap):")
        print(f"  observed   : {r['observed_diff']:+.3f}")
        print(f"  95% CI     : [{r['boot_ci_lo']:+.3f}, {r['boot_ci_hi']:+.3f}]")
        print(f"  bootstrap p: {r['boot_p_two_sided']:.4f}")

    (OUT_DIR / "rigor_report.txt").write_text(
        "FDR (BH alpha=0.05) and bootstrap rigor\n"
        "========================================\n"
        f"naive p<0.05  : {n_naive}/{len(fdr)} ({n_naive/len(fdr):.1%})\n"
        f"BH-FDR q<0.05 : {n_fdr}/{len(fdr)} ({n_fdr/len(fdr):.1%})\n\n"
        + boot.round(4).to_string(index=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
