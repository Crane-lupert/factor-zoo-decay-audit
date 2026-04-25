"""Day-6 statistical rigor: FDR multi-test correction + factor-level bootstrap.

FDR (Benjamini-Hochberg)
------------------------
With 212 factors, simple per-factor t-tests on post-pub Sharpe overcount
"significant" results due to multiple-hypothesis bias. BH-FDR controls the
expected proportion of false discoveries among rejected nulls. Standard
penalty for academic factor zoo claims.

Bootstrap-by-factor
-------------------
Group differences (e.g., behavioral vs risk_premium decay_diff) computed in
Day 3 used standard parametric tests. Re-confirm with non-parametric
factor-level resampling: draw factors with replacement within each group,
recompute mean difference, build empirical sampling distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fdr_bh(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR. Returns (reject_mask, q_values)."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    sorted_p = p[order]
    q_sorted = sorted_p * n / ranks
    # enforce monotone
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.clip(q_sorted, 0, 1)
    reject = q <= alpha
    return reject, q


def post_pub_pvalues(
    ls_returns: pd.DataFrame,
    sharpes: pd.DataFrame,
) -> pd.DataFrame:
    """Per-signal post-publication t-stat + two-sided p-value for H0: mean=0."""
    rets = ls_returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    by_signal = dict(tuple(rets.groupby("signalname", sort=False)))

    rows = []
    for r in sharpes.itertuples(index=False):
        if pd.isna(getattr(r, "post_start", None)) or pd.isna(getattr(r, "post_end", None)):
            continue
        sub = by_signal.get(r.Acronym)
        if sub is None:
            continue
        win = sub.loc[(sub["date"] >= r.post_start) & (sub["date"] <= r.post_end), "ret"].dropna()
        n = len(win)
        if n < 24:
            continue
        mean = float(win.mean())
        sd = float(win.std(ddof=1))
        if sd == 0 or not np.isfinite(sd):
            continue
        t = mean / (sd / np.sqrt(n))
        p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
        rows.append({
            "Acronym": r.Acronym,
            "post_n_months": n,
            "post_mean_pct": mean,
            "post_t": float(t),
            "post_p": float(p),
        })
    return pd.DataFrame(rows)


def apply_fdr(post_pvalues: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add FDR-BH q-values + reject flag."""
    df = post_pvalues.copy()
    reject, q = fdr_bh(df["post_p"].to_numpy(), alpha=alpha)
    df["bh_q"] = q
    df[f"fdr_reject_at_{alpha}"] = reject
    return df


def bootstrap_group_diff(
    group_a_values: np.ndarray,
    group_b_values: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Factor-level (cluster) bootstrap on mean(A) - mean(B).

    Returns {observed, ci_lo, ci_hi, p_two_sided}.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(group_a_values), np.asarray(group_b_values)
    obs = float(a.mean() - b.mean())
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.integers(0, len(a), len(a))
        bi = rng.integers(0, len(b), len(b))
        diffs[i] = a[ai].mean() - b[bi].mean()
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    # two-sided p from symmetric percentile null
    centered = diffs - obs
    p = float((np.abs(centered) >= abs(obs)).mean())
    return {"observed": obs, "ci_lo": lo, "ci_hi": hi, "p_two_sided": p, "n_boot": n_boot}


def pairwise_bootstrap(
    decay_df: pd.DataFrame, *, col: str = "decay_diff", n_boot: int = 5000, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap pairwise group differences across mechanisms."""
    src = decay_df[decay_df["analysis_eligible"]].dropna(subset=[col])
    groups = {m: g[col].dropna().to_numpy() for m, g in src.groupby("mechanism", sort=False)}
    mechs = sorted(groups.keys())
    rows = []
    for i, a in enumerate(mechs):
        for b in mechs[i + 1:]:
            xa, xb = groups[a], groups[b]
            if len(xa) < 2 or len(xb) < 2:
                continue
            res = bootstrap_group_diff(xa, xb, n_boot=n_boot, seed=seed)
            rows.append({
                "group_a": a, "group_b": b,
                "n_a": len(xa), "n_b": len(xb),
                "observed_diff": res["observed"],
                "boot_ci_lo": res["ci_lo"],
                "boot_ci_hi": res["ci_hi"],
                "boot_p_two_sided": res["p_two_sided"],
            })
    return pd.DataFrame(rows)
