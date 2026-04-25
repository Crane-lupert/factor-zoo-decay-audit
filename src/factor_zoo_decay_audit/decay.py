"""Day-3 post-publication decay analysis, conditioned on mechanism.

Definitions
-----------
Per signal, we already have annualized Sharpe in two windows:
    is_sharpe   over [SampleStartYear-01-01, SampleEndYear-12-31]
    post_sharpe over [(Year+1)-01-01, last available month]

Two decay measures:
    decay_diff  = post_sharpe - is_sharpe                         (additive)
    decay_ratio = post_sharpe / is_sharpe   (only when is_sharpe > 0)
                                            (mispricing > 1 means amplified;
                                             < 0 means sign reversal)

Engelberg-McLean-Pontiff (2024) finding (the result we want to reproduce
directionally):
    behavioral decay_diff < risk_premium decay_diff
    (i.e. behavioral factors lose more Sharpe post-publication)

Walk-forward annual Sharpe per signal: monthly returns -> calendar year
Sharpe (annualized within the year), then aligned to "years since
publication" so we can plot a unified survival curve per mechanism group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_IS_MONTHS = 36          # filter out signals with implausibly short IS window
MIN_POST_MONTHS = 60        # filter out signals where post-pub window is < 5 yrs
IS_SHARPE_FLOOR = 0.10      # for decay_ratio, require meaningful baseline


def join_sharpes_with_mechanism(
    sharpes: pd.DataFrame,
    ensemble_majority: pd.DataFrame,
    signal_doc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the per-signal decay analysis frame."""
    df = sharpes.merge(
        ensemble_majority.rename(columns={"acronym": "Acronym"})[
            ["Acronym", "ensemble_label", "votes_for", "total_votes", "unanimous"]
        ],
        on="Acronym",
        how="left",
    )
    if signal_doc is not None:
        df = df.merge(
            signal_doc[["Acronym", "Cat.Economic", "Signal Rep Quality", "Authors", "Journal"]],
            on="Acronym",
            how="left",
        )
    df = df.rename(columns={"ensemble_label": "mechanism"})
    df["decay_diff"] = df["post_sharpe"] - df["is_sharpe"]
    safe = (df["is_sharpe"].abs() >= IS_SHARPE_FLOOR) & (df["is_sharpe"] > 0)
    df["decay_ratio"] = np.where(safe, df["post_sharpe"] / df["is_sharpe"], np.nan)
    df["analysis_eligible"] = (
        (df["is_n_months"] >= MIN_IS_MONTHS)
        & (df["post_n_months"] >= MIN_POST_MONTHS)
        & df["mechanism"].notna()
        & (df["mechanism"] != "PARSE_FAIL")
    )
    return df


def group_decay_stats(decay_df: pd.DataFrame, *, eligible_only: bool = True) -> pd.DataFrame:
    """Per-mechanism summary statistics on decay_diff and decay_ratio."""
    src = decay_df[decay_df["analysis_eligible"]] if eligible_only else decay_df
    rows = []
    for mech, grp in src.groupby("mechanism", sort=False):
        diffs = grp["decay_diff"].dropna().to_numpy()
        ratios = grp["decay_ratio"].dropna().to_numpy()
        is_sh = grp["is_sharpe"].dropna().to_numpy()
        post_sh = grp["post_sharpe"].dropna().to_numpy()
        rows.append({
            "mechanism": mech,
            "n_factors": len(grp),
            "n_with_diff": len(diffs),
            "n_with_ratio": len(ratios),
            "mean_is_sharpe": float(np.mean(is_sh)) if len(is_sh) else np.nan,
            "mean_post_sharpe": float(np.mean(post_sh)) if len(post_sh) else np.nan,
            "mean_decay_diff": float(np.mean(diffs)) if len(diffs) else np.nan,
            "median_decay_diff": float(np.median(diffs)) if len(diffs) else np.nan,
            "sd_decay_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan,
            "mean_decay_ratio": float(np.mean(ratios)) if len(ratios) else np.nan,
            "median_decay_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mechanism").reset_index(drop=True)


def bootstrap_ci(
    values: np.ndarray, *, statistic=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on a 1D sample. Returns (point, lo, hi)."""
    if len(values) < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = statistic(values[rng.integers(0, n, n)])
    point = float(statistic(values))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def group_bootstrap_ci(decay_df: pd.DataFrame, col: str = "decay_diff", **kw) -> pd.DataFrame:
    """Per-mechanism bootstrap CI on `col` (mean by default)."""
    src = decay_df[decay_df["analysis_eligible"]]
    rows = []
    for mech, grp in src.groupby("mechanism", sort=False):
        v = grp[col].dropna().to_numpy()
        point, lo, hi = bootstrap_ci(v, **kw)
        rows.append({"mechanism": mech, "n": len(v), "point": point, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("mechanism").reset_index(drop=True)


def pairwise_tests(decay_df: pd.DataFrame, col: str = "decay_diff") -> pd.DataFrame:
    """Pairwise Welch's t and Mann-Whitney U on `col` between mechanism pairs."""
    from scipy import stats

    src = decay_df[decay_df["analysis_eligible"]]
    groups = {m: g[col].dropna().to_numpy() for m, g in src.groupby("mechanism", sort=False)}
    mechs = sorted(groups.keys())
    rows = []
    for i, a in enumerate(mechs):
        for b in mechs[i + 1:]:
            xa, xb = groups[a], groups[b]
            if len(xa) < 2 or len(xb) < 2:
                continue
            t = stats.ttest_ind(xa, xb, equal_var=False, alternative="two-sided")
            u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            rows.append({
                "group_a": a,
                "group_b": b,
                "n_a": len(xa),
                "n_b": len(xb),
                "mean_a": float(np.mean(xa)),
                "mean_b": float(np.mean(xb)),
                "diff_a_minus_b": float(np.mean(xa) - np.mean(xb)),
                "welch_t": float(t.statistic),
                "welch_p": float(t.pvalue),
                "mw_u": float(u.statistic),
                "mw_p": float(u.pvalue),
            })
    return pd.DataFrame(rows)


def walk_forward_annual_sharpes(
    ls_returns: pd.DataFrame,
    signal_doc: pd.DataFrame,
) -> pd.DataFrame:
    """Per (signal, calendar year) annualized Sharpe + years_since_pub.

    Output columns:
        Acronym, year, n_months, mean_ret, sd_ret, sharpe_ann, years_since_pub
    """
    rets = ls_returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    rets["year"] = rets["date"].dt.year

    pub_year = signal_doc.set_index("Acronym")["Year"].to_dict()

    rows = []
    for (signal, year), grp in rets.groupby(["signalname", "year"], sort=False):
        r = grp["ret"].dropna()
        n = len(r)
        if n < 6:  # need ~half-year data
            continue
        mean = float(r.mean())
        sd = float(r.std(ddof=1))
        if sd == 0 or not np.isfinite(sd):
            continue
        sharpe = mean / sd * np.sqrt(12)
        py = pub_year.get(signal, np.nan)
        years_since = (year - py) if pd.notna(py) else np.nan
        rows.append({
            "Acronym": signal, "year": int(year), "n_months": n,
            "mean_ret": mean, "sd_ret": sd, "sharpe_ann": sharpe,
            "years_since_pub": float(years_since) if pd.notna(years_since) else np.nan,
        })
    return pd.DataFrame(rows)
