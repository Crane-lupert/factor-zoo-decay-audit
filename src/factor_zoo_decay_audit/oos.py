"""Out-of-sample post-2020 cohort decay (Tier-2 improvement D).

For each predictor, we restrict the post-publication window to:

    oos_start = max(Year + 1, 2020-01-01)
    oos_end   = returns end (2024-12-31 in our cache)

For papers published before 2019, this is GENUINELY OOS relative to both
the original paper sample AND the post-pub window academic researchers
typically had access to when EMP-2024 was written. For papers published
2020+, OOS is just the normal post-pub window.

We then re-aggregate decay_diff by mechanism. If EMP 2024's behavioral-
decays-more finding survives in this 2020-2024 cohort, the result is
robust to the most recent 5 years of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OOS_START = pd.Timestamp("2020-01-01")
MIN_OOS_MONTHS = 24      # need at least 2 years for a usable Sharpe estimate
MONTHS_PER_YEAR = 12


def compute_oos_decay(
    ls_returns: pd.DataFrame,
    sharpes_df: pd.DataFrame,
    ensemble_majority: pd.DataFrame,
) -> pd.DataFrame:
    """Per-signal OOS post-2020 Sharpe + decay_diff vs IS Sharpe."""
    rets = ls_returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    rets_max = rets["date"].max()

    by_signal = dict(tuple(rets.groupby("signalname", sort=False)))
    sharpes = sharpes_df.set_index("Acronym")
    mech_map = ensemble_majority.set_index("acronym")["ensemble_label"].to_dict()

    rows = []
    for signal, sub in by_signal.items():
        if signal not in sharpes.index:
            continue
        s = sharpes.loc[signal]
        if pd.isna(s["Year"]) or pd.isna(s["is_sharpe"]):
            continue
        pub_start = pd.Timestamp(int(s["Year"]) + 1, 1, 1)
        oos_start = max(pub_start, OOS_START)
        if oos_start > rets_max:
            continue
        win = sub.loc[(sub["date"] >= oos_start) & (sub["date"] <= rets_max), "ret"].dropna()
        n = len(win)
        if n < MIN_OOS_MONTHS:
            continue
        sd = float(win.std(ddof=1))
        if sd == 0 or not np.isfinite(sd):
            continue
        oos_sharpe = float(win.mean() / sd * np.sqrt(MONTHS_PER_YEAR))
        rows.append({
            "Acronym": signal,
            "Year": int(s["Year"]),
            "mechanism": mech_map.get(signal),
            "oos_start": oos_start,
            "oos_end": rets_max,
            "oos_n_months": n,
            "is_sharpe": float(s["is_sharpe"]),
            "post_sharpe": float(s["post_sharpe"]) if pd.notna(s["post_sharpe"]) else float("nan"),
            "oos_sharpe": oos_sharpe,
            "is_minus_oos": float(s["is_sharpe"]) - oos_sharpe,
            "post_minus_oos": (float(s["post_sharpe"]) - oos_sharpe) if pd.notna(s["post_sharpe"]) else float("nan"),
        })
    return pd.DataFrame(rows)


def oos_group_summary(oos_decay: pd.DataFrame) -> pd.DataFrame:
    """Per-mechanism mean OOS Sharpe + decay_diff (is_sharpe - oos_sharpe)."""
    src = oos_decay.dropna(subset=["mechanism"])
    rows = []
    for mech, g in src.groupby("mechanism", sort=False):
        rows.append({
            "mechanism": mech,
            "n": len(g),
            "mean_is_sharpe": float(g["is_sharpe"].mean()),
            "mean_post_sharpe": float(g["post_sharpe"].mean()),
            "mean_oos_sharpe": float(g["oos_sharpe"].mean()),
            "mean_is_minus_oos": float(g["is_minus_oos"].mean()),
            "median_oos_sharpe": float(g["oos_sharpe"].median()),
            "frac_oos_positive": float((g["oos_sharpe"] > 0).mean()),
            "frac_oos_above_03": float((g["oos_sharpe"] >= 0.3).mean()),
        })
    return pd.DataFrame(rows).sort_values("mechanism").reset_index(drop=True)


def oos_pairwise_tests(oos_decay: pd.DataFrame) -> pd.DataFrame:
    """Pairwise tests on OOS Sharpe across mechanisms."""
    from scipy import stats

    src = oos_decay.dropna(subset=["mechanism"])
    groups = {m: g["oos_sharpe"].dropna().to_numpy() for m, g in src.groupby("mechanism", sort=False)}
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
                "group_a": a, "group_b": b,
                "n_a": len(xa), "n_b": len(xb),
                "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
                "diff_a_minus_b": float(np.mean(xa) - np.mean(xb)),
                "welch_p": float(t.pvalue), "mw_p": float(u.pvalue),
            })
    return pd.DataFrame(rows)
