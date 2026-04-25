"""IS / post-publication Sharpe ratios for Chen-Zimmermann long-short portfolios.

The OAP `ret` column is monthly long-short return in percent. Annualized
Sharpe = (mean / std) * sqrt(12), computed on percent units (units cancel
in the ratio).

Period definitions follow CLAUDE.md:
    in-sample        = [SampleStartYear-01-01, SampleEndYear-12-31]
    post-publication = [(Year+1)-01-01, returns end]

`Year` in the doc is the paper publication year. Using Year+1 as the post-pub
start is a conservative one-year buffer that avoids overlap with within-year
publication (Engelberg-McLean-Pontiff 2020 use publication month; OAP doc
gives only year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def _sharpe_annualized(ret: pd.Series) -> tuple[float, int]:
    """Return (annualized Sharpe, n_months). NaN Sharpe if <12 obs or zero std."""
    r = ret.dropna()
    n = int(r.size)
    if n < MONTHS_PER_YEAR:
        return float("nan"), n
    sd = float(r.std(ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan"), n
    return float(r.mean() / sd * np.sqrt(MONTHS_PER_YEAR)), n


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["date"] >= start) & (df["date"] <= end)]


def compute_period_sharpes(
    ls_returns: pd.DataFrame,
    signal_doc: pd.DataFrame,
) -> pd.DataFrame:
    """Per-signal IS and post-publication annualized Sharpe.

    Returns a DataFrame with columns:
        Acronym, Year, SampleStartYear, SampleEndYear,
        is_start, is_end, is_sharpe, is_n_months,
        post_start, post_end, post_sharpe, post_n_months,
        full_sharpe, full_n_months,
        reported_return_monthly_pct, reported_tstat,
        reported_sharpe_annualized.
    """
    cols = ["Acronym", "Year", "SampleStartYear", "SampleEndYear", "Return", "T-Stat", "Sign"]
    doc = signal_doc[cols].copy()
    for c in ("Year", "SampleStartYear", "SampleEndYear", "Return", "T-Stat", "Sign"):
        doc[c] = pd.to_numeric(doc[c], errors="coerce")

    rets = ls_returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    returns_max_date = rets["date"].max()

    by_signal = dict(tuple(rets.groupby("signalname", sort=False)))

    rows: list[dict[str, object]] = []
    for rec in doc.to_dict(orient="records"):
        signal = rec["Acronym"]
        sub = by_signal.get(signal)
        if sub is None or sub.empty:
            continue

        sample_start = rec["SampleStartYear"]
        sample_end = rec["SampleEndYear"]
        if pd.notna(sample_start) and pd.notna(sample_end):
            is_start = pd.Timestamp(int(sample_start), 1, 1)
            is_end = pd.Timestamp(int(sample_end), 12, 31)
            is_sharpe, is_n = _sharpe_annualized(_slice(sub, is_start, is_end)["ret"])
        else:
            is_start, is_end = None, None
            is_sharpe, is_n = float("nan"), 0

        pub_year = rec["Year"]
        if pd.notna(pub_year):
            post_start = pd.Timestamp(int(pub_year) + 1, 1, 1)
            if post_start <= returns_max_date:
                post_sharpe, post_n = _sharpe_annualized(_slice(sub, post_start, returns_max_date)["ret"])
                post_end = returns_max_date
            else:
                post_sharpe, post_n, post_end = float("nan"), 0, None
        else:
            post_start, post_end = None, None
            post_sharpe, post_n = float("nan"), 0

        full_sharpe, full_n = _sharpe_annualized(sub["ret"])

        # Reported Sharpe ≈ T-Stat / sqrt(N_IS_months) * sqrt(12).
        tstat = rec["T-Stat"]
        reported_sharpe = (
            float(tstat) / np.sqrt(is_n) * np.sqrt(MONTHS_PER_YEAR)
            if (is_n > 0 and pd.notna(tstat))
            else float("nan")
        )

        rows.append(
            {
                "Acronym": signal,
                "Year": pub_year,
                "SampleStartYear": sample_start,
                "SampleEndYear": sample_end,
                "is_start": is_start,
                "is_end": is_end,
                "is_sharpe": is_sharpe,
                "is_n_months": is_n,
                "post_start": post_start,
                "post_end": post_end,
                "post_sharpe": post_sharpe,
                "post_n_months": post_n,
                "full_sharpe": full_sharpe,
                "full_n_months": full_n,
                "reported_return_monthly_pct": float(rec["Return"]) if pd.notna(rec["Return"]) else float("nan"),
                "reported_tstat": float(tstat) if pd.notna(tstat) else float("nan"),
                "reported_sharpe_annualized": reported_sharpe,
            }
        )

    return pd.DataFrame(rows)
