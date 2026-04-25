"""Long/short leg decomposition (Tier-1 improvement B).

A long-short factor portfolio is two bets, not one. Implementation of the
short side has different mechanics (borrow cost, hard-to-borrow lists,
short-sale bans during stress), so HFs care which leg drives the alpha and
which leg fades after publication. Engelberg-McLean-Pontiff (2024) report
the short leg fades faster on average; we test the same here.

Convention
----------
For each predictor, OAP gives portfolios labeled '01' (lowest signal decile/
quintile/etc.) through 'NN' (highest), plus 'LS' (high - low, dollar-neutral
LS portfolio).

The doc's `Sign` column tells us which side the AUTHOR'S directional reading
is positive:
    Sign = +1  -> author says HIGH signal predicts higher returns
                  long_port  = port_max     (long the high-signal leg)
                  short_port = port_01      (short the low-signal leg)
    Sign = -1  -> author says LOW signal predicts higher returns
                  long_port  = port_01
                  short_port = port_max

Outputs
-------
Per-signal IS / post-publication annualized Sharpe for:
    long  : long-only return = mean(long_port_ret)            / std * sqrt(12)
    short : short-only return = mean(-short_port_ret)         / std * sqrt(12)
            (we EARN the negative of the short leg's return)
    ls    : full LS = long_port_ret - short_port_ret          (sign-applied)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def identify_legs(op_returns: pd.DataFrame, signal_doc: pd.DataFrame) -> pd.DataFrame:
    """For each signal, identify long/short port labels and Sign."""
    ports_per_signal = (
        op_returns[op_returns["port"] != "LS"]
        .groupby("signalname")["port"]
        .unique()
    )
    rows = []
    for signal, ports in ports_per_signal.items():
        port_list = sorted([p for p in ports if p != "LS"])
        max_port = port_list[-1]
        rows.append({"Acronym": signal, "min_port": "01", "max_port": max_port})
    legs = pd.DataFrame(rows)

    sign_map = signal_doc.set_index("Acronym")["Sign"].to_dict()
    legs["Sign"] = legs["Acronym"].map(sign_map)

    legs["long_port"] = np.where(legs["Sign"] >= 0, legs["max_port"], legs["min_port"])
    legs["short_port"] = np.where(legs["Sign"] >= 0, legs["min_port"], legs["max_port"])
    return legs


def _sharpe_annualized(ret: pd.Series) -> tuple[float, int]:
    r = ret.dropna()
    n = int(r.size)
    if n < MONTHS_PER_YEAR:
        return float("nan"), n
    sd = float(r.std(ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan"), n
    return float(r.mean() / sd * np.sqrt(MONTHS_PER_YEAR)), n


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["date"] >= start) & (df["date"] <= end)]


def compute_leg_sharpes(
    op_returns: pd.DataFrame,
    signal_doc: pd.DataFrame,
    legs: pd.DataFrame,
) -> pd.DataFrame:
    """For each signal, compute IS / post-pub annualized Sharpe per leg."""
    op = op_returns.copy()
    op["date"] = pd.to_datetime(op["date"])
    returns_max_date = op["date"].max()

    by_signal_port = {
        (s, p): g for (s, p), g in op.groupby(["signalname", "port"], sort=False)
    }
    doc = signal_doc.set_index("Acronym")[["Year", "SampleStartYear", "SampleEndYear", "Sign"]]

    rows = []
    for r in legs.itertuples(index=False):
        if r.Acronym not in doc.index:
            continue
        d = doc.loc[r.Acronym]
        if pd.isna(d["SampleStartYear"]) or pd.isna(d["SampleEndYear"]) or pd.isna(d["Year"]):
            continue
        is_start = pd.Timestamp(int(d["SampleStartYear"]), 1, 1)
        is_end = pd.Timestamp(int(d["SampleEndYear"]), 12, 31)
        post_start = pd.Timestamp(int(d["Year"]) + 1, 1, 1)
        if post_start > returns_max_date:
            continue

        long_df = by_signal_port.get((r.Acronym, r.long_port))
        short_df = by_signal_port.get((r.Acronym, r.short_port))
        if long_df is None or short_df is None:
            continue

        # Long-only: hold long_leg, earn its return.
        # Short-only: short the short_leg, earn -short_leg.ret.
        # LS: long minus short.
        long_is = _slice_window(long_df, is_start, is_end)["ret"]
        long_post = _slice_window(long_df, post_start, returns_max_date)["ret"]
        short_is = -_slice_window(short_df, is_start, is_end)["ret"]
        short_post = -_slice_window(short_df, post_start, returns_max_date)["ret"]

        long_is_aligned, short_is_aligned = (
            _slice_window(long_df, is_start, is_end).set_index("date")["ret"],
            _slice_window(short_df, is_start, is_end).set_index("date")["ret"],
        )
        ls_is = (long_is_aligned - short_is_aligned).dropna()
        long_post_aligned, short_post_aligned = (
            _slice_window(long_df, post_start, returns_max_date).set_index("date")["ret"],
            _slice_window(short_df, post_start, returns_max_date).set_index("date")["ret"],
        )
        ls_post = (long_post_aligned - short_post_aligned).dropna()

        rec = {
            "Acronym": r.Acronym,
            "Sign": int(d["Sign"]) if pd.notna(d["Sign"]) else 0,
            "long_port": r.long_port,
            "short_port": r.short_port,
            "is_start": is_start,
            "is_end": is_end,
            "post_start": post_start,
            "post_end": returns_max_date,
        }
        for label, series in [
            ("long_is", long_is), ("long_post", long_post),
            ("short_is", short_is), ("short_post", short_post),
            ("ls_is", ls_is), ("ls_post", ls_post),
        ]:
            sh, n = _sharpe_annualized(series)
            rec[f"{label}_sharpe"] = sh
            rec[f"{label}_n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_leg_decay(leg_sharpes: pd.DataFrame, ensemble_majority: pd.DataFrame) -> pd.DataFrame:
    """Per-signal per-leg decay_diff (post - IS), joined with mechanism."""
    df = leg_sharpes.merge(
        ensemble_majority.rename(columns={"acronym": "Acronym"})[["Acronym", "ensemble_label"]],
        on="Acronym",
        how="left",
    ).rename(columns={"ensemble_label": "mechanism"})

    df["long_decay_diff"] = df["long_post_sharpe"] - df["long_is_sharpe"]
    df["short_decay_diff"] = df["short_post_sharpe"] - df["short_is_sharpe"]
    df["ls_decay_diff"] = df["ls_post_sharpe"] - df["ls_is_sharpe"]
    df["short_minus_long_decay"] = df["short_decay_diff"] - df["long_decay_diff"]
    df["analysis_eligible"] = (
        (df["long_is_n"] >= 36) & (df["long_post_n"] >= 60)
        & (df["short_is_n"] >= 36) & (df["short_post_n"] >= 60)
        & df["mechanism"].notna() & (df["mechanism"] != "PARSE_FAIL")
    )
    return df


def leg_group_summary(leg_decay: pd.DataFrame) -> pd.DataFrame:
    """Per-mechanism mean of long/short/LS decay_diff + 'short - long' contrast."""
    src = leg_decay[leg_decay["analysis_eligible"]]
    rows = []
    for mech, g in src.groupby("mechanism", sort=False):
        rows.append({
            "mechanism": mech,
            "n": len(g),
            "mean_long_post_sharpe": float(g["long_post_sharpe"].mean()),
            "mean_short_post_sharpe": float(g["short_post_sharpe"].mean()),
            "mean_long_decay_diff": float(g["long_decay_diff"].mean()),
            "mean_short_decay_diff": float(g["short_decay_diff"].mean()),
            "mean_ls_decay_diff": float(g["ls_decay_diff"].mean()),
            "mean_short_minus_long_decay": float(g["short_minus_long_decay"].mean()),
        })
    return pd.DataFrame(rows).sort_values("mechanism").reset_index(drop=True)
