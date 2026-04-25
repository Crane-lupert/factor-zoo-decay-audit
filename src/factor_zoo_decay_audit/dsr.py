"""Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

Background
----------
With N strategies tested, even pure noise produces a maximum Sharpe ratio
that can look impressive. The Deflated Sharpe Ratio (DSR) penalizes a
candidate Sharpe by the expected maximum under N independent trials with
the same trial-Sharpe variance.

Formula
-------
The expected maximum Sharpe under N independent trials (Bailey-LdP eq. 9):

    E[max_SR] ~= sigma_SR_cross * (
        (1 - gamma) * Phi^{-1}(1 - 1/N)
        + gamma     * Phi^{-1}(1 - 1/(N*e))
    )

where gamma = Euler-Mascheroni (~0.5772), sigma_SR_cross = sd of trial SRs.

PSR(SR_thresh) (Bailey-LdP eq. 12) under non-normal returns:

    z = (SR_obs - SR_thresh) * sqrt(T - 1)
        / sqrt(1 - skew * SR_obs + ((kurt - 1) / 4) * SR_obs^2)

    DSR = Phi(z)        with SR_thresh = E[max_SR]

DSR is in [0, 1]. DSR > 0.95 means the IS Sharpe is significantly above the
no-skill maximum at 5%; DSR < 0.5 is suspect.

We compute DSR using:
    SR_obs    : per-signal monthly Sharpe (annualized / sqrt(12))
    T         : is_n_months (sample length)
    skew/kurt : per-signal sample skew / excess kurtosis on monthly returns
    sigma_SR_cross : sd of cross-sectional monthly SRs across all 212 signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def _sample_moments(monthly_ret: pd.Series) -> tuple[float, float, float, float, int]:
    r = monthly_ret.dropna().to_numpy()
    n = len(r)
    if n < 12:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    sr_monthly = mean / sd
    skew = float(stats.skew(r, bias=False)) if n > 2 else 0.0
    kurt_excess = float(stats.kurtosis(r, fisher=True, bias=False)) if n > 3 else 0.0
    return sr_monthly, sd, skew, kurt_excess, n


def expected_max_sharpe(n_trials: int, sigma_sr_cross_monthly: float) -> float:
    """E[max SR] in MONTHLY units under N independent trials with cross-sd sigma_sr."""
    if n_trials < 2:
        return 0.0
    inv1 = stats.norm.ppf(1 - 1 / n_trials)
    inv2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return sigma_sr_cross_monthly * ((1 - EULER_MASCHERONI) * inv1 + EULER_MASCHERONI * inv2)


def deflated_sharpe_per_signal(
    sharpes_df: pd.DataFrame,
    ls_returns: pd.DataFrame,
    n_trials: int | None = None,
) -> pd.DataFrame:
    """Compute DSR for each signal's IS window.

    sharpes_df : per-signal Day-1 Sharpe table (Acronym, is_start, is_end, is_sharpe, ...)
    ls_returns : monthly LS portfolio returns
    n_trials   : strategy-trial count for deflation (defaults to # signals here)
    """
    rets = ls_returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    by_signal = dict(tuple(rets.groupby("signalname", sort=False)))

    if n_trials is None:
        n_trials = sharpes_df["Acronym"].nunique()

    # Cross-sectional sd of monthly Sharpes (we compute moments per signal first
    # over full sample so the cross-sd is comparable across signals).
    cross_section_sr = []
    moments = {}
    for sig, sub in by_signal.items():
        sr_m, sd, sk, kt, n = _sample_moments(sub["ret"])
        moments[sig] = (sr_m, sd, sk, kt, n)
        if not np.isnan(sr_m):
            cross_section_sr.append(sr_m)
    sigma_sr_cross = float(np.nanstd(cross_section_sr, ddof=1)) if cross_section_sr else float("nan")
    e_max_sr_monthly = expected_max_sharpe(n_trials, sigma_sr_cross)

    rows = []
    for r in sharpes_df.itertuples(index=False):
        sig = r.Acronym
        is_start = getattr(r, "is_start", None)
        is_end = getattr(r, "is_end", None)
        if is_start is None or is_end is None or pd.isna(r.is_sharpe):
            continue
        sub = by_signal.get(sig)
        if sub is None:
            continue
        is_window = sub.loc[(sub["date"] >= is_start) & (sub["date"] <= is_end), "ret"]
        sr_m, sd, sk, kt, n = _sample_moments(is_window)
        if np.isnan(sr_m) or n < 24:
            continue
        denom_sq = 1 - sk * sr_m + ((kt) / 4) * sr_m ** 2  # kt is excess kurtosis
        if denom_sq <= 0 or not np.isfinite(denom_sq):
            denom_sq = 1.0
        z = (sr_m - e_max_sr_monthly) * np.sqrt(n - 1) / np.sqrt(denom_sq)
        dsr = float(stats.norm.cdf(z))
        rows.append({
            "Acronym": sig,
            "is_n_months": n,
            "is_sharpe_monthly": sr_m,
            "is_sharpe_annualized": sr_m * np.sqrt(12),
            "skew": sk,
            "excess_kurt": kt,
            "expected_max_sr_monthly": e_max_sr_monthly,
            "expected_max_sr_annualized": e_max_sr_monthly * np.sqrt(12),
            "deflated_sharpe": dsr,
            "robust_at_05": dsr > 0.95,
        })
    out = pd.DataFrame(rows)
    out.attrs["sigma_sr_cross_monthly"] = sigma_sr_cross
    out.attrs["n_trials"] = n_trials
    return out
