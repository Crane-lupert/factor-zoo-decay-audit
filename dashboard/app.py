"""Factor Zoo Decay + Capacity dashboard (MVP).

Run locally:
    cd d:/vscode/factor-zoo-decay-audit
    .venv/Scripts/streamlit run dashboard/app.py

Sections:
  1. Overview          : headline stats + survival-curve table
  2. Per-factor        : pick a factor, show rolling Sharpe / cumulative / DSR
  3. Mechanism         : decay distributions + leg-level decomposition
  4. Capacity          : v1 vs v2 cost model + per-tier breakdown
  5. Methodology       : explicit caveats & limitations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

st.set_page_config(page_title="Factor Zoo Decay + Capacity", layout="wide")


# ---------- data loading ----------

@st.cache_data(show_spinner=False)
def load_all() -> dict[str, pd.DataFrame]:
    base = REPO_ROOT
    out = {
        "doc": pd.read_parquet(base / "cache" / "oap_signal_doc.parquet"),
        "ls": pd.read_parquet(base / "cache" / "oap_ls_returns.parquet"),
        "sharpes": pd.read_parquet(base / "results" / "day1" / "sharpes.parquet"),
        "ensemble": pd.read_parquet(base / "results" / "day2" / "ensemble_majority.parquet"),
        "decay": pd.read_parquet(base / "results" / "day3" / "decay_per_factor.parquet"),
        "wf": pd.read_parquet(base / "results" / "day3" / "walk_forward_sharpes.parquet"),
        "cap_v1": pd.read_parquet(base / "results" / "day4" / "capacity_adjusted.parquet"),
        "cap_v2": pd.read_parquet(base / "results" / "day6" / "capacity_adjusted_v2.parquet"),
        "leg_decay": pd.read_parquet(base / "results" / "day6" / "leg_decay.parquet"),
        "oos": pd.read_parquet(base / "results" / "day6" / "oos_decay.parquet"),
        "dsr": pd.read_parquet(base / "results" / "day6" / "dsr.parquet"),
        "fdr": pd.read_parquet(base / "results" / "day6" / "fdr.parquet"),
    }
    out["ls"]["date"] = pd.to_datetime(out["ls"]["date"])
    out["wf"]["year"] = out["wf"]["year"].astype(int)
    return out


def fmt_aum(a: float) -> str:
    return f"${a/1e9:.0f}B" if a >= 1e9 else f"${a/1e6:.0f}M"


# ---------- pages ----------

def page_overview(d: dict[str, pd.DataFrame]) -> None:
    st.title("Factor Zoo Decay + Capacity")
    st.caption(
        "Independent reproduction of Chen-Zimmermann's 212 published equity-return "
        "predictors with mechanism classification, post-publication decay, and a "
        "capacity-adjusted survival curve."
    )

    decay = d["decay"][d["decay"]["analysis_eligible"]]
    cap = d["cap_v2"][d["cap_v2"]["analysis_eligible"]]
    dsr = d["dsr"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Factors", f"{decay['Acronym'].nunique()}")
    c2.metric("Mean post-pub Sharpe", f"{decay['post_sharpe'].mean():.3f}")
    c3.metric("Mean decay (post - IS)", f"{decay['decay_diff'].mean():.3f}")
    c4.metric("DSR-robust (>=0.95)", f"{int(dsr['robust_at_05'].sum())}/{len(dsr)}")
    cap_100m = cap[cap["aum_usd"] == 1e8]
    c5.metric("Viable @ $100M (v2)",
              f"{int((cap_100m['capacity_sharpe'] >= 0.30).sum())}/{cap_100m['Acronym'].nunique()}")

    st.subheader("Survival curve (capacity-adjusted Sharpe >= 0.30)")
    sc = (
        cap.assign(viable=lambda x: x["capacity_sharpe"] >= 0.30)
        .groupby(["aum_usd", "mechanism"])
        .agg(n=("Acronym", "count"), n_viable=("viable", "sum"))
        .reset_index()
    )
    sc["aum_label"] = sc["aum_usd"].map(fmt_aum)
    sc["viable_pct"] = sc["n_viable"] / sc["n"]
    pivot = sc.pivot(index="aum_label", columns="mechanism", values="viable_pct").reset_index()
    pivot = pivot[["aum_label"] + sorted([c for c in pivot.columns if c != "aum_label"])]

    fig = go.Figure()
    for mech in [c for c in pivot.columns if c != "aum_label"]:
        fig.add_trace(go.Scatter(
            x=pivot["aum_label"], y=(pivot[mech] * 100), mode="lines+markers",
            name=mech, line={"width": 3},
        ))
    fig.update_layout(
        xaxis_title="AUM scenario", yaxis_title="Viable factors (%)",
        height=350, margin={"t": 30, "b": 30, "l": 30, "r": 30},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean decay_diff by mechanism (post - IS, annualized Sharpe)")
    grp = (decay.groupby("mechanism")["decay_diff"]
           .agg(["mean", "median", "count"]).reset_index())
    grp.columns = ["mechanism", "mean", "median", "n"]
    st.dataframe(grp.style.format({"mean": "{:.3f}", "median": "{:.3f}"}),
                 hide_index=True, use_container_width=True)


def page_per_factor(d: dict[str, pd.DataFrame]) -> None:
    st.title("Per-factor view")
    eligible_acrs = sorted(d["decay"][d["decay"]["analysis_eligible"]]["Acronym"].unique())
    pick = st.selectbox("Factor", eligible_acrs, index=eligible_acrs.index("Mom6m") if "Mom6m" in eligible_acrs else 0)

    decay_row = d["decay"][d["decay"]["Acronym"] == pick].iloc[0]
    doc_row = d["doc"][d["doc"]["Acronym"] == pick].iloc[0]
    dsr_row = d["dsr"][d["dsr"]["Acronym"] == pick]
    leg_row = d["leg_decay"][d["leg_decay"]["Acronym"] == pick]
    oos_row = d["oos"][d["oos"]["Acronym"] == pick]

    c1, c2, c3 = st.columns([2, 2, 3])
    c1.markdown(f"**Authors / Year**: {doc_row['Authors']} / {int(doc_row['Year'])}")
    c1.markdown(f"**Journal**: {doc_row['Journal']}")
    c1.markdown(f"**Cat.Economic**: {doc_row['Cat.Economic']}")
    c1.markdown(f"**Mechanism (ensemble)**: `{decay_row['mechanism']}`")
    c2.metric("IS Sharpe", f"{decay_row['is_sharpe']:.3f}")
    c2.metric("Post-pub Sharpe", f"{decay_row['post_sharpe']:.3f}")
    c2.metric("Decay (post - IS)", f"{decay_row['decay_diff']:.3f}")
    if len(dsr_row):
        c3.metric("DSR (multi-test adj.)", f"{dsr_row['deflated_sharpe'].iat[0]:.3f}",
                  delta=("ROBUST" if dsr_row['robust_at_05'].iat[0] else "suspect"))
    if len(oos_row):
        c3.metric("OOS 2020-2024 Sharpe", f"{oos_row['oos_sharpe'].iat[0]:.3f}")

    sub = d["ls"][d["ls"]["signalname"] == pick].sort_values("date").copy()
    sub["cum_ret"] = (1 + sub["ret"] / 100).cumprod() - 1
    sub["rolling_60m_sharpe"] = (
        sub["ret"].rolling(60).mean() / sub["ret"].rolling(60).std() * np.sqrt(12)
    )
    pub_plus_1 = f"{int(decay_row['Year']) + 1}-01-01"
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sub["date"], y=sub["cum_ret"], name="Cumulative LS return"))
    fig1.add_shape(type="line", x0=pub_plus_1, x1=pub_plus_1, xref="x",
                   y0=0, y1=1, yref="paper", line={"dash": "dash", "color": "#94a3b8"})
    fig1.add_annotation(x=pub_plus_1, y=1, yref="paper", text="Pub +1yr", showarrow=False, yshift=10)
    fig1.update_layout(height=300, title="Cumulative LS return",
                       margin={"t": 40, "b": 30, "l": 30, "r": 30})
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sub["date"], y=sub["rolling_60m_sharpe"], name="60-month rolling annualized Sharpe"))
    fig2.add_hline(y=0, line_dash="dot")
    fig2.add_shape(type="line", x0=pub_plus_1, x1=pub_plus_1, xref="x",
                   y0=0, y1=1, yref="paper", line={"dash": "dash", "color": "#94a3b8"})
    fig2.update_layout(height=300, title="60-month rolling Sharpe",
                       margin={"t": 40, "b": 30, "l": 30, "r": 30})
    st.plotly_chart(fig2, use_container_width=True)

    if len(leg_row):
        leg = leg_row.iloc[0]
        st.subheader("Long vs short leg (Sharpe)")
        leg_table = pd.DataFrame({
            "leg": ["long-only", "short-only (when shorted)", "LS"],
            "IS Sharpe": [leg["long_is_sharpe"], leg["short_is_sharpe"], leg["ls_is_sharpe"]],
            "Post-pub Sharpe": [leg["long_post_sharpe"], leg["short_post_sharpe"], leg["ls_post_sharpe"]],
            "Decay (post - IS)": [leg["long_decay_diff"], leg["short_decay_diff"], leg["ls_decay_diff"]],
        })
        st.dataframe(leg_table.style.format({c: "{:.3f}" for c in leg_table.columns if c != "leg"}),
                     hide_index=True, use_container_width=True)

    st.subheader("Capacity-adjusted Sharpe by AUM (v2: per-factor turnover + tier ADV + sqrt impact + borrow)")
    cap_v2 = d["cap_v2"][d["cap_v2"]["Acronym"] == pick].copy()
    cap_v2["aum_label"] = cap_v2["aum_usd"].map(fmt_aum)
    cap_v1 = d["cap_v1"][d["cap_v1"]["Acronym"] == pick].copy()
    cap_v1["aum_label"] = cap_v1["aum_usd"].map(fmt_aum)
    if not cap_v2.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=cap_v2["aum_label"], y=cap_v2["capacity_sharpe"],
                              name="v2 (HF-grade)", marker_color="#2563eb"))
        if not cap_v1.empty:
            fig3.add_trace(go.Bar(x=cap_v1["aum_label"], y=cap_v1["capacity_sharpe"],
                                  name="v1 (academic)", marker_color="#94a3b8"))
        fig3.add_hline(y=0.30, line_dash="dash", annotation_text="0.30 viability")
        fig3.update_layout(height=300, barmode="group",
                           margin={"t": 30, "b": 30, "l": 30, "r": 30})
        st.plotly_chart(fig3, use_container_width=True)
        if not cap_v2.empty:
            st.caption(
                f"v2 cost model: cap_tier={cap_v2['cap_tier'].iat[0]}, "
                f"monthly turnover/side={cap_v2['monthly_turnover_per_side'].iat[0]:.2f}, "
                f"borrow={cap_v2['annual_borrow_bps'].iat[0]:.0f} bps/yr"
            )


def page_mechanism(d: dict[str, pd.DataFrame]) -> None:
    st.title("Mechanism aggregates")
    decay = d["decay"][d["decay"]["analysis_eligible"]]

    st.subheader("decay_diff (post - IS Sharpe) distribution by mechanism")
    fig = px.box(decay, x="mechanism", y="decay_diff", points="all",
                 color="mechanism", height=400)
    fig.update_layout(margin={"t": 20, "b": 30, "l": 30, "r": 30})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Long/short leg decomposition (mean per mechanism)")
    leg = d["leg_decay"][d["leg_decay"]["analysis_eligible"]]
    summary = (leg.groupby("mechanism")
               .agg(mean_long_decay=("long_decay_diff", "mean"),
                    mean_short_decay=("short_decay_diff", "mean"),
                    mean_ls_decay=("ls_decay_diff", "mean"),
                    mean_long_post_sharpe=("long_post_sharpe", "mean"),
                    mean_short_post_sharpe=("short_post_sharpe", "mean"),
                    n=("Acronym", "count"))
               .reset_index())
    st.dataframe(summary.style.format({c: "{:.3f}" for c in summary.columns
                                       if c not in ["mechanism", "n"]}),
                 hide_index=True, use_container_width=True)

    st.markdown(
        "**Reading**: Across all three mechanism categories, the long leg post-pub "
        "Sharpe is +0.4 to +0.5 (positive alpha survives) while the short leg's "
        "Sharpe-when-shorted is around -0.45 to -0.50 (the short side is what's broken "
        "post-publication). Consistent with Engelberg-McLean-Pontiff 2024."
    )

    st.subheader("OOS post-2020 cohort vs full post-pub")
    oos = d["oos"]
    oos_sum = (oos.dropna(subset=["mechanism"]).groupby("mechanism")
               .agg(n=("Acronym", "count"),
                    mean_post_sharpe=("post_sharpe", "mean"),
                    mean_oos_2020_sharpe=("oos_sharpe", "mean"),
                    frac_oos_above_03=("oos_sharpe", lambda s: (s >= 0.30).mean()))
               .reset_index())
    st.dataframe(oos_sum.style.format({c: "{:.3f}" for c in oos_sum.columns
                                       if c not in ["mechanism", "n"]}),
                 hide_index=True, use_container_width=True)
    st.caption(
        "Note: behavioral vs risk_premium gap is statistically significant in the "
        "FULL post-pub window (Welch p=0.003) but COMPRESSES TO INSIGNIFICANCE in "
        "the 2020-2024 OOS cohort (p=0.45). EMP 2024's directional finding does not "
        "persist into the most recent 5-year window — possibly due to crowding, "
        "behavioral-bias awareness, or COVID/2022 macro shocks."
    )


def page_capacity(d: dict[str, pd.DataFrame]) -> None:
    st.title("Capacity model: v1 (academic) vs v2 (HF-grade)")

    st.markdown(
        "**v1 (Day 4-5)**: single universe ADV, signal-agnostic 50%/month turnover, "
        "linear impact, no borrow.  \n"
        "**v2 (Day 6 upgrade)**: per-factor turnover (NMV-2016 calibration), "
        "cap-tier ADV (large_cap / ex_microcap / full_universe), sqrt-linear hybrid "
        "impact (Almgren-Chriss + FIM 2018), tier-conditional borrow cost on the "
        "short leg."
    )

    c1, c2 = st.columns(2)
    cap_v1 = d["cap_v1"][d["cap_v1"]["analysis_eligible"]]
    cap_v2 = d["cap_v2"][d["cap_v2"]["analysis_eligible"]]
    sc = lambda df: (df.groupby("aum_usd")
                     .apply(lambda x: pd.Series({
                         "n_viable": int((x["capacity_sharpe"] >= 0.30).sum()),
                         "n_total": len(x),
                         "viable_pct": (x["capacity_sharpe"] >= 0.30).mean(),
                         "mean_sharpe": x["capacity_sharpe"].mean(),
                     })).reset_index())
    sv1 = sc(cap_v1); sv2 = sc(cap_v2)
    sv1["model"] = "v1 (academic)"; sv2["model"] = "v2 (HF-grade)"
    both = pd.concat([sv1, sv2])
    both["aum_label"] = both["aum_usd"].map(fmt_aum)

    fig = px.bar(both, x="aum_label", y="viable_pct", color="model", barmode="group",
                 labels={"viable_pct": "Viable %", "aum_label": "AUM"})
    fig.update_layout(yaxis_tickformat=".0%", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost breakdown by AUM (v2)")
    cost_summary = (cap_v2.groupby("aum_usd")
                    .agg(mean_trading_bps=("annual_trading_bps", "mean"),
                         mean_borrow_bps=("annual_borrow_bps", "mean"),
                         mean_total_bps=("annual_total_cost_bps", "mean"))
                    .reset_index())
    cost_summary["aum_label"] = cost_summary["aum_usd"].map(fmt_aum)
    st.dataframe(cost_summary[["aum_label", "mean_trading_bps", "mean_borrow_bps", "mean_total_bps"]]
                 .style.format({c: "{:.0f}" for c in cost_summary.columns
                                if c.endswith("_bps")}),
                 hide_index=True, use_container_width=True)

    st.subheader("Per-tier breakdown (v2, $1B AUM)")
    tier_b1 = cap_v2[cap_v2["aum_usd"] == 1e9].copy()
    tier_summary = (tier_b1.groupby("cap_tier")
                    .agg(n=("Acronym", "count"),
                         mean_capacity_sharpe=("capacity_sharpe", "mean"),
                         viable_pct=("capacity_sharpe", lambda s: (s >= 0.30).mean()),
                         mean_total_cost_bps=("annual_total_cost_bps", "mean"))
                    .reset_index())
    st.dataframe(tier_summary.style.format({"mean_capacity_sharpe": "{:.3f}",
                                            "viable_pct": "{:.1%}",
                                            "mean_total_cost_bps": "{:.0f}"}),
                 hide_index=True, use_container_width=True)


def page_methodology(d: dict[str, pd.DataFrame]) -> None:
    st.title("Methodology & explicit caveats")
    st.markdown("""
### Positioning

This artifact does NOT claim original research. The mechanism-conditional decay
finding it reproduces was previously published by **Engelberg-McLean-Pontiff
(2024) "What Drives Anomaly Decay?"**. The contributions here are:

1. **Independent reproduction** of EMP-2024's headline directional finding
2. **Capacity overlay** (v2 with per-factor turnover, cap-tier ADV, sqrt impact, and short-side borrow)
3. **Public dashboard** (this app) consolidating decay + capacity + DSR in one place

### Universe

- 212 of OAP's 331-row signal-doc table have portfolio returns
  (`dl_port('op')`). The remaining 119 are binary signals, event-study factors,
  or signals requiring WRDS access. **"300 factor" in the original spec is
  honestly 212 factor coverage.**

### Reproduction fidelity vs published papers

- For the comparable subset (76 factors with `Test in OP` = `port sort` /
  `LS port`), median |rel_err| in IS Sharpe vs paper-reported Sharpe is **~12%**;
  ±20% covers 72%; sign agreement 98.7%; doc/our t-stat ratio median **1.003**.
- The Sharpe divergence is leg-construction noise (decile vs quintile, NYSE
  breakpoints, data-lag) between OAP's reproduction and the original papers,
  not a coding bug.

### Mechanism classification

- 3-model LLM ensemble (Anthropic Claude Haiku 4.5 + OpenAI GPT-4o-mini +
  Google Gemini 2.5 Flash Lite). Oracle κ vs hand labels = **0.85** on 20
  canonical factors (gate >0.70 PASS).
- 4-way labels: `behavioral` / `risk_premium` / `mispricing` / `data_mining_suspect`.
  In the final ensemble, `data_mining_suspect` is empty in majority — most OAP
  predictors have published economic stories.
- ~37% of factors are NOT 3/3 unanimous. Disagreement is concentrated on factors
  where the literature itself is divided (e.g., Investment, IdioVol3F).

### Decay analysis

- Post-pub start = `Year + 1` (1-year buffer). EMP-2024 use month-precision
  publication dates; OAP gives only year, so this is a coarser proxy.
- Eligibility: IS >= 36 months, post-pub >= 60 months. 4 of 212 excluded.
- LS-level decay only at first; **leg-level decomposition added in v2** (Day 6).

### Capacity model (v2)

- **Per-factor turnover** mapped from `Cat.Economic` per Novy-Marx-Velikov
  (2016) Table 5 (e.g., momentum 30%/month, value 4%/month).
- **Cap-tier universe ADV**: large_cap $30B/day, ex_microcap $40B/day,
  full_universe $50B/day. Tier inferred from OAP's `Filter` field.
- **Cost function**: `cost_bps = half_spread + 5*part% + 20*sqrt(part%)`,
  per side per rebalance. Calibrated to FIM-2018 / NMV-2016 realised costs.
- **Borrow cost** on short leg: 50/150/300 bps per year by tier.
- **Limitations**:
  - Single representative ADV per tier (microcap factors face higher costs).
  - Sqrt-linear hybrid (no concavity beyond sqrt).
  - Borrow cost is parametric, not estimated from real lending-fee data.

### Statistical rigor

- **FDR-BH at α=0.05**: 30% of factors are naive p<0.05 → only 17% survive.
- **Bootstrap (5000 reps)**: behavioral - risk_premium decay_diff = -0.254,
  95% CI [-0.42, -0.10], p=0.003. Confirms parametric Welch.
- **DSR (Bailey-LdP 2014)** with N_trials=212: only **23/212 (11%)** factors
  have DSR > 0.95. Behavioral 21% robust, mispricing 7%, risk_premium 7%.
- **OOS 2020-2024**: behavioral vs risk_premium gap COMPRESSES to insignificance
  (p=0.45) in the most recent 5-year cohort. EMP-2024's pre-2020 finding
  does not extend into 2020-2024.

### What was NOT done (intentional scope cuts)

- Per-stock yfinance ADV (delisted-ticker coverage gap, hours of fetch).
- Live microstructure cost calibration (no proprietary trading data).
- International factors (US CRSP only).
- Tax / financing / shorting-bans implementation drag.
- OOS update through 2025+ (data ends 2024-12).
""")


# ---------- main ----------

def main() -> None:
    data = load_all()
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Per-factor", "Mechanism aggregates", "Capacity v1 vs v2", "Methodology / caveats"],
    )
    {
        "Overview": page_overview,
        "Per-factor": page_per_factor,
        "Mechanism aggregates": page_mechanism,
        "Capacity v1 vs v2": page_capacity,
        "Methodology / caveats": page_methodology,
    }[page](data)
    st.sidebar.divider()
    st.sidebar.caption(
        "Data: Chen-Zimmermann Open Asset Pricing (`openassetpricing` 0.0.2). "
        "Mechanism labels: 3-model LLM ensemble (Anthropic / OpenAI / Google). "
        "Capacity: FIM-2018 / NMV-2016 calibrated."
    )


if __name__ == "__main__":
    main()
