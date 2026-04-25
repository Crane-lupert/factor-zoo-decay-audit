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
        "rolling": pd.read_parquet(base / "results" / "day6" / "rolling_viability.parquet"),
        "decade": pd.read_parquet(base / "results" / "day6" / "decade_breakdown.parquet"),
    }
    out["ls"]["date"] = pd.to_datetime(out["ls"]["date"])
    out["wf"]["year"] = out["wf"]["year"].astype(int)
    return out


def fmt_aum(a: float) -> str:
    return f"${a/1e9:.0f}B" if a >= 1e9 else f"${a/1e6:.0f}M"


# Display order for AUM scenarios; aum_label is a string so we must enforce
# numeric order explicitly (otherwise pandas/plotly alphasort gives 100B<100M<10B<1B).
AUM_ORDER = ["$100M", "$1B", "$10B", "$100B"]


@st.cache_data(show_spinner=False)
def _compute_multi_criterion_survival(d_repr: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Per-(AUM, mechanism) survival counts under 3 capacity-haircut criteria.

    crit_a : long-horizon capacity_sharpe >= 0.30
    crit_b : capacity-adjusted annual Sharpe positive in majority of post-pub years
    crit_c : capacity-adjusted annual Sharpe >= 0.30 in majority of post-pub years

    Annual capacity-adjusted Sharpe = annual_sharpe_gross - sharpe_haircut(AUM, factor),
    where the haircut is the long-horizon (annualized) value -- a conservative
    approximation that matches the v2 cost model used in capacity_adjusted_v2.parquet.
    """
    base = REPO_ROOT
    cap = pd.read_parquet(base / "results" / "day6" / "capacity_adjusted_v2.parquet")
    cap = cap[cap["analysis_eligible"]]
    wf = pd.read_parquet(base / "results" / "day3" / "walk_forward_sharpes.parquet")
    decay = pd.read_parquet(base / "results" / "day3" / "decay_per_factor.parquet")

    wf_post = wf[wf["years_since_pub"] > 0][["Acronym", "year", "sharpe_ann"]]

    rows: list[dict] = []
    for aum, sub in cap.groupby("aum_usd", sort=True):
        merged = wf_post.merge(
            sub[["Acronym", "mechanism", "sharpe_haircut", "capacity_sharpe"]],
            on="Acronym", how="inner",
        )
        merged["annual_net"] = merged["sharpe_ann"] - merged["sharpe_haircut"]
        per_factor = merged.groupby(["Acronym", "mechanism"]).agg(
            cap_sharpe=("capacity_sharpe", "first"),
            frac_pos=("annual_net", lambda s: (s > 0).mean()),
            frac_03=("annual_net", lambda s: (s >= 0.30).mean()),
        ).reset_index()
        per_factor["crit_a"] = per_factor["cap_sharpe"] >= 0.30
        per_factor["crit_b"] = per_factor["frac_pos"] >= 0.50
        per_factor["crit_c"] = per_factor["frac_03"] >= 0.50

        for mech, g in per_factor.groupby("mechanism"):
            for col in ("crit_a", "crit_b", "crit_c"):
                rows.append({
                    "aum_usd": aum, "mechanism": mech, "criterion": col,
                    "n_total": int(len(g)), "n_viable": int(g[col].sum()),
                    "viable_pct": float(g[col].mean()),
                })
        for col in ("crit_a", "crit_b", "crit_c"):
            rows.append({
                "aum_usd": aum, "mechanism": "ALL", "criterion": col,
                "n_total": int(len(per_factor)), "n_viable": int(per_factor[col].sum()),
                "viable_pct": float(per_factor[col].mean()),
            })
    return pd.DataFrame(rows)


# ---------- pages ----------

def page_overview(d: dict[str, pd.DataFrame]) -> None:
    st.title("Factor Zoo Decay + Capacity")
    st.caption(
        "Independent reproduction of Chen-Zimmermann's 212 published equity-return "
        "predictors with mechanism classification, post-publication decay, and a "
        "capacity-adjusted survival curve."
    )

    with st.expander("What is this dashboard? (read first)", expanded=False):
        st.markdown(
            """
Reproduction + decay + capacity + multi-test rigor for the 212 factors with
portfolio data in the **Chen-Zimmermann Open Asset Pricing** library.
Reproduces the EMP-2024 mechanism-conditional decay direction (Welch
p=0.003 on the full post-pub window) and reports the gap COMPRESSES to
insignificance (p=0.45) in 2020-2024 OOS. Capacity model is FIM-2018 /
NMV-2016 calibrated (per-factor turnover, cap-tier ADV, sqrt-linear
hybrid impact, tier-conditional borrow on the short leg). Not original
research; positioning is reproducible public artifact.
"""
        )

    decay = d["decay"][d["decay"]["analysis_eligible"]]
    cap = d["cap_v2"][d["cap_v2"]["analysis_eligible"]]
    dsr = d["dsr"]
    rolling = d["rolling"].merge(decay[["Acronym"]], on="Acronym", how="inner")  # eligible only

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Eligible factors", f"{decay['Acronym'].nunique()}")
    c2.metric("Mean post-pub Sharpe", f"{decay['post_sharpe'].mean():.3f}")
    c3.metric("Mean decay (post - IS)", f"{decay['decay_diff'].mean():.3f}")
    c4.metric("DSR-robust (>=0.95)", f"{int(dsr['robust_at_05'].sum())}/{len(dsr)}",
              help="DSR is computed on the IS Sharpe vs the multi-test Bailey-Lopez de Prado threshold; denominator is full universe (212), since DSR doesn't depend on post-pub eligibility.")
    cap_100m = cap[cap["aum_usd"] == 1e8]
    c5.metric("Viable @ $100M (v2)",
              f"{int((cap_100m['capacity_sharpe'] >= 0.30).sum())}/{cap_100m['Acronym'].nunique()}",
              help="Capacity-adjusted Sharpe >= 0.30 with HF-grade v2 cost model (per-factor turnover + tier ADV + sqrt impact + borrow).")

    n_elig = len(rolling)
    # New: per-mechanism comparison across the 3 viability criteria
    st.subheader("Rolling viability fraction by mechanism (3 criteria compared)")
    eligible_with_mech = (
        decay[["Acronym", "mechanism", "post_sharpe"]]
        .merge(d["rolling"][["Acronym", "frac_positive", "frac_above_03"]],
               on="Acronym", how="inner")
    )

    def _summ(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "Long-horizon Sharpe ≥ 0.30":
                float((g["post_sharpe"] >= 0.30).mean() * 100),
            "Positive in ≥50% of years":
                float((g["frac_positive"] >= 0.50).mean() * 100),
            "Sharpe ≥ 0.30 in ≥50% of years":
                float((g["frac_above_03"] >= 0.50).mean() * 100),
            "n": int(len(g)),
        })

    mech_summ = (
        eligible_with_mech.groupby("mechanism")
        .apply(_summ, include_groups=False).reset_index()
    )
    all_summ = pd.DataFrame([_summ(eligible_with_mech).rename({"n": "n"})])
    all_summ["mechanism"] = "ALL"
    rolling_by_mech = pd.concat([mech_summ, all_summ], ignore_index=True)

    rolling_long = rolling_by_mech.melt(
        id_vars=["mechanism", "n"],
        value_vars=["Long-horizon Sharpe ≥ 0.30",
                    "Positive in ≥50% of years",
                    "Sharpe ≥ 0.30 in ≥50% of years"],
        var_name="criterion", value_name="viable_pct",
    )
    color_map_full = {
        "behavioral": "#dc2626", "mispricing": "#2563eb",
        "risk_premium": "#059669", "ALL": "#374151",
    }
    fig_via = px.bar(
        rolling_long, x="criterion", y="viable_pct", color="mechanism",
        barmode="group", height=380, color_discrete_map=color_map_full,
        category_orders={"mechanism": ["ALL", "behavioral", "mispricing", "risk_premium"]},
        labels={"viable_pct": "Viable factors (%)", "criterion": ""},
        text=rolling_long["viable_pct"].round(0).astype(int).astype(str) + "%",
    )
    fig_via.update_traces(textposition="outside")
    fig_via.update_layout(margin={"t": 30, "b": 30, "l": 30, "r": 30},
                          yaxis_range=[0, 100])
    st.plotly_chart(fig_via, use_container_width=True)
    st.caption(
        "33pp gap between strictest (long-horizon ≥ 0.30, 46%) and loosest "
        "(positive-in-majority, 79%) shows the long-horizon tail effect; "
        "behavioral leads in the loose criteria but fades hardest in long-horizon."
    )

    # New: threshold sensitivity curve (true survival curve over Sharpe threshold)
    st.subheader("Threshold sensitivity -- survival under increasing Sharpe bar")
    wf_post = d["wf"][d["wf"]["years_since_pub"] > 0].merge(
        decay[["Acronym", "mechanism"]], on="Acronym", how="inner"
    )
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    for thr in thresholds:
        per_factor = (
            wf_post.assign(above=lambda x: x["sharpe_ann"] >= thr)
            .groupby(["Acronym", "mechanism"])["above"].mean()
            .reset_index()
        )
        for mech, g in per_factor.groupby("mechanism"):
            rows.append({"threshold": thr, "mechanism": mech,
                         "viable_pct": float((g["above"] >= 0.50).mean() * 100)})
        rows.append({"threshold": thr, "mechanism": "ALL",
                     "viable_pct": float((per_factor["above"] >= 0.50).mean() * 100)})
    sens = pd.DataFrame(rows)

    fig_sens = go.Figure()
    for mech in ["ALL", "behavioral", "mispricing", "risk_premium"]:
        sub = sens[sens["mechanism"] == mech]
        fig_sens.add_trace(go.Scatter(
            x=sub["threshold"], y=sub["viable_pct"], mode="lines+markers",
            name=mech, line={"width": 3, "color": color_map_full.get(mech)},
            hovertemplate=mech + "<br>threshold: %{x:.2f}<br>viable: %{y:.1f}%<extra></extra>",
        ))
    fig_sens.add_vline(x=0.30, line_dash="dot",
                       annotation_text="0.30 (default)")
    fig_sens.update_layout(
        height=380,
        xaxis_title="Annual Sharpe threshold (factor counted as 'still working' if it clears X in ≥50% of post-pub years)",
        yaxis_title="Viable factors (%)",
        margin={"t": 30, "b": 30, "l": 30, "r": 30}, yaxis_range=[0, 100],
    )
    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption(
        "Smooth monotone curves => headline rolling-viability rate is robust to "
        "small changes in threshold. Behavioral leads at low thresholds, converges "
        "to the other two by ~0.50."
    )

    st.subheader("Survival curves -- viable factors vs AUM under three criteria")
    st.caption(
        "Same X-axis (AUM) and Y-axis (% of factors viable) as a classical "
        "capacity survival curve. Each tab applies a DIFFERENT viability "
        "criterion to the after-cost (capacity-haircut) Sharpe. All criteria "
        "use the v2 cost model."
    )
    surv_df = _compute_multi_criterion_survival(d)
    tabs = st.tabs([
        "(A) capacity-adjusted Sharpe ≥ 0.30",
        "(B) annual Sharpe positive in ≥50% of years",
        "(C) annual Sharpe ≥ 0.30 in ≥50% of years",
    ])
    crit_specs = [
        ("crit_a", "(A) Long-horizon capacity-adjusted Sharpe ≥ 0.30"),
        ("crit_b", "(B) Annual capacity-adjusted Sharpe positive in ≥50% of post-pub years"),
        ("crit_c", "(C) Annual capacity-adjusted Sharpe ≥ 0.30 in ≥50% of post-pub years"),
    ]
    for tab, (col, title) in zip(tabs, crit_specs):
        with tab:
            sub = surv_df[surv_df["criterion"] == col]
            # Dynamic Y-axis: pad ~15% above the per-criterion max so the visual
            # uses the full chart height regardless of strictness level.
            y_max_pct = float(sub["viable_pct"].max() * 100) if len(sub) else 100.0
            y_top = max(5.0, y_max_pct * 1.15)
            fig = go.Figure()
            for mech in ["ALL", "behavioral", "mispricing", "risk_premium"]:
                ms = sub[sub["mechanism"] == mech].sort_values("aum_usd")
                if len(ms) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=[fmt_aum(a) for a in ms["aum_usd"]],
                    y=ms["viable_pct"] * 100, mode="lines+markers",
                    name=mech, line={"width": 3, "color": color_map_full.get(mech)},
                    customdata=ms[["n_viable", "n_total"]].values,
                    hovertemplate=mech + " @ %{x}<br>viable %{y:.1f}%<br>"
                                  "%{customdata[0]}/%{customdata[1]} factors<extra></extra>",
                ))
            fig.update_layout(
                title=title, height=360, yaxis_range=[0, y_top],
                xaxis={"categoryorder": "array", "categoryarray": AUM_ORDER},
                xaxis_title="AUM", yaxis_title="Viable factors (%)",
                margin={"t": 50, "b": 30, "l": 30, "r": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean decay_diff by mechanism (post - IS, annualized Sharpe)")
    grp = (decay.groupby("mechanism")["decay_diff"]
           .agg(["mean", "median", "count"]).reset_index())
    grp.columns = ["mechanism", "mean", "median", "n"]
    st.dataframe(grp.style.format({"mean": "{:.3f}", "median": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    st.caption(
        "decay_diff = post-pub minus IS Sharpe. Behavioral (-0.68) decays "
        "~1.6x more than the other two (-0.42) on the full post-pub window; "
        "Welch p = 0.003. See Mechanism page for OOS 2020-2024 contrast."
    )


def page_per_factor(d: dict[str, pd.DataFrame]) -> None:
    st.title("Per-factor view")
    st.caption(
        "Pick a factor for full lifecycle: IS / post-pub / OOS Sharpe, DSR, "
        "long vs short leg, cumulative LS, 60m rolling Sharpe, v1 vs v2 capacity. "
        "Dashed line = Pub+1yr."
    )
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
    cap_v2 = d["cap_v2"][d["cap_v2"]["Acronym"] == pick].sort_values("aum_usd").copy()
    cap_v2["aum_label"] = cap_v2["aum_usd"].map(fmt_aum)
    cap_v1 = d["cap_v1"][d["cap_v1"]["Acronym"] == pick].sort_values("aum_usd").copy()
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
                           xaxis={"categoryorder": "array", "categoryarray": AUM_ORDER},
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
    st.markdown(
        """
We classify each factor into one of three economic mechanism categories
based on the LLM ensemble (Anthropic Claude Haiku 4.5 + OpenAI GPT-4o-mini
+ Google Gemini 2.5 Flash Lite) reading the OAP `LongDescription`,
`Detailed Definition`, `Notes`, and `Cat.Economic` columns.

| category | story | canonical examples |
|---|---|---|
| **behavioral** | investor psychology causes systematic mispricing | momentum (under-reaction), short-term reversal (over-reaction), PEAD, MAX-return lottery preferences |
| **risk_premium** | rational compensation for systematic risk / discount-rate channel | size, book-to-market, gross profitability, low-beta (Frazzini-Pedersen), q-theory investment |
| **mispricing** | known mispricing sustained by limits-to-arbitrage / frictions | Sloan accruals, equity issuance, idiosyncratic-vol puzzle, Piotroski F-score |

Oracle κ vs 20 hand-labeled factors = **0.85** (Landis-Koch "almost
perfect"); 134/212 factors are unanimously labeled by all three models.
"""
    )

    decay = d["decay"][d["decay"]["analysis_eligible"]].copy()
    # 2-2 fix: hover shows Acronym + Year + decay_diff
    decay["hover_label"] = decay["Acronym"] + " (" + decay["Year"].astype(int).astype(str) + ")"

    st.subheader("decay_diff (post - IS Sharpe) distribution by mechanism")
    fig = px.box(decay, x="mechanism", y="decay_diff", points="all",
                 color="mechanism", height=400,
                 hover_data={"Acronym": True, "Year": True,
                             "is_sharpe": ":.3f", "post_sharpe": ":.3f",
                             "decay_diff": ":.3f", "mechanism": False})
    fig.update_traces(pointpos=0, jitter=0.3,
                      marker={"size": 6, "opacity": 0.7})
    fig.update_layout(margin={"t": 20, "b": 30, "l": 30, "r": 30},
                      yaxis_title="decay_diff (annualized Sharpe)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Hover any point for Acronym, Year, IS / post Sharpe, decay_diff. "
        "Behavioral box (red) clearly centers lower."
    )

    st.subheader("Long / short leg decomposition (mean per mechanism)")
    st.markdown(
        "A long-short factor portfolio is two bets at once. "
        "**EMP-2024's mechanism story**: post-publication decay is concentrated "
        "in the **short side**; long legs survive. Below confirms this for our 212-factor reproduction."
    )
    leg = d["leg_decay"][d["leg_decay"]["analysis_eligible"]]
    summary = (leg.groupby("mechanism")
               .agg(mean_long_decay=("long_decay_diff", "mean"),
                    mean_short_decay=("short_decay_diff", "mean"),
                    mean_ls_decay=("ls_decay_diff", "mean"),
                    mean_long_post_sharpe=("long_post_sharpe", "mean"),
                    mean_short_post_sharpe=("short_post_sharpe", "mean"),
                    n=("Acronym", "count"))
               .reset_index())
    # New leg-decomposition bar chart
    leg_long = summary.melt(
        id_vars=["mechanism"],
        value_vars=["mean_long_post_sharpe", "mean_short_post_sharpe"],
        var_name="leg", value_name="post_sharpe"
    )
    leg_long["leg"] = leg_long["leg"].map({
        "mean_long_post_sharpe": "long-only",
        "mean_short_post_sharpe": "short-only (when shorted)",
    })
    fig_leg = px.bar(leg_long, x="mechanism", y="post_sharpe", color="leg",
                     barmode="group", height=320,
                     color_discrete_map={"long-only": "#2563eb",
                                         "short-only (when shorted)": "#dc2626"},
                     labels={"post_sharpe": "Mean post-pub Sharpe"})
    fig_leg.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig_leg.update_layout(margin={"t": 20, "b": 30, "l": 30, "r": 30})
    st.plotly_chart(fig_leg, use_container_width=True)
    st.dataframe(summary.style.format({c: "{:.3f}" for c in summary.columns
                                       if c not in ["mechanism", "n"]}),
                 hide_index=True, use_container_width=True)
    st.caption(
        "Long legs deliver +0.45 to +0.53 (alpha survives long-only); short legs "
        "deliver -0.45 to -0.50 — post-pub fade is a short-side phenomenon."
    )

    st.subheader("Decade-level breakdown -- mean annual Sharpe by years_since_pub bin x mechanism")
    decade = d["decade"].copy()
    decade["decade_bin"] = decade["decade_bin"].astype(str)
    decade_order = ["0-5y", "5-10y", "10-15y", "15-20y", "20-30y", "30+y"]
    decade_pivot = decade.pivot_table(
        index="decade_bin", columns="mechanism", values="mean", aggfunc="first"
    ).reindex(decade_order).reset_index()

    # New: heatmap visualization
    heatmap_df = decade_pivot.set_index("decade_bin")
    fig_heat = px.imshow(
        heatmap_df.T, color_continuous_scale="RdBu", origin="lower",
        zmin=-0.4, zmax=0.7, aspect="auto",
        labels={"color": "mean annual Sharpe"},
    )
    fig_heat.update_layout(height=260, margin={"t": 20, "b": 30, "l": 30, "r": 30},
                           xaxis_title="years since publication",
                           yaxis_title="mechanism")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(
        decade_pivot.style.format(
            {c: "{:.3f}" for c in decade_pivot.columns if c != "decade_bin"}
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Years 0-10 sustain Sharpe ~0.4-0.6; the 20-30y / 30+y tail drags long-"
        "horizon averages down. This is why long-horizon ≥0.30 (46%) << positive-"
        "in-majority (79%)."
    )

    st.subheader("OOS post-2020 cohort vs full post-pub")
    oos = d["oos"]
    oos_sum = (oos.dropna(subset=["mechanism"]).groupby("mechanism")
               .agg(n=("Acronym", "count"),
                    mean_post_sharpe=("post_sharpe", "mean"),
                    mean_oos_2020_sharpe=("oos_sharpe", "mean"),
                    frac_oos_above_03=("oos_sharpe", lambda s: (s >= 0.30).mean()))
               .reset_index())
    # New: side-by-side bar chart
    oos_long = oos_sum.melt(
        id_vars=["mechanism"],
        value_vars=["mean_post_sharpe", "mean_oos_2020_sharpe"],
        var_name="window", value_name="mean_sharpe"
    )
    oos_long["window"] = oos_long["window"].map({
        "mean_post_sharpe": "Full post-pub",
        "mean_oos_2020_sharpe": "OOS 2020-2024",
    })
    fig_oos = px.bar(oos_long, x="mechanism", y="mean_sharpe", color="window",
                     barmode="group", height=320,
                     color_discrete_map={"Full post-pub": "#94a3b8",
                                         "OOS 2020-2024": "#2563eb"},
                     labels={"mean_sharpe": "Mean Sharpe"})
    fig_oos.add_hline(y=0.30, line_dash="dot", annotation_text="0.30 threshold")
    fig_oos.update_layout(margin={"t": 20, "b": 30, "l": 30, "r": 30})
    st.plotly_chart(fig_oos, use_container_width=True)
    st.dataframe(oos_sum.style.format({c: "{:.3f}" for c in oos_sum.columns
                                       if c not in ["mechanism", "n"]}),
                 hide_index=True, use_container_width=True)
    st.caption(
        "Honest OOS finding: behavioral-vs-risk-premium gap compresses from p=0.003 "
        "(full post-pub) to p=0.45 (2020-2024). Drivers not separated here: post-EMP "
        "crowding, COVID/2022 regime, or sample-size attenuation."
    )


def page_capacity(d: dict[str, pd.DataFrame]) -> None:
    st.title("Capacity model: v1 (academic) vs v2 (HF-grade)")

    with st.expander("What does 'capacity' mean? (read first)", expanded=False):
        st.markdown(
            """
A factor's published Sharpe ratio is gross — it ignores the costs you incur
when actually trading the strategy. Those costs grow with assets-under-
management (AUM): bigger trades push prices, leave more slippage, and
shorting the bad-side stocks costs borrow fees. **Capacity** is the AUM
beyond which the after-cost Sharpe drops below a deployment threshold
(here we use 0.30 — anything below ~0.30 is hard to justify after factoring
in fund-level fees and a hurdle return).

**Cost components in the v2 model:**

- **Half-spread (5–15 bps per side)** — the bid-ask spread you pay on
  entry and exit. Larger for small-cap stocks (15 bps) than blue-chips
  (5 bps).
- **Linear + sqrt impact (parametric)** — the price you push when you
  trade. Linear term: `5 × participation%`. Sqrt term: `20 × √participation%`.
  Calibrated to **Frazzini-Israel-Moskowitz (2018)** AQR live-trading data
  + **Almgren-Chriss (2005)** square-root impact theory.
- **Borrow cost (50 / 150 / 300 bps/yr by cap tier)** — the fee a stock
  lender charges to short the bad-side leg. Large-caps cost ~50 bps/yr to
  borrow; microcaps can cost 300 bps or more.
- **Per-factor turnover** — how often the portfolio is rebalanced. Calibrated
  to **Novy-Marx-Velikov (2016) Table 5** by `Cat.Economic`: momentum
  rebalances ~30% of the book per month, value ~4%, accruals ~8%.

**Why v1 vs v2?**

| | v1 (academic) | v2 (HF-grade) |
|---|---|---|
| universe ADV | single ($1T/month) | tiered ($30B / $40B / $50B per day by cap) |
| turnover | 50%/month signal-agnostic | per-factor (NMV-2016) |
| impact | linear only | linear + sqrt (Almgren-Chriss) |
| borrow cost | 0 | 50/150/300 bps/yr by tier |
| half-spread | 5 bps universal | 5/10/15 bps by tier |

v1 is the typical academic capacity exercise; v2 is what an HF risk team
would actually use for a deployment go/no-go decision. **For Mom6m at $1B
AUM**: v1 estimates 132 bps/yr cost, v2 estimates 841 bps/yr (6.4×).
"""
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
    both = pd.concat([sv1, sv2]).sort_values("aum_usd")
    both["aum_label"] = both["aum_usd"].map(fmt_aum)

    fig = px.bar(both, x="aum_label", y="viable_pct", color="model", barmode="group",
                 labels={"viable_pct": "Viable %", "aum_label": "AUM"},
                 category_orders={"aum_label": AUM_ORDER})
    fig.update_layout(yaxis_tickformat=".0%", height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "v2 wipes out ~80% of v1's apparent viability. Drivers: borrow (~10pp at "
        "$100M), per-factor turnover (momentum trades 6× v1's flat assumption), "
        "sqrt impact at very high AUM."
    )

    st.subheader("Cost breakdown by AUM (v2)")
    cost_summary = (cap_v2.groupby("aum_usd")
                    .agg(mean_trading_bps=("annual_trading_bps", "mean"),
                         mean_borrow_bps=("annual_borrow_bps", "mean"),
                         mean_total_bps=("annual_total_cost_bps", "mean"))
                    .reset_index()
                    .sort_values("aum_usd"))
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

### Long-horizon vs rolling Sharpe -- IMPORTANT framing caveat

The headline `Mean post-pub Sharpe` and `Viable @ $100M` numbers use a SINGLE
Sharpe computed over the full post-pub window (5 to 30+ years). This long-
horizon metric averages over good and bad regimes, and a factor with 10 strong
years followed by 20 weak ones will show low long-horizon Sharpe while having
delivered positive Sharpe in most years.

Concrete numbers for the 208 eligible factors:

  - Long-horizon post-pub Sharpe >= 0.30 viable    : **46%** (95/208)
  - Annual Sharpe positive in >=50% of post years : **79%** (164/208)
  - Annual Sharpe >= 0.30 in >=50% of post years   : **59%** (122/208)

The 33-percentage-point gap between "long-horizon viable" and "positive in
majority of years" is real and material -- 71 of the 113 factors flagged
"decayed" (long-horizon < 0.30) actually had positive Sharpe in MOST post-pub
years. Examples: `DivInit` (long-horizon 0.25, median annual 0.67),
`EarningsForecastDisparity` (long-horizon 0.20, median annual 0.96).

When using the dashboard:
  - "still works in most years" -> use the rolling metric (Overview page).
  - "deployable as a single long-horizon strategy" -> use the long-horizon
    metric (post_sharpe column).
  - The two are NOT equivalent.

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
