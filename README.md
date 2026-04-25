# factor-zoo-decay-audit

[![Live demo](https://img.shields.io/badge/Streamlit-live%20demo-2563eb?logo=streamlit&logoColor=white)](https://factor-zoo-decay-audit-260426ah.streamlit.app) &nbsp; [![GitHub](https://img.shields.io/badge/GitHub-Crane--lupert%2Ffactor--zoo--decay--audit-181717?logo=github)](https://github.com/Crane-lupert/factor-zoo-decay-audit)

**Live dashboard**: <https://factor-zoo-decay-audit-260426ah.streamlit.app>
**Repository**: <https://github.com/Crane-lupert/factor-zoo-decay-audit>

**Scope**: 5-7 days (extended) | **Status**: Day 6 complete + dashboard MVP deployed | **Positioning**: factor literature 유창성 배지 + capacity overlay (NOT original research)

## What this is

Independent reproduction of the Chen-Zimmermann Open Asset Pricing library (212 published equity-return predictors with portfolio data) plus:

1. Mechanism classification (3-model LLM ensemble; oracle κ = 0.85)
2. Post-publication decay analysis directionally reproducing Engelberg-McLean-Pontiff (2024)
3. **HF-grade capacity model** (per-factor turnover, cap-tier ADV, sqrt-linear hybrid impact, short-side borrow)
4. **Long/short leg decomposition** (short side is what fades, EMP-2024 mechanism story)
5. **Rigor layer**: BH-FDR, factor-level bootstrap, Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)
6. **OOS post-2020 cohort** (the EMP directional gap COMPRESSES to insignificance in 2020-2024)
7. Streamlit dashboard consolidating everything

**Honest positioning**: research novelty was scooped by EMP-2024. The artifact's value is (a) public dashboard, (b) capacity overlay with HF-grade calibration, (c) independent reproduction rigor.

## Install

```bash
cd d:/vscode/factor-zoo-decay-audit
uv venv
uv pip install -e .
uv pip install -e D:/vscode/portfolio-coordination/shared-utils
```

`OPENROUTER_API_KEY` should be in `D:/vscode/portfolio-coordination/.env` (already configured for the coord meta-repo). LLM calls only used in Day 2 mechanism classification (~$0.32 total under $3 cap).

## Re-run pipeline

```bash
.venv/Scripts/python scripts/day1_run.py            # cache OAP + IS/post-pub Sharpe + sanity
.venv/Scripts/python scripts/day2_run.py            # 3-model LLM ensemble (212 x 3 = 636 calls)
.venv/Scripts/python scripts/day2_analyze.py        # ensemble majority + oracle kappa
.venv/Scripts/python scripts/day3_run.py            # mechanism-conditional decay
.venv/Scripts/python scripts/day4_run.py            # capacity v1 (academic)
.venv/Scripts/python scripts/day6_legs.py           # long/short leg decomposition
.venv/Scripts/python scripts/day6_capacity_v2.py    # capacity v2 (HF-grade)
.venv/Scripts/python scripts/day6_oos.py            # OOS post-2020 cohort
.venv/Scripts/python scripts/day6_dsr.py            # Deflated Sharpe Ratio
.venv/Scripts/python scripts/day6_rigor.py          # BH-FDR + bootstrap
.venv/Scripts/streamlit run dashboard/app.py        # launch dashboard
```

## Key findings

| Test | Result |
|---|---|
| 212 factor load | PASS |
| Sign agreement vs OAP doc | 98.7% |
| Median t-stat ratio (ours / doc) | 1.003 |
| Oracle κ on 20 hand-labeled factors | 0.850 |
| EMP 2024 direction (full post-pub, behavioral - risk_premium decay) | -0.254 (Welch p=0.003, MW p=0.026) |
| Bootstrap factor-level CI on the same | [-0.42, -0.10], p=0.003 |
| OOS post-2020 cohort same gap | +0.069 (p=0.45) -- compresses to null |
| Long-leg post-pub mean Sharpe across mechanisms | 0.45 to 0.53 (alpha survives) |
| Short-leg post-pub mean Sharpe across mechanisms | -0.45 to -0.50 (this is what's broken) |
| BH-FDR at α=0.05 surviving | 36/210 (17%) |
| DSR > 0.95 (multi-test robust) | 23/212 (11%) |
| Capacity v2 viable @ $1B (Sharpe ≥ 0.30) | 9/208 (4%) |
| Capacity v2 viable @ $100M | 13/208 (6%) |

## Caveats (honest list — also displayed in the dashboard)

- "300 factor" claim in original spec is honestly **212 factor** coverage. Remaining 119 OAP rows are binary signals, event-study factors, or WRDS-required predictors.
- Sharpe reproduction vs originally-reported papers: median |rel_err| ~12% on the comparable subset (76 raw-LS / port-sort factors). Sign agreement 98.7%, t-stat ratio 1.003. Divergence is leg-construction noise (decile vs quintile, NYSE breakpoints, data-lag) between OAP's reproduction and the original papers.
- Mechanism classification: ~37% of factors are not 3-of-3 unanimous. Disagreement concentrated where the literature is itself divided (Investment, IdioVol3F).
- Post-pub start = `Year + 1`. EMP-2024 use month-precision; OAP gives only year.
- Capacity model is parametric (FIM-2018 / NMV-2016 calibration), not estimated from real lending fee or trading microstructure data. Sqrt-linear hybrid impact, single representative ADV per cap-tier.
- Borrow cost on short leg is parametric (50/150/300 bps/yr by tier), not from Markit short-interest data.
- US CRSP only (no international).

## Public deployment (Streamlit Community Cloud)

Repo is structured to deploy unchanged to [share.streamlit.io](https://share.streamlit.io):

1. **Create GitHub remote** (one-time, user action):
   ```bash
   gh repo create factor-zoo-decay-audit --public --source . --push
   # or set an existing remote:
   git remote add origin https://github.com/<USER>/factor-zoo-decay-audit.git
   git push -u origin main
   ```

2. **Connect Streamlit Cloud**:
   - Sign in with GitHub at [share.streamlit.io](https://share.streamlit.io)
   - "New app" → pick the repo, branch=`main`, `dashboard/app.py`
   - Python 3.11, `requirements.txt` is auto-detected
   - First build ~2 min; subsequent pushes auto-redeploy

3. **What's checked in for the cloud runtime**:
   - `cache/oap_signal_doc.parquet` (94 KB)
   - `cache/oap_ls_returns.parquet` (1.8 MB)
   - All `results/day*/*.parquet` (~1 MB)
   - Excluded: `cache/oap_op_returns.parquet` (22 MB; only used by Day-6 leg
     decomposition, whose result is already cached as `leg_decay.parquet`).

Total deployed footprint < 5 MB; well under GitHub free-tier per-file limit.

## Repository layout

```
src/factor_zoo_decay_audit/
    load.py          OAP data loaders + parquet cache
    sharpe.py        IS / post-pub annualized Sharpe
    sanity.py        Day-1 sanity gate (relaxed; structural reasoning documented)
    mechanism.py     LLM ensemble classification + prompt (v2 anchored)
    agreement.py     Cohen's kappa, majority vote, oracle kappa
    decay.py         Decay metrics + bootstrap CI + pairwise tests
    capacity.py      v1 (academic) + v2 (HF-grade) cost models, survival curve
    legs.py          Long/short leg decomposition
    oos.py           Post-2020 cohort decay
    dsr.py           Deflated Sharpe Ratio (Bailey-LdP 2014)
    rigor.py         BH-FDR + factor-level bootstrap

dashboard/app.py     Streamlit MVP with 5 pages

results/
    day1/  IS+post-pub Sharpe, sanity diagnostics
    day2/  ensemble classifications (v1 archived as v1.parquet) + oracle diagnostics
    day3/  decay_per_factor.parquet, walk_forward_sharpes.parquet
    day4/  capacity_adjusted.parquet (v1)
    day6/  capacity_adjusted_v2.parquet, leg_decay.parquet, oos_decay.parquet,
           dsr.parquet, fdr.parquet, factor_profiles.csv, sensitivity.csv
```

## License / attribution

Data: Chen & Zimmermann's [Open Asset Pricing](https://www.openassetpricing.com/). LLM mechanism labels via 3-model ensemble (Anthropic Claude Haiku 4.5 + OpenAI GPT-4o-mini + Google Gemini 2.5 Flash Lite) through OpenRouter.
