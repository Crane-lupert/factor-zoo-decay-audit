# factor-zoo-decay-audit (Project Factor, polished)

**Scope**: 5-7 days | **Status**: scaffolding | **Positioning**: Factor literature 유창성 배지 (not original research)

## What this is

Chen-Zimmermann Open Asset Pricing 의 ~300 공개 factor 를 재현 + post-publication decay 측정 + **capacity-adjusted survival curve** 추가 + 공개 대시보드.

**정직한 포지셔닝**: Research novelty 는 **Engelberg-McLean-Pontiff 2024 "What Drives Anomaly Decay?"** 에 scooped. 이 artifact 의 가치는 (a) 공개 대시보드 기여 (b) capacity-adjusted overlay (c) 독립 재현 rigor 증빙.

Full plan: [`CLAUDE.md`](CLAUDE.md)

## Install

```powershell
cd D:/vscode/factor-zoo-decay-audit
uv venv
uv pip install -e .
uv pip install -e D:/vscode/portfolio-coordination/shared-utils
copy .env.example .env
```

## Day 1 시작

1. `python -c "from openassetpricing import OpenAP; op = OpenAP(); print(op.list_signals()[:5])"` — 로드 sanity
2. `src/factor_zoo_decay_audit/load.py` — 300 factor returns + metadata DataFrame 로 consolidate
3. In-sample Sharpe 계산 + Chen-Zimmermann reported 대비 오차 ±5% 확인

## 규칙

- LLM 사용 적음 (mechanism 분류만) — `OpenRouterClient(project="F")` 4 slot 예약
- 체크포인트는 daily 1회로 축소 가능 (분량 작음)
- Dashboard 필수 — Streamlit + GitHub Pages fallback

자세한 하네스: [`CLAUDE.md`](CLAUDE.md)
