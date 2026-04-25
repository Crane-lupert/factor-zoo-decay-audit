# Project Factor — Factor Zoo Decay + Capacity Polished

## Purpose

Chen-Zimmermann Open Asset Pricing library 의 ~300 공개 factor 를 재현 + post-publication decay 측정 + **capacity-adjusted survival curve** 추가 + 공개 interactive dashboard.

**중요 positioning (정직)**:
- Research novelty 는 **Engelberg-McLean-Pontiff 2024 "What Drives Anomaly Decay?" 에 이미 scooped**. 이 프로젝트는 original research 주장 X
- 본 artifact 의 목적은 **"factor literature 유창성 배지"**: (a) 공개 대시보드 기여 (b) capacity-adjusted 추가 (c) 독립 재현 rigor 증빙
- CV 에 "replicated 300 published factors, built public dashboard with capacity adjustment" 로 1 줄 추가, 인터뷰에서 factor 대화 가능성 입증

**기여**: 공개 대시보드 형태로 300+ factor 의 decay + capacity survival 을 한 곳에 제공 (유사 리소스 부재 확인). LLM 은 factor paper abstract 의 mechanism 분류에만 사용 (최소).

---

## Dependencies

- Chen-Zimmermann: `pip install openassetpricing` (직접 설치, 외부 repo 의존 없음)
- Ken French Data Library: CSV download (MiroSalmon adapter 재사용)
- SEC 데이터: 거의 안 씀 (fundamental factor 소수에만)
- OpenRouter: `OpenRouterClient(project="F")`, project cap=4 (LLM 사용 매우 적음)

---

## 실행 계획 + Advance Gate

### Day 1 — 로드 + baseline 재현
작업:
- Repo 초기화 + `pip install openassetpricing shared-utils`
- 300 signal + portfolio returns 일괄 로드
- IS / OOS / post-publication Sharpe 계산 (사전 규정: in-sample=paper sample period, post-pub = paper publication date 이후)
- Chen-Zimmermann 이 reported 한 Sharpe 와 본 계산 일치 확인 (오차 < 5%)

**Advance Gate (Day 1 EOD)**:
- 300 factor 로드 성공
- Sanity: Chen-Zimmermann reported Sharpe 대비 재현 정합성 ≥ 95% factor 에서 ±5% 내

**미달 시**: 오차 원인 파악, openassetpricing 버전 또는 rebalance frequency 설정 재검토

### Day 2 — LLM mechanism 분류
작업:
- 각 factor 의 원 paper abstract + intro (openassetpricing 이 제공하는 metadata 또는 수동 수집)
- 3 모델 ensemble 이 mechanism label: `behavioral | risk_premium | mispricing | data_mining_suspect`
- Ensemble agreement (κ) 계산
- Oracle: 20 factor 수작업 label 과 일치 검증

**Advance Gate (Day 2 EOD)**:
- 전 factor 분류 완료
- Oracle κ > 0.7

### Day 3 — Decay 분석
작업:
- In-sample vs post-publication Sharpe 비교
- Decay ratio = Sharpe_post / Sharpe_IS
- Mechanism 카테고리별 decay 분포 (Engelberg-McLean-Pontiff 2024 결과와 일치 확인)
- Walk-forward annual Sharpe trajectory

**Advance Gate (Day 3 EOD)**:
- Engelberg-McLean-Pontiff 2024 의 "behavioral > risk-premium 간 decay 차이" 재현 확인 (독립 재현 rigor 증명)

### Day 4-5 — Capacity 확장 (핵심 contribution)
작업:
- Capacity model (Frazzini-Israel-Moskowitz 2018 기반 simplified): linear market impact (10 bps base + 0.05 × ADV participation rate)
- AUM scenario: $100M / $1B / $10B / $100B
- 각 factor 의 capacity-adjusted OOS Sharpe 계산
- Survival curve: AUM 이 증가함에 따라 Sharpe > 0.3 이상 유지되는 factor 수

**Advance Gate (Day 5 EOD)**: 300 factor × 4 AUM scenario capacity-adjusted Sharpe 완료

### Day 6-7 — Rigor + Dashboard 시작
작업:
- FDR 보정, DSR
- Bootstrap by factor
- Dashboard Streamlit:
  - Factor list with filter (mechanism, decay severity, capacity tier)
  - 개별 factor: cumulative return / rolling Sharpe / capacity haircut curve
  - Mechanism aggregate: decay boxplot

**Advance Gate (Day 7 EOD)**: dashboard MVP, 5 factor 예시 데모 가능

### Day 8-10 — Dashboard polish + deploy
- Streamlit Community Cloud 배포
- 정적 HTML fallback (GitHub Pages)

### Day 11-13 — README + writeup (lightweight)
- README 는 **데이터 + 재현 instructions + 한계 명시 (scoop 인정)** 중심
- Writeup 은 **blog post 수준 4-6p** (논문 아님). SSRN 업로드 가능하지만 short note.

---

## Data Pipeline Spec

- 주 data: `openassetpricing` Python API
- Fallback: Ken French Data Library CSV
- 추가 필요 데이터:
  - 논문 publication date (Chen-Zimmermann metadata 제공)
  - 공시 기반 fundamentals (SEC) — 소수 factor 에만
  - ADV (yfinance volume)

---

## 통계 Rigor Checklist

- [ ] Chen-Zimmermann reported Sharpe 와 재현 오차 ±5%
- [ ] Mechanism 분류 oracle κ > 0.7
- [ ] Engelberg-McLean-Pontiff 2024 재현 일치 확인
- [ ] Capacity model 합리성: ADV 비율 / 거래 비용 parameters 문헌 기반
- [ ] FDR / DSR / bootstrap

---

## Abandon Criteria

1. Chen-Zimmermann 로드 실패 → Ken French 로 축소 (n=30)
2. LLM mechanism oracle κ < 0.5 → 수동 label (범위 작아 가능)
3. Capacity model 이 문헌과 크게 다른 결과 → parameters 재검토 Day 5 까지
4. Budget: OpenRouter > $5 (LLM 적게 쓰므로 상한 낮음)

---

## Deliverables + Interview Demo

- Repo / dashboard / README / short note / CV 반영
- Paper draft 생략 (blog post / short note 만)

**Demo 5분 script**:
1. (30s) "Replicated 300 published factors via Chen-Zimmermann, built public dashboard with post-publication decay and capacity-adjusted survival curves at $100M/$1B/$10B/$100B AUM."
2. (90s) Dashboard walk: filter by mechanism, select behavioral factor, show decay + capacity haircut
3. (90s) "Scoop acknowledgment: Engelberg-McLean-Pontiff 2024 did mechanism-conditional decay research. My contribution is (a) open-source reproduction (b) capacity overlay (c) unified dashboard"
4. (60s) Methodology: cell-level rigor (DSR / FDR), Frazzini-Israel-Moskowitz capacity model
5. (30s) "Not an original paper. A literacy badge + public artifact."
