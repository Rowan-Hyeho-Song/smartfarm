# 🍓 berry_ai — 딸기 스마트팜 예측·제어

딸기 스마트팜의 **환경·생육 데이터**로 생육을 예측하고, 그 예측의 **불확실성**과 **제어 규칙**까지 도출하는 프로젝트입니다. 스마트팜 AI 경진대회(5회 예선) 데이터를 기준으로, **예측 → 불확실성 → 튜닝 → 제어 설계 → 제어 검증(A/B) → 환경 예보 → 실증 계획**까지 하나의 흐름으로 구성했습니다.

> 데이터: 온실 1~4 · 2024-11 ~ 2025-02 (겨울작) · 생육 주 1회 조사 + 환경 시간별 로깅

---

## 📊 핵심 결과

| 항목 | 결과 |
|---|---|
| **착과수 예측 v3** (개체 추적 + 개체 lag, huber·MAE 튜닝) | 집계 **R² 0.94** · **개체 R² 0.85** · MAE중앙 **0.73** |
| 착과수 예측 v2.1 (온실-날짜 평균 + lag + **전주 개체 분포** + 튜닝) | R² 0.89 (80% 돌파) |
| 착과수 예측 v2 (분포 없이) | R² 0.87 |
| 착과수 예측 v1 (개체 단위, 개체 lag 없음) | R² 0.64~0.73 |
| 착과수 **버킷 정확도** (결과지, 6구간) | **73%** (±1버킷 100%) |
| 예측 구간 적중률 (Conformal 보정 후) | **83%** (목표 80%) |
| 내부온도 예보 (6시간 뒤) | **R² 0.86** |
| CO₂ 제어 효과 (관측 준-A/B) | 고CO₂ 그룹 초장 **+1.57cm** (p<0.001) |

---

## 🗂️ 프로젝트 구조

```text
smartfarm/
├─ data/                     # 원본 데이터 (생육 1 + 환경 4 CSV, UTF-8)
│  └─ processed/             # 병합 학습표 (생성물)
├─ src/                      # 파이프라인 스크립트
│  ├─ data_prep.py           # 5개 CSV → 학습표 (집계·발달단계 feature)
│  ├─ train.py               # LightGBM 베이스라인 (검증방식 스위치)
│  ├─ train_advanced.py      # 앙상블·교차검증·불확실성
│  ├─ conformal.py           # Conformal 보정 (예측 구간 적중률 교정)
│  ├─ train_optuna.py        # Optuna 하이퍼파라미터 튜닝
│  ├─ predict.py             # 추론·제출 (예측값 + 신뢰 구간)
│  ├─ compare_experiments.py # 7개 지표 × 검증 × 집계 창 실험
│  ├─ forecast_env.py        # 환경 예보 (h시간 뒤 내부 온도·CO₂)
│  ├─ train_agg.py           # v2.1 — 온실-날짜 평균 + lag + 전주 분포 + 튜닝 (R² 0.89)
│  ├─ train_individual.py    # v3 — 개체 추적 + 개체 lag, huber·MAE 튜닝 (집계 0.94·개체 0.85·MAE 0.73)
│  └─ report.py              # 결과지 생성 (LOGO 행별 실제·예측·버킷·신뢰도 → CSV·HTML)
├─ models/                   # 학습된 모델
├─ docs/                     # 문서 (HTML, 브라우저로 열람)
└─ requirements.txt
```

---

## ⚙️ 설치 & 실행

```bash
# 1) 가상환경 (Miniforge/conda 권장)
conda create -n strawberry python=3.11 -y
conda activate strawberry
pip install pandas numpy scikit-learn lightgbm catboost optuna scipy

# (Mac은 LightGBM 전에)  brew install libomp

# 2) 데이터 병합 → 학습 → 추론
python src/data_prep.py       # data/processed/train_table.csv 생성
python src/train.py           # 베이스라인 학습·저장
python src/train_optuna.py    # 튜닝 모델 저장
python src/predict.py         # 예측 + 신뢰 구간 → predictions_*.csv
python src/forecast_env.py    # 환경 예보 성능
```

> 자세한 환경 구축(맥/윈도우 동일화)은 [docs/setup/연습환경_설치_데이터준비.html](docs/setup/연습환경_설치_데이터준비.html) 참고.

---

## 📚 문서 (docs/)

HTML 문서라 **브라우저로 열면** 도표·차트가 보입니다. (GitHub에서는 raw로 보이므로 로컬에서 열람 권장)

| 문서 | 내용 |
|---|---|
| [착과수 개체별 결과지 (v3)](docs/report/착과수_개체별_결과지.html) | 개체 40개 × 15주 궤적(실제 vs 예측) · 개체 R² 0.85 · MAE중앙 0.73 |
| [착과수 예측 결과지 (v2.1)](docs/report/착과수_예측_결과지.html) | 온실-날짜 60건 행별 실제·예측·버킷일치·신뢰도 |
| [파이프라인 흐름도](docs/pipeline/전체_예측_파이프라인_흐름도.html) | 전체 시스템 한눈에 |
| [모델 소개·설계 의도](docs/explain/모델_소개_설계의도.html) | 무엇을·왜·어떻게 (발표용) |
| [작업 이력 (원인·변경·이유)](docs/journal/작업이력_원인과변경.html) | 모델링 결정 11가지 |
| [생육 지표·집계 창 비교](docs/experiments/생육지표_집계창_비교.html) | 7개 지표 실험 |
| [A/B 제어 실험 (CO₂·온도)](docs/experiments/AB제어실험_CO2.html) | 제어 레버 검증 |
| [환경 예보 트랙](docs/experiments/환경예보_트랙.html) | h시간 뒤 환경 예측 |
| [룰 베이스 제어 규칙](docs/control/룰베이스_제어규칙.html) | 상관 기반 제어 규칙 |
| [실증 실험 계획서](docs/plan/실증실험_계획서.html) | 무작위 배정 A/B 프로토콜 |
| [딸기 생육·의사결정 쉬운 설명](docs/guide/딸기생육_지표_의사결정_쉬운설명.html) | 지표 → 생육 → 결정 |

---

## 🔬 방법론 하이라이트

- **검증방식을 타깃 성격에 맞춤** — 착과수(발달주기 지배)는 온실별 검증, 초장·엽병장(환경 반영)은 시간분할.
- **데이터 누수 차단** — 조사일 당일 환경은 요약에서 제외(미래 정보 차단).
- **불확실성 정량화** — 분위수 회귀 + Conformal(CQR)로 예측 구간을 정직하게 보정.
- **복합(다변량) 상관** — 부분상관·상호작용으로 진짜 제어 레버(CO₂) 규명, 계절 교란 제거.

---

*본 저장소는 대회 연습·연구 목적입니다. 성능 수치는 보유 데이터 기준이며 데이터 확대 시 달라질 수 있습니다.*
