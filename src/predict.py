# -*- coding: utf-8 -*-
"""
predict.py  (v3 경로 · 추론/제출)
=================================
[이 파일이 하는 일 — 한 문장]
최고 모델 v3(개체 추적)를 불러와 '새 데이터(여기선 온실 4)'의 착과수를
① 개체(표본)별로 예측하고 ② 온실-날짜 평균으로 집계하며,
③ 각 예측에 신뢰 구간(하한~상한)을 붙여 제출용 CSV로 저장합니다.

[모델 버전]
- 기본: lgbm_fruit_count_v3.txt (개체 추적 + 개체 lag + huber·MAE, 개체 R² 0.85 / 집계 0.94)
- 없으면 v2.1(온실-날짜 평균) → v1 튜닝 순으로 자동 폴백.

[실전에서 쓰는 법]
'온실 4' 자리에 새 주차의 실제 데이터(각 개체의 직전 조사 포함)를 넣으면
그 개체들의 예측값 + 구간이 나옵니다. 정직한 일반화 성능은 train_individual.py의
LOGO 기준(개체 R² 0.85 · 집계 0.94 · MAE중앙 0.73)입니다.

실행:  python src/predict.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUT_INDIV = PROJECT_DIR / "data" / "processed" / "predictions_v3_individual.csv"
OUT_AGG = PROJECT_DIR / "data" / "processed" / "predictions_v3_greenhouse.csv"

TARGET = "착과수"
GROWTH = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
KEYS = ["온실번호", "조사일자", "측정라인", "표본번호"]

# 모델 우선순위: v3(개체) → v2.1/v2(집계) → v1(튜닝)
MODEL_CANDIDATES = [
    ("v3", MODEL_DIR / "lgbm_fruit_count_v3.txt"),
    ("v2.1", MODEL_DIR / "lgbm_fruit_count_v2.txt"),
    ("v1", MODEL_DIR / "lgbm_fruit_count_tuned.txt"),
]

TRAIN_GREENHOUSES = [1, 2, 3]   # 학습에 쓴 온실
CALIB_GREENHOUSE = 3            # 구간 보정용 (split-conformal)
PREDICT_GREENHOUSE = 4          # '새 데이터'로 예측할 온실

ALPHA = 0.20
LO, HI = ALPHA / 2, 1 - ALPHA / 2
Q_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=15,
                min_child_samples=5, subsample=0.9, colsample_bytree=0.8)


def quantile_model(alpha):
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                             random_state=42, verbose=-1, **Q_PARAMS)


def conformal_Q(scores, alpha):
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def pick_model():
    for tag, path in MODEL_CANDIDATES:
        if path.exists():
            return tag, path
    return None, None


def main():
    print("=" * 62)
    print(f"예측(추론) + 신뢰 구간 — 대상: {TARGET}")
    print("=" * 62)

    tag, model_path = pick_model()
    if model_path is None:
        print("[오류] 모델이 없습니다. 먼저 train_individual.py(또는 train_agg.py)를 실행하세요.")
        return
    booster = lgb.Booster(model_file=str(model_path))
    feat = booster.feature_name()
    print(f"[모델] {model_path.name} ({tag}) · feature {len(feat)}개")

    # 데이터 로드 (개체 단위 학습표)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])

    # 모델 feature가 데이터에 다 있는지 확인 (버전 불일치 방지)
    missing = [c for c in feat if c not in df.columns]
    if missing:
        print(f"[오류] 모델이 요구하는 feature가 학습표에 없습니다: {missing[:5]}...")
        print("      data_prep.py를 다시 실행해 학습표를 최신화하세요.")
        return

    train = df[df["온실번호"].isin(TRAIN_GREENHOUSES)]
    new = df[df["온실번호"] == PREDICT_GREENHOUSE].copy()
    print(f"[데이터] 학습 온실 {TRAIN_GREENHOUSES} · 예측 온실 {PREDICT_GREENHOUSE}"
          f"({len(new)}개체행)")

    # (1) 점 예측 (저장된 모델)
    new["예측값"] = np.maximum(booster.predict(new[feat]), 0.0)

    # (2) 예측 구간 — split-conformal (개체 단위)
    proper = train[train["온실번호"] != CALIB_GREENHOUSE]
    calib = train[train["온실번호"] == CALIB_GREENHOUSE]
    m_lo = quantile_model(LO).fit(proper[feat], proper[TARGET])
    m_hi = quantile_model(HI).fit(proper[feat], proper[TARGET])
    yc = calib[TARGET].to_numpy()
    Q = conformal_Q(np.maximum(m_lo.predict(calib[feat]) - yc,
                               yc - m_hi.predict(calib[feat])), ALPHA)
    new["예측하한"] = np.maximum(m_lo.predict(new[feat]) - Q, 0.0)   # 착과수 ≥ 0
    new["예측상한"] = m_hi.predict(new[feat]) + Q
    # 점 예측(huber)과 구간(분위수)은 다른 모델 → 구간이 점 예측을 항상 감싸도록 보정
    new["예측하한"] = np.minimum(new["예측하한"], new["예측값"])
    new["예측상한"] = np.maximum(new["예측상한"], new["예측값"])

    # (3-a) 개체별 결과 저장
    indiv_cols = ["온실번호", "측정라인", "표본번호", "조사일자",
                  "예측하한", "예측값", "예측상한"]
    if TARGET in new.columns:
        indiv_cols.append(TARGET)
    indiv = new[indiv_cols].sort_values(["온실번호", "측정라인", "표본번호", "조사일자"])
    indiv.to_csv(OUT_INDIV, index=False, encoding="utf-8-sig")
    print(f"[저장] 개체별 예측 → {OUT_INDIV.name}")

    # (3-b) 온실-날짜 평균으로 집계 (온실 전체 예보)
    agg = (new.groupby(["온실번호", "조사일자"])
              .agg(예측하한=("예측하한", "mean"), 예측값=("예측값", "mean"),
                   예측상한=("예측상한", "mean"),
                   실제평균=(TARGET, "mean") if TARGET in new.columns else ("예측값", "mean"))
              .reset_index().sort_values("조사일자"))
    agg.to_csv(OUT_AGG, index=False, encoding="utf-8-sig")
    print(f"[저장] 온실-날짜 집계 예측 → {OUT_AGG.name}")

    # (4) 미리보기
    print("-" * 62)
    print("온실-날짜 집계 예측 미리보기(앞 6주):")
    prev = agg.copy(); prev["조사일자"] = prev["조사일자"].dt.date
    show = ["조사일자", "예측하한", "예측값", "예측상한"]
    if TARGET in new.columns:
        show.append("실제평균")
    print(prev[show].head(6).to_string(index=False))

    # (5) 참고 정확도 (실제값이 있을 때만)
    if TARGET in new.columns:
        y = new[TARGET].to_numpy(); p = new["예측값"].to_numpy()
        mae = np.mean(np.abs(y - p))
        cover = np.mean((y >= new["예측하한"]) & (y <= new["예측상한"]))
        gp = new.groupby(["온실번호", "조사일자"]).agg(a=(TARGET, "mean"), p=("예측값", "mean"))
        agg_rmse = np.sqrt(np.mean((gp["a"] - gp["p"]) ** 2))
        print("-" * 62)
        print(f"[참고] 온실 {PREDICT_GREENHOUSE} — 개체 MAE={mae:.3f} · "
              f"구간 적중률={cover*100:.0f}% · 집계 RMSE={agg_rmse:.3f}")
        print("       (저장 모델은 전체 학습이라 낙관적 — 정직한 성능은 "
              "LOGO 개체 R²=0.85 / 집계 0.94)")
    print("=" * 62)
    print("[실전] '온실 4' 자리에 새 주차 데이터(개체별 직전 조사 포함)를 넣으면 "
          "개체·온실 예측이 나옵니다.")


if __name__ == "__main__":
    main()
