# -*- coding: utf-8 -*-
"""
predict.py  (v2 경로)
=====================
[이 파일이 하는 일 — 한 문장]
v2 모델(온실-날짜 평균 + lag)을 불러와 새 데이터의 착과수를 예측하고,
'예측값 + 신뢰 구간(하한~상한)'을 제출용 CSV로 저장합니다.

[v1 → v2 무엇이 바뀌었나]
- 예측 단위: 개체(표본) → '온실-날짜 평균' (개체 노이즈 제거 → 정확도 R² 0.87)
- 입력에 lag(전주 생육) 포함. 모델은 lgbm_fruit_count_v2.txt.

[정직한 안내]
이 데모는 '저장된 v2 모델(전체 학습)'로 온실 4를 예측합니다. 실전에선 온실 4 자리에
'진짜 새 데이터'가 들어갑니다. 정직한 일반화 성능은 train_agg.py의 LOGO R²=0.87 입니다.

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

TARGET = "착과수"
MODEL_PATH = PROJECT_DIR / "models" / "lgbm_fruit_count_v2.txt"     # ★ v2 모델
FALLBACK = PROJECT_DIR / "models" / "lgbm_fruit_count_tuned.txt"    # 없으면 v1 튜닝

GROWTH = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
KEYS = ["온실번호", "조사일자", "측정라인", "표본번호"]

TRAIN_GREENHOUSES = [1, 2, 3]   # 학습에 쓴 온실
CALIB_GREENHOUSE = 3            # 구간 보정용 (split-conformal)
PREDICT_GREENHOUSE = 4          # '새 데이터'로 예측할 온실
OUT_PATH = PROJECT_DIR / "data" / "processed" / "predictions_fruit_count_v2.csv"

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


def load_aggregated():
    """학습표를 '온실-날짜 평균'으로 집계 (v2와 동일)."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    agg = df.groupby(["온실번호", "조사일자"]).mean(numeric_only=True).reset_index()
    return agg


def main():
    print("=" * 60)
    print(f"예측(추론) v2 + 신뢰 구간 — 대상: {TARGET}")
    print("=" * 60)

    # (1) v2 모델 로드
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK
    if not model_path.exists():
        print("[오류] 모델이 없습니다. 먼저 train_agg.py를 실행하세요.")
        return
    booster = lgb.Booster(model_file=str(model_path))
    feat = booster.feature_name()
    print(f"[모델] {model_path.name} · feature {len(feat)}개 (lag 포함)")

    # (2) 집계 데이터 + 예측 온실 분리
    agg = load_aggregated()
    train = agg[agg["온실번호"].isin(TRAIN_GREENHOUSES)]
    new = agg[agg["온실번호"] == PREDICT_GREENHOUSE].copy()
    print(f"[데이터] 집계 {len(agg)}행 · 예측 온실 {PREDICT_GREENHOUSE}({len(new)}주)")

    # (3) 점 예측 (저장된 v2 모델)
    new["예측값"] = booster.predict(new[feat])

    # (4) 예측 구간 — split-conformal (Q를 잰 모델과 예측 모델을 동일화)
    proper = train[train["온실번호"] != CALIB_GREENHOUSE]
    calib = train[train["온실번호"] == CALIB_GREENHOUSE]
    m_lo = quantile_model(LO).fit(proper[feat], proper[TARGET])
    m_hi = quantile_model(HI).fit(proper[feat], proper[TARGET])
    yc = calib[TARGET].to_numpy()
    Q = conformal_Q(np.maximum(m_lo.predict(calib[feat]) - yc, yc - m_hi.predict(calib[feat])), ALPHA)
    new["예측하한"] = np.maximum(m_lo.predict(new[feat]) - Q, 0.0)   # 착과수 ≥ 0
    new["예측상한"] = m_hi.predict(new[feat]) + Q

    # (5) 저장
    keep = ["온실번호", "조사일자", "예측하한", "예측값", "예측상한"]
    if TARGET in new.columns:
        keep.append(TARGET)
    result = new[keep].sort_values(["온실번호", "조사일자"])
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장] 예측+구간 → {OUT_PATH}")

    # (6) 미리보기 + 참고 정확도
    print("-" * 60)
    print("미리보기(앞 6주):")
    prev = result.copy()
    prev["조사일자"] = prev["조사일자"].dt.date
    print(prev.head(6).to_string(index=False))

    if TARGET in new.columns:
        y = new[TARGET].to_numpy(); p = new["예측값"].to_numpy()
        rmse = np.sqrt(np.mean((y - p) ** 2))
        cover = np.mean((y >= new["예측하한"]) & (y <= new["예측상한"]))
        print("-" * 60)
        print(f"[참고] 온실 {PREDICT_GREENHOUSE} RMSE={rmse:.3f} · 구간 적중률={cover*100:.0f}%")
        print("       (저장 모델은 전체 학습이라 낙관적 — 정직한 성능은 LOGO R²=0.87)")
    print("=" * 60)
    print("[실전] '온실 4' 자리에 새 주차 데이터를 넣으면 예측값+구간이 나옵니다.")


if __name__ == "__main__":
    main()
