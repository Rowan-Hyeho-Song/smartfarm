# -*- coding: utf-8 -*-
"""
predict.py
==========
[이 파일이 하는 일 — 한 문장]
튜닝해 저장해 둔 모델을 불러와 새 데이터의 생육을 예측하고,
'예측값 + 신뢰 구간(하한~상한)'을 제출용 CSV로 저장합니다.

[이번 버전에서 바뀐 점]
1) 기본 모델 → Optuna로 튜닝한 모델(lgbm_*_tuned.txt)을 사용.
2) 값 하나뿐 아니라 '예측 구간'을 함께 출력.
3) 구간을 최대한 '좁게' — 튜닝된 분위수 모델 + Conformal 보정 + (착과수는 음수 불가 → 하한 0).

[학습(train_*)과의 차이]
- train_* : 모델을 '만든다'(학습). predict : 만든 모델을 '쓴다'(추론, 제출).

[시연 방식]
저장된 착과수 모델(온실 1~3 학습)로 온실 4를 '처음 보는 데이터'처럼 예측합니다.
실전에선 온실 4 자리에 '진짜 새 데이터'를 넣으면 됩니다.

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

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent

TARGET = "착과수"
TARGET_SLUG = "fruit_count"

# ★ 튜닝 모델을 사용 (train_optuna.py가 저장한 것). 없으면 기본 모델로 자동 대체.
MODEL_PATH = PROJECT_DIR / "models" / f"lgbm_{TARGET_SLUG}_tuned.txt"
FALLBACK_MODEL = PROJECT_DIR / "models" / f"lgbm_{TARGET_SLUG}.txt"

FEATURE_TABLE = PROJECT_DIR / "data" / "processed" / "train_table.csv"

TRAIN_GREENHOUSES = [1, 2, 3]   # 학습에 쓴 온실
CALIB_GREENHOUSE = 3            # 구간 보정용으로 뺄 온실 (split-conformal)
PREDICT_GREENHOUSE = 4          # '새 데이터'로 예측할 온실
OUT_PATH = PROJECT_DIR / "data" / "processed" / f"predictions_{TARGET_SLUG}.csv"

# 예측 구간 설정: 목표 적중률 80% (=상하위 10% 지점)
ALPHA = 0.20
LO, HI = ALPHA / 2, 1 - ALPHA / 2

# Optuna로 찾은 최적 설정값 — 분위수(구간) 모델에도 그대로 적용해 구간을 더 좁게.
BEST_PARAMS = dict(
    n_estimators=573, learning_rate=0.1044, num_leaves=44, max_depth=7,
    min_child_samples=26, subsample=0.8609, colsample_bytree=0.6834,
    reg_alpha=0.001, reg_lambda=3.9446,
)
CLIP_MIN_ZERO = True   # 착과수는 음수가 될 수 없으므로 하한을 0으로 (구간이 더 좁고 현실적)


# ─────────────────────────────────────────────────────────────────────────────
def quantile_model(alpha):
    """튜닝된 설정으로 특정 분위수(alpha)를 예측하는 모델."""
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                             random_state=42, verbose=-1, **BEST_PARAMS)


def conformal_Q(scores, alpha):
    """보정량 Q = conformity score의 유한표본 (1-alpha) 분위수. 이만큼 구간을 넓혀 적중률을 맞춤."""
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def main():
    print("=" * 60)
    print(f"예측(추론) + 신뢰 구간 — 대상: {TARGET}")
    print("=" * 60)

    # (1) 튜닝 모델 불러오기 (없으면 기본 모델).
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    if not model_path.exists():
        print(f"[오류] 모델 파일이 없습니다. 먼저 train_optuna.py(또는 train.py)로 학습하세요.")
        return
    booster = lgb.Booster(model_file=str(model_path))
    feat = booster.feature_name()          # 학습 때와 같은 feature를 그대로 사용
    print(f"[모델] {model_path.name} · feature {len(feat)}개")

    # (2) 데이터 불러오기 + 학습/예측 온실 분리.
    df = pd.read_csv(FEATURE_TABLE, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    train_df = df[df["온실번호"].isin(TRAIN_GREENHOUSES)]
    new_df = df[df["온실번호"] == PREDICT_GREENHOUSE].copy()
    print(f"[데이터] 학습 온실 {TRAIN_GREENHOUSES} · 예측 온실 {PREDICT_GREENHOUSE}({len(new_df)}행)")

    # (3) 점 예측 — 저장된 튜닝 모델 사용.
    new_df["예측값"] = booster.predict(new_df[feat])

    # (4) 예측 구간 — 분위수 모델 + Conformal 보정 (정석 split-conformal).
    #     핵심: 'Q를 잰 모델'과 '예측에 쓰는 모델'이 같아야 적중률이 정확히 맞습니다.
    #     그래서 보정용 온실 1개(CALIB)를 학습에서 빼고, 나머지로 분위수 모델을 학습한 뒤,
    #     그 '같은 모델'을 보정과 예측에 모두 사용합니다. (앞서 재학습하면 구간이 과하게 넓어짐)
    proper_gs = [g for g in TRAIN_GREENHOUSES if g != CALIB_GREENHOUSE]
    proper = train_df[train_df["온실번호"].isin(proper_gs)]   # 분위수 모델 학습용
    calib = train_df[train_df["온실번호"] == CALIB_GREENHOUSE]  # 보정용

    m_lo = quantile_model(LO).fit(proper[feat], proper[TARGET])
    m_hi = quantile_model(HI).fit(proper[feat], proper[TARGET])

    # 보정용 온실에서 '구간이 빗나간 양'(conformity score)을 재서 보정량 Q 계산.
    y_cal = calib[TARGET].to_numpy()
    scores = np.maximum(m_lo.predict(calib[feat]) - y_cal, y_cal - m_hi.predict(calib[feat]))
    Q = conformal_Q(scores, ALPHA)

    # 같은 모델로 예측 온실에 구간 적용.
    lower = m_lo.predict(new_df[feat]) - Q      # 하한 = P10 - Q
    upper = m_hi.predict(new_df[feat]) + Q      # 상한 = P90 + Q
    if CLIP_MIN_ZERO:
        lower = np.maximum(lower, 0.0)          # 착과수는 0 미만 불가 → 구간이 더 좁아짐
    new_df["예측하한"] = lower
    new_df["예측상한"] = upper

    # (5) 저장.
    keep = ["온실번호", "조사일자", "측정라인", "표본번호", "예측하한", "예측값", "예측상한"]
    if TARGET in new_df.columns:
        keep.append(TARGET)
    result = new_df[keep].sort_values(["온실번호", "조사일자", "측정라인", "표본번호"])
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장] 예측+구간 → {OUT_PATH}")

    # (6) 미리보기 + 정확도/구간 품질.
    print("-" * 60)
    print("미리보기(앞 6행):")
    print(result.head(6).to_string(index=False))

    if TARGET in new_df.columns:
        y = new_df[TARGET].to_numpy(); p = new_df["예측값"].to_numpy()
        rmse = np.sqrt(np.mean((y - p) ** 2))
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        cover = np.mean((y >= new_df["예측하한"].to_numpy()) & (y <= new_df["예측상한"].to_numpy()))
        width = np.mean(new_df["예측상한"].to_numpy() - new_df["예측하한"].to_numpy())
        print("-" * 60)
        print(f"[점 예측]  RMSE={rmse:.3f} · R²={r2:.3f}")
        print(f"[구간]     적중률={cover*100:.0f}% (목표 {int((1-ALPHA)*100)}%) · 평균폭={width:.2f}")

    print("=" * 60)
    print("[실전 사용법] '온실 4' 자리에 새 데이터를 넣으면 예측값+구간이 그대로 나옵니다.")


if __name__ == "__main__":
    main()
