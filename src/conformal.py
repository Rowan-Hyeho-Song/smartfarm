# -*- coding: utf-8 -*-
"""
conformal.py
============
[이 파일이 하는 일 — 한 문장]
분위수 회귀로 만든 '예측 구간(P10~P90)'이 실제로는 목표(80%)만큼 안 맞았는데,
이를 'Conformal 보정(CQR)'으로 교정해 적중률을 80%에 맞춥니다.

[문제 복습]
train_advanced.py에서 P10~P90 구간의 '적중률(실제값이 구간 안에 든 비율)'이 52~58%였습니다.
목표는 80%인데 구간이 너무 좁았던 것입니다. → 의사결정에 쓰기엔 과신(overconfident).

[해결 아이디어 — CQR (Conformalized Quantile Regression)]
1) 학습에 안 쓴 '보정용(calibration) 데이터'를 따로 둡니다.
2) 그 데이터에서 '내 구간이 실제로 얼마나 빗나갔는지'(conformity score)를 잽니다.
3) 그 빗나간 양의 상위값(Q)만큼 구간을 좌우로 넓혀 줍니다.
   → 결과적으로 "약속한 80%가 실제로 지켜지는" 구간이 됩니다.

[우리 데이터에 맞춘 방식 — 온실 그룹 분할]
온실 4개 중: 1개=검증(test), 1개=보정(calibration), 2개=학습(train) 으로 나눠
모든 조합을 돌려 결과를 모읍니다. (같은 온실이 학습과 검증에 겹치지 않게 = 누수 차단)

실행:  python src/conformal.py
"""

import sys
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"

TARGETS = ["착과수", "초장"]
GREENHOUSES = [1, 2, 3, 4]
ALPHA = 0.20            # 1-ALPHA = 0.80 → 목표 적중률 80%
LO, HI = ALPHA / 2, 1 - ALPHA / 2   # 하위 10% ~ 상위 90% (P10~P90)
SEED = 42


def load():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    keys = ["온실번호", "조사일자", "측정라인", "표본번호"]
    growth = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
    feat = ["온실번호"] + [c for c in df.columns if c not in keys + growth]
    return df, feat


def quantile_model(alpha):
    """지정한 분위수(alpha)를 예측하는 LightGBM 모델."""
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )


def conformal_Q(scores, alpha):
    """
    보정량 Q 계산: conformity score들을 정렬해, 유한표본 보정된 (1-alpha) 분위수를 고름.
    이 Q만큼 구간을 양쪽으로 넓히면 목표 적중률이 보장됩니다.
    """
    scores = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))   # 유한표본 보정: (n+1) 사용
    if k > n:
        return np.inf                          # 보정 데이터가 너무 적으면 구간을 매우 넓게
    return scores[k - 1]


def evaluate(df, feat, target):
    """
    온실 그룹 분할로 '보정 전'과 '보정 후(CQR)' 예측구간의 적중률·평균폭을 비교.
    (test, calib)를 서로 다른 온실로 두는 모든 조합을 돌려 결과를 모읍니다.
    """
    cov_raw, cov_cqr, w_raw, w_cqr = [], [], [], []

    # 온실 4개에서 (test, calib) 순서쌍을 뽑고, 나머지 2개를 train으로.
    for test_g, calib_g in permutations(GREENHOUSES, 2):
        train_gs = [g for g in GREENHOUSES if g not in (test_g, calib_g)]

        tr = df[df["온실번호"].isin(train_gs)]
        cal = df[df["온실번호"] == calib_g]
        te = df[df["온실번호"] == test_g]

        # 1) 학습 데이터로 하한(P10)·상한(P90) 분위수 모델 학습
        m_lo = quantile_model(LO).fit(tr[feat], tr[target])
        m_hi = quantile_model(HI).fit(tr[feat], tr[target])

        # 2) 보정용 데이터에서 '구간이 빗나간 양' = conformity score
        #    E = max(하한 - 실제, 실제 - 상한). (구간 안에 잘 들면 음수)
        cal_lo, cal_hi = m_lo.predict(cal[feat]), m_hi.predict(cal[feat])
        y_cal = cal[target].to_numpy()
        scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
        Q = conformal_Q(scores, ALPHA)

        # 3) 검증 온실에 적용 — 보정 전[lo,hi] vs 보정 후[lo-Q, hi+Q]
        te_lo, te_hi = m_lo.predict(te[feat]), m_hi.predict(te[feat])
        y_te = te[target].to_numpy()

        # 보정 전
        lo0, hi0 = np.minimum(te_lo, te_hi), np.maximum(te_lo, te_hi)
        cov_raw.append(np.mean((y_te >= lo0) & (y_te <= hi0)))
        w_raw.append(np.mean(hi0 - lo0))

        # 보정 후 (CQR)
        lo1, hi1 = te_lo - Q, te_hi + Q
        cov_cqr.append(np.mean((y_te >= lo1) & (y_te <= hi1)))
        w_cqr.append(np.mean(hi1 - lo1))

    return (np.mean(cov_raw), np.mean(w_raw)), (np.mean(cov_cqr), np.mean(w_cqr))


def main():
    print("=" * 60)
    print(f"Conformal 보정(CQR) — 목표 적중률 {int((1-ALPHA)*100)}%")
    print("=" * 60)
    df, feat = load()

    for target in TARGETS:
        (cov0, w0), (cov1, w1) = evaluate(df, feat, target)
        print(f"\n■ {target}")
        print(f"  보정 전(순수 분위수)  적중률 {cov0*100:5.1f}%   평균폭 {w0:5.2f}")
        print(f"  보정 후(CQR)         적중률 {cov1*100:5.1f}%   평균폭 {w1:5.2f}")
        gap0, gap1 = abs(cov0 - (1 - ALPHA)), abs(cov1 - (1 - ALPHA))
        better = "개선됨 [OK]" if gap1 < gap0 else "차이 미미"
        print(f"  → 목표 80%에 {'더 가까워짐' if gap1 < gap0 else '비슷'} ({better})")

    print("\n" + "=" * 60)
    print("[해석] 보정 후 적중률이 80%에 가까우면, 예측 구간을 '신뢰할 수 있는")
    print(" 불확실성'으로 의사결정에 쓸 수 있습니다. (구간이 넓어지는 건 정상 — 정직해진 것)")


if __name__ == "__main__":
    main()
