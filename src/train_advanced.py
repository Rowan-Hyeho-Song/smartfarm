# -*- coding: utf-8 -*-
"""
train_advanced.py
=================
[이 파일이 하는 일 — 한 문장]
baseline(train.py)에서 한 걸음 더 나아가, 세 가지로 모델을 '심화'합니다.
  (1) 앙상블      : LightGBM + CatBoost 두 모델의 예측을 평균해 더 안정적으로.
  (2) 교차검증    : 온실 4개를 번갈아 검증(LOGO 4-fold)해 '운 좋은 한 번'이 아닌 평균 성능을.
  (3) 불확실성    : 값 하나만이 아니라 '이 정도 범위(P10~P90)일 것'까지 예측(분위수 회귀).

[왜 심화가 필요한가]
- 단일 모델·단일 검증은 우연에 흔들립니다. 앙상블+교차검증으로 신뢰도를 올립니다.
- 의사결정에는 '얼마일 것 같다'뿐 아니라 '얼마나 확신하나(범위)'가 중요합니다.

실행:  python src/train_advanced.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"

# 심화 실험을 볼 예측 대상들. (착과수=발달주기 지배, 초장=환경 서서히 반영)
TARGETS = ["착과수", "초장"]

GREENHOUSES = [1, 2, 3, 4]
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 & 입력열 준비
# ─────────────────────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    keys = ["온실번호", "조사일자", "측정라인", "표본번호"]
    growth = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
    # 입력(X) = 환경요약 + 시점 feature (+ 온실번호). 생육/키는 제외.
    feat = ["온실번호"] + [c for c in df.columns if c not in keys + growth]
    return df, feat


# ─────────────────────────────────────────────────────────────────────────────
# 모델 만들기 (매 폴드마다 새로 생성)
# ─────────────────────────────────────────────────────────────────────────────
def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )

def make_catboost():
    return CatBoostRegressor(
        iterations=400, learning_rate=0.05, depth=4,
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )

def make_lgbm_quantile(alpha):
    # objective='quantile' + alpha → 'P{alpha}' 분위수를 예측. (예: alpha=0.1 → 하위10%)
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 온실별 4-fold 교차검증 (LOGO: Leave-One-Greenhouse-Out)
#   각 온실을 한 번씩 '검증용'으로 빼고, 나머지 3개로 학습.
#   모든 온실의 예측을 모아 '전체 OOF(out-of-fold) 성능'을 계산.
# ─────────────────────────────────────────────────────────────────────────────
def logo_predict(model_factory, df, feat, target):
    """LOGO 방식으로 각 온실을 검증했을 때의 (예측값, 실제값)을 순서대로 모아 돌려줍니다."""
    preds, actual = [], []
    for g in GREENHOUSES:
        tr = df[df["온실번호"] != g]
        va = df[df["온실번호"] == g]
        model = model_factory()
        model.fit(tr[feat], tr[target])
        preds.append(np.asarray(model.predict(va[feat])))
        actual.append(va[target].to_numpy())
    return np.concatenate(preds), np.concatenate(actual)


def score(pred, actual):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# 불확실성 구간 (분위수 회귀) — LOGO로 P10/P50/P90 예측 후 '적중률(coverage)' 확인
# ─────────────────────────────────────────────────────────────────────────────
def logo_quantile(df, feat, target):
    """각 온실 검증에서 P10·P50·P90을 예측하고, 실제값이 [P10,P90] 안에 든 비율을 계산."""
    p10s, p90s, actuals = [], [], []
    for g in GREENHOUSES:
        tr = df[df["온실번호"] != g]
        va = df[df["온실번호"] == g]
        lo = make_lgbm_quantile(0.10); lo.fit(tr[feat], tr[target])
        hi = make_lgbm_quantile(0.90); hi.fit(tr[feat], tr[target])
        p10s.append(np.asarray(lo.predict(va[feat])))
        p90s.append(np.asarray(hi.predict(va[feat])))
        actuals.append(va[target].to_numpy())
    p10 = np.concatenate(p10s); p90 = np.concatenate(p90s); a = np.concatenate(actuals)
    # 하한이 상한보다 커지는 드문 경우 정렬로 방지.
    lo = np.minimum(p10, p90); hi = np.maximum(p10, p90)
    coverage = np.mean((a >= lo) & (a <= hi))   # 실제값이 구간 안에 든 비율 (목표: 약 80%)
    avg_width = np.mean(hi - lo)                 # 구간 평균 폭 (좁을수록 확신이 큼)
    return coverage, avg_width


# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 66)
    print("심화 실험 — 앙상블 · 온실별 교차검증 · 불확실성")
    print("=" * 66)
    df, feat = load()
    print(f"[로드] {len(df)}행 · 입력 feature {len(feat)}개\n")

    for target in TARGETS:
        print(f"■ 예측 대상: {target}")
        print(f"  {'모델':<16}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
        print("  " + "-" * 40)

        # 각 모델을 LOGO로 평가
        lgbm_p, a = logo_predict(make_lgbm, df, feat, target)
        cat_p, _ = logo_predict(make_catboost, df, feat, target)
        ens_p = (lgbm_p + cat_p) / 2.0   # 앙상블 = 두 모델 예측 평균

        for name, pred in [("LightGBM", lgbm_p), ("CatBoost", cat_p), ("앙상블(평균)", ens_p)]:
            rmse, mae, r2 = score(pred, a)
            print(f"  {name:<16}{rmse:>8.3f}{mae:>8.3f}{r2:>8.3f}")

        # 불확실성 구간
        cov, width = logo_quantile(df, feat, target)
        print(f"  └ 불확실성 P10~P90: 적중률 {cov*100:.0f}% · 평균폭 {width:.2f} "
              f"(목표 적중률 80%)\n")

    print("=" * 66)
    print("[해석 힌트]")
    print(" - 앙상블 R2가 단일모델보다 같거나 높으면 '안정성 확보'로 봄.")
    print(" - 적중률이 80%에 가까우면 불확실성 구간이 신뢰할 만함.")
    print(" - 착과수: 온실별 검증에서 R2 양(+) → 발달단계 feature가 작동 중.")


if __name__ == "__main__":
    main()
