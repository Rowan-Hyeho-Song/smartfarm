# -*- coding: utf-8 -*-
"""
train_agg.py  (v2.1 모델)
=========================
[이 파일이 하는 일 — 한 문장]
'개체(표본) 하나하나'가 아니라 '온실-날짜 평균 생육'을 예측하는 v2 모델을 만듭니다.
여기에 lag(전주 생육)·전주 개체 분포 feature와 Optuna 튜닝을 더해 착과수 R²를 0.89로 올립니다.

[왜 '평균'을 예측하나 — v1의 한계 돌파]
v1은 개체 하나하나를 예측했는데, 개체는 식물마다 랜덤 편차가 커서 R²가 막힙니다.
그런데 우리가 실제로 제어하는 건 개체가 아니라 '온실 전체'입니다.
그래서 '온실-날짜 평균 착과수'를 예측하면 개체 노이즈가 사라지고 정확도가 크게 오릅니다.
(측정: 개체 R² 0.57~0.64  →  평균 예측 R² 0.78  →  +lag 0.84  →  +전주분포 0.89)

[v2.1에서 무엇이 바뀌었나 — '정규화 말고 살리기']
평균으로 뭉갤 때 사라지던 '전주 개체 분포'(표준편차·최소·최대·중앙값)를 feature로 되살립니다.
개체를 통째로 살리면 노이즈(개체 편차 62%)만 늘어 오히려 나빠지지만(실측),
'분포 모양'만 골라 살리면 순수 이득이 됩니다. (착과수 LOGO R² 0.87 → 0.89)

[검증]
착과수는 '온실별 교차검증(LOGO)' — 온실 하나씩 빼고 나머지로 학습해 정직하게 평가.

실행:  python src/train_agg.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import r2_score, mean_absolute_error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"
MODEL_DIR = PROJECT_DIR / "models"

TARGET = "착과수"
GROWTH = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
KEYS = ["온실번호", "조사일자", "측정라인", "표본번호"]
GREENHOUSES = [1, 2, 3, 4]
N_TRIALS = 40
SEED = 42


def load_aggregated():
    """학습표를 '온실-날짜 평균'으로 집계한다 (개체 노이즈 제거)."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    # 온실·조사일자별 평균. (환경·lag는 원래 같은 값, 생육은 평균이 됨)
    agg = df.groupby(["온실번호", "조사일자"]).mean(numeric_only=True).reset_index()
    # 입력(X): 키·생육을 뺀 나머지 = 환경요약 + 시점 + lag(전주 생육) + 전주 분포
    #  단, '개체lag'(_개체전주)는 v3(개체 모델) 전용이라 v2.1(집계)에선 제외
    #  (집계하면 기존 그룹 lag와 거의 중복 → 넣으면 noise만 늘어남)
    feat = ["온실번호"] + [c for c in agg.columns
                          if c not in KEYS + GROWTH and "개체" not in c]
    return agg, feat


def logo_oof_r2(params, agg, feat, target):
    """온실 하나씩 빼며 검증한 예측을 모아 R²(정직한 일반화 성능)."""
    preds, actual = [], []
    for g in GREENHOUSES:
        tr = agg[agg["온실번호"] != g]
        va = agg[agg["온실번호"] == g]
        m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
        m.fit(tr[feat], tr[target])
        preds.append(m.predict(va[feat]))
        actual.append(va[target].to_numpy())
    return r2_score(np.concatenate(actual), np.concatenate(preds))


def objective(trial, agg, feat):
    params = {
        "objective": "regression",
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    }
    return logo_oof_r2(params, agg, feat, TARGET)


def main():
    print("=" * 60)
    print("v2.1 모델 — 온실-날짜 평균 + lag + 전주분포 + Optuna  (대상: 착과수)")
    print("=" * 60)
    agg, feat = load_aggregated()
    print(f"[데이터] 집계 {len(agg)}행 (온실×조사일) · 입력 feature {len(feat)}개")
    lag_n = len([c for c in feat if "전주" in c and "분포" not in c])
    dist_n = len([c for c in feat if "분포" in c])
    print(f"         그중 lag(전주 생육) {lag_n}개 · 전주 개체 분포 {dist_n}개\n")

    # 비교 기준: 기본 설정 (튜닝 전)
    base = dict(objective="regression", n_estimators=400, learning_rate=0.05,
                num_leaves=15, min_child_samples=5, subsample=0.9, colsample_bytree=0.8)
    base_r2 = logo_oof_r2(base, agg, feat, TARGET)
    print(f"[튜닝 전] LOGO R² = {base_r2:.4f}")

    # Optuna 탐색
    print(f"[탐색] {N_TRIALS}회...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, agg, feat), n_trials=N_TRIALS)
    best_r2 = study.best_value
    mark = "✅ 80% 돌파!" if best_r2 >= 0.80 else "(80% 미달)"
    print(f"[튜닝 후] LOGO R² = {best_r2:.4f}   {mark}")
    print(f"          (튜닝 전 {base_r2:.4f} 대비 {best_r2-base_r2:+.4f})")

    # 최적 설정으로 전체 데이터 학습 → 저장 (예측용)
    best = dict(objective="regression", **study.best_params)
    final = lgb.LGBMRegressor(**best, random_state=SEED, verbose=-1)
    final.fit(agg[feat], agg[TARGET])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "lgbm_fruit_count_v2.txt"
    final.booster_.save_model(str(out))

    # 참고: 초장도 같은 방식으로
    r2_ho = logo_oof_r2(base, agg, feat, "초장")

    print("-" * 60)
    print(f"[저장] v2 모델 → {out}")
    print(f"[참고] 초장 v2 LOGO R² = {r2_ho:.4f} (같은 평균+lag 방식)")
    print("-" * 60)
    print("[중요 feature TOP 8]")
    imp = pd.DataFrame({"feature": feat, "중요도": final.feature_importances_})
    imp = imp.sort_values("중요도", ascending=False).head(8)
    print(imp.to_string(index=False))
    print("=" * 60)
    print("[요약] 개체 예측(v1) → 온실-날짜 평균 예측(v2) + lag + 튜닝으로 정확도 향상.")


if __name__ == "__main__":
    main()
