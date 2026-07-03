# -*- coding: utf-8 -*-
"""
train_optuna.py
===============
[이 파일이 하는 일 — 한 문장]
Optuna로 LightGBM의 '하이퍼파라미터'(모델의 설정값들)를 자동 탐색해,
착과수 예측의 R²(정확도)를 baseline보다 끌어올립니다.

[하이퍼파라미터 튜닝이란?]
모델에는 사람이 정해줘야 하는 설정값이 많습니다(트리 깊이, 학습률, 잎 개수 등).
이 값을 잘 고르면 성능이 오릅니다. Optuna는 이 조합을 '똑똑하게 여러 번 시도'해
가장 좋은 조합을 찾아 줍니다. (수동으로 하나씩 바꾸는 것보다 빠르고 체계적)

[중요 — 무엇을 기준으로 '좋다'고 판단하나]
착과수는 '온실별 검증(LOGO)'이 맞는 타깃이므로,
튜닝의 점수도 LOGO 교차검증 R²로 매깁니다. (검증방식과 튜닝기준을 일치)

[R² vs 적중률 메모]
- 여기서 올리는 건 R²(예측값의 정확도).
- 예측 '구간'의 적중률(80% 목표)은 conformal.py가 담당. 둘은 다른 지표입니다.
  단, 모델이 좋아지면 같은 적중률을 '더 좁은 구간'으로 달성해 구간이 유용해집니다.

실행:  python src/train_optuna.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import r2_score, mean_squared_error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)  # 탐색 로그 최소화

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"
MODEL_DIR = PROJECT_DIR / "models"

TARGET = "착과수"
TARGET_SLUG = "fruit_count"
GREENHOUSES = [1, 2, 3, 4]
N_TRIALS = 60          # Optuna가 시도할 횟수 (많을수록 촘촘하지만 느림)
SEED = 42


def load():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    keys = ["온실번호", "조사일자", "측정라인", "표본번호"]
    growth = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
    feat = ["온실번호"] + [c for c in df.columns if c not in keys + growth]
    return df, feat


def logo_oof_r2(params, df, feat, target):
    """
    주어진 설정값(params)으로 LOGO 교차검증을 돌려, 모든 온실의 예측을 모아 R²를 계산.
    (온실 하나씩 빼고 학습→예측 → 4번 → 예측 전부 합쳐 한 번에 채점)
    """
    preds, actual = [], []
    for g in GREENHOUSES:
        tr = df[df["온실번호"] != g]
        va = df[df["온실번호"] == g]
        model = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
        model.fit(tr[feat], tr[target])
        preds.append(model.predict(va[feat]))
        actual.append(va[target].to_numpy())
    return r2_score(np.concatenate(actual), np.concatenate(preds))


def objective(trial, df, feat):
    """Optuna가 매 시도마다 부르는 함수. 설정값을 제안받아 → R²를 돌려줌(높을수록 좋음)."""
    params = {
        "objective": "regression",
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    return logo_oof_r2(params, df, feat, TARGET)


def main():
    print("=" * 60)
    print(f"Optuna 하이퍼파라미터 튜닝 — 대상: {TARGET} (기준: LOGO R²)")
    print("=" * 60)
    df, feat = load()

    # 비교용 baseline: train_advanced.py에서 쓰던 기본 설정.
    base_params = dict(objective="regression", n_estimators=400, learning_rate=0.05,
                       num_leaves=15, min_child_samples=10, subsample=0.8, colsample_bytree=0.8)
    base_r2 = logo_oof_r2(base_params, df, feat, TARGET)
    print(f"[baseline] LOGO R² = {base_r2:.4f}\n")

    # Optuna 탐색 시작.
    print(f"[탐색] {N_TRIALS}회 시도 중...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, df, feat), n_trials=N_TRIALS, show_progress_bar=False)

    best_r2 = study.best_value
    print(f"\n[결과] 최적 LOGO R² = {best_r2:.4f}  (baseline {base_r2:.4f} 대비 "
          f"{'+' if best_r2>=base_r2 else ''}{best_r2-base_r2:.4f})")
    print("[최적 설정값]")
    for k, v in study.best_params.items():
        print(f"  {k:<18}= {round(v, 4) if isinstance(v, float) else v}")

    # 최적 설정으로 '온실4 검증' 모델을 학습해 저장 (predict.py와 동일한 홀드아웃).
    best = dict(objective="regression", **study.best_params)
    tr = df[df["온실번호"] != 4]
    final = lgb.LGBMRegressor(**best, random_state=SEED, verbose=-1)
    final.fit(tr[feat], tr[TARGET])
    va = df[df["온실번호"] == 4]
    holdout_r2 = r2_score(va[TARGET], final.predict(va[feat]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"lgbm_{TARGET_SLUG}_tuned.txt"
    final.booster_.save_model(str(out))
    print("-" * 60)
    print(f"[저장] 튜닝 모델 → {out}")
    print(f"       (온실4 홀드아웃 R² = {holdout_r2:.4f})")
    print("=" * 60)
    print("[다음] predict.py의 MODEL_PATH를 이 _tuned 모델로 바꾸면 그대로 사용됩니다.")


if __name__ == "__main__":
    main()
