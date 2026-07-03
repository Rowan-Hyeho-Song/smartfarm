# -*- coding: utf-8 -*-
"""
train.py
========
[이 파일이 하는 일 — 한 문장]
data_prep.py가 만든 학습표(data/processed/train_table.csv)를 읽어,
"직전 7일의 환경(X) → 그 날 딸기 생육(Y)"을 예측하는
LightGBM 베이스라인 모델을 학습하고 성능을 평가합니다.

[베이스라인이란?]
가장 먼저 만드는 '기준점' 모델입니다. 화려하지 않아도 됩니다.
이 점수를 기준으로 "앞으로 개선한 모델이 정말 더 나은가"를 비교합니다.

[LightGBM이란?]
표(정형) 데이터에서 가장 잘 먹히는 대표적인 모델(그래디언트 부스팅 트리).
숫자 컬럼 여러 개로 하나의 값을 예측하는 데 강하고, 빠르며, 결측값(NaN)도 알아서 처리합니다.

[초보자를 위한 읽는 법]
- 위에서 아래로 흐릅니다. 각 단계는 def(함수)로 나뉘고 맨 아래 main()이 순서대로 부릅니다.
- 실행:  python src/train.py
- 예측 대상을 바꾸려면 아래 TARGET 값만 고치세요 (예: "착과수" → "초장").
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 윈도우 콘솔(cp949)에서도 한글/기호가 깨지거나 오류나지 않도록 출력 인코딩을 UTF-8로 고정.
#   (콘솔에서 'chcp 65001'까지 해두면 화면에도 또렷하게 보입니다)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1) 설정값 — 여기 숫자/이름만 바꾸면 실험을 조정할 수 있습니다.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"
MODEL_DIR = PROJECT_DIR / "models"          # 학습된 모델을 저장할 곳

# ★ 예측하고 싶은 생육 지표. 바꿔서 다른 걸 예측해 볼 수 있습니다.
#    후보: 착과수 / 초장 / 엽수 / 엽장 / 엽폭 / 엽병장 / 관부직경
TARGET = "착과수"

# ★ 검증(성능 평가) 방식. 타깃 성격에 맞게 고릅니다.
#    - "time"       : 뒤쪽 N주를 '미래'로 떼어 검증. '과거로 미래를 예측'하는 실전과 같음.
#                     → 초장·엽폭처럼 '환경 따라 서서히 변하는' 크기 지표에 적합.
#    - "greenhouse" : 온실 하나를 통째로 빼서 검증(나머지 온실로 학습).
#                     → 착과수처럼 '재배 주차(발달단계)'에 크게 좌우되는 값에 적합.
#                       (모든 주차가 학습에 들어가야 발달단계 feature가 힘을 씀)
VALIDATION = "greenhouse"

# "greenhouse" 방식일 때, 검증용으로 뺄 온실 번호.
HOLDOUT_GREENHOUSE = 4

# 모델 파일 이름에 쓸 영문 슬러그.
#   (LightGBM은 윈도우에서 '한글 파일 경로'로 저장하면 오류가 나므로, 파일명만 영문으로.)
TARGET_SLUG = {
    "착과수": "fruit_count",
    "초장": "plant_height",
    "엽수": "leaf_count",
    "엽장": "leaf_length",
    "엽폭": "leaf_width",
    "엽병장": "petiole_length",
    "관부직경": "crown_diameter",
}

# 검증(validation)용으로 떼어 둘 '마지막 조사일 수'.
#   총 15개 조사일 중 뒤쪽 며칠을 '아직 못 본 미래'로 취급해 성능을 확인합니다.
N_VALID_DATES = 4

# 재현성을 위한 난수 고정값. (같은 seed면 매번 같은 결과)
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2) 학습표 불러오기
# ─────────────────────────────────────────────────────────────────────────────
def load_table() -> pd.DataFrame:
    """train_table.csv를 읽어 옵니다. (utf-8-sig = BOM 있어도/없어도 안전)"""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 조사일자를 진짜 날짜형으로 (뒤에서 날짜 기준으로 나누기 위해).
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    print(f"[로드] 학습표 {len(df):,}행 × {df.shape[1]}열")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3) 입력(X)과 정답(Y) 열을 고르기
# ─────────────────────────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame):
    """
    어떤 열을 '입력(X)'으로 쓰고, 어떤 열이 '정답(Y)'인지 정합니다.

    [가장 중요한 원칙 — 정답 힌트를 입력에 넣지 않기]
    생육 열들(초장·엽수·착과수 등)은 '예측 대상'입니다.
    이걸 입력에 같이 넣으면 모델이 '정답을 보고 정답을 맞히는' 반칙이 됩니다.
    그래서 입력에는 '환경 요약 열'만 쓰고, 생육 열은 전부 제외합니다.
    (예외: 온실번호는 '이 온실은 원래 이런 경향' 정보라 입력에 포함합니다.)
    """
    # 표를 구분하는 키 열들 (모델 입력이 아님).
    key_cols = ["온실번호", "조사일자", "측정라인", "표본번호"]

    # 생육 열 = 예측 대상 후보들. 전부 입력에서 뺍니다.
    growth_cols = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]

    # 입력(X) = (키도 아니고 생육도 아닌) 나머지 = 환경 요약 열들.
    feature_cols = [c for c in df.columns if c not in key_cols and c not in growth_cols]

    # 온실 고유 경향을 반영하도록 온실번호를 입력에 다시 추가.
    feature_cols = ["온실번호"] + feature_cols

    print(f"[열 선택] 예측 대상 Y = '{TARGET}'")
    print(f"[열 선택] 입력 X = {len(feature_cols)}개 열 (환경요약 + 온실번호)")
    return feature_cols, TARGET


# ─────────────────────────────────────────────────────────────────────────────
# 4) 학습용 / 검증용으로 '시간 순서대로' 나누기
# ─────────────────────────────────────────────────────────────────────────────
def make_split(df: pd.DataFrame):
    """
    데이터를 학습(train)과 검증(valid)으로 나눕니다.
    위의 VALIDATION 설정에 따라 두 방식 중 하나를 씁니다.

    [공통 원칙 — 미래/정답 누수 방지]
    학습과 검증이 섞이면 '미리 본' 셈이 되어 점수가 부풀려집니다.
    그래서 두 방식 모두 '깨끗하게 분리되는 경계'로 나눕니다.
    """
    if VALIDATION == "time":
        # [시간 분할] 앞쪽 날짜로 학습 → 뒤쪽 N주로 검증 (과거→미래 예측).
        all_dates = sorted(df["조사일자"].unique())
        cutoff = all_dates[-N_VALID_DATES:][0]  # 이 날짜부터가 검증 구간
        train_df = df[df["조사일자"] < cutoff]
        valid_df = df[df["조사일자"] >= cutoff]
        print(f"[분할] 방식=time · 학습 {len(train_df)}행 (~{pd.Timestamp(cutoff).date()} 이전) / "
              f"검증 {len(valid_df)}행 ({pd.Timestamp(cutoff).date()} 이후)")
    else:
        # [온실 분할] 온실 하나를 통째로 빼서 검증 → 모든 조사주차가 학습에 포함됨.
        #   착과수처럼 '발달단계(주차)'가 중요한 값에 적합.
        train_df = df[df["온실번호"] != HOLDOUT_GREENHOUSE]
        valid_df = df[df["온실번호"] == HOLDOUT_GREENHOUSE]
        print(f"[분할] 방식=greenhouse · 학습 {len(train_df)}행 (온실 1~4 중 {HOLDOUT_GREENHOUSE}번 제외) / "
              f"검증 {len(valid_df)}행 (온실 {HOLDOUT_GREENHOUSE}번)")

    return train_df, valid_df


# ─────────────────────────────────────────────────────────────────────────────
# 5) LightGBM 모델 학습
# ─────────────────────────────────────────────────────────────────────────────
def train_model(train_df, valid_df, feature_cols, target):
    """
    LightGBM 회귀(숫자 값 예측) 모델을 학습합니다.
    검증 데이터로 '조기 종료(early stopping)'를 사용해 과적합을 막습니다.
    """
    # 입력(X)과 정답(y)을 각각 뽑아냅니다.
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_valid, y_valid = valid_df[feature_cols], valid_df[target]

    # 모델 설정.  (베이스라인이라 무난하고 보수적인 값으로)
    model = lgb.LGBMRegressor(
        objective="regression",   # 숫자 값을 예측하는 문제
        n_estimators=500,         # 트리를 최대 500개까지 (조기종료로 자동 조절)
        learning_rate=0.05,       # 한 걸음의 크기 (작을수록 신중하게 학습)
        num_leaves=15,            # 트리 복잡도. 데이터가 작으니 작게(과적합 방지)
        min_child_samples=10,     # 잎 하나에 최소 몇 개 샘플이 있어야 하는지
        subsample=0.8,            # 매 트리마다 데이터 80%만 사용 (일반화에 도움)
        colsample_bytree=0.8,     # 매 트리마다 열 80%만 사용
        random_state=SEED,
        verbose=-1,               # 학습 중 로그 최소화
    )

    # 실제 학습. eval_set에 검증데이터를 주면 성능이 안 늘 때 알아서 멈춥니다.
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),  # 30번 개선 없으면 중단
        ],
    )
    print(f"[학습] 완료 · 실제 사용한 트리 수 = {model.best_iteration_}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6) 성능 평가
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, valid_df, feature_cols, target):
    """
    검증 데이터로 예측해 보고, 세 가지 지표로 성능을 봅니다.

    - RMSE: 평균적으로 몇 '개/㎝'나 틀리는지 (작을수록 좋음). 큰 오차에 민감.
    - MAE : 평균 절대 오차 (작을수록 좋음). 해석이 직관적.
    - R²  : 1에 가까울수록 좋음. 0이면 '그냥 평균값 찍기' 수준, 음수면 그보다 못함.
    """
    X_valid, y_valid = valid_df[feature_cols], valid_df[target]
    pred = model.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    mae = mean_absolute_error(y_valid, pred)
    r2 = r2_score(y_valid, pred)

    print("-" * 60)
    print(f"[검증 성능]  (예측 대상: {target})")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  R2   = {r2:.3f}")

    # 비교 기준: '무조건 학습데이터 평균으로 찍었을 때'의 오차.
    #   우리 모델이 이보다 나아야 의미가 있습니다.
    baseline_pred = np.full(len(y_valid), y_valid.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_valid, baseline_pred))
    verdict = "우리 모델이 더 좋음 [OK]" if rmse < baseline_rmse else "평균 찍기보다 못함 [주의]"
    print(f"  (참고) 평균값만 찍었을 때 RMSE = {baseline_rmse:.3f}  ->  {verdict}")
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 7) 어떤 환경 지표가 예측에 중요했나 (feature importance)
# ─────────────────────────────────────────────────────────────────────────────
def show_importance(model, feature_cols, top_n=10):
    """
    모델이 예측할 때 어떤 입력 열을 많이 참고했는지 순위를 보여 줍니다.
    → "착과수는 어떤 환경 지표와 관련이 큰가"를 사람이 해석하는 데 도움.
    """
    imp = pd.DataFrame({
        "지표": feature_cols,
        "중요도": model.feature_importances_,
    }).sort_values("중요도", ascending=False)

    print("-" * 60)
    print(f"[중요 지표 TOP {top_n}]")
    print(imp.head(top_n).to_string(index=False))
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# 8) 전체 순서 실행
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"LightGBM 베이스라인 학습 시작  (예측: {TARGET})")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # (1) 학습표 로드
    df = load_table()

    # 예측 대상에 값이 비어 있는 행은 학습에서 제외 (정답이 없으면 못 배움).
    df = df.dropna(subset=[TARGET])

    # (2) 입력/정답 열 정하기 → (3) 시간 순 분할
    feature_cols, target = select_features(df)
    train_df, valid_df = make_split(df)

    # (4) 학습 → (5) 평가 → (6) 중요도
    model = train_model(train_df, valid_df, feature_cols, target)
    evaluate(model, valid_df, feature_cols, target)
    show_importance(model, feature_cols)

    # (7) 모델 저장 (나중에 predict.py에서 불러 쓸 수 있게).
    #     파일명은 영문 슬러그 사용 (윈도우 + LightGBM 한글경로 오류 회피).
    slug = TARGET_SLUG.get(target, "target")
    model_path = MODEL_DIR / f"lgbm_{slug}.txt"
    model.booster_.save_model(str(model_path))
    print("-" * 60)
    print(f"[저장] 모델 → {model_path}")
    print("=" * 60)
    print("[다음 단계 힌트]")
    print(" - TARGET을 '초장' 등으로 바꿔 다른 지표도 학습해 보세요.")
    print(" - 성능이 아쉬우면: 창(WINDOW_DAYS) 조절, feature 추가, Optuna로 하이퍼파라미터 탐색.")


if __name__ == "__main__":
    main()
