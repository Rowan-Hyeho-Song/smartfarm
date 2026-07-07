# -*- coding: utf-8 -*-
"""
exp_fruit_setting_2022.py  (신규 데이터 · 착과수 최적환경 2022 첫 분석)
=====================================================================
[이 파일이 하는 일 — 한 문장]
신규 테스트 데이터(STRAWBERRY_FRUIT_SETTING_ENV_20221209.csv)로 착과수(FRST_TREE_CNT)를
예측하되, '조사기간(세그먼트) 단위 GroupKFold'와 '시간순 forward-chaining' 두 가지로
정직하게 평가하고, 순진한 기준선(평균찍기·지속성)과 비교한다.

[왜 이렇게 평가하나 — 이 데이터의 함정]
- 단일 구역(ZONE 66) · 2022-01~06 · 10분 간격 로깅(26,063행).
- 착과수는 '주 1회 조사값'이 그 주 내내 상수로 반복되는 계단형(고유값 22, 세그먼트 27개).
- 그래서 행 단위 랜덤 분할은 '시간블록 암기'로 R²가 가짜로 치솟는다(누수).
  → 같은 조사기간 행이 학습/검증에 쪼개지지 않게 '세그먼트 단위'로 나눈다.
- GroupKFold(랜덤 세그먼트 배정)도 과거↔미래를 보간해 낙관적일 수 있어,
  '과거로 미래 1주를 예측'하는 forward-chaining을 함께 본다(실전에 가까움).

실행:  python src/exp_fruit_setting_2022.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "STRAWBERRY_FRUIT_SETTING_ENV_20221209.csv"
TARGET = "FRST_TREE_CNT"
ID_COLS = ["ZONE_NM", "STRG_DT", TARGET]
PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.9,
              colsample_bytree=0.8, random_state=42, verbose=-1)


def load():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["STRG_DT"] = pd.to_datetime(df["STRG_DT"])
    df = df.sort_values("STRG_DT").reset_index(drop=True)
    # 조사기간 세그먼트: 착과수 값이 유지되는 연속 구간 = 한 번의 조사
    df["seg"] = (df[TARGET] != df[TARGET].shift()).cumsum()
    feat = [c for c in df.columns if c not in ID_COLS + ["seg"]]
    return df, feat


def eval_groupkfold(df, feat, n_splits=5):
    """세그먼트 단위 GroupKFold OOF — 같은 조사기간 누수 차단."""
    X, y, grp = df[feat], df[TARGET].to_numpy(), df["seg"].to_numpy()
    oof = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=grp):
        m = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    g = pd.DataFrame({"seg": grp, "y": y, "p": oof}).groupby("seg").mean()
    return dict(row_r2=r2_score(y, oof), row_mae=mean_absolute_error(y, oof),
                seg_r2=r2_score(g["y"], g["p"]), seg_mae=mean_absolute_error(g["y"], g["p"]),
                naive_r2=r2_score(y, np.full_like(y, y.mean(), dtype=float)))


def eval_forward(df, feat, start=8):
    """시간순 forward-chaining: 앞 start개 세그먼트 학습 후 한 주씩 앞으로 예측."""
    order = df.drop_duplicates("seg")["seg"].tolist()
    seg_val = df.drop_duplicates("seg").set_index("seg")[TARGET]
    rows = []
    for i in range(start, len(order)):
        tr = df[df["seg"].isin(order[:i])]
        te = df[df["seg"] == order[i]]
        m = lgb.LGBMRegressor(**PARAMS).fit(tr[feat], tr[TARGET])
        rows.append((order[i], te[TARGET].iloc[0], float(m.predict(te[feat]).mean()),
                     seg_val.loc[order[i - 1]]))  # 지속성=직전 세그먼트 값
    r = pd.DataFrame(rows, columns=["seg", "actual", "pred", "persist"])
    return dict(model_r2=r2_score(r["actual"], r["pred"]),
                model_mae=mean_absolute_error(r["actual"], r["pred"]),
                persist_r2=r2_score(r["actual"], r["persist"]),
                persist_mae=mean_absolute_error(r["actual"], r["persist"]),
                n=len(r))


def main():
    print("=" * 64)
    print("신규 데이터 · 착과수 최적환경(2022) 첫 분석")
    print("=" * 64)
    df, feat = load()
    print(f"[데이터] {len(df):,}행 × 피처 {len(feat)}개 · "
          f"구역 {df['ZONE_NM'].nunique()}개 · 세그먼트(조사기간) {df['seg'].nunique()}개")
    print(f"         기간 {df['STRG_DT'].min().date()} ~ {df['STRG_DT'].max().date()} "
          f"(10분 간격) · 착과수 {df[TARGET].min()}~{df[TARGET].max()}")

    k = eval_groupkfold(df, feat)
    print("\n[세그먼트 GroupKFold OOF]")
    print(f"  행 단위      R² {k['row_r2']:.3f} · MAE {k['row_mae']:.3f}")
    print(f"  세그먼트평균 R² {k['seg_r2']:.3f} · MAE {k['seg_mae']:.3f}   (조사기간 단위 실질)")
    print(f"  naive(평균)  R² {k['naive_r2']:.3f}")

    f = eval_forward(df, feat)
    print(f"\n[시간순 forward-chaining · 미래 1주 예측 · {f['n']}주 평가]")
    print(f"  모델          R² {f['model_r2']:.3f} · MAE {f['model_mae']:.3f}")
    print(f"  지속성(직전주) R² {f['persist_r2']:.3f} · MAE {f['persist_mae']:.3f}")

    print("\n[해석] forward-chaining에서 모델이 지속성을 거의 못 이기면,")
    print("       환경 피처는 발달단계·자기상관 위에 추가 정보를 별로 못 준다는 뜻.")
    print("       → 착과수는 발달단계 지배(기존 프로젝트 결론과 일치). 개화수 merge·")
    print("          발달단계 feature·제어/에너지 트랙으로 확장 필요.")
    print("=" * 64)


if __name__ == "__main__":
    main()
