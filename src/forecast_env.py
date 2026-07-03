# -*- coding: utf-8 -*-
"""
forecast_env.py
===============
[이 파일이 하는 일 — 한 문장]
시간별 환경 데이터로 "지금부터 h시간 뒤의 내부 환경(온도·CO₂)"을 예측하는
단기 예보 모델을 만듭니다. (제어 결정 = '앞으로 어떻게 될지'를 알아야 미리 대응)

[왜 필요한가 — 제어와의 연결]
폐루프 제어는 '지금'만 보고 반응하면 늦습니다. 예: 한파가 오기 '전에' 난방을 켜야 함.
그래서 "앞으로 몇 시간 환경이 어떻게 변할지"를 예보하는 모델이 제어의 눈이 됩니다.
(다른 팀의 '문제2: 환경 예보 h+1~h+12'에 해당하는 트랙)

[방법 — 딥러닝(LSTM) 대신 부스팅 + 시차(lag) feature]
시간별 데이터는 많지만(온실당 3천여 시각), 우리는 표(부스팅)에 강합니다.
과거 몇 시점의 값(시차 feature)과 외부기상·시간대를 입력해 미래 값을 예측합니다.
정직한 설정: 예측 시점에 '아는 정보'만 사용(미래 내부값은 안 씀).

[성능 판단 기준 — 지속성(persistence)]
"h시간 뒤도 지금과 같다"고 찍는 게 순진한 기준선. 우리 모델이 이보다 나아야 의미.

실행:  python src/forecast_env.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TARGETS = ["내부온도", "내부CO2"]     # 예보할 내부 환경
HORIZONS = [1, 3, 6, 12]             # 몇 시간 뒤를 볼지
LAGS = [1, 2, 3, 6, 12]             # 과거 몇 시점을 입력으로 쓸지
SEED = 42


def load_hourly():
    """온실 4개의 시간별 환경을 읽어 합치고 시간순 정렬."""
    frames = []
    for g in [1, 2, 3, 4]:
        e = pd.read_csv(DATA_DIR / f"24_환경정보-{g}온실.csv", encoding="utf-8-sig")
        e["측정일시"] = pd.to_datetime(e["측정일시"], format="%m/%d/%Y %H:%M")
        e["온실번호"] = g
        frames.append(e)
    env = pd.concat(frames, ignore_index=True)
    env = env.sort_values(["온실번호", "측정일시"]).reset_index(drop=True)
    return env


def make_features(env, target, horizon):
    """
    시차(lag) feature와 예측 대상(미래 값)을 만든다.
    - 입력 X: 과거~현재 값들(내부온도·CO₂·습도의 시차) + 외부기상 + 시간대 + 온실.
    - 정답 y: horizon 시간 뒤의 target 값.
    ※ shift는 온실별로 따로 (다른 온실 값이 섞이지 않게).
    """
    df = env.copy()
    g = df.groupby("온실번호")

    feat_cols = ["온실번호", "외부온도", "외부일사량", "외부풍속"]
    # 시간대(주기성): 시(hour)를 sin/cos로 (0시와 23시가 이어지도록)
    hour = df["측정일시"].dt.hour
    df["시_sin"] = np.sin(2 * np.pi * hour / 24)
    df["시_cos"] = np.cos(2 * np.pi * hour / 24)
    feat_cols += ["시_sin", "시_cos"]

    # 내부 상태의 시차(현재 t 및 과거) — '지금까지 아는 정보'
    for col in ["내부온도", "내부CO2", "내부습도"]:
        for L in [0] + LAGS:
            name = f"{col}_t-{L}"
            df[name] = g[col].shift(L)
            feat_cols.append(name)

    # 정답: horizon 시간 뒤의 target (미래로 shift = shift(-h))
    df["y"] = g[target].shift(-horizon)

    # 시차/미래 생성으로 생긴 결측 행 제거
    df = df.dropna(subset=feat_cols + ["y"]).reset_index(drop=True)
    return df, feat_cols


def time_split(df):
    """시간 순 분할: 앞 75% 학습 / 뒤 25% 검증 (미래 예측 상황 재현)."""
    cutoff = df["측정일시"].quantile(0.75)
    tr = df[df["측정일시"] <= cutoff]
    va = df[df["측정일시"] > cutoff]
    return tr, va


def main():
    print("=" * 64)
    print("환경 예보 — 시간별 데이터로 h시간 뒤 내부 환경 예측")
    print("=" * 64)
    env = load_hourly()
    print(f"[로드] 시간별 {len(env):,}행 (온실 1~4)\n")

    for target in TARGETS:
        unit = "℃" if target == "내부온도" else "ppm"
        print(f"■ {target} 예보  (기준선=지속성: 'h시간 뒤도 지금과 같다')")
        print(f"  {'지평':>5}{'모델 RMSE':>12}{'지속성 RMSE':>14}{'모델 R²':>10}   판정")
        print("  " + "-" * 52)
        for h in HORIZONS:
            df, feat = make_features(env, target, h)
            tr, va = time_split(df)
            model = lgb.LGBMRegressor(objective="regression", n_estimators=500,
                learning_rate=0.05, num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
            model.fit(tr[feat], tr["y"])
            pred = model.predict(va[feat])
            rmse = np.sqrt(mean_squared_error(va["y"], pred))
            r2 = r2_score(va["y"], pred)
            # 지속성 기준선: 현재값(t-0)으로 미래를 찍음
            persist = va[f"{target}_t-0"].to_numpy()
            rmse_p = np.sqrt(mean_squared_error(va["y"], persist))
            verd = "모델 우세 O" if rmse < rmse_p else "지속성 우세 X"
            print(f"  {('h+'+str(h)):>5}{rmse:>10.2f}{unit:<2}{rmse_p:>12.2f}{unit:<2}{r2:>10.3f}   {verd}")
        print()

    print("=" * 64)
    print("[해석] 지평이 멀수록 어려워지지만(불확실성↑), 지속성보다 나으면 예보 가치가 있음.")
    print("       이 예보를 제어 규칙의 '조건(IF)'에 넣으면 선제 대응이 가능해집니다.")


if __name__ == "__main__":
    main()
