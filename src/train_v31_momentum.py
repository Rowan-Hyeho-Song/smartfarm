# -*- coding: utf-8 -*-
"""
train_v31_momentum.py  (v3.1 실험 · 모멘텀 파생 추가)
====================================================
[이 파일이 하는 일 — 한 문장]
v3(개체 추적)의 피처 집합은 그대로 두고, 거기에 '증가속도(모멘텀)' 파생 7개를 더한
v3.1을 만들어, 동일 조건(같은 하이퍼파라미터·같은 LOGO)에서 v3과 나란히 성능을 비교합니다.

[비교 방식 — 정직한 ablation]
- 피처만 다르고 나머지는 똑같아야 '피처의 순수 효과'가 보입니다.
- 그래서 BASE(v3 피처)에서 튜닝해 얻은 하이퍼파라미터 P를 BASE와 V31 '둘 다'에 똑같이 적용.
- 그 다음 V31을 따로 튜닝했을 때의 '천장'도 참고로 같이 찍습니다.

[누수 없음]
모멘텀은 전주·전전주(과거)에서만 파생 → 미래 정보 아님.

실행:  python src/train_v31_momentum.py
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
GHS = [1, 2, 3, 4]
SEED = 42
N_TRIALS = 30

# v3.1에서 새로 더할 모멘텀 파생 (data_prep.add_momentum_features 산출물)
MOMENTUM = ["착과수_개체증가량", "초장_개체증가량", "엽수_개체증가량", "관부직경_개체증가량",
            "착과수_전주증가량", "초장_전주증가량", "착과수_개체대비온실_전주"]


def load():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    # v3 BASE 피처: 환경요약+시점(전주/개체 제외) + 개체 lag  ← train_individual.py와 동일 규칙
    env_time = [c for c in df.columns
                if c not in KEYS + GROWTH and "전주" not in c and "개체" not in c
                and c not in MOMENTUM]
    plant_lag = [c for c in df.columns if "개체전" in c]
    base = ["온실번호", "측정라인"] + env_time + plant_lag
    v31 = base + MOMENTUM
    return df, base, v31


def logo_oof(df, feat, params):
    """온실 하나씩 빼며 개체 예측 → out-of-fold 표."""
    parts = []
    for g in GHS:
        tr = df[df["온실번호"] != g]
        va = df[df["온실번호"] == g].copy()
        m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
        m.fit(tr[feat], tr[TARGET])
        va["예측"] = np.maximum(m.predict(va[feat]), 0.0)
        parts.append(va)
    return pd.concat(parts).reset_index(drop=True)


def scores(oof):
    """개체 R² · 집계 R²(온실-날짜 평균) · 전체 MAE · 개체 MAE 중앙값."""
    ind = r2_score(oof[TARGET], oof["예측"])
    gp = oof.groupby(["온실번호", "조사일자"])[[TARGET, "예측"]].mean()
    agg = r2_score(gp[TARGET], gp["예측"])
    mae = mean_absolute_error(oof[TARGET], oof["예측"])
    pm = oof.groupby(["온실번호", "측정라인", "표본번호"]).apply(
        lambda s: np.mean(np.abs(s[TARGET] - s["예측"])), include_groups=False)
    return ind, agg, mae, float(np.median(pm))


def tune(df, feat):
    """huber 손실 + 전체 MAE 최소화 (v3과 동일 목적함수)."""
    def obj(t):
        p = dict(objective="huber",
                 n_estimators=t.suggest_int("n_estimators", 200, 700),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                 num_leaves=t.suggest_int("num_leaves", 15, 63),
                 min_child_samples=t.suggest_int("min_child_samples", 5, 30),
                 subsample=t.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 5.0, log=True))
        o = logo_oof(df, feat, p)
        return mean_absolute_error(o[TARGET], o["예측"])
    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    return dict(objective="huber", **st.best_params)


def row(tag, s):
    ind, agg, mae, med = s
    print(f"  {tag:<22} 개체R² {ind:.4f} · 집계R² {agg:.4f} · "
          f"전체MAE {mae:.3f} · MAE중앙 {med:.3f}")


def main():
    print("=" * 70)
    print("v3.1 실험 — 모멘텀(증가속도) 파생 7개 추가 vs v3 (대상: 착과수)")
    print("=" * 70)
    df, base, v31 = load()
    print(f"[데이터] {len(df)}행 · v3 피처 {len(base)}개 → v3.1 피처 {len(v31)}개 (+{len(MOMENTUM)})")
    print(f"[추가 피처] {MOMENTUM}")

    # ── (1) 정직한 ablation: BASE에서 튜닝한 params를 둘 다에 동일 적용 ──
    print("-" * 70)
    print(f"[튜닝] v3 BASE 기준 huber+MAE {N_TRIALS}회 탐색 → 그 params를 둘 다에 적용")
    params = tune(df, base)
    print("\n[동일 하이퍼파라미터 · 피처만 다름]")
    s_base = scores(logo_oof(df, base, params)); row("v3  (BASE)", s_base)
    s_v31 = scores(logo_oof(df, v31, params));  row("v3.1(+모멘텀)", s_v31)
    d_ind, d_agg = s_v31[0] - s_base[0], s_v31[1] - s_base[1]
    d_med = s_v31[3] - s_base[3]
    print(f"  → Δ 개체R² {d_ind:+.4f} · Δ 집계R² {d_agg:+.4f} · Δ MAE중앙 {d_med:+.3f}")

    # ── (2) v3.1 따로 튜닝했을 때의 천장 ──
    print("-" * 70)
    print(f"[참고] v3.1 자체 튜닝 {N_TRIALS}회 (피처에 맞춰 하이퍼파라미터 재탐색)")
    params_v31 = tune(df, v31)
    oof_v31 = logo_oof(df, v31, params_v31)
    s_v31t = scores(oof_v31); row("v3.1 (튜닝됨)", s_v31t)

    # ── (3) 모멘텀 피처가 실제로 쓰였나 (중요도 순위) ──
    final = lgb.LGBMRegressor(**params_v31, random_state=SEED, verbose=-1)
    final.fit(df[v31], df[TARGET])
    imp = (pd.DataFrame({"feature": v31, "중요도": final.feature_importances_})
           .sort_values("중요도", ascending=False).reset_index(drop=True))
    imp["순위"] = imp.index + 1
    print("-" * 70)
    print(f"[중요도 상위 10]  (전체 {len(v31)}개 중)")
    print(imp.head(10)[["순위", "feature", "중요도"]].to_string(index=False))
    print(f"\n[모멘텀 7개의 순위]")
    print(imp[imp["feature"].isin(MOMENTUM)][["순위", "feature", "중요도"]].to_string(index=False))

    # ── (4) 모델 저장 ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "lgbm_fruit_count_v31.txt"
    final.booster_.save_model(str(out))
    print("-" * 70)
    print(f"[저장] v3.1 모델 → {out}")

    # ── (5) 판정 ──
    print("=" * 70)
    verdict = ("채택 검토 — 집계/개체 R²가 의미있게 개선" if (d_agg > 0.005 or d_ind > 0.005)
               else "보류 — 동일 params에서 개선 미미(GBDT가 이미 차분을 학습). v3.2(환경 델타)로 이동 권장")
    print(f"[판정] {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
