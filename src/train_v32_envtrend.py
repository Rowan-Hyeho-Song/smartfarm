# -*- coding: utf-8 -*-
"""
train_v32_envtrend.py  (v3.2 실험 · 환경 추세(델타) 파생 추가)
============================================================
[이 파일이 하는 일 — 한 문장]
v3.1(개체 추적 + 모멘텀)에 '환경 추세(델타)' 파생 10개를 더한 v3.2를 만들어,
동일 조건(같은 하이퍼파라미터·같은 LOGO)에서 v3 / v3.1 / v3.2를 나란히 비교합니다.

[왜 이게 v3.1보다 기대값이 큰가]
v3.1의 모멘텀은 '이미 학습표에 있던 lag'에서 파생 → GBDT가 일부 스스로 만들 수 있었음.
v3.2의 환경 추세는 '직전 7일 vs 그 이전 7일'이라, 지금 학습표에 '아예 없던 새 정보'.
(최근 주가 지난 주보다 더웠나/건조했나/밝았나 — 방향성)

[누수 없음]
모든 추세는 조사일 이전(< d) 환경만 사용.

실행:  python src/train_v32_envtrend.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v31_momentum import (load, logo_oof, scores, tune, row,
                                 MOMENTUM, TARGET, SEED)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

# v3.2에서 새로 더할 환경 추세 파생 (data_prep.summarize_env_before_survey 산출물)
ENVTREND = ["내부온도_추세", "내부습도_추세", "내부CO2_추세", "외부온도_추세",
            "VPD_추세", "급액EC_추세", "외부일사량_추세", "주야온도차_DIF_추세",
            "내부온도_14일평균", "외부일사량_14일평균"]


def main():
    print("=" * 70)
    print("v3.2 실험 — 환경 추세(델타) 파생 10개 추가 vs v3.1 / v3 (대상: 착과수)")
    print("=" * 70)
    df, base, v31 = load()
    v32 = v31 + ENVTREND
    print(f"[데이터] {len(df)}행 · v3 {len(base)} → v3.1 {len(v31)} → v3.2 {len(v32)} 피처")
    print(f"[추가 피처] {ENVTREND}")

    # ── (1) 정직한 ablation: BASE(v3)에서 튜닝한 params를 셋 다에 동일 적용 ──
    print("-" * 70)
    print("[튜닝] v3 BASE 기준 huber+MAE 탐색 → 그 params를 v3/v3.1/v3.2에 동일 적용")
    params = tune(df, base)
    print("\n[동일 하이퍼파라미터 · 피처만 다름]")
    s_base = scores(logo_oof(df, base, params)); row("v3   (BASE)", s_base)
    s_v31 = scores(logo_oof(df, v31, params));   row("v3.1 (+모멘텀)", s_v31)
    s_v32 = scores(logo_oof(df, v32, params));   row("v3.2 (+환경추세)", s_v32)
    print(f"  → v3.2 vs v3.1  Δ개체R² {s_v32[0]-s_v31[0]:+.4f} · "
          f"Δ집계R² {s_v32[1]-s_v31[1]:+.4f} · ΔMAE중앙 {s_v32[3]-s_v31[3]:+.3f}")
    print(f"  → v3.2 vs v3    Δ개체R² {s_v32[0]-s_base[0]:+.4f} · "
          f"Δ집계R² {s_v32[1]-s_base[1]:+.4f} · ΔMAE중앙 {s_v32[3]-s_base[3]:+.3f}")

    # ── (2) v3.2 자체 튜닝 천장 ──
    print("-" * 70)
    print("[참고] v3.2 자체 튜닝 (피처에 맞춰 하이퍼파라미터 재탐색)")
    params_v32 = tune(df, v32)
    s_v32t = scores(logo_oof(df, v32, params_v32)); row("v3.2 (튜닝됨)", s_v32t)

    # ── (3) 환경 추세 피처가 실제로 쓰였나 (중요도 순위) ──
    final = lgb.LGBMRegressor(**params_v32, random_state=SEED, verbose=-1)
    final.fit(df[v32], df[TARGET])
    imp = (pd.DataFrame({"feature": v32, "중요도": final.feature_importances_})
           .sort_values("중요도", ascending=False).reset_index(drop=True))
    imp["순위"] = imp.index + 1
    print("-" * 70)
    print(f"[중요도 상위 12]  (전체 {len(v32)}개 중)")
    print(imp.head(12)[["순위", "feature", "중요도"]].to_string(index=False))
    print(f"\n[환경 추세 10개의 순위]")
    print(imp[imp["feature"].isin(ENVTREND)][["순위", "feature", "중요도"]].to_string(index=False))

    # ── (4) 모델 저장 ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "lgbm_fruit_count_v32.txt"
    final.booster_.save_model(str(out))
    print("-" * 70)
    print(f"[저장] v3.2 모델 → {out}")

    # ── (5) 판정 ──
    d_ind, d_agg, d_med = s_v32[0]-s_v31[0], s_v32[1]-s_v31[1], s_v32[3]-s_v31[3]
    print("=" * 70)
    if d_agg > 0.005 or d_ind > 0.005 or d_med < -0.02:
        verdict = "채택 검토 — v3.1 대비 의미있는 개선"
    elif abs(d_ind) < 0.003 and abs(d_med) < 0.01:
        verdict = "보류 — 환경 추세는 착과수(발달 지배)에 거의 무효. 주간 환경 변동이 착과에 약함을 시사"
    else:
        verdict = "혼조 — 지표별 엇갈림, 개별 피처 취사선택 필요"
    print(f"[판정] {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
