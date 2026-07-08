# -*- coding: utf-8 -*-
"""
train_v33_developmental.py  (v3.3 실험 · 발달축 파생 추가)
========================================================
[이 파일이 하는 일 — 한 문장]
v3.1(개체 추적 + 모멘텀)에 '발달축(발달단계 근사)' 파생 6개를 더한 v3.3을 만들어,
동일 조건(같은 하이퍼파라미터·같은 LOGO)에서 v3 / v3.1 / v3.3을 나란히 비교합니다.

[왜 발달축인가 — v3.2가 준 힌트]
v3.2에서 '환경 추세'는 착과수에 무효였습니다(착과수는 발달단계 지배). 그렇다면 도움은
'환경'이 아니라 '발달' 쪽 파생에서 나올 것 — 개체 누적·착과 템포·소스싱크 균형·자기
궤적상 위치·증가 국면. 모두 트리가 스스로 만들기 어려운 누적·비율·국면으로 설계.

[누수 없음]
모든 발달 파생은 '전주까지'의 과거만 사용.

실행:  python src/train_v33_developmental.py
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
                                 write_result_sheet, TARGET, SEED)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
CSV_OUT = PROJECT_DIR / "data" / "processed" / "result_sheet_v3_3.csv"
HTML_OUT = PROJECT_DIR / "docs" / "report" / "착과수_개체별_결과지_v3_3.html"

# v3.3에서 새로 더할 발달축 파생 (data_prep.add_developmental_features 산출물)
DEVEL = ["착과수_개체누적_전주", "착과수_개체누적률_전주", "착과대비엽수_전주",
         "착과대비관부_전주", "착과수_자기최고대비_전주", "착과수_증가국면_전주"]


def main():
    print("=" * 70)
    print("v3.3 실험 — 발달축 파생 6개 추가 vs v3.1 / v3 (대상: 착과수)")
    print("=" * 70)
    df, base, v31 = load()
    v33 = v31 + DEVEL
    print(f"[데이터] {len(df)}행 · v3 {len(base)} → v3.1 {len(v31)} → v3.3 {len(v33)} 피처")
    print(f"[추가 피처] {DEVEL}")

    # ── (1) 정직한 ablation: BASE(v3)에서 튜닝한 params를 셋 다에 동일 적용 ──
    print("-" * 70)
    print("[튜닝] v3 BASE 기준 huber+MAE 탐색 → 그 params를 v3/v3.1/v3.3에 동일 적용")
    params = tune(df, base)
    print("\n[동일 하이퍼파라미터 · 피처만 다름]")
    s_base = scores(logo_oof(df, base, params)); row("v3   (BASE)", s_base)
    s_v31 = scores(logo_oof(df, v31, params));   row("v3.1 (+모멘텀)", s_v31)
    s_v33 = scores(logo_oof(df, v33, params));   row("v3.3 (+발달축)", s_v33)
    print(f"  → v3.3 vs v3.1  Δ개체R² {s_v33[0]-s_v31[0]:+.4f} · "
          f"Δ집계R² {s_v33[1]-s_v31[1]:+.4f} · ΔMAE중앙 {s_v33[3]-s_v31[3]:+.3f}")
    print(f"  → v3.3 vs v3    Δ개체R² {s_v33[0]-s_base[0]:+.4f} · "
          f"Δ집계R² {s_v33[1]-s_base[1]:+.4f} · ΔMAE중앙 {s_v33[3]-s_base[3]:+.3f}")

    # ── (2) v3.3 자체 튜닝 천장 ──
    print("-" * 70)
    print("[참고] v3.3 자체 튜닝 (피처에 맞춰 하이퍼파라미터 재탐색)")
    params_v33 = tune(df, v33)
    oof_v33 = logo_oof(df, v33, params_v33)
    s_v33t = scores(oof_v33); row("v3.3 (튜닝됨)", s_v33t)

    # ── (3) 발달 파생이 실제로 쓰였나 (중요도 순위) ──
    final = lgb.LGBMRegressor(**params_v33, random_state=SEED, verbose=-1)
    final.fit(df[v33], df[TARGET])
    imp = (pd.DataFrame({"feature": v33, "중요도": final.feature_importances_})
           .sort_values("중요도", ascending=False).reset_index(drop=True))
    imp["순위"] = imp.index + 1
    print("-" * 70)
    print(f"[중요도 상위 12]  (전체 {len(v33)}개 중)")
    print(imp.head(12)[["순위", "feature", "중요도"]].to_string(index=False))
    print(f"\n[발달축 6개의 순위]")
    print(imp[imp["feature"].isin(DEVEL)][["순위", "feature", "중요도"]].to_string(index=False))

    # ── (4) 모델 저장 ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "lgbm_fruit_count_v33.txt"
    final.booster_.save_model(str(out))
    print("-" * 70)
    print(f"[저장] v3.3 모델 → {out}")

    # ── (5) 판정 + (채택 시) 결과지 ──
    d_ind, d_agg, d_med = s_v33[0]-s_v31[0], s_v33[1]-s_v31[1], s_v33[3]-s_v31[3]
    adopt = (d_agg > 0.003 or d_ind > 0.003 or d_med < -0.01)
    print("=" * 70)
    print(f"[판정] {'채택 검토 — v3.1 대비 개선' if adopt else '보류 — v3.1 대비 개선 미미'}")
    print("=" * 70)

    if adopt:
        # 결과지 경로를 v3.3용으로 바꿔 재사용
        import train_v31_momentum as t
        t.CSV_OUT, t.HTML_OUT = CSV_OUT, HTML_OUT
        med, dmed = write_result_sheet(oof_v33, s_v33t[0], s_v33t[1], base_med=s_base[3])
        print(f"[저장] v3.3 결과지 → {HTML_OUT}  (MAE중앙 {med:.3f}, v3 대비 ▼{dmed:.3f})")


if __name__ == "__main__":
    main()
