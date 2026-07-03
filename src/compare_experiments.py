# -*- coding: utf-8 -*-
"""
compare_experiments.py
======================
[이 파일이 하는 일]
① 7개 생육 지표를 '시간분할'과 '온실별(LOGO)' 두 검증으로 모두 평가해 한 표로 비교.
② 집계 창(직전 3일 / 7일 / 14일)을 바꿔가며 성능이 어떻게 달라지는지 실험.

목적: "왜 착과수·초장만 봤나", "왜 7일인가" 같은 질문에 데이터로 답하기 위함.
실행:  python src/compare_experiments.py
"""

import sys, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import data_prep as dp   # 같은 src 폴더의 병합 파이프라인 재사용

GROWTH = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
GH = [1, 2, 3, 4]
# 모든 지표를 '같은 조건'으로 공정 비교하기 위한 공통 설정값.
BASE = dict(objective="regression", n_estimators=400, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, subsample=0.8, colsample_bytree=0.8)


def feats(df):
    keys = ["온실번호", "조사일자", "측정라인", "표본번호"]
    return ["온실번호"] + [c for c in df.columns if c not in keys + GROWTH]


def r2_time(df, f, t):
    """시간분할: 앞 기간 학습 → 뒤 4주 검증."""
    d = df.dropna(subset=[t])
    dates = sorted(d["조사일자"].unique()); cut = dates[-4]
    tr, va = d[d["조사일자"] < cut], d[d["조사일자"] >= cut]
    if len(tr) == 0 or len(va) == 0:
        return float("nan")
    m = lgb.LGBMRegressor(**BASE, random_state=42, verbose=-1).fit(tr[f], tr[t])
    return r2_score(va[t], m.predict(va[f]))


def r2_logo(df, f, t):
    """온실별(LOGO): 온실 하나씩 빼며 검증한 예측을 모아 R²."""
    d = df.dropna(subset=[t]); preds, act = [], []
    for g in GH:
        tr, va = d[d["온실번호"] != g], d[d["온실번호"] == g]
        if len(va) == 0:
            continue
        m = lgb.LGBMRegressor(**BASE, random_state=42, verbose=-1).fit(tr[f], tr[t])
        preds.append(m.predict(va[f])); act.append(va[t].to_numpy())
    return r2_score(np.concatenate(act), np.concatenate(preds))


# ── 환경/생육 원본은 한 번만 로드(창 실험 때 재사용) ──
with contextlib.redirect_stdout(io.StringIO()):
    ENV = dp.add_env_features(dp.load_environment())
    GROWTH_WIDE = dp.load_growth()


def build_table(window_days):
    """지정한 집계 창으로 학습표를 새로 만든다(로그는 숨김)."""
    dp.WINDOW_DAYS = window_days
    with contextlib.redirect_stdout(io.StringIO()):
        summ = dp.summarize_env_before_survey(ENV, GROWTH_WIDE)
        table = dp.build_training_table(GROWTH_WIDE, summ)
        table = dp.add_time_features(table)
    return table


def main():
    # ───────── ① 7개 생육 지표 × 두 검증 ─────────
    print("=" * 64)
    print("① 7개 생육 지표 비교 (집계 창 7일 · 검증방식 2종)")
    print("=" * 64)
    tbl7 = build_table(7)
    f = feats(tbl7)
    print(f"{'생육 지표':<10}{'시간분할 R²':>13}{'온실별 R²':>12}   추천 검증")
    print("-" * 64)
    rows = []
    for t in GROWTH:
        rt, rl = r2_time(tbl7, f, t), r2_logo(tbl7, f, t)
        rows.append((t, rt, rl))
    # 보기 좋게 R²(둘 중 큰 값) 내림차순
    for t, rt, rl in sorted(rows, key=lambda x: max(x[1], x[2]), reverse=True):
        rec = "시간분할" if rt >= rl else "온실별"
        print(f"{t:<10}{rt:>13.3f}{rl:>12.3f}   {rec}")

    # ───────── ② 집계 창 길이 실험 ─────────
    print("\n" + "=" * 64)
    print("② 집계 창 길이 실험 (직전 3 / 7 / 14일)")
    print("=" * 64)
    print(f"{'창(일)':>6}{'착과수(온실별)':>16}{'초장(시간분할)':>16}{'엽병장(시간분할)':>17}")
    print("-" * 64)
    for w in [3, 7, 14]:
        tbl = build_table(w); ff = feats(tbl)
        a = r2_logo(tbl, ff, "착과수")
        b = r2_time(tbl, ff, "초장")
        c = r2_time(tbl, ff, "엽병장")
        print(f"{w:>6}{a:>16.3f}{b:>16.3f}{c:>17.3f}")

    print("\n[해석] 각 지표는 성격에 맞는 검증에서 가장 잘 나온다.")
    print("       창 길이는 지표마다 최적이 다를 수 있으니 위 표로 선택.")


if __name__ == "__main__":
    main()
