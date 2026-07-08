# -*- coding: utf-8 -*-
"""
train_v34_ensemble.py  (v3.4 실험 · 규제·앙상블로 과적합 억제)
=============================================================
[이 파일이 하는 일 — 한 문장]
v3.3의 rich 피처(v3.1 + 발달축 = 47개, '정보량 최상위인데 과적합')를 그대로 쓰되,
'규제 강화·시드 배깅·모델 다양성(LGBM+CatBoost+Ridge) 블렌딩'으로 과적합을 눌러
v3.1(생산 최고)을 넘을 수 있는지 봅니다. (데이터는 그대로 — 지금 데이터를 최대한 짜냄)

[왜 이게 통할 수 있나]
v3.3의 문제는 강한 발달 신호에 트리 하나가 과하게 기대는 것(과적합). 해법 3종:
 (1) 강규제 LGBM — num_leaves↓·min_child↑·reg↑ 로 한 모델의 과신을 억제
 (2) 시드 배깅   — 여러 seed 평균 → 분산(variance) 감소
 (3) 다양성 블렌드 — 결이 다른 모델 평균(트리 2종 + 선형). 특히 발달 '비율' 피처
      (착과대비엽수·자기최고대비)는 선형에 가까워 Ridge가 트리보다 안정적일 수 있음.

[검증]  전부 온실별 LOGO out-of-fold. Ridge/ExtraTrees는 NaN 불가라 fold 안에서 중앙값 대치.

실행:  python src/train_v34_ensemble.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v31_momentum import load, logo_oof, scores, tune, TARGET, GHS, SEED
from train_v33_developmental import DEVEL

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

# 강규제 LGBM (소데이터 과적합 억제 방향으로 고정)
REG_LGBM = dict(objective="huber", n_estimators=500, learning_rate=0.03,
                num_leaves=15, min_child_samples=25, subsample=0.8,
                colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0)


def make_cat():
    # 얕은 depth + L2 규제 + MAE 손실. CatBoost는 NaN 네이티브 처리.
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4,
                             l2_leaf_reg=6.0, loss_function="MAE",
                             random_seed=SEED, verbose=0, allow_writing_files=False)


# ── fit_predict 빌더들: (Xtr, ytr, Xva) → 예측 ──
def fp_lgbm(params, seeds=(SEED,)):
    def f(Xtr, ytr, Xva):
        ps = []
        for s in seeds:
            m = lgb.LGBMRegressor(**params, random_state=s, verbose=-1)
            m.fit(Xtr, ytr); ps.append(m.predict(Xva))
        return np.mean(ps, axis=0)
    return f


def fp_cat(seeds=(SEED,)):
    def f(Xtr, ytr, Xva):
        ps = []
        for s in seeds:
            m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4,
                                  l2_leaf_reg=6.0, loss_function="MAE",
                                  random_seed=s, verbose=0, allow_writing_files=False)
            m.fit(Xtr, ytr); ps.append(m.predict(Xva))
        return np.mean(ps, axis=0)
    return f


def fp_ridge(alpha):
    def f(Xtr, ytr, Xva):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("rg", Ridge(alpha=alpha))])
        pipe.fit(Xtr, ytr); return pipe.predict(Xva)
    return f


def fp_extratrees(n=500):
    def f(Xtr, ytr, Xva):
        imp = SimpleImputer(strategy="median")
        Xt = imp.fit_transform(Xtr); Xv = imp.transform(Xva)
        m = ExtraTreesRegressor(n_estimators=n, min_samples_leaf=3,
                                random_state=SEED, n_jobs=-1)
        m.fit(Xt, ytr); return m.predict(Xv)
    return f


def fp_blend(parts):
    """parts = [(fit_predict, weight), ...] → 가중 평균."""
    def f(Xtr, ytr, Xva):
        tot, wsum = 0.0, 0.0
        for fp, w in parts:
            tot = tot + w * fp(Xtr, ytr, Xva); wsum += w
        return tot / wsum
    return f


def logo_eval(df, feat, fit_predict):
    """온실별 LOGO OOF → (개체R², 집계R², 전체MAE, MAE중앙), oof."""
    parts = []
    for g in GHS:
        tr = df[df["온실번호"] != g]
        va = df[df["온실번호"] == g].copy()
        pred = fit_predict(tr[feat].copy(), tr[TARGET].to_numpy(), va[feat].copy())
        va["예측"] = np.maximum(pred, 0.0)
        parts.append(va)
    oof = pd.concat(parts).reset_index(drop=True)
    return scores(oof), oof


def row(tag, s, ref=None):
    ind, agg, mae, med = s
    extra = ""
    if ref is not None:
        extra = f"   (ΔMAE중앙 {med-ref[3]:+.3f} · Δ개체R² {ind-ref[0]:+.4f})"
    print(f"  {tag:<26} 개체R² {ind:.4f} · 집계R² {agg:.4f} · MAE {mae:.3f} · MAE중앙 {med:.3f}{extra}")


def main():
    print("=" * 78)
    print("v3.4 실험 — 규제·앙상블로 v3.3 rich 피처 과적합 억제 (대상: 착과수)")
    print("=" * 78)
    df, base, v31 = load()
    rich = v31 + DEVEL
    print(f"[데이터] {len(df)}행 · base(v3) {len(base)} · v3.1 {len(v31)} · rich(v3.3) {len(rich)} 피처")

    # ── 기준선: v3(base) · v3.1(생산 최고) · v3.3(rich 단일 LGBM 튜닝) ──
    print("-" * 78)
    print("[기준선] v3 · v3.1 · v3.3 (각자 튜닝된 단일 LGBM)")
    s_v3 = scores(logo_oof(df, base, tune(df, base))); row("v3   (base 34)", s_v3)
    s31 = scores(logo_oof(df, v31, tune(df, v31))); row("v3.1 (v31 41·생산최고)", s31, ref=s_v3)
    s33 = scores(logo_oof(df, rich, tune(df, rich))); row("v3.3 (rich 47·튜닝)", s33, ref=s31)
    print(f"  ↑ 이후 모든 v3.4 후보는 '생산 최고 v3.1'(MAE중앙 {s31[3]:.3f}) 기준으로 비교")

    # ── Ridge alpha 선택 (rich, LOGO 전체MAE 최소) ──
    print("-" * 78)
    print("[Ridge] alpha 그리드 (rich, NaN→중앙값 대치)")
    best_a, best_mae = None, 1e9
    for a in [3, 10, 30, 100]:
        s, _ = logo_eval(df, rich, fp_ridge(a))
        print(f"  alpha={a:<4} MAE {s[2]:.3f} · MAE중앙 {s[3]:.3f} · 개체R² {s[0]:.4f}")
        if s[2] < best_mae:
            best_mae, best_a = s[2], a
    print(f"  → 선택 alpha={best_a}")

    # ── v3.4 후보들 (전부 rich 피처) ──
    print("-" * 78)
    print("[v3.4 후보] rich 피처 + 규제/배깅/다양성 블렌드   (기준: v3.1)")
    seeds5 = (11, 22, 33, 44, 55)
    lgbm_bag = fp_lgbm(REG_LGBM, seeds=seeds5)
    cat_bag = fp_cat(seeds=(11, 22, 33))
    ridge = fp_ridge(best_a)

    cfgs = {
        "a. 강규제 LGBM(단일)": fp_lgbm(REG_LGBM),
        "b. LGBM 시드배깅×5":  lgbm_bag,
        "c. CatBoost(배깅×3)": cat_bag,
        "d. Ridge(선형)":       ridge,
        "e. ExtraTrees":        fp_extratrees(),
        "f. 블렌드 LGBM+Cat":   fp_blend([(lgbm_bag, .5), (cat_bag, .5)]),
        "g. 블렌드 LGBM+Cat+Ridge": fp_blend([(lgbm_bag, .4), (cat_bag, .4), (ridge, .2)]),
    }
    results = {}
    for name, fp in cfgs.items():
        s, oof = logo_eval(df, rich, fp)
        results[name] = (s, oof)
        row(name, s, ref=s31)

    # ── 최고 후보 선정 (전체 MAE 최소, v3의 튜닝 목표와 일치) ──
    best_name = min(results, key=lambda k: results[k][0][2])
    bs = results[best_name][0]
    print("-" * 78)
    print(f"[최고 후보] {best_name} — 개체R² {bs[0]:.4f} · 집계R² {bs[1]:.4f} · "
          f"MAE {bs[2]:.3f} · MAE중앙 {bs[3]:.3f}")

    # ── 판정 (v3.1 대비) ──
    d_ind, d_agg, d_med = bs[0]-s31[0], bs[1]-s31[1], bs[3]-s31[3]
    print("-" * 78)
    if d_med < -0.01 or d_ind > 0.005 or d_agg > 0.005:
        verdict = f"채택 검토 — v3.1 대비 개선 (ΔMAE중앙 {d_med:+.3f} · Δ개체R² {d_ind:+.4f})"
    else:
        verdict = (f"보류 — 앙상블도 v3.1을 못 넘음 (ΔMAE중앙 {d_med:+.3f}). "
                   f"600행에서 발달 신호의 과적합은 규제로도 완전 해소 불가")
    print("=" * 78)
    print(f"[판정] {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
