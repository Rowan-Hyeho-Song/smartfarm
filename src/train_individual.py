# -*- coding: utf-8 -*-
"""
train_individual.py  (v3 모델 · 개체 추적)
==========================================
[이 파일이 하는 일 — 한 문장]
(온실+측정라인+표본번호)를 '하나의 개체'로 보고, 그 개체 '자신의 과거(개체 lag)'로
개체 단위 착과수를 예측하는 v3 모델을 만들고, 개체별 결과지(CSV·HTML)를 생성합니다.

[왜 개체 추적인가 — life2vec식 발상]
같은 표본은 매 조사일 같은 물리적 개체입니다(40개체가 15회 모두 등장).
'그 개체가 지난주 몇 개 달았나'는 온실 평균보다 훨씬 강한 신호 → 개체 예측이 크게 향상.
(측정: 개체 R² 0.64 → 0.81, 집계 R² 0.78 → 0.90)

[검증]
온실별 LOGO. 개체 R²(개체 하나하나)와 집계 R²(온실-날짜 평균) 둘 다 보고.

실행:  python src/train_individual.py
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
CSV_OUT = PROJECT_DIR / "data" / "processed" / "result_sheet_individual.csv"
HTML_OUT = PROJECT_DIR / "docs" / "report" / "착과수_개체별_결과지.html"

TARGET = "착과수"
GROWTH = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
KEYS = ["온실번호", "조사일자", "측정라인", "표본번호"]
GHS = [1, 2, 3, 4]
SEED = 42
N_TRIALS = 40


def load_individual():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    # 환경요약 + 시점 (그룹/분포/개체 lag 제외한 순수 환경·시점)
    #  '_추세'·'_14일'(v3.2 파생)은 v3 피처가 아니므로 제외 → v3 모델을 동결.
    env_time = [c for c in df.columns
                if c not in KEYS + GROWTH and "전주" not in c and "개체" not in c
                and "추세" not in c and "14일" not in c]
    plant_lag = [c for c in df.columns if "개체전" in c]
    feat = ["온실번호", "측정라인"] + env_time + plant_lag
    return df, feat


def logo_oof(df, feat, params):
    """온실 하나씩 빼며 개체 예측 → out-of-fold 예측 붙인 표."""
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
    """개체 R²(개체 하나하나) + 집계 R²(온실-날짜 평균)."""
    ind = r2_score(oof[TARGET], oof["예측"])
    gp = oof.groupby(["온실번호", "조사일자"])[[TARGET, "예측"]].mean()
    agg = r2_score(gp[TARGET], gp["예측"])
    return ind, agg


def tune(df, feat):
    # 손실은 huber(L2·L1 절충), 튜닝 목표는 '전체 MAE 최소화'.
    #  → MAE(특히 개체 MAE 중앙값)를 직접 낮추면서도 huber라 R²까지 지킨다.
    #    (실측: L2/R²튜닝 대비 MAE 중앙 0.78→0.73, 개체R² 0.84→0.85, 집계R² 0.94 유지)
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
        return mean_absolute_error(o[TARGET], o["예측"])   # 전체 MAE 최소화
    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    return dict(objective="huber", **st.best_params)


def sparkline(actual, pred, w=150, h=34, pad=4):
    """개체 궤적: 실제(회색) vs 예측(초록) 15주 미니 그래프."""
    vals = np.concatenate([actual, pred])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    rng = hi - lo if hi > lo else 1.0
    n = len(actual)
    def pts(arr):
        p = []
        for i, v in enumerate(arr):
            x = pad + (i / (n - 1)) * (w - 2 * pad) if n > 1 else w / 2
            y = (h - pad) - ((v - lo) / rng) * (h - 2 * pad)
            p.append(f"{x:.1f},{y:.1f}")
        return " ".join(p)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline points="{pts(actual)}" fill="none" stroke="#9ca3af" stroke-width="1.5"/>'
            f'<polyline points="{pts(pred)}" fill="none" stroke="#16a34a" stroke-width="1.8"/>'
            f'</svg>')


def main():
    print("=" * 62)
    print("v3 모델 — 개체 추적(온실+라인+표본) + 개체 lag  (대상: 착과수)")
    print("=" * 62)
    df, feat = load_individual()
    plant_n = len([c for c in feat if "개체전" in c])
    print(f"[데이터] {len(df)}행(개체×조사) · feature {len(feat)}개 (그중 개체 lag {plant_n}개)")

    def mae_median(o):
        pm = o.groupby(["온실번호", "측정라인", "표본번호"]).apply(
            lambda s: np.mean(np.abs(s[TARGET] - s["예측"])), include_groups=False)
        return mean_absolute_error(o[TARGET], o["예측"]), float(np.median(pm))

    base = dict(objective="regression", n_estimators=400, learning_rate=0.05,
                num_leaves=31, min_child_samples=10, subsample=0.9, colsample_bytree=0.8)
    o0 = logo_oof(df, feat, base); bi, ba = scores(o0); bmae, bmed = mae_median(o0)
    print(f"[튜닝 전 L2] 개체 R² {bi:.4f} · 집계 R² {ba:.4f} · 전체 MAE {bmae:.3f} · MAE중앙 {bmed:.3f}")

    print(f"[탐색] huber + MAE 최소화 {N_TRIALS}회...")
    params = tune(df, feat)
    oof = logo_oof(df, feat, params)
    ind, agg = scores(oof); tmae, tmed = mae_median(oof)
    print(f"[튜닝 후 huber] 개체 R² {ind:.4f} · 집계 R² {agg:.4f} · "
          f"전체 MAE {tmae:.3f} · MAE중앙 {tmed:.3f}")

    # 최종 모델(전체 학습) 저장
    final = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
    final.fit(df[feat], df[TARGET])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "lgbm_fruit_count_v3.txt"
    final.booster_.save_model(str(out))
    print(f"[저장] v3 모델 → {out}")

    # 개체별 결과지
    oof["오차"] = (oof[TARGET] - oof["예측"]).abs()
    save = oof[["온실번호", "측정라인", "표본번호", "조사일자", TARGET, "예측", "오차"]].copy()
    save["조사일자"] = save["조사일자"].dt.date
    save = save.sort_values(["온실번호", "측정라인", "표본번호", "조사일자"])
    save.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[저장] 개체별 결과지 CSV → {CSV_OUT}")

    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(render_html(oof, ind, agg), encoding="utf-8")
    print(f"[저장] 개체별 결과지 HTML → {HTML_OUT}")

    print("-" * 62)
    print("[중요 feature TOP 8]")
    imp = pd.DataFrame({"feature": feat, "중요도": final.feature_importances_})
    print(imp.sort_values("중요도", ascending=False).head(8).to_string(index=False))
    print("=" * 62)


def render_html(oof, ind, agg):
    mae = mean_absolute_error(oof[TARGET], oof["예측"])
    plant_mae = oof.groupby(["온실번호", "측정라인", "표본번호"]).apply(
        lambda s: np.mean(np.abs(s[TARGET] - s["예측"])), include_groups=False)
    mae_med = float(np.median(plant_mae))
    n_plant = oof.groupby(["온실번호", "측정라인", "표본번호"]).ngroups

    # 개체별 요약 행
    rows = []
    grp = oof.sort_values("조사일자").groupby(["온실번호", "측정라인", "표본번호"])
    for (gh, ln, sp), sub in grp:
        a = sub[TARGET].to_numpy(); p = sub["예측"].to_numpy()
        pmae = np.mean(np.abs(a - p))
        spark = sparkline(a, p)
        mae_bg = "#e8f5e9" if pmae < 0.7 else ("#fff8e1" if pmae < 1.2 else "#fdeaea")
        rows.append(f"""<tr>
<td>{int(gh)}</td><td>{int(ln)}</td><td>{int(sp)}</td>
<td class="num">{a.mean():.2f}</td><td class="num strong">{p.mean():.2f}</td>
<td class="num" style="background:{mae_bg}">{pmae:.2f}</td>
<td class="spark">{spark}</td>
</tr>""")
    rows_html = "\n".join(rows)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>착과수 개체별 결과지 (v3)</title>
<style>
:root{{--bd:#e3e6ea;--mut:#6b7280;--ink:#111827;--accent:#16a34a}}
*{{box-sizing:border-box}}
body{{font-family:"Segoe UI","Malgun Gothic",system-ui,sans-serif;margin:0;background:#f7f8fa;color:var(--ink)}}
.wrap{{max-width:960px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 20px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}}
.chip{{background:#eef2ff;color:#3730a3;border-radius:999px;padding:4px 12px;font-size:12.5px;font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:14px;padding:16px 14px;text-align:center}}
.card .v{{font-size:26px;font-weight:800;line-height:1}}
.card .l{{font-size:12px;color:var(--mut);margin-top:6px}}
.card.main .v{{color:var(--accent)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:0 2px 14px;align-items:center}}
.legend .ln{{display:inline-block;width:18px;height:0;border-top:2px solid #9ca3af;vertical-align:middle;margin-right:4px}}
.legend .lp{{display:inline-block;width:18px;height:0;border-top:2px solid var(--accent);vertical-align:middle;margin-right:4px}}
.tablewrap{{max-height:74vh;overflow:auto;background:#fff;border:1px solid var(--bd);border-radius:14px}}
table{{border-collapse:collapse;width:100%;font-size:13px;min-width:640px}}
th,td{{padding:6px 10px;border-bottom:1px solid #eef0f3;text-align:center;white-space:nowrap}}
th{{background:#eef1f5;color:#374151;font-weight:700;position:sticky;top:0;z-index:2;box-shadow:inset 0 -1px 0 #d5d9df}}
td.num{{font-variant-numeric:tabular-nums}}
td.strong{{font-weight:700;color:var(--accent)}}
td.spark{{padding:2px 10px}}
.note{{margin-top:20px;font-size:12.5px;color:var(--mut);line-height:1.7;background:#fff;border:1px solid var(--bd);border-radius:12px;padding:14px 16px}}
a.back{{font-size:13px;color:#4f46e5;text-decoration:none}}
</style></head><body><div class="wrap">
<a class="back" href="../index.html">← 문서 인덱스</a>
<h1>🍓 착과수 개체별 결과지 <span style="font-size:15px;color:var(--mut)">v3 · 개체 추적</span></h1>
<p class="sub">(온실+측정라인+표본번호) = 하나의 개체로 보고, 그 개체 자신의 과거로 예측 (LOGO out-of-fold)</p>
<div class="chips">
<span class="chip">모델 LightGBM v3</span>
<span class="chip">손실 huber · MAE 최소화</span>
<span class="chip">개체 lag(자기 전주·전전주)</span>
<span class="chip">검증 LOGO (온실별)</span>
<span class="chip">개체 {n_plant}개 × 15주</span>
</div>
<div class="cards">
<div class="card main"><div class="v">{mae_med:.2f}</div><div class="l">개체 MAE 중앙값<br>(huber·MAE 튜닝)</div></div>
<div class="card"><div class="v">{ind:.2f}</div><div class="l">개체 R²<br>(개체 하나하나)</div></div>
<div class="card"><div class="v">{agg:.2f}</div><div class="l">집계 R²<br>(온실-날짜 평균)</div></div>
<div class="card"><div class="v">{n_plant}</div><div class="l">추적 개체 수<br>(온실당 10)</div></div>
</div>
<div class="legend">
<span><span class="ln"></span>실제 착과수</span>
<span><span class="lp"></span>예측 착과수</span>
<span>· 궤적은 15주간 변화 (왼쪽 정식 초기 → 오른쪽 최근)</span>
<span>· 오차 칸 색: 초록 낮음 / 노랑 중간 / 빨강 높음</span>
</div>
<div class="tablewrap"><table>
<thead><tr>
<th>온실</th><th>라인</th><th>표본</th><th>평균 실제</th><th>평균 예측</th><th>MAE</th><th>15주 궤적 (실제 vs 예측)</th>
</tr></thead>
<tbody>
{rows_html}
</tbody></table></div>
<div class="note">
<b>이 결과지가 v2.1과 다른 점</b> — v2.1은 '온실 전체 평균' 한 값을 예측했지만, v3은 <b>표본 개체 하나하나</b>를 예측합니다.
그 개체 자신의 지난주 착과수(<code>착과수_개체전주</code>)가 가장 강한 단서라, 개체 예측 R²가 <b>0.64 → {ind:.2f}</b>로 크게 올랐습니다.
집계로 올리면 R² <b>{agg:.2f}</b>로 v2.1(0.893)과 대등하면서, <b>개체 해상도</b>(어느 라인·표본이 부진한지)를 추가로 얻습니다.<br>
<b>정직한 한계</b> — 개체 40개·600행·온실 4개 LOGO 기준이라 데이터가 늘면 수치는 달라질 수 있습니다. 궤적의 실제선이 예측선과 붙어 있을수록 그 개체를 잘 맞힌 것입니다.
</div>
</div></body></html>"""


if __name__ == "__main__":
    main()
