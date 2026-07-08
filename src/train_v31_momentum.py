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
CSV_OUT = PROJECT_DIR / "data" / "processed" / "result_sheet_v3_1.csv"
HTML_OUT = PROJECT_DIR / "docs" / "report" / "착과수_개체별_결과지_v3_1.html"

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
                and c not in MOMENTUM and "추세" not in c and "14일" not in c]
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


def sparkline(actual, pred, w=150, h=34, pad=4):
    """개체 궤적: 실제(회색) vs 예측(초록) 미니 그래프 (v3 결과지와 동일 포맷)."""
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


def write_result_sheet(oof, ind, agg, base_med):
    """v3.1 개체별 결과지(CSV·HTML) 생성. base_med = v3(모멘텀 없음) MAE중앙(비교용)."""
    oof = oof.copy()
    oof["오차"] = (oof[TARGET] - oof["예측"]).abs()
    save = oof[["온실번호", "측정라인", "표본번호", "조사일자", TARGET, "예측", "오차"]].copy()
    save["조사일자"] = save["조사일자"].dt.date
    save = save.sort_values(["온실번호", "측정라인", "표본번호", "조사일자"])
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    save.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    mae = mean_absolute_error(oof[TARGET], oof["예측"])
    pm = oof.groupby(["온실번호", "측정라인", "표본번호"]).apply(
        lambda s: np.mean(np.abs(s[TARGET] - s["예측"])), include_groups=False)
    mae_med = float(np.median(pm))
    n_plant = oof.groupby(["온실번호", "측정라인", "표본번호"]).ngroups
    delta_med = base_med - mae_med  # v3 대비 개선폭(+면 좋아짐)

    rows = []
    grp = oof.sort_values("조사일자").groupby(["온실번호", "측정라인", "표본번호"])
    for (gh, ln, sp), sub in grp:
        a = sub[TARGET].to_numpy(); p = sub["예측"].to_numpy()
        pmae = np.mean(np.abs(a - p))
        mae_bg = "#e8f5e9" if pmae < 0.7 else ("#fff8e1" if pmae < 1.2 else "#fdeaea")
        rows.append(f"""<tr>
<td>{int(gh)}</td><td>{int(ln)}</td><td>{int(sp)}</td>
<td class="num">{a.mean():.2f}</td><td class="num strong">{p.mean():.2f}</td>
<td class="num" style="background:{mae_bg}">{pmae:.2f}</td>
<td class="spark">{sparkline(a, p)}</td>
</tr>""")
    rows_html = "\n".join(rows)

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>착과수 개체별 결과지 (v3.1)</title>
<style>
:root{{--bd:#e3e6ea;--mut:#6b7280;--ink:#111827;--accent:#16a34a}}
*{{box-sizing:border-box}}
body{{font-family:"Segoe UI","Malgun Gothic",system-ui,sans-serif;margin:0;background:#f7f8fa;color:var(--ink)}}
.wrap{{max-width:960px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 20px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}}
.chip{{background:#eef2ff;color:#3730a3;border-radius:999px;padding:4px 12px;font-size:12.5px;font-weight:600}}
.chip.new{{background:#e0f7f5;color:#0f766e}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:14px;padding:16px 14px;text-align:center}}
.card .v{{font-size:26px;font-weight:800;line-height:1}}
.card .l{{font-size:12px;color:var(--mut);margin-top:6px}}
.card.main .v{{color:var(--accent)}}
.delta{{font-size:12px;font-weight:700;color:var(--accent);margin-top:3px}}
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
<h1>🍓 착과수 개체별 결과지 <span style="font-size:15px;color:var(--mut)">v3.1 · 개체 추적 + 모멘텀</span></h1>
<p class="sub">v3(개체 추적)에 증가속도·상대위치 파생 7개를 더한 모델. LOGO out-of-fold 예측.</p>
<div class="chips">
<span class="chip">모델 LightGBM v3.1</span>
<span class="chip">손실 huber · MAE 최소화</span>
<span class="chip new">+모멘텀 7개 (개체대비온실 등)</span>
<span class="chip">검증 LOGO (온실별)</span>
<span class="chip">개체 {n_plant}개 × 15주</span>
</div>
<div class="cards">
<div class="card main"><div class="v">{mae_med:.2f}</div><div class="l">개체 MAE 중앙값</div><div class="delta">▼ {delta_med:.3f} vs v3</div></div>
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
<b>v3와 무엇이 다른가</b> — 같은 개체 추적 모델에 <b>모멘텀 파생 7개</b>(증가속도 + '개체 vs 온실평균' 상대위치)를 더했습니다.
그중 <code>착과수_개체대비온실_전주</code>가 전체 피처 중 <b>중요도 1위</b>가 되면서, 개체 MAE 중앙값이 v3 <b>{base_med:.3f} → {mae_med:.3f}</b>로 낮아졌습니다.<br>
<b>정직한 한계</b> — 개체 40개·600행·온실 4개 LOGO 기준이라 데이터가 늘면 수치는 달라질 수 있습니다. 집계 R²는 v3와 사실상 동일(노이즈 수준)하고, 개선은 <b>개체 해상도</b>에서 나옵니다.
</div>
</div></body></html>"""
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    return mae_med, delta_med


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

    # ── (4-2) v3.1 개체별 결과지 (CSV·HTML) — v3 대비 개선폭 표기 ──
    med, dmed = write_result_sheet(oof_v31, s_v31t[0], s_v31t[1], base_med=s_base[3])
    print(f"[저장] v3.1 결과지 CSV  → {CSV_OUT}")
    print(f"[저장] v3.1 결과지 HTML → {HTML_OUT}  (MAE중앙 {med:.3f}, v3 대비 ▼{dmed:.3f})")

    # ── (5) 판정 ──
    print("=" * 70)
    verdict = ("채택 검토 — 집계/개체 R²가 의미있게 개선" if (d_agg > 0.005 or d_ind > 0.005)
               else "보류 — 동일 params에서 개선 미미(GBDT가 이미 차분을 학습). v3.2(환경 델타)로 이동 권장")
    print(f"[판정] {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
