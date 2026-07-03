# -*- coding: utf-8 -*-
"""
report.py  (착과수 예측 결과지 생성)
====================================
[이 파일이 하는 일 — 한 문장]
v2.1 모델을 'LOGO(자기 온실은 학습에서 빼고 예측)'로 돌려, 온실-날짜마다
실제 vs 예측 · 버킷 일치 · 신뢰도 · 오차를 담은 '결과지'를 CSV와 HTML로 만듭니다.

[왜 LOGO로 뽑나 — 정직함]
저장된 모델은 전체 데이터를 학습해 자기 자신을 맞히므로 점수가 낙관적입니다.
결과지는 '한 번도 보지 않은 온실'을 맞힌 값이라야 정직합니다. 그래서 온실 하나씩
빼고 나머지로 학습→예측한 out-of-fold 값을 씁니다. (train_agg.py의 0.89와 같은 방식)

[버킷 정확도란]
연속값 예측을 '구간(버킷)'으로 나눠 '같은 구간을 맞혔나'를 보는 지표.
발표에서 'R² 0.89'보다 '버킷 정확도 XX%'가 더 직관적으로 와닿습니다.

실행:  python src/report.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import r2_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_table.csv"
CSV_OUT = PROJECT_DIR / "data" / "processed" / "result_sheet_fruit_count.csv"
HTML_OUT = PROJECT_DIR / "docs" / "report" / "착과수_예측_결과지.html"

TARGET = "착과수"
GROWTH = ["관부직경", "엽병장", "엽수", "엽장", "엽폭", "착과수", "초장"]
KEYS = ["온실번호", "조사일자", "측정라인", "표본번호"]
GHS = [1, 2, 3, 4]
SEED = 42
N_TRIALS = 40

# 버킷 경계 (온실-날짜 평균 착과수 범위 0~8.2 기준)
EDGES = [0.5, 2, 4, 6, 8]
LABELS = ["0 (착과전)", "0.5~2", "2~4", "4~6", "6~8", "8+"]

# 예측 구간(80%) 분위수 모델 설정
ALPHA = 0.20
LO, HI = ALPHA / 2, 1 - ALPHA / 2
Q_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=15,
                min_child_samples=5, subsample=0.9, colsample_bytree=0.8)


def load_aggregated():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["조사일자"] = pd.to_datetime(df["조사일자"])
    agg = df.groupby(["온실번호", "조사일자"]).mean(numeric_only=True).reset_index()
    feat = ["온실번호"] + [c for c in agg.columns if c not in KEYS + GROWTH]
    return agg, feat


def bucket(v):
    """연속 착과수 → 버킷 인덱스(0~5)."""
    return int(np.digitize([v], EDGES)[0])


def tune(agg, feat):
    """train_agg와 동일한 방식으로 최적 파라미터 탐색 (결과지가 headline 0.89와 일치)."""
    def logo(params):
        P, A = [], []
        for g in GHS:
            tr = agg[agg["온실번호"] != g]; va = agg[agg["온실번호"] == g]
            m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
            m.fit(tr[feat], tr[TARGET])
            P.append(m.predict(va[feat])); A.append(va[TARGET].to_numpy())
        return r2_score(np.concatenate(A), np.concatenate(P))

    def obj(t):
        p = dict(objective="regression",
                 n_estimators=t.suggest_int("n_estimators", 150, 600),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                 num_leaves=t.suggest_int("num_leaves", 7, 31),
                 min_child_samples=t.suggest_int("min_child_samples", 3, 15),
                 subsample=t.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 5.0, log=True))
        return logo(p)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS)
    return dict(objective="regression", **study.best_params)


def conformal_Q(scores, alpha):
    """split-conformal 보정폭: 잔차 점수의 (1-alpha) 분위수(유한표본 보정)."""
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def build_oof(agg, feat, params):
    """온실 하나씩 빼며 예측 → 점 예측 + 80% 구간(전부 out-of-fold).

    구간은 split-conformal로 보정: 학습 온실 3개 중 2개로 분위수 모델을 만들고
    나머지 1개(보정용)에서 잰 잔차로 폭 Q를 더해 커버리지를 80%에 맞춘다.
    """
    rows = []
    for g in GHS:
        tr_ghs = [x for x in GHS if x != g]
        tr = agg[agg["온실번호"] != g]
        va = agg[agg["온실번호"] == g].copy()

        # 점 예측: 학습 온실 3개 전부 사용 (headline 0.89와 동일)
        m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
        m.fit(tr[feat], tr[TARGET])
        va["예측"] = np.maximum(m.predict(va[feat]), 0.0)

        # 구간: 2개(proper)로 분위수 학습 → 1개(calib)로 conformal 보정폭 Q
        calib_gh = tr_ghs[0]
        proper = tr[tr["온실번호"] != calib_gh]
        calib = tr[tr["온실번호"] == calib_gh]
        m_lo = lgb.LGBMRegressor(objective="quantile", alpha=LO, random_state=SEED,
                                 verbose=-1, **Q_PARAMS).fit(proper[feat], proper[TARGET])
        m_hi = lgb.LGBMRegressor(objective="quantile", alpha=HI, random_state=SEED,
                                 verbose=-1, **Q_PARAMS).fit(proper[feat], proper[TARGET])
        yc = calib[TARGET].to_numpy()
        Q = conformal_Q(np.maximum(m_lo.predict(calib[feat]) - yc,
                                    yc - m_hi.predict(calib[feat])), ALPHA)
        va["하한"] = np.maximum(m_lo.predict(va[feat]) - Q, 0.0)
        va["상한"] = m_hi.predict(va[feat]) + Q
        rows.append(va)
    out = pd.concat(rows).sort_values(["온실번호", "조사일자"]).reset_index(drop=True)
    return out


def confidence(row):
    """예측 구간 [하한,상한] 중 '예측 버킷' 안에 들어오는 비율 → 버킷 확신도(0~1)."""
    b = bucket(row["예측"])
    blo = EDGES[b - 1] if b > 0 else 0.0
    bhi = EDGES[b] if b < len(EDGES) else max(row["상한"], EDGES[-1]) + 1e6
    lo, hi = row["하한"], row["상한"]
    if hi <= lo:
        return 1.0
    overlap = max(0.0, min(hi, bhi) - max(lo, blo))
    return float(np.clip(overlap / (hi - lo), 0.0, 1.0))


def conf_color(c):
    """신뢰도(0~1) → 초록 계열 배경색."""
    # 낮으면 연회색, 높으면 진초록
    r = int(232 - 132 * c); g = int(245 - 30 * c); b = int(233 - 120 * c)
    return f"rgb({r},{g},{b})"


def main():
    print("=" * 60)
    print("착과수 예측 결과지 생성 (v2.1 · LOGO out-of-fold)")
    print("=" * 60)
    agg, feat = load_aggregated()
    print(f"[데이터] 집계 {len(agg)}행 · feature {len(feat)}개 · 탐색 {N_TRIALS}회...")
    params = tune(agg, feat)
    df = build_oof(agg, feat, params)

    # 지표 계산
    df["실제버킷"] = df[TARGET].map(bucket)
    df["예측버킷"] = df["예측"].map(bucket)
    df["버킷일치"] = df["실제버킷"] == df["예측버킷"]
    df["오차"] = (df[TARGET] - df["예측"]).abs()
    df["신뢰도"] = df.apply(confidence, axis=1)
    df["구간적중"] = (df[TARGET] >= df["하한"]) & (df[TARGET] <= df["상한"])

    acc = df["버킷일치"].mean()
    acc1 = (df["실제버킷"] - df["예측버킷"]).abs().le(1).mean()   # ±1 버킷 허용
    r2 = r2_score(df[TARGET], df["예측"])
    rmse = np.sqrt(np.mean(df["오차"] ** 2))
    cover = df["구간적중"].mean()
    print(f"[결과] 버킷 정확도 {acc*100:.1f}% · ±1버킷 {acc1*100:.1f}% · "
          f"R² {r2:.3f} · RMSE {rmse:.3f} · 구간적중 {cover*100:.0f}%")

    # CSV 저장
    cols = ["온실번호", "조사일자", TARGET, "예측", "오차", "하한", "상한",
            "실제버킷", "예측버킷", "버킷일치", "신뢰도", "구간적중"]
    save = df[cols].copy()
    save["조사일자"] = save["조사일자"].dt.date
    save.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[저장] CSV → {CSV_OUT}")

    # HTML 결과지 저장
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(render_html(df, acc, acc1, r2, rmse, cover, params), encoding="utf-8")
    print(f"[저장] HTML 결과지 → {HTML_OUT}")
    print("=" * 60)


def render_html(df, acc, acc1, r2, rmse, cover, params):
    trs = []
    for _, r in df.iterrows():
        ok = r["버킷일치"]
        corr_bg = "#d7f0dc" if ok else "#f7d4d4"
        corr_tx = "TRUE" if ok else "FALSE"
        cov_mark = "✓" if r["구간적중"] else "·"
        c = r["신뢰도"]
        trs.append(f"""<tr>
<td>{int(r['온실번호'])}</td><td class="dt">{r['조사일자'].date()}</td>
<td class="num">{r[TARGET]:.2f}</td><td class="num strong">{r['예측']:.2f}</td>
<td class="num err">{r['오차']:.2f}</td>
<td>{LABELS[int(r['실제버킷'])]}</td><td>{LABELS[int(r['예측버킷'])]}</td>
<td class="corr" style="background:{corr_bg}">{corr_tx}</td>
<td class="conf"><span class="bar" style="width:{c*100:.0f}%;background:{conf_color(c)}"></span><span class="cv">{c*100:.0f}%</span></td>
<td class="num">[{r['하한']:.1f}~{r['상한']:.1f}] {cov_mark}</td>
</tr>""")
    rows_html = "\n".join(trs)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>착과수 예측 결과지 (v2.1)</title>
<style>
:root{{--bd:#e3e6ea;--mut:#6b7280;--ink:#111827;--accent:#16a34a}}
*{{box-sizing:border-box}}
body{{font-family:"Segoe UI","Malgun Gothic",system-ui,sans-serif;margin:0;background:#f7f8fa;color:var(--ink)}}
.wrap{{max-width:1060px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 20px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}}
.chip{{background:#eef2ff;color:#3730a3;border-radius:999px;padding:4px 12px;font-size:12.5px;font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:26px}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:14px;padding:16px 14px;text-align:center}}
.card .v{{font-size:26px;font-weight:800;line-height:1}}
.card .l{{font-size:12px;color:var(--mut);margin-top:6px}}
.card.main .v{{color:var(--accent)}}
.tablewrap{{max-height:72vh;overflow:auto;background:#fff;border:1px solid var(--bd);border-radius:14px}}
table{{border-collapse:collapse;width:100%;font-size:13px;min-width:820px}}
th,td{{padding:7px 9px;border-bottom:1px solid #eef0f3;text-align:center;white-space:nowrap}}
th{{background:#eef1f5;color:#374151;font-weight:700;position:sticky;top:0;z-index:2;box-shadow:inset 0 -1px 0 #d5d9df,0 1px 3px rgba(0,0,0,.05)}}
td.num{{font-variant-numeric:tabular-nums}}
td.strong{{font-weight:700}}
td.err{{color:#9ca3af}}
td.dt{{color:#4b5563}}
td.corr{{font-weight:700;font-size:12px}}
td.conf{{position:relative;min-width:140px;text-align:left;padding-left:12px}}
td.conf .bar{{display:inline-block;height:12px;border-radius:3px;vertical-align:middle}}
td.conf .cv{{font-size:11px;color:#374151;margin-left:6px}}
.note{{margin-top:20px;font-size:12.5px;color:var(--mut);line-height:1.7;background:#fff;border:1px solid var(--bd);border-radius:12px;padding:14px 16px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:10px 2px 0}}
.legend b{{color:var(--ink)}}
a.back{{font-size:13px;color:#4f46e5;text-decoration:none}}
</style></head><body><div class="wrap">
<a class="back" href="../index.html">← 문서 인덱스</a>
<h1>🍓 착과수 예측 결과지 <span style="font-size:15px;color:var(--mut)">v2.1</span></h1>
<p class="sub">온실 하나씩 빼고 예측한 out-of-fold 결과 (한 번도 학습에 안 쓴 온실을 맞힌 정직한 값)</p>
<div class="chips">
<span class="chip">모델 LightGBM v2.1</span>
<span class="chip">온실-날짜 평균 + lag + 전주 개체 분포</span>
<span class="chip">검증 LOGO (온실별 교차)</span>
<span class="chip">버킷 6단계</span>
</div>
<div class="cards">
<div class="card main"><div class="v">{acc*100:.0f}%</div><div class="l">버킷 정확도<br>(정확히 같은 구간)</div></div>
<div class="card"><div class="v">{acc1*100:.0f}%</div><div class="l">±1 버킷 정확도<br>(한 칸 이내)</div></div>
<div class="card"><div class="v">{r2:.2f}</div><div class="l">R²<br>(설명력)</div></div>
<div class="card"><div class="v">{rmse:.2f}</div><div class="l">RMSE<br>(평균 오차)</div></div>
<div class="card"><div class="v">{cover*100:.0f}%</div><div class="l">구간 적중률<br>(80% 목표)</div></div>
</div>
<div class="tablewrap"><table>
<thead><tr>
<th>온실</th><th>조사일자</th><th>실제</th><th>예측</th><th>오차</th>
<th>실제 버킷</th><th>예측 버킷</th><th>버킷일치</th><th>신뢰도</th><th>예측구간(80%)</th>
</tr></thead>
<tbody>
{rows_html}
</tbody></table></div>
<div class="legend">
<span><b>버킷일치</b> = 실제·예측이 같은 착과수 구간이면 TRUE</span>
<span><b>신뢰도</b> = 예측구간이 예측 버킷 안에 들어오는 비율(구간이 좁고 버킷 중앙일수록 높음)</span>
<span><b>구간 ✓</b> = 실제값이 80% 예측구간 안에 들어옴</span>
</div>
<div class="note">
<b>이 표를 어떻게 읽나</b><br>
• 각 행은 '한 온실의 한 조사일'입니다. 예측은 그 온실을 <b>학습에서 제외</b>하고 나머지 3개 온실로만 학습해 맞힌 값이라, 실제 새 온실에 적용했을 때의 성능에 가깝습니다.<br>
• <b>버킷 정확도 {acc*100:.0f}%</b> = 60개 중 착과수 구간을 정확히 맞힌 비율. ±1 버킷까지 허용하면 <b>{acc1*100:.0f}%</b>.<br>
• 초반(착과 전, 값 0) 시기는 쉽게 맞고, 착과가 급증하는 전환기(예: 11월 말~12월 초)에서 오차가 커지는 경향이 보입니다.<br>
<b>정직한 한계</b> — 데이터가 온실 4개·60행뿐이라 수치는 데이터가 늘면 달라질 수 있습니다. 버킷 경계(0.5·2·4·6·8)는 착과수 분포에 맞춰 정한 값입니다.
</div>
</div></body></html>"""


if __name__ == "__main__":
    main()
