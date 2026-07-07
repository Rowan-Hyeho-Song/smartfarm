# -*- coding: utf-8 -*-
"""
report_fruit_setting_2022.py  (신규 데이터 · 주차별 착과수 결과지)
================================================================
[이 파일이 하는 일 — 한 문장]
착과수 최적환경(2022) 데이터를 '주차(조사기간) 단위'로, forward-chaining(과거로
미래 1주 예측)의 실제 vs 예측·오차·적중을 담은 결과지(HTML·CSV)로 만든다.

[왜 forward-chaining인가]
계단형 타깃이라 랜덤 분할은 누수. 앞 8주 학습 후 한 주씩 앞으로 예측한 값이라야
'실전(다음 주 예측)'에 정직하다. 앞 8주는 학습 워밍업이라 예측 공란.

[적중 기준]
착과수는 0~8.75 소수 평균값이라, 회귀 오차와 함께 '±0.5 이내/±1.0 이내 적중률'을 본다.

실행:  python src/report_fruit_setting_2022.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "STRAWBERRY_FRUIT_SETTING_ENV_20221209.csv"
CSV_OUT = PROJECT_DIR / "data" / "processed" / "result_sheet_fruit_2022.csv"
HTML_OUT = PROJECT_DIR / "docs" / "report" / "착과수_결과지_신규2022.html"
TARGET = "FRST_TREE_CNT"
START = 8
PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.05, num_leaves=31,
              min_child_samples=20, subsample=0.9, colsample_bytree=0.8, random_state=42, verbose=-1)


def build_table():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["STRG_DT"] = pd.to_datetime(df["STRG_DT"]); df = df.sort_values("STRG_DT").reset_index(drop=True)
    df["seg"] = (df[TARGET] != df[TARGET].shift()).cumsum()
    feat = [c for c in df.columns if c not in ["ZONE_NM", "STRG_DT", TARGET, "seg"]]
    order = df.drop_duplicates("seg")["seg"].tolist()
    seg_val = df.drop_duplicates("seg").set_index("seg")[TARGET]
    seg_start = df.groupby("seg")["STRG_DT"].min()

    rows = []
    for i, s in enumerate(order):
        actual = float(seg_val.loc[s])
        pred = persist = np.nan
        if i >= START:
            tr = df[df["seg"].isin(order[:i])]
            m = lgb.LGBMRegressor(**PARAMS).fit(tr[feat], tr[TARGET])
            pred = float(m.predict(df[df["seg"] == s][feat]).mean())
            persist = float(seg_val.loc[order[i - 1]])
        rows.append(dict(주차=i + 1, 시작일=seg_start.loc[s].date(), 실제=round(actual, 2),
                         예측=(round(pred, 2) if pred == pred else None),
                         지속성=(round(persist, 2) if persist == persist else None),
                         오차=(round(abs(actual - pred), 2) if pred == pred else None)))
    return pd.DataFrame(rows)


def sparkline(actual, pred, w=760, h=150, pad=28):
    """27주 궤적: 실제(회색) vs forward 예측(보라). None은 끊어 그림."""
    xs = np.arange(len(actual))
    vals = [v for v in list(actual) + list(pred) if v is not None]
    lo, hi = 0.0, max(vals) * 1.1
    rng = hi - lo if hi > lo else 1.0
    def pts(arr):
        seg, out = [], []
        for i, v in enumerate(arr):
            if v is None:
                if seg: out.append(seg); seg = []
                continue
            x = pad + (i / (len(arr) - 1)) * (w - 2 * pad)
            y = (h - pad) - ((v - lo) / rng) * (h - 2 * pad)
            seg.append(f"{x:.1f},{y:.1f}")
        if seg: out.append(seg)
        return out
    def polylines(arr, color, wd):
        return "".join(f'<polyline points="{" ".join(s)}" fill="none" stroke="{color}" stroke-width="{wd}"/>'
                       for s in pts(arr))
    # y축 눈금
    grid = "".join(f'<line x1="{pad}" y1="{(h-pad)-(g/rng)*(h-2*pad):.1f}" x2="{w-pad}" '
                   f'y2="{(h-pad)-(g/rng)*(h-2*pad):.1f}" stroke="#242a38" stroke-width="1"/>'
                   f'<text x="{pad-6}" y="{(h-pad)-(g/rng)*(h-2*pad)+3:.1f}" text-anchor="end" '
                   f'fill="#7b8294" font-size="10">{g:g}</text>'
                   for g in [0, 2, 4, 6, 8])
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="주차별 실제 vs 예측 착과수">'
            f'{grid}{polylines(list(actual),"#9ca3af",2)}{polylines(list(pred),"#b69bff",2.4)}</svg>')


CSS = """
:root{--bg:#0e1014;--card:#1b1f2a;--card-2:#222736;--line:#2c3242;--text:#e7e9ee;
--text-dim:#a4abba;--text-faint:#7b8294;--berry:#ff5d73;--berry-soft:#ff8a9c;--leaf:#5fd39a;
--gold:#ffc861;--sky:#5cc8ff;--violet:#b69bff;}
*{box-sizing:border-box;}
body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#1d2230 0%,var(--bg) 55%);
color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Malgun Gothic","Segoe UI",Roboto,sans-serif;line-height:1.7;-webkit-font-smoothing:antialiased;}
.wrap{max-width:900px;margin:0 auto;padding:40px 22px 90px;}
a.back{font-size:13.5px;color:var(--sky);text-decoration:none;}
h1{font-size:clamp(23px,4vw,32px);margin:14px 0 4px;font-weight:800;letter-spacing:-.01em;}
h1 .v{font-size:15px;color:var(--text-dim);font-weight:600;}
.sub{color:var(--text-dim);font-size:14.5px;margin:0 0 20px;}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;}
.chip{background:rgba(182,155,255,.1);color:#c9b6ff;border:1px solid rgba(182,155,255,.3);border-radius:999px;padding:4px 12px;font-size:12.5px;font-weight:600;}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px;}
@media(max-width:620px){.cards{grid-template-columns:repeat(2,1fr);}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 14px;text-align:center;}
.card .n{font-size:25px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;}
.card.m .n{color:var(--violet);}.card.a .n{color:var(--leaf);}.card.b .n{color:var(--sky);}.card.c .n{color:var(--gold);}
.card .l{font-size:12px;color:var(--text-faint);margin-top:6px;line-height:1.4;}
.fig{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 12px;overflow-x:auto;}
.fig svg{display:block;width:100%;height:auto;min-width:560px;}
.legend{display:flex;gap:16px;justify-content:center;font-size:12.5px;color:var(--text-dim);margin-top:8px;}
.legend .ln{display:inline-block;width:16px;height:0;border-top:2px solid #9ca3af;vertical-align:middle;margin-right:5px;}
.legend .lp{display:inline-block;width:16px;height:0;border-top:2px solid var(--violet);vertical-align:middle;margin-right:5px;}
.tablewrap{max-height:70vh;overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:14px;margin-top:22px;}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px;}
th,td{padding:8px 12px;border-bottom:1px solid #242a38;text-align:center;white-space:nowrap;}
th{background:#23283a;color:var(--berry-soft);font-weight:700;position:sticky;top:0;z-index:2;}
td.num{font-variant-numeric:tabular-nums;} td.d{color:var(--text-dim);}
td.strong{font-weight:700;color:var(--violet);}
.note{margin-top:20px;font-size:13px;color:var(--text-dim);line-height:1.7;background:var(--card);border:1px solid var(--line);border-left:3px solid var(--gold);border-radius:12px;padding:14px 16px;}
.note b{color:var(--text);}
"""


def render(tbl, r2, mae, r2p, hit05, hit10, n_eval):
    spark = sparkline(tbl["실제"].tolist(),
                      [v if v == v else None for v in tbl["예측"].tolist()])
    trs = []
    for _, r in tbl.iterrows():
        p = "—" if r["예측"] is None else f'{r["예측"]:.2f}'
        e = "" if r["오차"] is None else f'{r["오차"]:.2f}'
        ecls = ""
        if r["오차"] is not None:
            ecls = "background:rgba(95,211,154,.14)" if r["오차"] <= 0.5 else (
                "background:rgba(255,200,97,.12)" if r["오차"] <= 1.0 else "background:rgba(255,93,115,.12)")
        warm = ' class="d"' if r["예측"] is None else ""
        trs.append(f'<tr{warm}><td class="num">{r["주차"]}</td><td class="d">{r["시작일"]}</td>'
                   f'<td class="num">{r["실제"]:.2f}</td><td class="num strong">{p}</td>'
                   f'<td class="num" style="{ecls}">{e}</td></tr>')
    rows = "\n".join(trs)
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>착과수 결과지 — 신규 데이터(2022) forward-chaining</title>
<style>{CSS}</style></head><body><div class="wrap">
<a class="back" href="../index.html">← 문서 인덱스</a>
<h1>🍓 착과수 결과지 <span class="v">신규 데이터 2022 · forward-chaining</span></h1>
<p class="sub">단일 온실(ZONE 66)·주차 단위. 앞 8주 학습 후 한 주씩 '미래 1주'를 예측한 정직한 결과지.</p>
<div class="chips"><span class="chip">주차 {len(tbl)}개</span><span class="chip">예측 평가 {n_eval}주</span>
<span class="chip">forward-chaining</span><span class="chip">LightGBM</span></div>
<div class="cards">
<div class="card m"><div class="n">{r2:.2f}</div><div class="l">forward R²<br>(미래 1주 예측)</div></div>
<div class="card a"><div class="n">{mae:.2f}</div><div class="l">MAE<br>(평균 오차, 개)</div></div>
<div class="card b"><div class="n">{hit05:.0f}%</div><div class="l">±0.5 이내<br>적중률</div></div>
<div class="card c"><div class="n">{hit10:.0f}%</div><div class="l">±1.0 이내<br>적중률</div></div>
</div>
<div class="fig">{spark}
<div class="legend"><span><span class="ln"></span>실제 착과수</span><span><span class="lp"></span>forward 예측</span></div>
</div>
<div class="tablewrap"><table><thead><tr><th>주차</th><th>시작일</th><th>실제</th><th>예측</th><th>오차</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class="note"><b>읽는 법</b> — 예측선(보라)이 실제선(회색)에 붙을수록 잘 맞힌 것. 오차 칸 색: 초록 ±0.5 이내 · 노랑 ±1.0 · 빨강 그 이상.
앞 8주는 학습 워밍업이라 공란(예측 없음). <b>정직한 한계</b>: forward R² {r2:.2f}는 '직전 주값 그대로'(지속성 {r2p:.2f})와 사실상 동급 —
환경·제어가 발달단계·자기상관 위에 추가정보를 별로 못 준다는 뜻(예측 트랙 결론과 일치). n=27주·단일 구역 기준.</div>
</div></body></html>"""


def main():
    tbl = build_table()
    ev = tbl.dropna(subset=["예측"])
    r2 = r2_score(ev["실제"], ev["예측"]); mae = mean_absolute_error(ev["실제"], ev["예측"])
    r2p = r2_score(ev["실제"], ev["지속성"])
    hit05 = (ev["오차"] <= 0.5).mean() * 100
    hit10 = (ev["오차"] <= 1.0).mean() * 100
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(render(tbl, r2, mae, r2p, hit05, hit10, len(ev)), encoding="utf-8")
    print("=" * 60)
    print("착과수 결과지(신규 2022) 생성")
    print(f"  주차 {len(tbl)}개 · 예측평가 {len(ev)}주")
    print(f"  forward R² {r2:.3f} · MAE {mae:.3f} · 지속성 R² {r2p:.3f}")
    print(f"  ±0.5 적중 {hit05:.0f}% · ±1.0 적중 {hit10:.0f}%")
    print(f"  CSV  → {CSV_OUT}")
    print(f"  HTML → {HTML_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
