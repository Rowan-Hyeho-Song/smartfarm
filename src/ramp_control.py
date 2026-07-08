# -*- coding: utf-8 -*-
"""
ramp_control.py  (제어 실행 정책 · 점진 보정 rate-limiter)
========================================================
[이 파일이 하는 일 — 한 문장]
"편차를 한 번에 되돌리지 말고, 하루 변화폭 상한을 두고 2~5일에 걸쳐 나눠 보정하라"는
온실 기후제어 원리를 rate-limiter로 형식화하고, 급격 vs 점진 스케줄을 비교·시각화한다.

[왜 필요한가 — 층위]
기존 제어 트랙(control_recommend·룰베이스)은 '무엇을 어느 방향으로' 바꿀지(방향)를 준다.
이 모듈은 그 위에 얹는 '어떻게' 층 — 목표 편차를 며칠에 걸쳐 얼마씩 옮길지(실행).

[핵심 원리]
당일 주간 온도가 평소보다 1~2℃ 높게 측정됐다고, 그날 밤 바로 1~2℃ 낮추면 주야 스윙이
급격해 식물에 열 스트레스가 누적된다. 대신 하루 상한(cap, 예: 0.5℃/일)을 두고 2~5일에
나눠 완만히 전이하면 같은 목표에 도달하면서 스트레스를 크게 줄인다.

실행:  python src/ramp_control.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
HTML_OUT = PROJECT_DIR / "docs" / "control" / "점진제어_ramp.html"

# ── 기본 정책 파라미터 ──
CAP_PER_DAY = 0.5     # 야간온도 일일 변화 상한 (℃/일)
MIN_DAYS = 2          # 최소 분산 일수
MAX_DAYS = 5          # 최대 분산 일수
DIF_SAFE = (5.0, 8.0) # 주야온도차(DIF) 스트레스 안전밴드 (℃) — 참고용


def ramp_schedule(current, target, cap=CAP_PER_DAY, min_days=MIN_DAYS, max_days=MAX_DAYS):
    """
    current → target 로 '하루 최대 cap씩' 완만히 옮기는 일별 설정값 스케줄.

    - 필요한 일수 = ceil(|편차|/cap), 단 [min_days, max_days]로 제한.
    - max_days로도 cap을 못 지키면(편차가 너무 큼) 그 사실을 meta.capped_exceeded=True로 알림.
    반환: (설정값 배열[day1..dayN], meta dict)
    """
    delta = float(target) - float(current)
    if abs(delta) < 1e-9:
        return np.array([float(current)]), dict(days=1, step=0.0, delta=0.0,
                                                 capped_exceeded=False)
    days = int(np.clip(int(np.ceil(abs(delta) / cap)), min_days, max_days))
    step = delta / days
    setpoints = float(current) + step * np.arange(1, days + 1)
    return setpoints, dict(days=days, step=step, delta=delta,
                           capped_exceeded=abs(step) > cap + 1e-9)


def compare(current, target, cap=CAP_PER_DAY):
    """급격(1일) vs 점진(rate-limited) 비교 지표."""
    sched, meta = ramp_schedule(current, target, cap)
    abrupt_peak = abs(target - current)          # 하루에 다 바꿀 때의 일일 변화폭
    ramp_peak = abs(meta["step"])                # 점진일 때의 일일 변화폭
    # 스트레스 프록시: 일일 변화폭은 초과분에 대해 초선형으로 부담이 커진다고 보고 제곱 합.
    def stress(daily_changes):
        return float(np.sum(np.square(np.maximum(np.abs(daily_changes) - cap, 0.0))))
    abrupt_stress = stress([abrupt_peak])
    ramp_stress = stress(np.diff(np.concatenate([[current], sched])))
    return sched, meta, dict(abrupt_peak=abrupt_peak, ramp_peak=ramp_peak,
                             abrupt_stress=abrupt_stress, ramp_stress=ramp_stress)


def fmt_sched(current, sched):
    seq = " → ".join([f"{current:.1f}"] + [f"{v:.1f}" for v in sched])
    return seq


def main():
    print("=" * 68)
    print("점진 제어 (ramp-rate 제한) — 급격 vs 점진 야간온도 보정")
    print("=" * 68)
    print(f"[정책] 일일 상한 {CAP_PER_DAY}℃/일 · 분산 {MIN_DAYS}~{MAX_DAYS}일 · "
          f"DIF 안전밴드 {DIF_SAFE[0]}~{DIF_SAFE[1]}℃")
    print("-" * 68)

    night0 = 8.0  # 평소 야간 설정온도(겨울 딸기 예시)
    scenarios = [1.0, 1.5, 2.0]  # 주간이 평소보다 +Δ℃ 높게 측정 → 야간을 Δ만큼 낮추고 싶은 상황
    rows = []
    for d in scenarios:
        target = night0 - d
        sched, meta, m = compare(night0, target)
        print(f"[주간 +{d:.1f}℃ 이상 감지 → 야간 목표 {target:.1f}℃]")
        print(f"  급격(1일): {night0:.1f} → {target:.1f}  일일변화 {m['abrupt_peak']:.1f}℃  (스트레스 프록시 {m['abrupt_stress']:.2f})")
        print(f"  점진({meta['days']}일): {fmt_sched(night0, sched)}  일일변화 {m['ramp_peak']:.2f}℃  (스트레스 프록시 {m['ramp_stress']:.2f})")
        print()
        rows.append((d, target, meta, m, sched))

    write_doc(night0, rows)
    print("-" * 68)
    print(f"[저장] 점진 제어 문서 → {HTML_OUT}")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# 문서(HTML) 생성 — 다크 하우스 스타일 + 급격 vs 점진 궤적 SVG
# ─────────────────────────────────────────────────────────────────────────────
def _svg_compare(night0, target, sched, w=560, h=250, pad=44):
    days = len(sched)
    xmax = max(days, MAX_DAYS)
    xs = list(range(0, xmax + 1))
    lo = min(night0, target) - 0.6
    hi = max(night0, target) + 0.6
    def X(i): return pad + i / xmax * (w - 2 * pad)
    def Y(v): return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
    # 급격: day0 그대로, day1에 target, 이후 유지
    abrupt = [night0] + [target] * xmax
    # 점진: day0 current, day1..N ramp, 이후 target 유지
    ramp = [night0] + list(sched) + [target] * (xmax - days)
    def poly(vals, color, wdt, dash=""):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{wdt}"{d}/>'
    # 축·격자
    grid = "".join(f'<line x1="{pad}" y1="{Y(v):.1f}" x2="{w-pad}" y2="{Y(v):.1f}" stroke="#2c3242" stroke-width="1"/>'
                   f'<text x="{pad-8}" y="{Y(v)+4:.1f}" text-anchor="end" fill="#7b8294" font-size="11">{v:.1f}</text>'
                   for v in np.arange(np.ceil(lo), np.floor(hi) + 0.01, 1.0))
    xlab = "".join(f'<text x="{X(i):.1f}" y="{h-pad+18:.1f}" text-anchor="middle" fill="#7b8294" font-size="11">D{i}</text>'
                   for i in xs)
    dots = "".join(f'<circle cx="{X(i+1):.1f}" cy="{Y(v):.1f}" r="3.2" fill="#5fd39a"/>'
                   for i, v in enumerate(sched))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="급격 vs 점진 야간온도 보정">'
            f'{grid}{xlab}'
            f'{poly(abrupt, "#ff5d73", 2.4, "6 5")}'
            f'{poly(ramp, "#5fd39a", 2.6)}{dots}'
            f'<text x="{w-pad}" y="{Y(target)-8:.1f}" text-anchor="end" fill="#5fd39a" font-size="12" font-weight="700">점진(안전)</text>'
            f'<text x="{X(1)+6:.1f}" y="{Y(target)+16:.1f}" fill="#ff5d73" font-size="12" font-weight="700">급격(스트레스)</text>'
            f'</svg>')


def write_doc(night0, rows):
    # 대표 시나리오: +1.5℃
    rep = [r for r in rows if abs(r[0] - 1.5) < 1e-6][0]
    _, rep_target, rep_meta, rep_m, rep_sched = rep
    svg = _svg_compare(night0, rep_target, rep_sched)

    srows = "".join(
        f"<tr><td class='term'>주간 +{d:.1f}℃</td><td>{night0:.1f} → {tg:.1f}℃</td>"
        f"<td class='neg'>{m['abrupt_peak']:.1f}℃ / 1일</td>"
        f"<td class='pos'>{m['ramp_peak']:.2f}℃ / {meta['days']}일</td>"
        f"<td>{fmt_sched(night0, sc)}</td></tr>"
        for (d, tg, meta, m, sc) in rows)

    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>제어 — 점진 보정(ramp-rate 제한)</title>
<style>
  :root{{--bg:#0e1014;--card:#1b1f2a;--card-2:#222736;--line:#2c3242;--text:#e7e9ee;
    --text-dim:#a4abba;--text-faint:#7b8294;--berry:#ff5d73;--berry-soft:#ff8a9c;
    --leaf:#5fd39a;--gold:#ffc861;--sky:#5cc8ff;--violet:#b69bff;--radius:16px;--maxw:900px;}}
  *{{box-sizing:border-box;}} a{{color:inherit;text-decoration:none;}}
  body{{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#1d2230 0%,var(--bg) 55%);
    color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Malgun Gothic","Segoe UI",Roboto,sans-serif;line-height:1.75;}}
  .wrap{{max-width:var(--maxw);margin:0 auto;padding:0 22px 120px;}}
  header.hero{{max-width:var(--maxw);margin:0 auto;padding:56px 22px 22px;}}
  .back{{font-size:13.5px;color:var(--sky);margin-bottom:16px;display:inline-block;}}
  .eyebrow{{display:inline-block;font-size:13px;letter-spacing:.14em;font-weight:700;color:var(--violet);
    text-transform:uppercase;border:1px solid rgba(182,155,255,.35);border-radius:999px;padding:6px 14px;margin-bottom:18px;background:rgba(182,155,255,.07);}}
  h1.title{{font-size:clamp(25px,5vw,38px);line-height:1.2;margin:0 0 14px;font-weight:800;}}
  h1.title .accent{{color:var(--violet);}}
  .lede{{font-size:clamp(15px,2.3vw,17.5px);color:var(--text-dim);max-width:720px;margin:0;}}
  .tldr{{background:linear-gradient(135deg,rgba(182,155,255,.13),rgba(95,211,154,.09));
    border:1px solid rgba(182,155,255,.30);border-radius:var(--radius);padding:20px 24px;margin:24px auto 0;max-width:var(--maxw);}}
  .tldr h2{{margin:0 0 10px;font-size:15.5px;color:var(--violet);}} .tldr ul{{margin:0;padding-left:20px;}}
  .tldr li{{margin:7px 0;font-size:15px;color:var(--text);}} .tldr b{{color:#fff;}}
  section{{margin-top:44px;}} .kicker{{font-size:13px;font-weight:700;letter-spacing:.12em;color:var(--violet);text-transform:uppercase;margin:0 0 8px;}}
  h2.sec{{font-size:clamp(19px,3.4vw,25px);font-weight:800;margin:0 0 8px;}} .sec-sub{{color:var(--text-dim);margin:0 0 20px;font-size:15px;}}
  p{{margin:0 0 14px;}} .dim{{color:var(--text-dim);}}
  .fig{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 14px;}}
  .cap{{font-size:12.5px;color:var(--text-faint);text-align:center;margin:10px 4px 0;}}
  .tbl-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:14px;margin:6px 0 0;}}
  table{{border-collapse:collapse;width:100%;min-width:600px;font-size:14px;}}
  thead th{{background:#23283a;color:var(--violet);text-align:left;font-weight:700;padding:11px 14px;border-bottom:1px solid var(--line);white-space:nowrap;}}
  tbody td{{padding:10px 14px;border-bottom:1px solid var(--line);color:var(--text-dim);font-variant-numeric:tabular-nums;}}
  tbody tr:last-child td{{border-bottom:none;}} tbody tr:nth-child(even){{background:rgba(255,255,255,.018);}}
  td.term{{color:#fff;font-weight:700;white-space:nowrap;}} .pos{{color:var(--leaf);font-weight:700;}} .neg{{color:var(--berry-soft);font-weight:700;}}
  code{{background:var(--card-2);border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-family:Consolas,monospace;font-size:12.5px;color:var(--sky);}}
  pre{{background:var(--card-2);border:1px solid var(--line);border-radius:12px;padding:16px 18px;overflow-x:auto;color:var(--text);font-size:13px;line-height:1.6;}}
  .plain{{border-left:3px solid var(--leaf);background:rgba(95,211,154,.07);border-radius:10px;padding:13px 17px;margin:16px 0 0;color:var(--text);font-size:15px;}}
  .plain b{{color:var(--leaf);}} .plain::before{{content:"🍓 요점";display:block;font-size:12.5px;font-weight:700;color:var(--leaf);margin-bottom:5px;}}
  .warn{{border-left:3px solid var(--gold);background:rgba(255,200,97,.07);border-radius:10px;padding:13px 17px;margin:16px 0 0;color:var(--text);font-size:14.5px;}}
  .warn b{{color:var(--gold);}} .warn::before{{content:"⚠ 정직한 한계";display:block;font-size:12.5px;font-weight:700;color:var(--gold);margin-bottom:5px;}}
  footer{{max-width:var(--maxw);margin:52px auto 0;padding:20px 22px 0;border-top:1px solid var(--line);color:var(--text-faint);font-size:13px;}}
</style></head><body>
<header class="hero">
  <a class="back" href="../index.html">← 문서 인덱스</a>
  <span class="eyebrow">베리디시전 · 제어 실행 정책</span>
  <h1 class="title">점진 보정 — <span class="accent">한 번에 되돌리지 않는다</span></h1>
  <p class="lede">기존 제어 규칙은 '무엇을 어느 방향으로' 바꿀지를 줍니다. 이 문서는 그 위에 얹는 <b style="color:var(--text)">'어떻게' 층</b> — 목표 편차를 하루 상한(0.5℃/일)을 두고 <b style="color:var(--text)">2~5일에 나눠</b> 완만히 옮기는 rate-limiter입니다.</p>
</header>
<div class="tldr">
  <h2>핵심 3가지</h2>
  <ul>
    <li><b>급격한 주야 스윙 = 스트레스</b> — 주간이 평소보다 1~2℃ 높았다고 그날 밤 바로 1~2℃ 낮추면 식물에 열 스트레스가 누적됩니다.</li>
    <li><b>편차를 나눠 흘린다</b> — 하루 최대 <b>{CAP_PER_DAY}℃</b>씩, <b>{MIN_DAYS}~{MAX_DAYS}일</b>에 걸쳐 목표에 도달. 같은 날 반대방향 과보정은 금지.</li>
    <li><b>DIF는 안전밴드 안에서</b> — 주야온도차(DIF)를 {DIF_SAFE[0]}~{DIF_SAFE[1]}℃ 밴드 안에 유지하며 전이해, 발달 균형을 깨지 않습니다.</li>
  </ul>
</div>
<div class="wrap">

<section>
  <div class="kicker">그림으로</div>
  <h2 class="sec">급격(1일) vs 점진({rep_meta['days']}일) — 야간 {night0:.1f}℃ → {rep_target:.1f}℃</h2>
  <p class="sec-sub">주간이 평소보다 +1.5℃ 높게 측정돼 야간을 1.5℃ 낮추려는 상황. 빨강은 하루에 1.5℃ 떨어뜨리는 급격 보정, 초록은 {CAP_PER_DAY}℃/일씩 {rep_meta['days']}일에 나눈 점진 보정.</p>
  <div class="fig">{svg}</div>
  <p class="cap">가로축 D0(오늘)~D5, 세로축 야간 설정온도(℃). 점진(초록)은 목표에 늦게 닿지만 하루 변화폭이 작아 스트레스를 줄입니다.</p>
</section>

<section>
  <div class="kicker">시나리오 표</div>
  <h2 class="sec">감지 편차별 보정 스케줄</h2>
  <div class="tbl-wrap"><table>
    <thead><tr><th>감지</th><th>야간 목표</th><th>급격 일일변화</th><th>점진 일일변화</th><th>점진 스케줄(℃)</th></tr></thead>
    <tbody>{srows}</tbody>
  </table></div>
  <div class="plain">같은 목표에 도달하되, 하루 변화폭을 <b>{CAP_PER_DAY}℃ 이하</b>로 눌러 스트레스를 낮춥니다. 편차가 클수록 분산 일수가 늘어납니다(최대 {MAX_DAYS}일).</div>
</section>

<section>
  <div class="kicker">형식화</div>
  <h2 class="sec">rate-limiter 한 줄</h2>
  <pre>일일 보정 = clip(목표 − 현재, −cap, +cap)      # cap = {CAP_PER_DAY}℃/일
분산 일수 = clip(⌈|편차| / cap⌉, {MIN_DAYS}, {MAX_DAYS})
규칙: 같은 날 반대방향 과보정 금지 · DIF는 {DIF_SAFE[0]}~{DIF_SAFE[1]}℃ 밴드 유지</pre>
  <p class="dim" style="font-size:13.5px;">구현: <code>src/ramp_control.py</code> · <code>ramp_schedule(current, target, cap, min_days, max_days)</code> → 일별 설정값 배열. 기존 <a href="룰베이스_제어규칙.html" style="color:var(--sky)">룰베이스 제어규칙</a>이 '방향'을 정하면, 이 모듈이 '속도'를 정합니다.</p>
</section>

<section>
  <div class="kicker">위치</div>
  <h2 class="sec">제어 트랙에서의 자리</h2>
  <p>예측 성능은 데이터 크기(600행)에 막혔지만, 이 정책은 <b>도메인 지식으로 바로 값어치를 더하는</b> 층입니다. <a href="../experiments/AB제어실험_CO2.html" style="color:var(--sky)">A/B 제어 실험</a>에서 CO₂를 독립 레버로 규명했고 온도는 DIF 트레이드오프 레버였는데, 이 점진 정책은 그 온도 레버를 <b>안전하게 움직이는 법</b>을 규정합니다.</p>
  <div class="warn">이 스케줄은 농학 원리(온도적산·완만 전이) 기반 <b>휴리스틱</b>이며, 우리 데이터로 A/B 검증한 값은 아닙니다. cap·분산 일수·DIF 밴드는 작물·시즌에 따라 조정해야 하고, <a href="../plan/실증실험_계획서.html" style="color:var(--gold)">실증 계획</a>의 무작위 배정 프로토콜로 <b>급격 vs 점진</b>을 직접 비교해 검증할 대상입니다.</div>
</section>

<footer>berry_ai · 딸기 스마트팜 예측·제어 · 점진 보정(ramp-rate) 정책 · 기본값 {CAP_PER_DAY}℃/일 · {MIN_DAYS}~{MAX_DAYS}일 분산</footer>
</div></body></html>""", encoding="utf-8")


if __name__ == "__main__":
    main()
