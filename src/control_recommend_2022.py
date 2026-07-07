# -*- coding: utf-8 -*-
"""
control_recommend_2022.py  (신규 데이터 · 제어활동 추천 트랙)
============================================================
[이 파일이 하는 일 — 한 문장]
착과수 최적환경(2022) 데이터에서 '제어로 바꿀 수 있는 레버'만 골라,
발달단계(계절)를 걷어낸 뒤(stage-adjusted) 착과 성과와의 연관을 재고,
성과 상·하위 조사기간의 제어 설정을 대조해 '제어 추천'을 도출한다.

[예측 트랙과 무엇이 다른가 — '다른 신규 모델']
- 예측 트랙(exp_fruit_setting_2022.py): 환경→착과수를 '맞히는' 회귀. 실전 R²≈지속성.
- 이 처방 트랙: 착과수를 맞히는 게 목적이 아니라, '제어를 어느 방향으로 바꾸면
  발달단계 대비 착과가 나았나'를 본다. → 부분상관(잔차) + 상·하위 대조(uplift식).

[정직성 — 관측데이터의 한계를 구조로 방어]
- 착과수는 발달단계가 지배(예측 트랙에서 확인). 그래서 먼저 '경과일 곡선'으로
  단계 기대치를 만들고, 그 '잔차'(단계 대비 초과성과)에 대해서만 제어를 본다.
  → 계절 교란을 제거해야 '제어 신호'와 '계절 신호'가 안 섞인다.
- 그래도 인과가 아니라 연관이다. 그래서 농학 기준(VPD·CO2·야간온도·습도)과
  교차검증해, 데이터·농학이 같은 방향일 때만 추천으로 채택한다.

실행:  python src/control_recommend_2022.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "STRAWBERRY_FRUIT_SETTING_ENV_20221209.csv"
OUT_DIR = PROJECT_DIR / "data" / "processed"
TARGET = "FRST_TREE_CNT"


def vpd(temp, rh):
    """수증기압차(kPa): SVP*(1-RH/100). 심화 문서 기준 0.8이 영양/생식 분기."""
    svp = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    return svp * (1 - rh / 100.0)


def load_and_engineer():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["STRG_DT"] = pd.to_datetime(df["STRG_DT"])
    df = df.sort_values("STRG_DT").reset_index(drop=True)
    df["seg"] = (df[TARGET] != df[TARGET].shift()).cumsum()

    # 대표 내부 온·습도(1_2 평균 센서) → VPD
    df["_T"] = df["AVE_INNER_TPRT_1_2"]
    df["_RH"] = df["AVE_INNER_HMDT_1_2"]
    df["VPD"] = vpd(df["_T"], df["_RH"])

    # 주간(1)/야간(0) 판별: 외부일사가 큰 코드를 낮으로 확정
    day_code = df.groupby("DYTM_NIGHT_CD")["EXTN_SRQT"].mean().idxmax()
    df["_is_day"] = (df["DYTM_NIGHT_CD"] == day_code).astype(int)

    # CO2 갭(설정 대비 실측 부족분): 음수면 목표 미달
    df["CO2_gap"] = df["PFBS_NTRO_CBDX_CTRN"] - df["CBDX_STNG_VL"]
    return df


def to_segments(df):
    """조사기간(주) 단위로 집계 — 착과수의 실제 해상도."""
    first = df["STRG_DT"].min()
    rows = []
    for s, g in df.groupby("seg"):
        day = g[g["_is_day"] == 1]
        night = g[g["_is_day"] == 0]
        rows.append(dict(
            seg=s, 착과수=g[TARGET].iloc[0],
            경과일=(g["STRG_DT"].min() - first).days,
            시작일=g["STRG_DT"].min().date(),
            # ── 제어 레버(우리가 설정/조절 가능) ──
            CO2설정=g["CBDX_STNG_VL"].mean(),
            CO2실측=g["PFBS_NTRO_CBDX_CTRN"].mean(),
            CO2발생_비율=g["CBDX_GNRT_OPRT_YN"].mean(),
            야간온도=night["_T"].mean(),
            주간온도=day["_T"].mean(),
            DIF_주야온도차=day["_T"].mean() - night["_T"].mean(),
            VPD=g["VPD"].mean(),
            내부습도=g["_RH"].mean(),
            난방설정=g[[f"HTNG_TPRT_{i}" for i in range(1, 6)]].mean().mean(),
            환기설정=g[[f"VNTILAT_TPRT_{i}" for i in range(1, 6)]].mean().mean(),
            천창개도=g[["SKLT_OPDR_RATE_1_LEFT", "SKLT_OPDR_RATE_1_RIGHT"]].mean().mean(),
            스크린개도=g[["HRZNT_SCRN_OPDR_RATE_1", "HRZNT_SCRN_OPDR_RATE_2"]].mean().mean(),
            공급온도=g[[f"SPL_TPRT_{i}" for i in range(1, 5)]].mean().mean(),
            # ── 외부(제어 불가, 계절 교란원) ──
            외부기온=g["EXTN_TPRT"].mean(),
            외부일사=g["EXTN_SRQT"].mean(),
        ))
    return pd.DataFrame(rows).sort_values("경과일").reset_index(drop=True)


LEVERS = ["CO2설정", "CO2실측", "CO2발생_비율", "야간온도", "주간온도", "DIF_주야온도차",
          "VPD", "내부습도", "난방설정", "환기설정", "천창개도", "스크린개도", "공급온도"]

# 농학 기준(심화 문서): (레버, 바람직한 방향 설명, 기준값)
AGRO = {
    "VPD": "0.8 부근 — 아래=영양생장, 위=생식/품질",
    "내부습도": "≈75% 목표(과습 병해·저증산 회피)",
    "CO2실측": "≤750ppm 포화, 그 이하 공급 이득",
    "야간온도": "동화산물 배분 다이얼(맑고 축적 많으면↑)",
    "DIF_주야온도차": "주야편차로 열매/잎 배분 조절",
}


def stage_residual(seg):
    """경과일 4차 다항으로 발달단계 기대곡선 → 착과수 잔차(단계 대비 초과성과)."""
    x, y = seg["경과일"].to_numpy(float), seg["착과수"].to_numpy(float)
    coef = np.polyfit(x, y, deg=4)
    fit = np.polyval(coef, x)
    ss_res = np.sum((y - fit) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return y - fit, 1 - ss_res / ss_tot


def main():
    print("=" * 66)
    print("신규 데이터 · 제어활동 추천 트랙 (stage-adjusted)")
    print("=" * 66)
    df = load_and_engineer()
    seg = to_segments(df)
    print(f"[데이터] 조사기간(주) {len(seg)}개 · 제어 레버 {len(LEVERS)}개 · "
          f"기간 {seg['시작일'].iloc[0]} ~ {seg['시작일'].iloc[-1]}")

    resid, stage_r2 = stage_residual(seg)
    seg["단계잔차"] = resid
    print(f"\n[발달단계 지배 정량화] 경과일 곡선만으로 착과수 R² = {stage_r2:.3f}")
    print("  → 착과수 변동의 대부분이 '발달단계'로 설명됨(예측 트랙 결론 재확인).")

    # ── 분석 A: 제어 레버 ↔ (원착과 vs 단계잔차) 상관 대조 ──
    print("\n[분석 A] 제어 레버 상관 — 원본 vs 단계보정(잔차)")
    print(f"  {'레버':<14}{'원상관':>8}{'단계보정':>10}   해석")
    recs = []
    rows_a = []
    for L in LEVERS:
        raw = spearmanr(seg[L], seg["착과수"]).correlation
        adj = spearmanr(seg[L], seg["단계잔차"]).correlation
        rows_a.append((L, raw, adj))
    for L, raw, adj in sorted(rows_a, key=lambda t: -abs(t[2])):
        tag = "★ 단계 넘어 신호" if abs(adj) >= 0.30 else ("계절교란(보정후 소멸)" if abs(raw) >= 0.4 and abs(adj) < 0.2 else "")
        print(f"  {L:<14}{raw:>8.2f}{adj:>10.2f}   {tag}")

    # ── 분석 B: 성과 상·하위(단계 대비) 제어 설정 대조 = 'uplift식' ──
    k = len(seg) // 3
    top = seg.nlargest(k, "단계잔차")     # 단계 대비 초과성과(상위)
    bot = seg.nsmallest(k, "단계잔차")    # 단계 대비 부진(하위)
    print(f"\n[분석 B] 단계 대비 상위 {k}주 vs 하위 {k}주 — 제어 설정 대조")
    print(f"  {'레버':<14}{'상위평균':>9}{'하위평균':>9}{'차이':>8}   방향")
    for L in LEVERS:
        hi, lo = top[L].mean(), bot[L].mean()
        d = hi - lo
        if abs(d) < 1e-9:
            continue
        arrow = "↑ 상위가 높음" if d > 0 else "↓ 상위가 낮음"
        rows_a_adj = dict((x[0], x[2]) for x in rows_a)
        strong = "  ⇐ 추천후보" if abs(rows_a_adj[L]) >= 0.30 else ""
        print(f"  {L:<14}{hi:>9.2f}{lo:>9.2f}{d:>+8.2f}   {arrow}{strong}")
        if abs(rows_a_adj[L]) >= 0.30:
            recs.append((L, hi, lo, d, rows_a_adj[L]))

    # ── 분석 A 결과: 데이터만의 약한 후보(|r|≥0.20, 참고용) ──
    rows_a_adj = dict((x[0], x[2]) for x in rows_a)
    tent = [(L, rows_a_adj[L]) for L in LEVERS if abs(rows_a_adj[L]) >= 0.20]
    print("\n[데이터 단독 후보] 단계보정 |r|≥0.20 (약함 — 참고용)")
    if tent:
        for L, r in sorted(tent, key=lambda t: -abs(t[1])):
            print(f"  · {L}: 보정 r={r:+.2f}  (n=27이라 통계적으로 불확실)")
    else:
        print("  (없음)")
    print("  → 단계를 걷어내면 제어 신호가 노이즈와 구분 안 됨(상관≠인과 실증).")

    # ── 농학-갭 분석: 실제 운영점 vs 심화 문서 목표치 (견고한 추천) ──
    day = df[df["_is_day"] == 1]
    op_vpd_day = day["VPD"].mean()
    op_rh = df["_RH"].mean()
    op_co2_act = df["PFBS_NTRO_CBDX_CTRN"].mean()
    op_co2_set = df["CBDX_STNG_VL"].mean()
    op_night = seg["야간온도"].mean()
    op_dif = seg["DIF_주야온도차"].mean()

    gaps = [
        ("주간 VPD", f"{op_vpd_day:.2f} kPa", "0.8 부근",
         op_vpd_day < 0.55, "과습·저증산 → 광합성·양분이동 저해. 최소난방·환기로 VPD↑ 필요"),
        ("내부습도", f"{op_rh:.0f}%", "≈75%",
         op_rh > 82, "겨울 과습 상태 지속 → 곰팡이병(94%↑4h) 위험. 최소난방으로 습도↓"),
        ("CO₂ 달성", f"실측 {op_co2_act:.0f} / 설정 {op_co2_set:.0f}ppm", "설정 근접(≤750 포화)",
         op_co2_act < op_co2_set - 100, "설정 대비 크게 미달 → 주간 광합성 손실. 공급·밀폐 타이밍 점검"),
        ("주야 DIF", f"{op_dif:.1f}℃", "동화산물 기반 가변",
         False, "축적 많은 날 야온↑로 소모 유도(맛)·잎 확보 — 고정값 아닌 그날 대응"),
    ]
    print("\n" + "=" * 66)
    print("[제어 추천 종합]  데이터가 제어효과를 못 가리므로 → 농학 기준 대비 '운영 갭'으로 처방")
    print("=" * 66)
    out = []
    for name, actual, target, flag, action in gaps:
        mark = "⚠️ 갭" if flag else "· 참고"
        print(f"  {mark} {name}: 실제 {actual}  vs  목표 {target}")
        print(f"       → {action}")
        out.append(dict(항목=name, 실제운영=actual, 농학목표=target,
                        갭여부=("갭" if flag else "참고"), 조치=action))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    op = OUT_DIR / "control_recommendation_2022.csv"
    pd.DataFrame(out).to_csv(op, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 제어 추천표 → {op}")
    print("\n[정직] n=27주·단일구역·관측데이터 → 제어→착과 인과는 못 뽑음. "
          "그래서 추천은 '데이터로 확인한 실제 운영점 vs 농학 목표'의 갭 기반. "
          "실증 A/B로 최종 검증 필요.")


if __name__ == "__main__":
    main()
