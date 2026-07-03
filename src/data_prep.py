# -*- coding: utf-8 -*-
"""
data_prep.py
============
[이 파일이 하는 일 — 한 문장]
보유한 5개 CSV(생육 1개 + 온실별 환경 4개)를 읽어서,
"직전 7일의 환경 요약(X)  →  그 날 조사한 딸기 생육(Y)" 형태의
학습용 표 1개(data/processed/train_table.csv)로 합쳐 줍니다.

[왜 이런 모양이 필요한가]
- 환경 데이터는 '1시간 간격'으로 아주 촘촘하고(온실당 3천여 행),
  생육 데이터는 '일주일에 한 번'만 조사됩니다(총 15일).
- 시간 단위가 다르기 때문에 그냥 붙일 수 없습니다.
- 그래서 "생육을 조사한 날"을 기준점으로 삼고,
  그 직전 7일 동안의 환경을 '평균/합계/최소/최대' 같은 숫자 몇 개로 '요약'해서 붙입니다.
- 이렇게 하면 한 줄이 곧 "이런 환경이었을 때 → 딸기가 이만큼 자랐다"라는
  하나의 학습 예시가 됩니다.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0) 준비물 불러오기
#    - pandas: 표(엑셀 같은 데이터)를 다루는 대표 라이브러리. 관례상 pd 로 줄여 씀.
#    - pathlib.Path: 파일 경로를 OS(맥/윈도우) 상관없이 안전하게 다루는 도구.
#    - numpy: 수학 계산용. 여기선 VPD(수증기압차) 계산에만 잠깐 사용.
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 윈도우 콘솔(cp949)에서도 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 고정.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1) 설정값(상수) — 바꿀 만한 값들을 맨 위에 모아 둡니다.
#    나중에 창(window)을 7일이 아니라 14일로 바꾸고 싶으면 여기 숫자만 고치면 됩니다.
# ─────────────────────────────────────────────────────────────────────────────

# 이 파일(src/data_prep.py)의 위치를 기준으로 프로젝트 폴더를 찾습니다.
#   __file__            = .../smartfarm/src/data_prep.py
#   .parent             = .../smartfarm/src
#   .parent.parent      = .../smartfarm   ← 프로젝트 루트
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"                 # 원본 CSV들이 있는 곳
OUT_DIR = DATA_DIR / "processed"                # 결과물을 저장할 곳(없으면 아래에서 자동 생성)

# 생육 조사일 '직전 며칠'의 환경을 요약할지. (생육이 주1회라 7일이 자연스러움)
WINDOW_DAYS = 7

# 파일들이 CP949/UTF-8 섞여 있어도 안전하게 읽히도록 'utf-8-sig' 사용.
#   (utf-8-sig = 앞에 BOM이 붙은 UTF-8도, 안 붙은 UTF-8도 모두 잘 읽음)
ENCODING = "utf-8-sig"

# 환경 CSV의 '측정일시' 글자 형식.  예) "9/30/2024 0:00"
#   %m=월 %d=일 %Y=4자리연도 %H=24시간제 시 %M=분
ENV_DATETIME_FORMAT = "%m/%d/%Y %H:%M"

# 생육 CSV의 '조사일자' 글자 형식.  예) "2024-11-01"
GROWTH_DATE_FORMAT = "%Y-%m-%d"


# ─────────────────────────────────────────────────────────────────────────────
# 2) 환경 데이터 불러오기 (온실별 4개 파일 → 하나로 합치기)
# ─────────────────────────────────────────────────────────────────────────────
def load_environment() -> pd.DataFrame:
    """
    24_환경정보-1온실.csv ~ -4온실.csv 4개를 읽어 세로로 이어붙인 뒤,
    '온실번호' 열을 새로 만들어 붙인 표를 돌려줍니다.

    [핵심 의도]
    환경 파일 안에는 '몇 번 온실인지' 알려주는 열이 없습니다.
    대신 '파일 이름'에 온실 번호가 들어 있으므로(예: -1온실),
    파일을 읽을 때마다 그 번호를 열로 직접 붙여 줍니다.
    그래야 나중에 생육 데이터의 '온실번호'와 짝을 맞출 수 있습니다.
    """
    frames = []  # 온실별 표를 잠깐 담아 둘 리스트

    for greenhouse_no in [1, 2, 3, 4]:
        file_path = DATA_DIR / f"24_환경정보-{greenhouse_no}온실.csv"

        # 파일을 표로 읽어 옵니다.
        df = pd.read_csv(file_path, encoding=ENCODING)

        # '측정일시'가 지금은 그냥 글자입니다. 이걸 진짜 '날짜/시간' 값으로 바꿔야
        # 나중에 "이 날짜의 직전 7일" 같은 시간 계산이 가능해집니다.
        df["측정일시"] = pd.to_datetime(df["측정일시"], format=ENV_DATETIME_FORMAT)

        # 이 파일이 몇 번 온실인지 열로 새로 추가.
        df["온실번호"] = greenhouse_no

        frames.append(df)

    # 4개 표를 세로로(행을 이어서) 하나로 합칩니다. ignore_index=True 는 행 번호를 새로 매김.
    env = pd.concat(frames, ignore_index=True)

    print(f"[환경] {len(env):,}행 로드 (온실 4개 합침, "
          f"{env['측정일시'].min().date()} ~ {env['측정일시'].max().date()})")
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 3) 생육 데이터 불러오기 + 가로로 펼치기 (long → wide)
# ─────────────────────────────────────────────────────────────────────────────
def load_growth() -> pd.DataFrame:
    """
    24_생육정보.csv 를 읽어서, '한 표본의 모든 조사항목이 한 줄'이 되도록 펼칩니다.

    [원본 모양 — long(세로)]
      한 줄에 항목이 딱 하나씩 들어 있습니다.
        온실1 | 11-01 | 라인1 | 표본1 | 초장  | 23
        온실1 | 11-01 | 라인1 | 표본1 | 엽수  | 7
    [바꿀 모양 — wide(가로)]
      한 줄에 그 표본의 모든 항목이 열로 쭉 늘어섭니다.
        온실1 | 11-01 | 라인1 | 표본1 | 초장=23 | 엽수=7 | 착과수=0 | ...
    [왜?]
    머신러닝 모델은 "한 줄 = 하나의 관측치"를 기대합니다.
    초장, 엽수 등이 서로 다른 줄에 흩어져 있으면 학습에 못 쓰므로 한 줄로 모읍니다.
    """
    file_path = DATA_DIR / "24_생육정보.csv"
    df = pd.read_csv(file_path, encoding=ENCODING)

    # '조사일자'도 글자 → 진짜 날짜로 변환 (환경과 날짜를 맞추기 위해).
    df["조사일자"] = pd.to_datetime(df["조사일자"], format=GROWTH_DATE_FORMAT)

    # '조사항목값'이 혹시 글자로 읽혔을 수 있으니 숫자로 강제 변환.
    #   errors="coerce" = 숫자로 못 바꾸는 값은 비움(NaN) 처리 → 뒤에서 걸러짐.
    df["조사항목값"] = pd.to_numeric(df["조사항목값"], errors="coerce")

    # 여기가 'long → wide'로 펼치는 핵심.
    #   index   : 한 줄을 구분하는 기준(같은 표본을 한 줄로 모음)
    #   columns : 이 열의 값들이 '새 열 이름'이 됨 (초장, 엽수, ...)
    #   values  : 새 열에 채울 값
    #   aggfunc="mean": 혹시 같은 칸에 값이 둘 이상이면 평균으로. (지금 데이터는 중복 없음 — 안전장치)
    wide = df.pivot_table(
        index=["온실번호", "조사일자", "측정라인", "표본번호"],
        columns="조사항목",
        values="조사항목값",
        aggfunc="mean",
    )

    # pivot_table 결과는 index가 '여러 열로 이루어진 이름표' 상태입니다.
    # reset_index()로 그 이름표들을 다시 보통 열로 되돌립니다(평범한 표로).
    wide = wide.reset_index()

    # pivot 뒤 남는 '조사항목'이라는 열 묶음 이름표는 지워서 깔끔하게.
    wide.columns.name = None

    print(f"[생육] {len(wide):,}행 (표본 단위) · 생육항목 {list(wide.columns[4:])}")
    return wide


# ─────────────────────────────────────────────────────────────────────────────
# 4) 파생 변수 만들기 (있는 값으로 '더 쓸모 있는 값'을 계산)
# ─────────────────────────────────────────────────────────────────────────────
def add_env_features(env: pd.DataFrame) -> pd.DataFrame:
    """
    환경 표에 도메인 지식으로 만든 새 열 몇 개를 추가합니다.
    (요약하기 '전에' 시간 단위에서 먼저 계산해 두는 게 정확합니다.)

    - 주간여부: 낮인지 밤인지. 외부일사량이 0보다 크면 낮(1), 아니면 밤(0).
                → 딸기는 낮/밤 온도의 '차이(DIF)'가 생육에 중요하기 때문.
    - VPD    : 수증기압차. 온도와 습도를 합쳐 '식물이 느끼는 건조함'을 나타내는 값.
                단순 습도(%)보다 병해·증산 판단에 더 잘 맞는 지표라 미리 만들어 둡니다.
    """
    # 낮(1) / 밤(0) 표시.  (True/False를 int로 바꿔 1/0으로)
    env["주간여부"] = (env["외부일사량"] > 0).astype(int)

    # VPD 계산 (단위: kPa). 공식은 농업기상에서 널리 쓰는 표준식입니다.
    #   1) 포화수증기압 SVP = 0.6108 * e^(17.27*T / (T+237.3))   (T = 내부온도 ℃)
    #   2) VPD = SVP * (1 - 상대습도/100)
    T = env["내부온도"]
    RH = env["내부습도"]
    svp = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    env["VPD"] = svp * (1 - RH / 100.0)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# 5) 생육 조사일마다 '직전 7일 환경'을 요약하기  ← 이 파일에서 가장 중요한 단계
# ─────────────────────────────────────────────────────────────────────────────
def summarize_env_before_survey(env: pd.DataFrame, survey_dates: pd.DataFrame) -> pd.DataFrame:
    """
    각 (온실번호, 조사일자)에 대해, 그 날 직전 WINDOW_DAYS(7)일간의 환경을
    숫자 몇 개로 압축(요약)한 표를 만듭니다.

    [핵심 의도 — '데이터 누수' 방지]
    생육을 예측할 때는 '그 날 이후'의 환경을 절대 쓰면 안 됩니다(미래를 미리 본 셈이 되므로).
    그래서 딱 '직전 7일 ~ 조사일 자정 전'까지만 잘라서 요약합니다.

    [무엇을 요약하나]
    - 평균/최소/최대: 온도·습도·CO₂·EC·VPD 등 전반적 수준과 변동폭.
    - 합계: 외부일사량의 합 = '그 기간 받은 빛의 총량'(누적광량 개념).
    - 주간/야간 평균온도: 낮/밤 온도를 따로. (딸기 생육에서 특히 중요)
    """
    # 요약할 대상 열과, 각 열에 적용할 요약 방법(평균/최소/최대/합계)을 정합니다.
    #   숫자로 요약이 의미 있는 열만 고릅니다.
    agg_plan = {
        "내부온도": ["mean", "min", "max"],
        "내부습도": ["mean", "min", "max"],
        "내부CO2": ["mean", "max"],
        "외부일사량": ["mean", "sum"],   # sum = 누적광량(빛의 총량)
        "외부온도": ["mean", "min"],     # min = 기간 중 최저 외기온(저온장해 위험)
        "급액EC": ["mean"],
        "급액PH": ["mean"],
        "VPD": ["mean", "max"],
    }

    rows = []  # 요약 결과를 한 줄씩 담아 둘 리스트

    # 생육이 조사된 (온실, 날짜) 조합만 반복합니다. (4온실 × 15일 = 60개뿐이라 충분히 빠름)
    #   drop_duplicates: 표본이 여러 개라도 (온실,날짜) 조합은 한 번씩만.
    for _, key in survey_dates[["온실번호", "조사일자"]].drop_duplicates().iterrows():
        g = key["온실번호"]
        d = key["조사일자"]

        # 요약할 시간 구간: [조사일 - 7일, 조사일 자정)  ← 조사일 당일은 포함하지 않음(미래 차단)
        start = d - pd.Timedelta(days=WINDOW_DAYS)

        # 해당 온실 & 구간에 들어오는 환경 행만 골라냅니다.
        mask = (env["온실번호"] == g) & (env["측정일시"] >= start) & (env["측정일시"] < d)
        window = env.loc[mask]

        # 이 조합에 대한 요약값을 담을 딕셔너리. 먼저 키(온실/날짜)부터 넣습니다.
        summary = {"온실번호": g, "조사일자": d, "환경관측수": len(window)}

        if len(window) == 0:
            # 직전 7일 환경이 아예 없으면(예: 데이터 시작 초반) 요약은 비워 둡니다.
            rows.append(summary)
            continue

        # agg_plan대로 각 열을 요약해서 '열이름_방법' 형태의 새 열로 저장.
        #   예) 내부온도 mean → "내부온도_mean"
        for col, funcs in agg_plan.items():
            for func in funcs:
                summary[f"{col}_{func}"] = window[col].agg(func)

        # 낮/밤을 나눠 내부온도 평균을 따로 계산 (DIF 관련 feature).
        day = window.loc[window["주간여부"] == 1, "내부온도"]
        night = window.loc[window["주간여부"] == 0, "내부온도"]
        summary["내부온도_주간평균"] = day.mean() if len(day) else np.nan
        summary["내부온도_야간평균"] = night.mean() if len(night) else np.nan
        # 낮밤 온도차(DIF): 값이 클수록 열매 쪽으로 무게가 실리는 경향.
        summary["주야온도차_DIF"] = summary["내부온도_주간평균"] - summary["내부온도_야간평균"]

        rows.append(summary)

    env_summary = pd.DataFrame(rows)
    print(f"[요약] 환경 요약 {len(env_summary):,}줄 생성 "
          f"(온실×조사일 조합, 각 조사일 직전 {WINDOW_DAYS}일 기준)")
    return env_summary


# ─────────────────────────────────────────────────────────────────────────────
# 6) 생육(Y)과 환경요약(X)을 하나로 합치기
# ─────────────────────────────────────────────────────────────────────────────
def build_training_table(growth_wide: pd.DataFrame, env_summary: pd.DataFrame) -> pd.DataFrame:
    """
    표본 단위 생육 표(왼쪽)에, (온실번호+조사일자)가 같은 환경요약(오른쪽)을 붙입니다.

    [merge = 엑셀의 VLOOKUP 같은 것]
    - on=["온실번호","조사일자"] : 이 두 열의 값이 같은 것끼리 짝지어 붙입니다.
    - how="left" : 왼쪽(생육)을 모두 살리고, 짝이 있으면 환경요약을 채웁니다.
      → 같은 (온실,날짜)의 여러 표본에는 '같은 환경요약'이 복사되어 붙습니다(정상).
    """
    merged = growth_wide.merge(env_summary, on=["온실번호", "조사일자"], how="left")

    # 보기 좋게 정렬 (온실 → 날짜 순).
    merged = merged.sort_values(["온실번호", "조사일자", "측정라인", "표본번호"]).reset_index(drop=True)

    print(f"[합침] 최종 학습표 {len(merged):,}행 × {merged.shape[1]}열")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 6-2) 발달단계(시점) 파생변수 추가
# ─────────────────────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    '조사일자'만으로 만들 수 있는 '재배가 얼마나 진행됐나'(발달단계) 정보를 추가합니다.

    [왜 필요한가]
    착과수 같은 값은 '직전 7일 환경'보다 '지금이 재배 몇 주차인가(화방 발달 단계)'에
    훨씬 크게 좌우됩니다. 예: 정식 초기엔 열매가 없고, 1화방 시기에 확 늘었다가,
    수확 후 잠시 줄고, 2화방에서 다시 오릅니다. 이 '시간 흐름'을 모델에 알려줘야 합니다.

    [누수(반칙)가 아닌 이유]
    이 값들은 전부 '날짜'만 보고 계산합니다. 예측 시점에도 '오늘이 며칠인지'는 당연히
    아는 정보이므로, 미래를 훔쳐보는 게 아닙니다. (환경/생육 측정값을 미리 보는 게 아님)

    - 재배경과일수 : 그 온실의 '첫 조사일'로부터 며칠 지났는지.
    - 재배경과주차 : 위를 7로 나눈 주 단위 (0주차, 1주차, ...).
    - 조사회차     : 그 온실에서 몇 번째 조사인지 (1, 2, 3, ... 15).
    - 월           : 계절성(겨울 저광 등)을 반영.
    """
    df = df.copy()

    # 각 '온실번호'별로 가장 이른 조사일자를 그 온실의 '기준(0일)'으로 삼습니다.
    #   transform("min") = 그룹의 최솟값을 각 행에 다시 붙여 줌.
    first_date = df.groupby("온실번호")["조사일자"].transform("min")

    # 기준일로부터 며칠/몇 주 지났는지.
    df["재배경과일수"] = (df["조사일자"] - first_date).dt.days
    df["재배경과주차"] = df["재배경과일수"] // 7

    # 온실별 조사 순번(1,2,3,...). dense = 같은 날짜는 같은 번호, 빈틈없이 매김.
    df["조사회차"] = df.groupby("온실번호")["조사일자"].rank(method="dense").astype(int)

    # 월 (계절성).
    df["월"] = df["조사일자"].dt.month

    print(f"[시점] 발달단계 feature 추가: 재배경과일수 · 재배경과주차 · 조사회차 · 월")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7) 전체 순서 실행 (이 파일을 직접 실행했을 때만 동작)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("딸기 데이터 병합 파이프라인 시작")
    print("=" * 60)

    # 결과 저장 폴더가 없으면 만듭니다. (exist_ok=True: 이미 있어도 에러 안 냄)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # (1) 환경 4파일 로드 → (2) 파생변수(주간여부·VPD) 추가
    env = load_environment()
    env = add_env_features(env)

    # (3) 생육 로드 + 가로로 펼치기
    growth_wide = load_growth()

    # (4) 각 조사일 직전 7일 환경 요약
    env_summary = summarize_env_before_survey(env, growth_wide)

    # (5) 생육(Y) + 환경요약(X) 합치기
    train_table = build_training_table(growth_wide, env_summary)

    # (5-2) 발달단계(시점) feature 추가
    train_table = add_time_features(train_table)

    # (6) 파일로 저장. 엑셀에서도 한글이 안 깨지도록 utf-8-sig(BOM)로 저장.
    out_path = OUT_DIR / "train_table.csv"
    train_table.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("-" * 60)
    print(f"저장 완료 → {out_path}")
    print("미리보기(앞 5행):")
    # 열이 많으므로 몇 개만 뽑아서 보여 줍니다.
    preview_cols = [c for c in ["온실번호", "조사일자", "초장", "착과수",
                                "내부온도_mean", "외부일사량_sum", "주야온도차_DIF"]
                    if c in train_table.columns]
    print(train_table[preview_cols].head().to_string(index=False))
    print("=" * 60)
    print("[다음 단계 힌트]")
    print(" - X(입력): '_mean/_sum/_min/_max' 로 끝나는 환경요약 열들")
    print(" - Y(예측 대상): 초장 / 착과수 / 엽수 등 생육 열 중 하나를 골라 시작")
    print(" - 학습/검증은 반드시 '조사일자' 순서로 나누세요(랜덤 분할 금지 = 미래 누수 방지)")


# 파이썬 관례: 이 파일을 'python src/data_prep.py'로 직접 실행하면 main()이 돌고,
# 다른 파일이 이 파일을 import할 때는 main()이 자동 실행되지 않도록 막아 줍니다.
if __name__ == "__main__":
    main()
