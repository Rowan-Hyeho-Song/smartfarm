# -*- coding: utf-8 -*-
"""
run_all.py  (전체 파이프라인 원커맨드 재현)
===========================================
[이 파일이 하는 일 — 한 문장]
데이터 병합 → v2.1 학습 → v3 학습 → 추론 → 결과지까지 순서대로 한 번에 실행합니다.
새 환경(예: 맥북)에서 'python src/run_all.py' 하나로 전체를 재현할 때 씁니다.

각 단계는 별도 파이썬 프로세스로 돌리며, 하나라도 실패하면 즉시 멈추고 알려줍니다.

실행:  python src/run_all.py
"""
import sys
import subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = Path(__file__).resolve().parent
PY = sys.executable  # 지금 이 스크립트를 돌린 파이썬(=올바른 가상환경)

# (표시이름, 스크립트파일) 순서대로 실행
STEPS = [
    ("① 데이터 병합 (5 CSV → 학습표)", "data_prep.py"),
    ("② v2.1 학습 (온실-날짜 평균 + 분포)", "train_agg.py"),
    ("③ v3 학습 (개체 추적 + huber·MAE)", "train_individual.py"),
    ("④ 추론 (v3 개체·온실 예측 + 구간)", "predict.py"),
    ("⑤ 결과지 (v2.1 버킷 결과지)", "report.py"),
]


def main():
    print("=" * 64)
    print("berry_ai 전체 파이프라인 재현 시작")
    print(f"  파이썬: {PY}")
    print("=" * 64)

    for i, (label, script) in enumerate(STEPS, 1):
        path = SRC / script
        print(f"\n[{i}/{len(STEPS)}] {label}")
        print(f"      → {script} 실행 중...")
        result = subprocess.run([PY, str(path)], cwd=str(SRC.parent))
        if result.returncode != 0:
            print(f"\n[중단] '{script}'에서 오류(종료코드 {result.returncode}). "
                  f"위 로그를 확인하세요.")
            sys.exit(result.returncode)
        print(f"      ✓ 완료")

    print("\n" + "=" * 64)
    print("전체 파이프라인 재현 완료 ✓")
    print("  - 학습표: data/processed/train_table.csv")
    print("  - 모델:   models/lgbm_fruit_count_v2.txt (v2.1) · lgbm_fruit_count_v3.txt (v3)")
    print("  - 예측:   data/processed/predictions_v3_individual.csv · _greenhouse.csv")
    print("  - 결과지: docs/report/착과수_예측_결과지.html · 착과수_개체별_결과지.html")
    print("=" * 64)


if __name__ == "__main__":
    main()
