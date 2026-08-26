"""
성별에 따른 화자 차이가 감정 신호를 가리고 있는 것으로 보이는 feature들을 다듬는다.

원본 메타데이터(4차년도.csv/5차년도_2차.csv)에 화자 고유 ID는 없지만 '성별' 컬럼은
있고, 감정별 성별 비율이 16.5%~41.7%로 상당히 불균형하다는 걸 확인했다. 절대 Hz
기반 feature 두 종류가 특히 영향을 받는다:

1. 포먼트(F1~F4): 성도 길이라는 화자 해부학적 특성을 그대로 반영하는 값이라, 음성학
   표준 화자 정규화 기법대로 절대 Hz 대신 인접 포먼트 비율(F2/F1, F3/F2, F4/F3)로
   바꾼다 - 화자의 성도 크기와 무관하게 모음의 "형태"를 비교할 수 있게 된다.
2. formant_roughness / pitch_mean·std·median: 값 자체는 유지하되, 성별별 평균/
   표준편차로 z-score를 추가한다. 실측 결과 formant_roughness는 성별 내부로 나누면
   효과크기가 남성 기준 거의 3배(0.004 -> 0.011)로 뛰었다 - 성별 간 차이(eps^2=0.019)가
   감정 효과보다 커서 신호를 가리고 있었다는 뜻. pitch_mean은 성별 영향이 극단적으로
   크지만(eps^2=0.43) 감정 순위 자체는 두 성별 모두 동일하게 유지되어 "고장난" 신호는
   아니었다 - 그래도 z-score를 추가해두면 이 데이터셋의 특정 성별 구성에 덜 의존하는
   feature가 되어 일반화에 유리하다.

harmony_features.py/extract_features.py가 이미 뽑아둔 값을 재사용하는 가벼운
후처리라 오디오를 다시 읽지 않는다 - LPC 기반 포먼트 재추출(수 시간)을 피하기 위함.
"""

import csv
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "speaker_norm")

HARMONY_CSV = os.path.join(BASE_DIR, "output", "harmony", "per_file_harmony.csv")
FEATURES_CSV = os.path.join(BASE_DIR, "output", "features", "per_file_features.csv")
RAW_CSVS = [
    (os.path.join(BASE_DIR, "4차년도.csv"), "cp949"),
    (os.path.join(BASE_DIR, "5차년도_2차.csv"), "cp949"),
]

FIELDNAMES = [
    "wav_id", "situation",
    "f2_f1_ratio", "f3_f2_ratio", "f4_f3_ratio",
    "formant_roughness_gender_z",
    "pitch_mean_gender_z", "pitch_std_gender_z", "pitch_median_gender_z",
    "gender",
]


def load_gender_map() -> dict:
    gender_map = {}
    for path, encoding in RAW_CSVS:
        with open(path, encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                gender_map[row["wav_id"].strip()] = row["성별"].strip()
    return gender_map


def load_csv_by_id(path: str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["wav_id"]: row for row in csv.DictReader(f)}


def compute_gender_stats(rows_by_gender: dict, col: str) -> dict:
    """성별 -> (mean, std) 딕셔너리."""
    stats = {}
    for gender, values in rows_by_gender.items():
        arr = np.array(values)
        stats[gender] = (float(arr.mean()), float(arr.std()) if arr.std() > 0 else 1.0)
    return stats


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gender_map = load_gender_map()
    harmony = load_csv_by_id(HARMONY_CSV)
    features = load_csv_by_id(FEATURES_CSV)

    common_ids = sorted(set(harmony) & set(features) & set(gender_map))
    print(f"공통 wav_id: {len(common_ids)}개")

    # 1차 패스: 성별별 formant_roughness_mean / pitch_mean/std/median 값 모아서 성별 평균·표준편차 계산
    by_gender = {
        "formant_roughness_mean": {},
        "pitch_mean": {}, "pitch_std": {}, "pitch_median": {},
    }
    for wid in common_ids:
        gender = gender_map[wid]
        if gender not in ("male", "female"):
            continue
        by_gender["formant_roughness_mean"].setdefault(gender, []).append(float(harmony[wid]["formant_roughness_mean"]))
        by_gender["pitch_mean"].setdefault(gender, []).append(float(features[wid]["pitch_mean"]))
        by_gender["pitch_std"].setdefault(gender, []).append(float(features[wid]["pitch_std"]))
        by_gender["pitch_median"].setdefault(gender, []).append(float(features[wid]["pitch_median"]))

    gender_stats = {col: compute_gender_stats(vals, col) for col, vals in by_gender.items()}
    for col, stats in gender_stats.items():
        for gender, (mean, std) in stats.items():
            print(f"  {col} [{gender}]: mean={mean:.3f}, std={std:.3f}")

    # 2차 패스: 실제 feature 계산 및 저장
    per_file_path = os.path.join(OUTPUT_DIR, "per_file_speaker_norm.csv")
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        n_skipped = 0
        for wid in common_ids:
            gender = gender_map[wid]
            if gender not in ("male", "female"):
                n_skipped += 1
                continue

            h = harmony[wid]
            feat = features[wid]
            f1, f2, f3, f4 = (float(h[c]) for c in ("f1_mean", "f2_mean", "f3_mean", "f4_mean"))

            row = {
                "wav_id": wid,
                "situation": h["situation"],
                "gender": gender,
                "f2_f1_ratio": f2 / f1,
                "f3_f2_ratio": f3 / f2,
                "f4_f3_ratio": f4 / f3,
            }

            fr_mean, fr_std = gender_stats["formant_roughness_mean"][gender]
            row["formant_roughness_gender_z"] = (float(h["formant_roughness_mean"]) - fr_mean) / fr_std

            for col in ("pitch_mean", "pitch_std", "pitch_median"):
                mean, std = gender_stats[col][gender]
                row[f"{col}_gender_z"] = (float(feat[col]) - mean) / std

            writer.writerow(row)

    print(f"\n건너뜀(성별 정보 없음): {n_skipped}개")
    print(f"저장 완료: {per_file_path}")


if __name__ == "__main__":
    main()
