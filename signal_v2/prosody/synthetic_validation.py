"""SP-V2-007: CWT multi-scale prosody의 physical validation.

알려진 하나의 시간 스케일에서만 진동하는 semitone 궤적을 합성해서, band_features()가
그 스케일에 해당하는 대역(microprosody/syllable/word/phrase/utterance)에 실제로
에너지를 몰아주는지 확인한다 - "어느 스케일에서 변화가 일어나는가"를 이 표현이
올바르게 구분하는지에 대한 최소한의 물리적 검증이다.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.prosody.cwt_prosody import SCALE_BANDS, extract_cwt_prosody_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "cwt_prosody")


def make_single_scale_f0(period_sec: float, duration: float = 2.0, hop_sec: float = 0.01,
                          f0_ref: float = 150.0, depth_semitones: float = 2.0) -> tuple:
    times = np.arange(0, duration, hop_sec)
    semitone = depth_semitones * np.sin(2 * np.pi * times / period_sec)
    f0 = f0_ref * (2.0 ** (semitone / 12.0))
    return times, f0


def expected_band_for_period(period_sec: float) -> str:
    for band_name, lo, hi in SCALE_BANDS:
        if lo <= period_sec < hi:
            return band_name
    return "utterance" if period_sec >= SCALE_BANDS[-1][2] else "microprosody"


def run_validation() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_periods = [0.04, 0.15, 0.4, 1.0]  # microprosody / syllable / word / phrase 각 대역 중심 근처

    rows = []
    for period in test_periods:
        expected = expected_band_for_period(period)
        times, f0 = make_single_scale_f0(period)
        features = extract_cwt_prosody_features(times, f0)
        dominant = features.get("dominant_band", "?")

        ratios = {band_name: features.get(f"{band_name}_energy_ratio", 0.0) for band_name, _, _ in SCALE_BANDS}
        print(f"injected period={period:.3f}s (expected band={expected}) -> dominant_band={dominant}")
        print("   energy ratios:", {k: round(v, 3) for k, v in ratios.items()})

        row = {"injected_period_sec": period, "expected_band": expected, "dominant_band": dominant}
        row.update({f"{k}_energy_ratio": v for k, v in ratios.items()})
        rows.append(row)

    per_file_path = os.path.join(OUTPUT_DIR, "synthetic_validation.csv")
    fieldnames = list(rows[0].keys())
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장 완료: {per_file_path}")
    return rows


if __name__ == "__main__":
    run_validation()
