"""SP-V2-008 driver: aperiodicity/MVF feature의 physical validation(합성 신호) +
기존 hnr_mean/std와의 비교, 실제 오디오 소규모 pilot.

합성 신호로 먼저 검증한다: 깨끗한 배음 스택은 MVF가 Nyquist 근처(전 대역 주기적)여야
하고, breathy(잡음 섞인) 신호는 MVF가 낮아야(저주파부터 이미 비주기적) 한다 -
HNR 하나로는 이 둘을 구분하기 어렵다는 게 기존 파이프라인의 한계였다(audit §13).
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.f0 import synthetic_signals as synth
from signal_v2.voice_quality.aperiodicity import extract_aperiodicity_features
from voice_quality_features import compute_voice_quality_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "aperiodicity")


def synthetic_conditions() -> list:
    return [
        synth.harmonic_stack(f0=150.0, n_harmonics=8, rolloff=0.88),
        synth.additive_white_noise(f0=150.0, snr_db=20.0),
        synth.additive_white_noise(f0=150.0, snr_db=10.0),
        synth.additive_white_noise(f0=150.0, snr_db=0.0),
        synth.breathy_noisy_harmonic(f0=150.0, noise_mix=0.2),
        synth.breathy_noisy_harmonic(f0=150.0, noise_mix=0.4),
        synth.breathy_noisy_harmonic(f0=150.0, noise_mix=0.6),
    ]


def run_validation() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for gt in synthetic_conditions():
        ap_features = extract_aperiodicity_features(gt.waveform, gt.sr)
        existing = compute_voice_quality_features(gt.waveform, gt.sr)

        row = {"condition": gt.name, "description": gt.description}
        row.update(ap_features)
        row["existing_hnr_mean"] = existing["hnr_mean"]
        row["existing_hnr_std"] = existing["hnr_std"]
        rows.append(row)

        print(f"\n=== {gt.name} ({gt.description}) ===")
        print(f"  mvf_mean={ap_features['mvf_mean']:.0f} Hz  "
              f"low_band_periodicity={ap_features['low_band_periodicity_mean']:.3f}  "
              f"mid_band_periodicity={ap_features['mid_band_periodicity_mean']:.3f}  "
              f"high_band_periodicity={ap_features['high_band_periodicity_mean']:.3f}  "
              f"existing_hnr_mean={existing['hnr_mean']:.1f} dB")

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
