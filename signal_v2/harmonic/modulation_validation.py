"""SP-V2-010: harmonic modulation spectrum의 physical validation.

알려진 속도(mod_rate_hz)로 전체 배음 진폭을 함께 진폭변조(AM)한 합성 신호를 만들어,
harmonic_modulation_features가 그 변조 속도를 실제로 복원하는지 확인한다. 정지
(변조 없는) 조건과 비교해 "변조가 없으면 dominant frequency가 뚜렷하지 않아야
한다"도 함께 확인한다.

주의(스코프 한계, 정직하게 기록): 여기서는 모든 배음에 동일하게 적용되는 공유
진폭변조만 테스트한다. 배음마다 서로 다른 변조(harmonic-specific modulation)는
테스트하지 않았다 - inter_harmonic_modulation_coherence가 그 경우를 구분하도록
설계되어 있지만 이 검증에서 직접 확인하지는 않았다.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.f0.synthetic_signals import _synth_harmonic_stack, SR
from signal_v2.harmonic.harmonic_modulation import harmonic_modulation_features
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "harmonic_modulation")


def make_am_modulated_signal(f0: float = 150.0, sr: int = SR, duration: float = 2.0,
                              mod_rate_hz: float = 4.0, mod_depth: float = 0.5,
                              n_harmonics: int = 5, rolloff: float = 0.88) -> tuple:
    n = int(duration * sr)
    t = np.arange(n) / sr
    f0_per_sample = np.full(n, f0)
    voiced = np.ones(n, dtype=bool)
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_rate_hz * t)
    y = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff, amplitude=envelope)

    gt_times = np.arange(0, duration, 0.001)
    gt_f0 = np.full_like(gt_times, f0)
    return y, sr, gt_times, gt_f0


def run_validation() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conditions = [
        ("no_modulation", 0.0),
        ("slow_am_2hz", 2.0),
        ("syllabic_am_4hz", 4.0),
        ("fast_am_8hz", 8.0),
    ]

    rows = []
    for name, mod_rate in conditions:
        y, sr, gt_times, gt_f0 = make_am_modulated_signal(mod_rate_hz=mod_rate, mod_depth=0.5 if mod_rate > 0 else 0.0)
        traj = extract_harmonic_trajectory(y, sr, gt_times, gt_f0, mode="adaptive", c=3, n_harmonics=5)
        features = harmonic_modulation_features(traj)

        row = {"condition": name, "true_mod_rate_hz": mod_rate}
        row["recovered_dominant_freq_mean"] = features["modulation_dominant_freq_mean"]
        row["h3_dominant_freq"] = features["h3_modulation_dominant_freq"]
        rows.append(row)
        print(f"{name:<18} true={mod_rate:.1f}Hz  recovered(mean over harmonics)={features['modulation_dominant_freq_mean']:.2f}Hz  "
              f"h3={features['h3_modulation_dominant_freq']:.2f}Hz")

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
