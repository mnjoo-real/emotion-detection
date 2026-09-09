"""SP-V2-009: harmonic phase feature의 physical validation.

깨끗한 배음 스택(harmonic_stack)은 배음 간 위상 관계가 시간에 걸쳐 거의 완벽히
일정해야 하므로(진짜 주기 신호) mean_phase_coherence(R_k 평균)가 높아야 하고,
잡음이 많이 섞인 신호(breathy_noisy_harmonic, noise_mix 높게)는 매 분석 윈도우마다
위상 추정이 잡음에 흔들려 coherence가 낮아져야 한다.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.f0 import synthetic_signals as synth
from signal_v2.harmonic.harmonic_phase import harmonic_phase_features
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "harmonic_phase")


def run_validation() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conditions = [
        ("clean_harmonic_stack", synth.harmonic_stack(f0=150.0, n_harmonics=5, rolloff=0.88)),
        ("breathy_20pct", synth.breathy_noisy_harmonic(f0=150.0, n_harmonics=5, rolloff=0.88, noise_mix=0.2)),
        ("breathy_40pct", synth.breathy_noisy_harmonic(f0=150.0, n_harmonics=5, rolloff=0.88, noise_mix=0.4)),
        ("breathy_60pct", synth.breathy_noisy_harmonic(f0=150.0, n_harmonics=5, rolloff=0.88, noise_mix=0.6)),
        ("additive_noise_0db", synth.additive_white_noise(f0=150.0, n_harmonics=5, rolloff=0.88, snr_db=0.0)),
    ]

    rows = []
    for name, gt in conditions:
        traj = extract_harmonic_trajectory(gt.waveform, gt.sr, gt.times, gt.f0_true, mode="adaptive", c=3, n_harmonics=5)
        features = harmonic_phase_features(traj)
        row = {"condition": name}
        row.update(features)
        rows.append(row)
        print(f"{name:<20} mean_phase_coherence={features['mean_phase_coherence']:.3f}  "
              f"temporal_phase_instability={features['temporal_phase_instability']:.3f}")

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
