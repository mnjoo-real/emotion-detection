"""SP-V2-006: 진짜 jitter/shimmer와 의도된 F0 glide/vibrato를 독립적으로 조절할 수
있는 합성 조건 A-E (사용자 브리핑 그대로).

signal_v2.f0.synthetic_signals.py는 위상을 F0(t)를 그대로 적분해서 만들기 때문에
"진짜 jitter"가 존재하지 않는다(주기 하나하나가 매끄러운 F0(t)를 정확히 따른다).
여기서는 반대로 **주기 단위(period-synchronous)로 직접 신호를 합성**해서, 각
주기의 실제 길이를 의도된(smooth) F0 궤적에 무작위 승수(jitter)를 곱해 만들고,
각 주기의 진폭에도 별도의 무작위 승수(shimmer)를 곱한다 - 이렇게 하면 "의도된
매끄러운 궤적"과 "주기별 실제 편차"를 합성 단계에서부터 정확히 분리해 알고 있다.

조건:
  A. stable F0 + true jitter
  B. smooth pitch glide + no jitter
  C. smooth pitch glide + true jitter
  D. vibrato + no jitter
  E. vibrato + true jitter
"""

from dataclasses import dataclass, field

import numpy as np

SR = 16000


@dataclass
class JitterConditionSignal:
    name: str
    waveform: np.ndarray
    sr: int
    cycle_start_samples: np.ndarray   # 각 주기의 시작 샘플 인덱스 (ground-truth pitch mark)
    true_f0_intended: np.ndarray      # 각 주기에서 "의도된"(jitter 반영 전) F0
    true_jitter_std: float
    true_shimmer_std: float
    description: str = ""


def _period_synchronous_synthesis(f0_fn, sr: int, duration: float, jitter_std: float = 0.0,
                                   shimmer_std: float = 0.0, n_harmonics: int = 6, rolloff: float = 0.88,
                                   seed: int = 0) -> tuple:
    """주기 단위로 직접 파형을 합성한다. f0_fn(t)는 "의도된"(jitter 없는) 순간 F0.
    각 주기마다 실제 사용되는 F0 = f0_fn(t) * (1 + jitter), 진폭 = base * (1 + shimmer).
    위상은 주기 경계에서 연속(끊기지 않음) - 주파수만 주기 단위로 바뀐다."""
    rng = np.random.RandomState(seed)
    n_total = int(duration * sr)
    out = np.zeros(n_total)

    cycle_starts = []
    true_f0_per_cycle = []

    idx = 0
    t_cursor = 0.0
    phase = 0.0
    while idx < n_total:
        f0_intended = f0_fn(t_cursor)
        if f0_intended <= 0:
            break
        period_intended = 1.0 / f0_intended
        jitter_factor = 1.0 + (rng.randn() * jitter_std if jitter_std > 0 else 0.0)
        period_actual = max(period_intended * jitter_factor, 1.0 / sr)

        n_samples_cycle = int(round(period_actual * sr))
        n_samples_cycle = max(1, min(n_samples_cycle, n_total - idx))
        f0_actual = 1.0 / (n_samples_cycle / sr)

        shimmer_factor = 1.0 + (rng.randn() * shimmer_std if shimmer_std > 0 else 0.0)

        sample_idx = np.arange(n_samples_cycle)
        cycle_phase = phase + 2 * np.pi * f0_actual * sample_idx / sr
        cycle_signal = np.zeros(n_samples_cycle)
        for k in range(1, n_harmonics + 1):
            if f0_actual * k >= sr / 2 - 50:
                continue
            cycle_signal += (rolloff ** (k - 1)) * np.sin(k * cycle_phase)

        out[idx: idx + n_samples_cycle] = cycle_signal * shimmer_factor
        cycle_starts.append(idx)
        true_f0_per_cycle.append(f0_intended)

        phase = (phase + 2 * np.pi * f0_actual * n_samples_cycle / sr) % (2 * np.pi)
        idx += n_samples_cycle
        t_cursor += n_samples_cycle / sr

    peak = np.max(np.abs(out)) if np.any(out) else 1.0
    out = out / peak if peak > 0 else out
    return out, np.array(cycle_starts), np.array(true_f0_per_cycle)


JITTER_STD = 0.012   # 상대 표준편차 ~1.2% - 병리적으로 뚜렷한 수준의 "진짜" jitter
SHIMMER_STD = 0.035  # 상대 표준편차 ~3.5%


def condition_a_stable_with_jitter(duration: float = 1.0, f0: float = 150.0, sr: int = SR) -> JitterConditionSignal:
    waveform, starts, true_f0 = _period_synchronous_synthesis(
        lambda t: f0, sr, duration, jitter_std=JITTER_STD, shimmer_std=SHIMMER_STD, seed=1,
    )
    return JitterConditionSignal(
        name="A_stable_with_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=JITTER_STD, true_shimmer_std=SHIMMER_STD,
        description="Stable F0 + true cycle-to-cycle jitter/shimmer.",
    )


def condition_b_glide_no_jitter(duration: float = 1.0, f0_start: float = 100.0, f0_end: float = 250.0,
                                 sr: int = SR) -> JitterConditionSignal:
    f0_fn = lambda t: f0_start + (f0_end - f0_start) * (t / duration)
    waveform, starts, true_f0 = _period_synchronous_synthesis(f0_fn, sr, duration, jitter_std=0.0, shimmer_std=0.0, seed=2)
    return JitterConditionSignal(
        name="B_glide_no_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=0.0, true_shimmer_std=0.0,
        description="Smooth linear F0 glide, no true jitter/shimmer.",
    )


def condition_c_glide_with_jitter(duration: float = 1.0, f0_start: float = 100.0, f0_end: float = 250.0,
                                   sr: int = SR) -> JitterConditionSignal:
    f0_fn = lambda t: f0_start + (f0_end - f0_start) * (t / duration)
    waveform, starts, true_f0 = _period_synchronous_synthesis(
        f0_fn, sr, duration, jitter_std=JITTER_STD, shimmer_std=SHIMMER_STD, seed=3,
    )
    return JitterConditionSignal(
        name="C_glide_with_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=JITTER_STD, true_shimmer_std=SHIMMER_STD,
        description="Smooth linear F0 glide + true cycle-to-cycle jitter/shimmer.",
    )


def condition_d_vibrato_no_jitter(duration: float = 1.0, f0_center: float = 180.0, vibrato_rate: float = 5.5,
                                   vibrato_depth_semitones: float = 0.5, sr: int = SR) -> JitterConditionSignal:
    f0_fn = lambda t: f0_center * (2.0 ** (vibrato_depth_semitones * np.sin(2 * np.pi * vibrato_rate * t) / 12.0))
    waveform, starts, true_f0 = _period_synchronous_synthesis(f0_fn, sr, duration, jitter_std=0.0, shimmer_std=0.0, seed=4)
    return JitterConditionSignal(
        name="D_vibrato_no_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=0.0, true_shimmer_std=0.0,
        description="Vibrato, no true jitter/shimmer.",
    )


def condition_e_vibrato_with_jitter(duration: float = 1.0, f0_center: float = 180.0, vibrato_rate: float = 5.5,
                                     vibrato_depth_semitones: float = 0.5, sr: int = SR) -> JitterConditionSignal:
    f0_fn = lambda t: f0_center * (2.0 ** (vibrato_depth_semitones * np.sin(2 * np.pi * vibrato_rate * t) / 12.0))
    waveform, starts, true_f0 = _period_synchronous_synthesis(
        f0_fn, sr, duration, jitter_std=JITTER_STD, shimmer_std=SHIMMER_STD, seed=5,
    )
    return JitterConditionSignal(
        name="E_vibrato_with_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=JITTER_STD, true_shimmer_std=SHIMMER_STD,
        description="Vibrato + true cycle-to-cycle jitter/shimmer.",
    )


def condition_f_fast_accent_no_jitter(duration: float = 1.0, f0_center: float = 180.0,
                                       rate: float = 15.0, depth_semitones: float = 2.0,
                                       sr: int = SR) -> JitterConditionSignal:
    """D/E보다 훨씬 빠르고 큰 폭의 pitch 변화(빠른 억양 강세/glissando 근사) - SP-V2-002가
    이미 지적한 "테스트한 glide가 너무 느렸을 수 있다"는 우려를 직접 겨냥한 stress test."""
    f0_fn = lambda t: f0_center * (2.0 ** (depth_semitones * np.sin(2 * np.pi * rate * t) / 12.0))
    waveform, starts, true_f0 = _period_synchronous_synthesis(f0_fn, sr, duration, jitter_std=0.0, shimmer_std=0.0, seed=6)
    return JitterConditionSignal(
        name="F_fast_accent_no_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=0.0, true_shimmer_std=0.0,
        description=f"Fast large-excursion F0 modulation ({rate:.0f} Hz, +/-{depth_semitones:.1f} semitones), no true jitter.",
    )


def condition_g_fast_accent_with_jitter(duration: float = 1.0, f0_center: float = 180.0,
                                         rate: float = 15.0, depth_semitones: float = 2.0,
                                         sr: int = SR) -> JitterConditionSignal:
    f0_fn = lambda t: f0_center * (2.0 ** (depth_semitones * np.sin(2 * np.pi * rate * t) / 12.0))
    waveform, starts, true_f0 = _period_synchronous_synthesis(
        f0_fn, sr, duration, jitter_std=JITTER_STD, shimmer_std=SHIMMER_STD, seed=7,
    )
    return JitterConditionSignal(
        name="G_fast_accent_with_jitter", waveform=waveform, sr=sr, cycle_start_samples=starts,
        true_f0_intended=true_f0, true_jitter_std=JITTER_STD, true_shimmer_std=SHIMMER_STD,
        description=f"Fast large-excursion F0 modulation ({rate:.0f} Hz, +/-{depth_semitones:.1f} semitones) + true jitter.",
    )


def all_conditions() -> list:
    return [
        condition_a_stable_with_jitter(),
        condition_b_glide_no_jitter(),
        condition_c_glide_with_jitter(),
        condition_d_vibrato_no_jitter(),
        condition_e_vibrato_with_jitter(),
        condition_f_fast_accent_no_jitter(),
        condition_g_fast_accent_with_jitter(),
    ]
