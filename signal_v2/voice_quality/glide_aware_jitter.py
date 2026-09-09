"""SP-V2-006: glide-aware(잔차 기반) jitter/shimmer.

기존 base 저장소가 이미 실험적으로 발견한 문제(README/브리핑): 빠른 F0 glide가
관습적 jitter/shimmer 값을 크게 오염시킨다 - voice_quality_features.py의 주기
추적이 "다음 주기 예측 위치 = 현재 프레임 F0로 선형 외삽"을 가정하기 때문에,
진짜 glide가 있으면 그 예측이 빗나가서 (실제로는 성대가 완벽히 규칙적인데도)
큰 jitter로 측정된다.

핵심 아이디어(사용자 브리핑 그대로, 문헌의 정확한 공식이라고 주장하지 않는다):
  관측 주기 T_n, 독립적으로 추정한 매끄러운 F0 궤적(pyin)으로부터 얻은 기대 주기
  T_hat_n 이 있을 때, 잔차 epsilon_n = T_n - T_hat_n 만으로 jitter를 정의한다.
  진폭도 동일하게 A_n = 매끄러운 진폭 궤적(관측 진폭의 이동평균으로 근사) + 잔차로
  분해해 residual shimmer를 정의한다.

conventional_jitter_shimmer는 기존 voice_quality_features.py와 동일한 수식(관측
주기/진폭의 연속 차분)을 이 모듈 안에서 독립적으로 재현한 것 - 비교 기준(baseline
arm)으로 쓰기 위함이며 기존 파일을 수정하지 않는다.
"""

import numpy as np


def _segment_amplitudes(mark_samples: np.ndarray, waveform: np.ndarray) -> np.ndarray:
    """mark_samples 사이 구간의 peak-to-peak 진폭. 실제 오디오의 pitch marking(특히
    voice_quality_features.mark_pitch_periods)은 드물게 연속된 mark가 같은 샘플
    위치를 가리키거나(0-length) 역전되는(negative-length) 경우가 있다 - 합성
    신호(항상 단조증가)에서는 나타나지 않아 초기 버전에서 놓쳤던 실제 오디오 전용
    edge case. 그런 구간은 np.ptp가 "zero-size array" 에러를 내므로 진폭 0으로
    처리한다(그 구간에 정보가 없다고 보는 것이 유일하게 안전한 처리)."""
    amps = np.zeros(len(mark_samples) - 1)
    for i in range(len(mark_samples) - 1):
        seg = waveform[mark_samples[i]: mark_samples[i + 1]]
        amps[i] = np.ptp(seg) if seg.size > 0 else 0.0
    return amps


def _smooth_series(x: np.ndarray, window: int = 7) -> np.ndarray:
    """이동평균으로 매끄러운 추세를 추정한다(가장자리는 반사 패딩)."""
    if len(x) < 2:
        return x.copy()
    window = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    window = max(3, window)
    pad = window // 2
    padded = np.pad(x, pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def conventional_jitter_shimmer(mark_samples: np.ndarray, waveform: np.ndarray, sr: int) -> dict:
    """기존 voice_quality_features.py와 동일한 수식(관측 주기의 연속 차분)을 이
    모듈 안에서 독립 재현 - glide로 인한 오염 여부를 대조하기 위한 baseline."""
    if len(mark_samples) < 3:
        return {"jitter": np.nan, "shimmer": np.nan, "n_cycles": 0}

    periods = np.diff(mark_samples) / sr
    amps = _segment_amplitudes(mark_samples, waveform)

    jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) if np.mean(periods) > 0 else np.nan
    shimmer = np.mean(np.abs(np.diff(amps))) / np.mean(amps) if np.mean(amps) > 0 else np.nan
    return {"jitter": float(jitter), "shimmer": float(shimmer), "n_cycles": len(periods)}


def residual_jitter_shimmer(mark_samples: np.ndarray, waveform: np.ndarray, sr: int,
                             f0_smooth_times: np.ndarray = None, f0_smooth_values: np.ndarray = None,
                             amp_smooth_window: int = 7) -> dict:
    """관측 주기에서 "의도된 매끄러운 F0 궤적이 예측하는 기대 주기"를 뺀 잔차만으로
    jitter를 정의한다. f0_smooth_*가 주어지면 그것(예: pyin 프레임 F0, glide/vibrato는
    따라가지만 주기 단위 jitter는 따라가지 못할 만큼 성긴 시간 해상도)을 기대 주기의
    근거로 쓰고, 없으면 관측 주기 자체의 이동평균으로 대체한다(fallback)."""
    if len(mark_samples) < 3:
        return {"residual_jitter": np.nan, "residual_shimmer": np.nan, "n_cycles": 0}

    periods = np.diff(mark_samples) / sr
    mark_times = mark_samples[:-1] / sr

    if f0_smooth_times is not None and f0_smooth_values is not None:
        valid = ~np.isnan(f0_smooth_values)
        if valid.sum() >= 2:
            f0_at_marks = np.interp(mark_times, f0_smooth_times[valid], f0_smooth_values[valid])
            expected_periods = np.where(f0_at_marks > 0, 1.0 / np.where(f0_at_marks > 0, f0_at_marks, 1.0), np.nan)
        else:
            expected_periods = _smooth_series(periods)
    else:
        expected_periods = _smooth_series(periods)

    epsilon = periods - expected_periods
    valid_eps = ~np.isnan(epsilon)
    if valid_eps.sum() < 3:
        residual_jitter = np.nan
    else:
        eps = epsilon[valid_eps]
        residual_jitter = float(np.mean(np.abs(np.diff(eps))) / np.mean(periods)) if np.mean(periods) > 0 else np.nan

    amps = _segment_amplitudes(mark_samples, waveform)
    amp_smooth = _smooth_series(amps, amp_smooth_window)
    delta = amps - amp_smooth
    residual_shimmer = float(np.mean(np.abs(np.diff(delta))) / np.mean(amps)) if np.mean(amps) > 0 else np.nan

    return {
        "residual_jitter": residual_jitter, "residual_shimmer": residual_shimmer,
        "n_cycles": int(valid_eps.sum()),
    }


def praat_jitter_shimmer(waveform: np.ndarray, sr: int, f0min: float = 65.0, f0max: float = 900.0) -> dict:
    """독립적인 Praat(parselmouth) 참조값 - 이 저장소의 자체 jitter/shimmer 구현과
    별개의, 임상 표준에 가까운 구현. README가 이미 명시한 대로 기존 voice_quality_features.py는
    Praat-equivalent가 아니므로, 이 비교가 처음으로 실제 Praat 값과 대조하는 지점이다."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.ascontiguousarray(waveform, dtype=np.float64), sampling_frequency=sr)
    point_process = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return {"praat_jitter": float(jitter_local), "praat_shimmer": float(shimmer_local)}
