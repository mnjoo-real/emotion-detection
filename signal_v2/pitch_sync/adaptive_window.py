"""SP-V2-002: 고정 윈도우 vs pitch-synchronous(피치 적응적) 윈도우 배음 추출.

기존 파이프라인(harmony_features.py 등)은 화자 F0와 무관한 고정 길이 윈도우
(LPC 25ms, pyin ~256ms, FFT-peak 128ms)를 쓴다. 여기서는 그 중 하나를 대표하는
"고정 40ms" baseline과, 화자 F0에 따라 길이가 변하는 L(t) = c*T0(t) (T0=1/F0,
c in {2,3,4}, min/max로 clamp) 윈도우를 같은 신호에 대해 나란히 돌려 배음
주파수/진폭 추정의 안정성을 비교한다.

핵심 설계 결정:
  - 윈도우 길이가 달라지면 FFT 피크의 절대 크기도 달라지므로(윈도우 계수 합에
    비례), sum(window)/2로 나눠 진폭을 정규화한다 - 그래야 윈도우 길이가 다른
    두 방법의 진폭 추정치를 공정하게 비교할 수 있다.
  - 배음 주파수는 포물선(2차) 보간으로 sub-bin 정밀도를 얻는다 - zero-padding을
    과도하게 키우지 않고도 cents 단위로 의미 있는 정밀도를 확보하기 위함.
  - ground-truth F0(t)를 그대로 사용해서(F0 추정 오차와 분리) "F0를 정확히 안다고
    가정했을 때 윈도우 선택 자체가 배음 추정에 미치는 영향"만 isolate한다.
"""

from dataclasses import dataclass

import numpy as np


FIXED_WINDOW_SEC = 0.04   # 기존 파이프라인이 쓰는 "고정 40ms" 스타일 윈도우의 대표값
MIN_WINDOW_SEC = 0.015
MAX_WINDOW_SEC = 0.100
DEFAULT_N_HARMONICS = 5
DEFAULT_HOP_SEC = 0.01
NFFT = 4096


@dataclass
class HarmonicTrajectory:
    times: np.ndarray            # (T,)
    freq: np.ndarray             # (T, n_harmonics) - 추정 배음 주파수(Hz), 못 찾으면 nan
    amp: np.ndarray               # (T, n_harmonics) - 정규화된 절대 진폭(윈도우 길이 보정됨)
    amp_db_rel: np.ndarray       # (T, n_harmonics) - 20*log10(A_k/A_1), 1차 배음 기준 상대 진폭
    phase: np.ndarray            # (T, n_harmonics) - 라디안, 정수 bin 근사(SP-V2-009)
    window_len_sec: np.ndarray   # (T,) - 실제 사용된 윈도우 길이(적응형에서는 시간별로 다름)


def _parabolic_peak(mag: np.ndarray, idx: int) -> tuple:
    """mag[idx] 주변 3점 포물선 보간으로 (refined_index, refined_amplitude)를 구한다."""
    if idx <= 0 or idx >= len(mag) - 1:
        return float(idx), float(mag[idx])
    y1, y2, y3 = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = (y1 - 2 * y2 + y3)
    if denom == 0:
        return float(idx), float(y2)
    delta = 0.5 * (y1 - y3) / denom
    delta = np.clip(delta, -1.0, 1.0)
    refined_amp = y2 - 0.25 * (y1 - y3) * delta
    return float(idx + delta), float(refined_amp)


def _extract_frame_harmonics(waveform: np.ndarray, sr: int, center_sample: int, window_len_samples: int,
                              f0: float, n_harmonics: int, nfft: int = NFFT) -> tuple:
    """center_sample을 중심으로 한 윈도우에서 1..n_harmonics번째 배음의 주파수/진폭/위상을
    추정한다. 위상은 (SP-V2-009) 정수 bin에서 읽은 근사치 - 진폭/주파수처럼 포물선
    보간을 하지 않는다(복소수 보간은 더 복잡해 1차 구현에서는 생략, 문서화된 한계)."""
    half = window_len_samples // 2
    lo, hi = center_sample - half, center_sample - half + window_len_samples
    if lo < 0 or hi > len(waveform):
        nan_arr = np.full(n_harmonics, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy()

    frame = waveform[lo:hi]
    window = np.hamming(len(frame))
    windowed = frame * window

    complex_spectrum = np.fft.rfft(windowed, n=nfft)
    spectrum = np.abs(complex_spectrum)
    freqs = np.fft.rfftfreq(nfft, d=1 / sr)
    norm = np.sum(window) / 2.0
    if norm <= 0:
        nan_arr = np.full(n_harmonics, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy()

    freq_bin_hz = sr / nfft
    out_freq = np.full(n_harmonics, np.nan)
    out_amp = np.full(n_harmonics, np.nan)
    out_phase = np.full(n_harmonics, np.nan)
    for k in range(1, n_harmonics + 1):
        target_freq = k * f0
        if target_freq >= sr / 2 - freq_bin_hz:
            continue
        tolerance_hz = max(15.0, 0.12 * f0)
        lo_bin = max(1, int((target_freq - tolerance_hz) / freq_bin_hz))
        hi_bin = min(len(spectrum) - 2, int((target_freq + tolerance_hz) / freq_bin_hz))
        if hi_bin <= lo_bin:
            continue
        local = spectrum[lo_bin:hi_bin + 1]
        peak_local_idx = int(np.argmax(local))
        peak_idx = lo_bin + peak_local_idx
        refined_idx, refined_amp = _parabolic_peak(spectrum, peak_idx)
        out_freq[k - 1] = refined_idx * freq_bin_hz
        out_amp[k - 1] = refined_amp / norm
        out_phase[k - 1] = np.angle(complex_spectrum[peak_idx])

    return out_freq, out_amp, out_phase


def extract_harmonic_trajectory(waveform: np.ndarray, sr: int, gt_times: np.ndarray, gt_f0: np.ndarray,
                                 mode: str = "fixed", c: float = 3.0,
                                 n_harmonics: int = DEFAULT_N_HARMONICS,
                                 hop_sec: float = DEFAULT_HOP_SEC) -> HarmonicTrajectory:
    """mode='fixed': 고정 FIXED_WINDOW_SEC 윈도우. mode='adaptive': L(t)=c*T0(t), clamp됨."""
    duration = gt_times[-1]
    analysis_times = np.arange(hop_sec, duration - hop_sec, hop_sec)

    freqs_out = np.full((len(analysis_times), n_harmonics), np.nan)
    amps_out = np.full((len(analysis_times), n_harmonics), np.nan)
    phases_out = np.full((len(analysis_times), n_harmonics), np.nan)
    window_lens = np.full(len(analysis_times), np.nan)

    for i, t in enumerate(analysis_times):
        f0 = np.interp(t, gt_times, gt_f0)
        if np.isnan(f0) or f0 <= 0:
            continue

        if mode == "fixed":
            window_sec = FIXED_WINDOW_SEC
        elif mode == "adaptive":
            t0 = 1.0 / f0
            window_sec = np.clip(c * t0, MIN_WINDOW_SEC, MAX_WINDOW_SEC)
        else:
            raise ValueError(f"unknown mode: {mode}")

        window_len_samples = int(round(window_sec * sr))
        if window_len_samples < 8:
            continue
        center_sample = int(round(t * sr))

        freq_k, amp_k, phase_k = _extract_frame_harmonics(waveform, sr, center_sample, window_len_samples, f0, n_harmonics)
        freqs_out[i] = freq_k
        amps_out[i] = amp_k
        phases_out[i] = phase_k
        window_lens[i] = window_sec

    # 1차 배음 기준 상대 진폭(dB) - amps_out[:,0]이 0이거나 nan이면 그 프레임은 nan 유지
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_db_rel = 20 * np.log10(amps_out / amps_out[:, [0]])

    return HarmonicTrajectory(
        times=analysis_times, freq=freqs_out, amp=amps_out,
        amp_db_rel=amp_db_rel, phase=phases_out, window_len_sec=window_lens,
    )
