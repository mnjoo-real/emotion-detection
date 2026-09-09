"""SP-V2-007: 연속 웨이블릿 변환(CWT) 기반 multi-scale prosody.

기존 melody_rhythm_features.py의 음(note) 분리는 그대로 유지한다(discrete,
음소/음절 수준의 이산적 표현). 여기서는 그와 별개로, F0를 화자-상대적
반음(semitone) 궤적으로 바꾼 뒤 CWT로 여러 시간 스케일(microprosody, 음절,
단어, 구, 발화 전체)로 분해하는 **연속** 표현을 추가한다 - Suni, Aalto, Vainio
등의 "Wavelet Prosody Analyzer" 계열 연구가 쓰는 것과 같은 일반적인 접근(Mexican
hat CWT)이며, 이 알고리즘 자체를 novelty로 주장하지 않는다. 우리의 목표는 "감정이
어떤 시간 스케일에서 F0/음질 궤적을 바꾸는가"를 보는 파생 feature 설계다.
"""

from dataclasses import dataclass, field

import numpy as np
import pywt

WAVELET = "mexh"  # Mexican hat - prosody CWT 문헌에서 흔히 쓰는 선택
UNIFORM_HOP_SEC = 0.01  # F0 궤적을 균일 시간축으로 재샘플링(CWT는 균일 샘플링 전제)

# 시간 스케일 대역 경계(초 단위 "의사 주기" pseudo-period) - micro/syllable/word/phrase/utterance
SCALE_BANDS = [
    ("microprosody", 0.02, 0.06),
    ("syllable", 0.06, 0.25),
    ("word", 0.25, 0.6),
    ("phrase", 0.6, 1.5),
    ("utterance", 1.5, 4.0),
]


@dataclass
class CwtProsodyResult:
    times: np.ndarray
    signal: np.ndarray            # 분석에 쓰인 균일 샘플링 신호(예: 반음 F0 궤적)
    scales: np.ndarray
    pseudo_periods: np.ndarray    # scales에 대응하는 의사 주기(초)
    coefficients: np.ndarray      # (n_scales, n_times)


def to_semitone_relative(f0_times: np.ndarray, f0_values: np.ndarray, ref: str = "median") -> tuple:
    """F0(NaN=무성)를 화자-상대적 반음 궤적으로 바꾼다. 무성 구간은 선형보간으로
    채운다(CWT는 균일하고 결측 없는 신호를 전제 - Suni et al. 계열 연구의 표준 전처리)."""
    voiced = ~np.isnan(f0_values) & (f0_values > 0)
    if voiced.sum() < 2:
        return np.array([]), np.array([])

    f0_ref = np.median(f0_values[voiced]) if ref == "median" else ref
    interp_f0 = np.interp(f0_times, f0_times[voiced], f0_values[voiced])
    semitone = 12 * np.log2(interp_f0 / f0_ref)
    return f0_times, semitone


def resample_uniform(times: np.ndarray, values: np.ndarray, hop_sec: float = UNIFORM_HOP_SEC) -> tuple:
    if len(times) < 2:
        return np.array([]), np.array([])
    uniform_times = np.arange(times[0], times[-1], hop_sec)
    uniform_values = np.interp(uniform_times, times, values)
    return uniform_times, uniform_values


def compute_cwt(signal: np.ndarray, hop_sec: float = UNIFORM_HOP_SEC,
                 min_period: float = 0.02, max_period: float = 4.0, n_scales: int = 40) -> tuple:
    """signal(균일 샘플링)에 대해 min_period~max_period(초) 범위를 로그 간격으로
    덮는 CWT를 계산한다. 반환: coefficients(n_scales, n_times), scales, pseudo_periods(초)."""
    fs = 1.0 / hop_sec
    center_freq = pywt.central_frequency(WAVELET)  # mexh의 정규화된 중심 주파수

    target_periods = np.geomspace(min_period, max_period, n_scales)
    target_freqs = 1.0 / target_periods
    scales = center_freq * fs / target_freqs

    coefficients, freqs = pywt.cwt(signal, scales, WAVELET, sampling_period=1.0 / fs)
    pseudo_periods = 1.0 / freqs
    return coefficients, scales, pseudo_periods


def analyze_prosody_trajectory(f0_times: np.ndarray, f0_values: np.ndarray) -> CwtProsodyResult:
    raw_times, semitone = to_semitone_relative(f0_times, f0_values)
    if semitone.size < 4:
        return CwtProsodyResult(times=np.array([]), signal=np.array([]), scales=np.array([]),
                                 pseudo_periods=np.array([]), coefficients=np.zeros((0, 0)))

    uniform_times, uniform_signal = resample_uniform(raw_times, semitone)
    duration = uniform_times[-1] - uniform_times[0] if uniform_times.size else 0.0
    max_period = min(4.0, max(0.5, duration))

    coeffs, scales, pseudo_periods = compute_cwt(uniform_signal, max_period=max_period)
    return CwtProsodyResult(times=uniform_times, signal=uniform_signal, scales=scales,
                             pseudo_periods=pseudo_periods, coefficients=coeffs)


def _entropy_of_magnitudes(x: np.ndarray) -> float:
    mag = np.abs(x)
    total = mag.sum()
    if total <= 0:
        return 0.0
    p = mag / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def band_features(result: CwtProsodyResult) -> dict:
    """SCALE_BANDS 각각에 대해 energy/variance/entropy/peak_density/sign_change_rate를
    계산하고, band 간 cross-scale energy ratio와 전체 dominant scale도 함께 낸다."""
    out = {}
    if result.coefficients.size == 0:
        for band_name, _, _ in SCALE_BANDS:
            out.update({
                f"{band_name}_energy": 0.0, f"{band_name}_variance": 0.0,
                f"{band_name}_entropy": 0.0, f"{band_name}_peak_density": 0.0,
                f"{band_name}_sign_change_rate": 0.0,
            })
        out["dominant_scale_period_sec"] = 0.0
        for band_name, _, _ in SCALE_BANDS:
            out[f"{band_name}_energy_ratio"] = 0.0
        return out

    duration = result.times[-1] - result.times[0] if result.times.size > 1 else 1.0
    total_energy_all = float(np.sum(result.coefficients ** 2)) + 1e-12

    band_energies = {}
    for band_name, lo, hi in SCALE_BANDS:
        mask = (result.pseudo_periods >= lo) & (result.pseudo_periods < hi)
        if not np.any(mask):
            out.update({
                f"{band_name}_energy": 0.0, f"{band_name}_variance": 0.0,
                f"{band_name}_entropy": 0.0, f"{band_name}_peak_density": 0.0,
                f"{band_name}_sign_change_rate": 0.0,
            })
            band_energies[band_name] = 0.0
            continue

        band_signal = result.coefficients[mask].mean(axis=0)  # 이 대역의 시간에 따른 대표 신호
        energy = float(np.sum(band_signal ** 2))
        band_energies[band_name] = energy

        out[f"{band_name}_energy"] = energy
        out[f"{band_name}_variance"] = float(np.var(band_signal))
        out[f"{band_name}_entropy"] = _entropy_of_magnitudes(band_signal)

        peaks = np.where((band_signal[1:-1] > band_signal[:-2]) & (band_signal[1:-1] > band_signal[2:]))[0]
        out[f"{band_name}_peak_density"] = float(len(peaks) / duration) if duration > 0 else 0.0

        signs = np.sign(band_signal)
        signs[signs == 0] = 1
        sign_changes = np.sum(np.diff(signs) != 0)
        out[f"{band_name}_sign_change_rate"] = float(sign_changes / duration) if duration > 0 else 0.0

    total_band_energy = sum(band_energies.values()) + 1e-12
    for band_name in band_energies:
        out[f"{band_name}_energy_ratio"] = band_energies[band_name] / total_band_energy

    dominant_band = max(band_energies.items(), key=lambda kv: kv[1])[0]
    dominant_scale_idx = np.argmax(np.sum(result.coefficients ** 2, axis=1))
    out["dominant_scale_period_sec"] = float(result.pseudo_periods[dominant_scale_idx])
    out["dominant_band"] = dominant_band

    return out


def extract_cwt_prosody_features(f0_times: np.ndarray, f0_values: np.ndarray) -> dict:
    result = analyze_prosody_trajectory(f0_times, f0_values)
    return band_features(result)
