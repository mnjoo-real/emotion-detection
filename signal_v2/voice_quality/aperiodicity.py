"""SP-V2-008: HNR 단일 스칼라 대신 band-wise periodicity / MVF(Maximum Voiced
Frequency) 기반 표현.

기존 voice_quality_features.py의 hnr_mean/std는 프레임 전체(broadband)를 하나의
숫자로 압축한다 - breathy(고주파 잡음 위주)와 harsh/tense(전대역 불안정)를 구분하지
못한다. 여기서는 WORLD vocoder의 D4C 알고리즘(Morise)이 주는 주파수별
aperiodicity(0=완전 주기적/배음, 1=완전 비주기적/잡음)를 그대로 이용해 band-wise
구조를 만든다. D4C 자체는 기존 논문의 알고리즘이므로(pyworld.d4c), 우리는 이
출력을 emotion-관련 voice-quality 축(breathiness, harshness 등)에 대응하는
파생 feature로 가공하는 부분만 새로 설계한다.
"""

from dataclasses import dataclass

import numpy as np
import pyworld

LOW_BAND_HZ = (0, 2000)
MID_BAND_HZ = (2000, 4000)
HIGH_BAND_HZ = (4000, 8000)
MVF_APERIODICITY_THRESHOLD = 0.5  # 이 값 이상이면 그 주파수부터 "비주기적 우세"로 간주


@dataclass
class AperiodicityResult:
    times: np.ndarray
    f0: np.ndarray
    aperiodicity: np.ndarray   # (n_frames, n_freq_bins), 0..1
    freq_axis: np.ndarray
    mvf: np.ndarray             # (n_frames,) - 프레임별 MVF(Hz), 전 대역 비주기적이면 0


def compute_raw(waveform: np.ndarray, sr: int, f0_floor: float = 65.0, f0_ceil: float = 900.0,
                 frame_period_ms: float = 10.0) -> AperiodicityResult:
    x = np.ascontiguousarray(waveform, dtype=np.float64)
    f0, t = pyworld.harvest(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period_ms)
    ap = pyworld.d4c(x, f0, t, sr)
    n_bins = ap.shape[1]
    freq_axis = np.linspace(0, sr / 2, n_bins)

    mvf = np.zeros(len(t))
    for i in range(len(t)):
        above_thresh = ap[i] >= MVF_APERIODICITY_THRESHOLD
        if not np.any(above_thresh):
            mvf[i] = freq_axis[-1]  # 전 대역이 주기적 우세
            continue
        first_idx = np.argmax(above_thresh)
        mvf[i] = freq_axis[first_idx]

    return AperiodicityResult(times=t, f0=f0, aperiodicity=ap, freq_axis=freq_axis, mvf=mvf)


def _band_mean_periodicity(ap: np.ndarray, freq_axis: np.ndarray, band: tuple, voiced: np.ndarray) -> tuple:
    mask = (freq_axis >= band[0]) & (freq_axis < band[1])
    if not np.any(mask) or not np.any(voiced):
        return 0.0, 0.0
    band_periodicity = 1.0 - ap[voiced][:, mask].mean(axis=1)
    return float(band_periodicity.mean()), float(band_periodicity.std())


def summarize_aperiodicity_features(result: AperiodicityResult) -> dict:
    voiced = result.f0 > 0
    out = {
        "mvf_mean": 0.0, "mvf_std": 0.0,
        "low_band_periodicity_mean": 0.0, "low_band_periodicity_std": 0.0,
        "mid_band_periodicity_mean": 0.0, "mid_band_periodicity_std": 0.0,
        "high_band_periodicity_mean": 0.0, "high_band_periodicity_std": 0.0,
        "aperiodicity_spectral_slope_mean": 0.0,
        "voiced_to_aperiodic_transition_rate": 0.0,
        "n_voiced_frames": int(voiced.sum()),
    }
    if not np.any(voiced):
        return out

    out["mvf_mean"] = float(result.mvf[voiced].mean())
    out["mvf_std"] = float(result.mvf[voiced].std())

    out["low_band_periodicity_mean"], out["low_band_periodicity_std"] = _band_mean_periodicity(
        result.aperiodicity, result.freq_axis, LOW_BAND_HZ, voiced)
    out["mid_band_periodicity_mean"], out["mid_band_periodicity_std"] = _band_mean_periodicity(
        result.aperiodicity, result.freq_axis, MID_BAND_HZ, voiced)
    out["high_band_periodicity_mean"], out["high_band_periodicity_std"] = _band_mean_periodicity(
        result.aperiodicity, result.freq_axis, HIGH_BAND_HZ, voiced)

    # 프레임별 aperiodicity(주파수 축)의 선형 기울기 - 저주파는 주기적, 고주파로 갈수록
    # 비주기적(양의 기울기)인 정상적 성문 여기 패턴에서 얼마나 벗어나는지 요약
    slopes = []
    for i in np.where(voiced)[0]:
        coeffs = np.polyfit(result.freq_axis, result.aperiodicity[i], 1)
        slopes.append(coeffs[0])
    if slopes:
        out["aperiodicity_spectral_slope_mean"] = float(np.mean(slopes)) * 1000  # per-kHz 단위로 스케일

    mvf_voiced = result.mvf[voiced]
    if len(mvf_voiced) > 1:
        transitions = np.abs(np.diff(mvf_voiced)) > (0.1 * result.freq_axis[-1])
        out["voiced_to_aperiodic_transition_rate"] = float(np.mean(transitions))

    return out


def extract_aperiodicity_features(waveform: np.ndarray, sr: int) -> dict:
    result = compute_raw(waveform, sr)
    return summarize_aperiodicity_features(result)
