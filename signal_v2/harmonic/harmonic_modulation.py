"""SP-V2-010: harmonic modulation spectrum.

각 배음의 진폭 궤적 A_k(t)(HarmonicTrajectory.amp - 절대, 윈도우 길이로 정규화된
진폭)에 대해 "그 진폭 자체가 시간에 따라 얼마나 빠르게 변하는가"(temporal
modulation spectrum)를 계산한다. 목표는 harmonic structure의 순간값이 아니라
변화 속도(rate)를 모델링하는 것 - 예를 들어 syllable 리듬에 맞춰 천천히
변하는지, 그보다 빠른 떨림이 있는지.

코드에서는 임시로 `harmonic_modulation_features`로 부른다(브리핑 그대로) -
논문에서 쓸 최종 용어는 prior-art 검토 후 결정한다는 브리핑의 지시를 따른다.
"""

import numpy as np

from signal_v2.pitch_sync.adaptive_window import HarmonicTrajectory

LOW_RATE_MAX_HZ = 5.0     # 음절(syllable) 리듬대 - 이하는 "느린" 변조
HIGH_RATE_MAX_HZ = 50.0   # 이 위는 modulation spectrum 분석 범위 밖(청각적으로도 별 의미 없음)


MIN_RELATIVE_VARIATION = 0.01  # 궤적의 변동계수(std/|mean|)가 이보다 작으면 "사실상 평탄"으로 간주


def _modulation_spectrum(trajectory: np.ndarray, hop_sec: float) -> tuple:
    """trajectory(시간에 따른 배음 진폭, nan 허용)의 변조 스펙트럼을 FFT로 계산.
    nan(추출 실패 프레임)은 선형보간으로 채우고, 평균을 제거(detrend)한 뒤 Hamming
    윈도우를 적용한다.

    거의 평탄한(변동계수가 매우 작은) 궤적은 스펙트럼을 계산하지 않고 빈 배열을
    반환한다 - 검증 중 발견한 문제: 진짜 변조가 전혀 없는 평탄한 궤적도 Nyquist
    인접 bin에 (수치적 잔여물로 인해) 국소적으로 뚜렷한 - 즉 주변 bin 대비
    prominent한 - 가짜 피크를 만들 수 있어서, "주변 대비 얼마나 튀는가"만으로는
    걸러지지 않았다(peak-vs-median 방식의 1차 시도가 이 사례를 놓쳤음 - 아래
    harmonic_modulation_features의 min_peak_ratio 필터 참고). 절대적 기준(궤적
    자체가 애초에 변하는지)을 먼저 확인하는 게 더 안전하다는 걸 확인한 사례."""
    valid = ~np.isnan(trajectory)
    if valid.sum() < 4:
        return np.array([]), np.array([])

    values = trajectory[valid]
    mean_val = np.mean(values)
    cv = np.std(values) / (abs(mean_val) + 1e-12)
    if cv < MIN_RELATIVE_VARIATION:
        return np.array([]), np.array([])

    t = np.arange(len(trajectory))
    filled = np.interp(t, t[valid], trajectory[valid])
    detrended = filled - filled.mean()
    windowed = detrended * np.hamming(len(detrended))

    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=hop_sec)
    return freqs, spectrum


def _band_energy(freqs: np.ndarray, spectrum: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask] ** 2))


def _spectral_centroid_bandwidth(freqs: np.ndarray, spectrum: np.ndarray) -> tuple:
    power = spectrum ** 2
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * power) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
    return centroid, bandwidth


def _significant_dominant_freq(freqs: np.ndarray, spectrum: np.ndarray, min_peak_ratio: float = 4.0) -> float:
    """가장 강한 bin이 나머지 스펙트럼의 중앙값보다 min_peak_ratio배 이상 크지 않으면
    "뚜렷한 변조가 없다"고 보고 0.0을 반환한다(진짜 변조가 없는 평탄한 궤적에서
    Nyquist 인접 bin에 뜨는 수치적 잔여물을 그럴듯한 dominant frequency로
    잘못 보고하는 것을 막기 위함 - 검증 중 no_modulation 조건에서 실제로 발견된
    문제, docs/signal_processing_v2_log.md SP-V2-010 참고)."""
    nonzero = freqs > 0
    if not np.any(nonzero):
        return 0.0
    sub_spectrum = spectrum[nonzero]
    sub_freqs = freqs[nonzero]
    peak_idx = int(np.argmax(sub_spectrum))
    peak_mag = sub_spectrum[peak_idx]
    median_mag = np.median(sub_spectrum) + 1e-12
    if peak_mag < min_peak_ratio * median_mag:
        return 0.0
    return float(sub_freqs[peak_idx])


def _spectral_entropy(spectrum: np.ndarray) -> float:
    power = spectrum ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def harmonic_modulation_features(traj: HarmonicTrajectory, hop_sec: float = None) -> dict:
    if hop_sec is None:
        hop_sec = float(np.median(np.diff(traj.times))) if len(traj.times) > 1 else 0.01

    n_harmonics = traj.amp.shape[1]
    out = {}
    per_harmonic_dominant = []
    per_harmonic_low_energy_ratio = []

    for k in range(1, n_harmonics + 1):
        freqs, spectrum = _modulation_spectrum(traj.amp[:, k - 1], hop_sec)
        prefix = f"h{k}_modulation"
        if freqs.size == 0:
            out[f"{prefix}_dominant_freq"] = 0.0
            out[f"{prefix}_low_rate_energy"] = 0.0
            out[f"{prefix}_high_rate_energy"] = 0.0
            out[f"{prefix}_centroid"] = 0.0
            out[f"{prefix}_bandwidth"] = 0.0
            out[f"{prefix}_entropy"] = 0.0
            continue

        # DC(0Hz) 성분은 detrend로 이미 제거했으므로 그 다음 bin부터 dominant를 찾되,
        # 뚜렷한 피크가 없으면(=평탄한 궤적) 0.0으로 보고한다(수치적 잔여물 방지)
        dominant_freq = _significant_dominant_freq(freqs, spectrum)

        low_energy = _band_energy(freqs, spectrum, 0, LOW_RATE_MAX_HZ)
        high_energy = _band_energy(freqs, spectrum, LOW_RATE_MAX_HZ, HIGH_RATE_MAX_HZ)
        total_energy = low_energy + high_energy + 1e-12

        centroid, bandwidth = _spectral_centroid_bandwidth(freqs, spectrum)
        entropy = _spectral_entropy(spectrum)

        out[f"{prefix}_dominant_freq"] = dominant_freq
        out[f"{prefix}_low_rate_energy"] = low_energy
        out[f"{prefix}_high_rate_energy"] = high_energy
        out[f"{prefix}_centroid"] = centroid
        out[f"{prefix}_bandwidth"] = bandwidth
        out[f"{prefix}_entropy"] = entropy

        per_harmonic_dominant.append(dominant_freq)
        per_harmonic_low_energy_ratio.append(low_energy / total_energy)

    out["modulation_dominant_freq_mean"] = float(np.mean(per_harmonic_dominant)) if per_harmonic_dominant else 0.0
    out["modulation_low_rate_ratio_mean"] = float(np.mean(per_harmonic_low_energy_ratio)) if per_harmonic_low_energy_ratio else 0.0

    # inter-harmonic modulation coherence: 서로 다른 배음의 진폭 궤적이 함께 오르내리는 정도
    # (단순하고 해석 가능한 대응 지표로 pairwise Pearson 상관의 평균을 쓴다)
    correlations = []
    for i in range(n_harmonics):
        for j in range(i + 1, n_harmonics):
            a, b = traj.amp[:, i], traj.amp[:, j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() < 4 or a[valid].std() == 0 or b[valid].std() == 0:
                continue
            correlations.append(np.corrcoef(a[valid], b[valid])[0, 1])
    out["inter_harmonic_modulation_coherence"] = float(np.mean(correlations)) if correlations else 0.0

    return out
