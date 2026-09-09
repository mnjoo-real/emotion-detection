"""SP-V2-004: Raw-spectrum vs source(글로탈)-정규화 roughness.

브리핑의 Phase F - 이 저장소의 잠재적 original contribution 후보. 기존
partial_roughness.py는 **관측된(source*filter 합성) 스펙트럼**에서 FFT 피크를
뽑아 Sethares dissonance를 계산한다(audit §6). 여기서는 같은 Sethares 모델을
그대로 재사용하되(harmony_features.sethares_dissonance, 모델 자체는 바꾸지
않는다), 입력 신호만 세 가지로 바꿔서 비교한다:

  1. raw: 원 waveform에서 뽑은 배음(성도 공진 + 성문파 + 채널 특성이 다 섞인
     관측 스펙트럼) - 기존 partial_roughness.py와 동일한 성격(다른 추출
     알고리즘이지만), ablation의 baseline arm.
  2. source: IAIF로 역필터링한 글로탈 유량 미분 신호에서 뽑은 배음 -
     성도(포먼트) 영향을 최대한 제거한, 성문 여기(excitation) 자체의 harmonic
     구조.
  3. source_norm: source와 같지만 진폭을 프레임별 H1(1차 배음) 기준 비율로
     정규화 - 발화 강도(loudness)/거리 등 절대 진폭 변동을 제거하고 배음 간
     "형태"만 남긴다.

세 변형 모두 signal_v2.pitch_sync.adaptive_window.extract_harmonic_trajectory로
배음 주파수/진폭을 뽑는 동일한 절차를 쓴다 - 알고리즘을 고정하고 입력 신호만
바꿔야 "source-filter 분리 자체의 효과"만 isolate할 수 있기 때문이다.
"""

import numpy as np

from harmony_features import sethares_dissonance
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory, HarmonicTrajectory


def _frame_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    valid = ~np.isnan(freqs) & ~np.isnan(amps) & (amps > 0)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return np.nan
    scores = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            scores.append(sethares_dissonance(freqs[i], freqs[j], amps[i], amps[j]))
    return float(np.mean(scores)) if scores else np.nan


def trajectory_roughness(traj: HarmonicTrajectory, use_normalized_amp: bool = False) -> dict:
    """HarmonicTrajectory(주파수/진폭 궤적)에서 프레임별 roughness를 계산하고
    발화 전체 평균/표준편차로 요약한다.

    use_normalized_amp=True면 amp_db_rel(H1 기준 상대 진폭, dB)을 선형 비율로
    바꿔 사용한다 - 절대 진폭(발화 강도) 영향을 제거한 "형태만" 버전(Phase F의
    source_norm 변형)."""
    n_frames = traj.freq.shape[0]
    if use_normalized_amp:
        amps = 10.0 ** (traj.amp_db_rel / 20.0)
    else:
        amps = traj.amp

    scores = np.full(n_frames, np.nan)
    for t in range(n_frames):
        scores[t] = _frame_roughness(traj.freq[t], amps[t])

    valid = scores[~np.isnan(scores)]
    return {
        "mean": float(np.mean(valid)) if valid.size else 0.0,
        "std": float(np.std(valid)) if valid.size else 0.0,
        "n_valid_frames": int(valid.size),
    }


def compute_roughness_variants(raw_waveform: np.ndarray, glottal_signal: np.ndarray, sr: int,
                                f0_times: np.ndarray, f0_values: np.ndarray,
                                n_harmonics: int = 5, c: float = 3.0) -> dict:
    """raw / source / source_norm 세 roughness 변형을 한 번에 계산."""
    raw_traj = extract_harmonic_trajectory(
        raw_waveform, sr, f0_times, f0_values, mode="adaptive", c=c, n_harmonics=n_harmonics,
    )
    source_traj = extract_harmonic_trajectory(
        glottal_signal, sr, f0_times, f0_values, mode="adaptive", c=c, n_harmonics=n_harmonics,
    )

    raw_rough = trajectory_roughness(raw_traj, use_normalized_amp=False)
    source_rough = trajectory_roughness(source_traj, use_normalized_amp=False)
    source_norm_rough = trajectory_roughness(source_traj, use_normalized_amp=True)

    return {
        "raw_roughness_mean": raw_rough["mean"], "raw_roughness_std": raw_rough["std"],
        "source_roughness_mean": source_rough["mean"], "source_roughness_std": source_rough["std"],
        "source_roughness_norm_mean": source_norm_rough["mean"], "source_roughness_norm_std": source_norm_rough["std"],
        "n_valid_frames_raw": raw_rough["n_valid_frames"],
        "n_valid_frames_source": source_rough["n_valid_frames"],
    }
