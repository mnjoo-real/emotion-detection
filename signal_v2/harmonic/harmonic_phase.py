"""SP-V2-009: 배음 간 상대 위상(harmonic phase) feature.

기존 파이프라인은 전부 magnitude/F0 기반이라 위상 정보를 전혀 쓰지 않는다(audit
§13, weakness #6). 배음 k의 위상 phi_k(t)를 1차 배음 위상의 k배와 비교한 상대
위상

    Delta phi_k(t) = phi_k(t) - k * phi_1(t)   (mod 2*pi, [-pi, pi]로 wrap)

를 계산한다. 순수 주기 신호(진짜 배음 구조)라면 Delta phi_k는 시간에 따라
거의 일정해야 한다(위상이 잠겨 있음, phase-locked) - 반대로 배음이 아닌 잡음이나
서로 무관한 성분이면 Delta phi_k가 매 프레임 무작위로 흩어진다.

위상은 원형량(circular quantity, 2*pi에서 wrap)이라 일반 평균/표준편차를 쓰면
안 되고(예: -pi와 +pi는 실제로는 거의 같은 값인데 산술평균은 이걸 정반대로
취급함) circular statistics(복소 지수 평균)를 써야 한다:

    R_k = | mean_t( exp(i * Delta phi_k(t)) ) |   (0~1, 1=완전 위상-잠김, 0=완전 무작위)
    circular_variance_k = 1 - R_k
"""

import numpy as np

from signal_v2.pitch_sync.adaptive_window import HarmonicTrajectory


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def relative_harmonic_phase(traj: HarmonicTrajectory) -> np.ndarray:
    """Delta phi_k(t) = phi_k(t) - k*phi_1(t), (T, n_harmonics), 1차 배음(k=1)은
    정의상 0. wrap된 라디안값을 반환."""
    n_harmonics = traj.phase.shape[1]
    k = np.arange(1, n_harmonics + 1)
    phi_1 = traj.phase[:, [0]]
    delta = traj.phase - k[np.newaxis, :] * phi_1
    return wrap_to_pi(delta)


def circular_stats(angles: np.ndarray) -> dict:
    """1차원 각도 배열(라디안, nan 허용)의 circular mean/circular variance(=1-R)."""
    valid = angles[~np.isnan(angles)]
    if valid.size == 0:
        return {"circular_mean": np.nan, "resultant_length": np.nan, "circular_variance": np.nan}
    z = np.mean(np.exp(1j * valid))
    r = np.abs(z)
    return {
        "circular_mean": float(np.angle(z)),
        "resultant_length": float(r),
        "circular_variance": float(1 - r),
    }


def harmonic_phase_features(traj: HarmonicTrajectory) -> dict:
    """배음별 phase-locking(R_k, circular_variance_k)과, 각 시점에서 여러 배음
    사이의 순간적 위상 일관성(temporal phase instability의 대응 개념)을 요약."""
    delta_phi = relative_harmonic_phase(traj)
    n_harmonics = delta_phi.shape[1]

    out = {}
    resultant_lengths = []
    for k in range(2, n_harmonics + 1):  # k=1은 정의상 Delta phi=0이라 제외
        stats = circular_stats(delta_phi[:, k - 1])
        out[f"h{k}_phase_resultant_length"] = stats["resultant_length"]
        out[f"h{k}_phase_circular_variance"] = stats["circular_variance"]
        if not np.isnan(stats["resultant_length"]):
            resultant_lengths.append(stats["resultant_length"])

    out["mean_phase_coherence"] = float(np.mean(resultant_lengths)) if resultant_lengths else 0.0

    # 시점별로(여러 배음에 걸쳐) 위상이 서로 일관되는지 - 그 시계열의 변동성
    # ("temporal phase instability"에 대응하는 값: 시점별 R이 시간에 따라 얼마나
    # 흔들리는지의 표준편차)
    per_frame_r = []
    for t in range(delta_phi.shape[0]):
        row = delta_phi[t, 1:]  # k=2..n
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            continue
        per_frame_r.append(np.abs(np.mean(np.exp(1j * valid))))
    if per_frame_r:
        out["temporal_phase_instability"] = float(np.std(per_frame_r))
        out["cross_harmonic_coherence_mean"] = float(np.mean(per_frame_r))
    else:
        out["temporal_phase_instability"] = 0.0
        out["cross_harmonic_coherence_mean"] = 0.0

    return out
