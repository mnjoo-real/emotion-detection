"""SP-V2-003: Iterative Adaptive Inverse Filtering (Alku, 1992).

이 파일은 새 알고리즘이 아니라 **기존 논문(Alku 1992)의 iterative glottal
inverse filtering 절차를 최대한 충실히 구현**한 것이다 - 브리핑에서 요구한 대로,
이 구현 자체를 우리의 novelty로 주장하지 않는다. 우리의 잠재적 novelty는 이 위에서
계산하는 source-normalized roughness(source_roughness.py, SP-V2-004) 쪽이다.

절차(문헌에서 널리 재현되는 2-pass 구조 - Alku 1992; Drugman, Bozkurt & Dutoit의
review 등에서 같은 4단계로 요약됨):

  0. DC 제거, (선택) leaky 적분으로 입술 방사(lip radiation, 관습적으로
     ~+6dB/oct 미분으로 근사됨) 효과를 상쇄한다.
  1. 적분된 신호에 저차(p_gl) LPC를 적용해 성문파의 대략적인 스펙트럴 기울기를
     추정하고, 그 역필터로 "성도 관련 신호" 1차 후보를 얻는다.
  2. 그 후보에 고차(p_vt) LPC를 적용해 성도(vocal tract) 필터 1차 추정치를 얻고,
     그 역필터를 **원(비적분) 신호**에 적용해 성문파 미분(dg/dt) 1차 후보를 얻는다.
  3. 1차 성문파 후보에 다시 저차 LPC를 적용해 성문파 추정을 정제한다.
  4. 정제된 성문파 관련 신호에 고차 LPC를 다시 적용해 성도 필터를 정제하고,
     그 역필터를 원 신호에 다시 적용해 최종 성문파 미분 추정치를 얻는다.

출력은 (성문파 유량 추정 g(t), 성문파 유량 미분 추정 dg(t)/dt, 성도 LPC 계수)이다.
harmony_features.py의 Levinson-Durbin 구현을 그대로 재사용해서 LPC 계산 로직이
중복되지 않게 한다.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from harmony_features import levinson_durbin

DEFAULT_LEAK = 0.99  # leaky integrator 계수 - 1.0(완전 적분)은 DC 드리프트가 커서 실무적으로 <1 사용


@dataclass
class IaifResult:
    glottal_flow: np.ndarray
    glottal_flow_derivative: np.ndarray
    vt_lpc: np.ndarray            # 최종 성도 LPC 계수 (a[0]=1)
    glottal_lpc: np.ndarray       # 최종 성문파 LPC 계수 (a[0]=1)


def _lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    """자기상관 + Levinson-Durbin으로 order차 LPC 계수를 구한다 (a[0]=1)."""
    if len(x) <= order:
        a = np.zeros(order + 1)
        a[0] = 1.0
        return a
    windowed = x * np.hamming(len(x))
    r_full = np.correlate(windowed, windowed, mode="full")
    r = r_full[len(windowed) - 1:][: order + 1]
    if r[0] <= 0:
        a = np.zeros(order + 1)
        a[0] = 1.0
        return a
    a, err = levinson_durbin(r, order)
    if err <= 0 or np.any(np.isnan(a)):
        a = np.zeros(order + 1)
        a[0] = 1.0
    return a


def default_vt_order(sr: int) -> int:
    """성도 LPC 차수의 경험적 규칙(2 + fs(kHz)) - 표준 LPC 차수 선택 관례."""
    return int(round(2 + sr / 1000.0))


def iaif(x: np.ndarray, sr: int, p_vt: int = None, p_gl: int = 4, leak: float = DEFAULT_LEAK) -> IaifResult:
    """한 프레임(또는 짧은 발화 전체)에 대해 IAIF를 적용한다.

    p_vt: 성도 LPC 차수 (기본: 2+fs_kHz, 예: 16kHz -> 18차)
    p_gl: 성문파 LPC 차수 (기본 4 - Alku 1992가 쓰는 낮은 차수, 성문파의
          완만한 스펙트럴 기울기만 잡기 위함)
    leak: 방사 효과를 상쇄하는 leaky integrator 계수
    """
    if p_vt is None:
        p_vt = default_vt_order(sr)

    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))

    integrator = ([1.0], [1.0, -leak])

    # 0. 입술 방사 상쇄 (leaky 적분)
    x_int = lfilter(*integrator, x)

    # 1. 성문파 1차 추정(저차 LPC) -> 성도 관련 신호 1차 후보
    a_g1 = _lpc_coeffs(x_int, p_gl)
    h1 = lfilter(a_g1, [1.0], x_int)

    # 2. 성도 1차 추정(고차 LPC) -> 원신호를 역필터링해 성문파 미분 1차 후보
    a_vt1 = _lpc_coeffs(h1, p_vt)
    dg1 = lfilter(a_vt1, [1.0], x)
    g1 = lfilter(*integrator, dg1)

    # 3. 성문파 추정 정제(저차 LPC, 1차 성문파 후보 기반)
    a_g2 = _lpc_coeffs(g1, p_gl)
    h2 = lfilter(a_g2, [1.0], x_int)

    # 4. 성도 추정 정제(고차 LPC) -> 최종 성문파 미분/유량 추정
    a_vt2 = _lpc_coeffs(h2, p_vt)
    dg2 = lfilter(a_vt2, [1.0], x)
    g2 = lfilter(*integrator, dg2)

    return IaifResult(glottal_flow=g2, glottal_flow_derivative=dg2, vt_lpc=a_vt2, glottal_lpc=a_g2)


def lpc_to_formants(a: np.ndarray, sr: int, freq_min: float = 90.0, bandwidth_max: float = 500.0,
                     max_formants: int = 4) -> list:
    """성도 LPC 계수의 근을 formants(freq, bandwidth) 목록으로 변환한다.
    harmony_features.extract_formants_from_frame의 근 처리 로직과 동일한 방식
    (각도->Hz, -0.5*(sr/pi)*log|root|->bandwidth)을 재사용한다."""
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    freqs = np.angle(roots) * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / np.pi) * np.log(np.abs(roots).clip(1e-6, None))

    formants = [
        (f, bw) for f, bw in zip(freqs, bandwidths)
        if freq_min < f < sr / 2 - 100 and bw < bandwidth_max
    ]
    formants.sort(key=lambda fb: fb[0])
    return formants[:max_formants]


def iaif_windowed(waveform: np.ndarray, sr: int, frame_sec: float = 0.032, hop_sec: float = 0.016,
                   p_vt: int = None, p_gl: int = 4, leak: float = DEFAULT_LEAK) -> dict:
    """짧은 프레임(기본 32ms/16ms hop)으로 IAIF를 반복 적용해 overlap-add로 이어붙인
    글로탈 유량/미분 신호 전체를 만든다. 프레임별 성도 LPC 계수도 함께 반환한다
    (formant 궤적/시각화용)."""
    if p_vt is None:
        p_vt = default_vt_order(sr)

    n = len(waveform)
    frame_len = int(round(frame_sec * sr))
    hop_len = int(round(hop_sec * sr))
    window = np.hamming(frame_len)

    glottal_flow = np.zeros(n)
    glottal_deriv = np.zeros(n)
    weight = np.zeros(n)
    vt_lpc_frames = []
    frame_times = []

    n_frames = 1 + max(0, (n - frame_len) // hop_len)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        if end > n:
            break
        frame = waveform[start:end]
        result = iaif(frame, sr, p_vt=p_vt, p_gl=p_gl, leak=leak)

        glottal_flow[start:end] += result.glottal_flow * window
        glottal_deriv[start:end] += result.glottal_flow_derivative * window
        weight[start:end] += window
        vt_lpc_frames.append(result.vt_lpc)
        frame_times.append((start + frame_len / 2) / sr)

    safe_weight = np.where(weight > 1e-6, weight, 1.0)
    glottal_flow = glottal_flow / safe_weight
    glottal_deriv = glottal_deriv / safe_weight

    return {
        "glottal_flow": glottal_flow,
        "glottal_flow_derivative": glottal_deriv,
        "vt_lpc_frames": vt_lpc_frames,
        "frame_times": np.array(frame_times),
        "frame_len": frame_len,
        "hop_len": hop_len,
    }
