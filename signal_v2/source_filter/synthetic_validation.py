"""SP-V2-003: IAIF를 알려진 source-filter 합성 신호로 먼저 검증한다.

실제 음성에는 ground-truth 성문파가 없으므로, Rosenberg 모델(1971)로 만든 알려진
성문파 유량(flow) 펄스열을 알려진 성도 공진(포먼트) 필터에 통과시키고 방사
효과(미분 근사)를 적용해 "정답을 아는 합성 발화"를 만든다. IAIF가 이 합성 발화에서
(1) 원래의 성문파 유량 파형 모양을 얼마나 잘 복원하는지, (2) 필터에 사용한 실제
포먼트 주파수를 얼마나 정확히 복원하는지를 정량 검증한다.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.source_filter.iaif import iaif_windowed, lpc_to_formants

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "source_filter")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def rosenberg_pulse_train(f0: float, sr: int, duration: float, open_quotient: float = 0.7) -> np.ndarray:
    """Rosenberg(1971) B 모델 성문파 유량(flow, 미분 아님) 펄스열.
    한 주기 T 안에서: [0, t1] 여는 국면(raised-cosine 상승), [t1, t1+t2] 닫는
    국면(cosine 하강, t1+t2에서 완전 폐쇄 - 여기서 미분이 불연속적으로 꺾이는
    지점이 실제 성문 폐쇄 순간(GCI)에 해당), 나머지는 폐쇄 구간(flow=0)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    period = 1.0 / f0
    phase = np.mod(t, period)

    t1 = 0.6 * open_quotient * period
    t2 = 0.4 * open_quotient * period

    g = np.zeros(n)
    opening = phase <= t1
    g[opening] = 0.5 * (1 - np.cos(np.pi * phase[opening] / t1))

    closing = (phase > t1) & (phase <= t1 + t2)
    g[closing] = np.cos(np.pi * (phase[closing] - t1) / (2 * t2))

    return g


def cascade_formant_filter(x: np.ndarray, sr: int, formants: list) -> np.ndarray:
    """포먼트(freq, bandwidth) 목록을 2차 공진기 캐스케이드로 적용해 성도 필터를 근사."""
    y = x.copy()
    for freq, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * freq / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(theta), r ** 2]
        y = lfilter(b, a, y)
    return y


def make_synthetic_speech(f0: float = 120.0, sr: int = 16000, duration: float = 0.6,
                           formants: list = None, open_quotient: float = 0.7) -> dict:
    if formants is None:
        formants = [(700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0)]  # 대략 /a/ 모음 근사

    glottal_flow = rosenberg_pulse_train(f0, sr, duration, open_quotient)
    vt_out = cascade_formant_filter(glottal_flow, sr, formants)
    # 방사(lip radiation) 효과 근사: leaky differentiator (~+6dB/oct)
    speech = lfilter([1, -0.98], [1], vt_out)
    peak = np.max(np.abs(speech))
    if peak > 0:
        speech = speech / peak

    return {"speech": speech, "glottal_flow": glottal_flow, "formants": formants, "sr": sr, "f0": f0}


def best_lag_correlation(true_sig: np.ndarray, est_sig: np.ndarray, max_lag: int) -> tuple:
    """그룹 지연(LPC+적분 단계에서 생기는) 보정을 위해 -max_lag..max_lag 범위에서
    최적 시차를 찾고, 그 시차에서의 피어슨 상관계수를 반환."""
    true_n = (true_sig - true_sig.mean())
    est_n = (est_sig - est_sig.mean())

    xcorr = correlate(est_n, true_n, mode="full")
    center = len(true_n) - 1
    window = xcorr[center - max_lag: center + max_lag + 1]
    best_offset = int(np.argmax(np.abs(window))) - max_lag

    if best_offset >= 0:
        a, b = true_n[: len(true_n) - best_offset], est_n[best_offset:]
    else:
        a, b = true_n[-best_offset:], est_n[: len(est_n) + best_offset]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 2 or a.std() == 0 or b.std() == 0:
        return 0.0, best_offset
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr, best_offset


def run_validation() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    synth = make_synthetic_speech()
    sr = synth["sr"]
    result = iaif_windowed(synth["speech"], sr, frame_sec=0.032, hop_sec=0.016)

    corr, lag = best_lag_correlation(synth["glottal_flow"], result["glottal_flow"], max_lag=int(0.01 * sr))

    # 성도 LPC 프레임들에서 얻은 포먼트를 프레임별로 모아 중앙값으로 요약(정상상태 신호이므로
    # 프레임 간 formant 값이 크게 다르지 않아야 한다 - 다르면 그 자체가 불안정성의 증거)
    all_formants = [lpc_to_formants(a, sr) for a in result["vt_lpc_frames"]]
    n_formants_expected = len(synth["formants"])
    per_formant_estimates = [[] for _ in range(n_formants_expected)]
    for frame_formants in all_formants:
        for idx in range(min(len(frame_formants), n_formants_expected)):
            per_formant_estimates[idx].append(frame_formants[idx][0])

    formant_errors_hz = []
    formant_medians = []
    for idx, (true_freq, _) in enumerate(synth["formants"]):
        estimates = per_formant_estimates[idx]
        if estimates:
            median_est = float(np.median(estimates))
            formant_medians.append(median_est)
            formant_errors_hz.append(abs(median_est - true_freq))
        else:
            formant_medians.append(None)
            formant_errors_hz.append(None)

    metrics = {
        "glottal_flow_correlation_at_best_lag": corr,
        "best_lag_samples": lag,
        "true_formants_hz": [f for f, _ in synth["formants"]],
        "estimated_formants_hz_median": formant_medians,
        "formant_abs_error_hz": formant_errors_hz,
        "n_frames": len(result["vt_lpc_frames"]),
    }

    print("=== IAIF synthetic validation ===")
    print(f"glottal flow correlation (best-lag aligned): {corr:.3f} (lag={lag} samples)")
    for true_f, est_f, err in zip(metrics["true_formants_hz"], formant_medians, formant_errors_hz):
        est_str = f"{est_f:.0f} Hz" if est_f is not None else "N/A"
        err_str = f"{err:.0f} Hz" if err is not None else "N/A"
        print(f"  formant true={true_f:.0f} Hz -> estimated median={est_str} (abs error {err_str})")

    with open(os.path.join(OUTPUT_DIR, "synthetic_validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 시각화: 원 waveform / true vs estimated glottal flow / 스펙트럼(포먼트 확인용)
    t = np.arange(len(synth["speech"])) / sr
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    axes[0].plot(t, synth["speech"], linewidth=0.8)
    axes[0].set_title("synthetic speech waveform (known glottal source + known formants)")
    axes[0].set_xlabel("time (s)")

    shift = max(0, -lag)
    axes[1].plot(t, synth["glottal_flow"] / np.max(np.abs(synth["glottal_flow"])), label="true glottal flow", linewidth=1.2)
    est_norm = result["glottal_flow"] / (np.max(np.abs(result["glottal_flow"])) + 1e-9)
    axes[1].plot(t, est_norm, label="IAIF-estimated glottal flow", linewidth=1.0, alpha=0.8)
    axes[1].set_title(f"true vs. estimated glottal flow (corr={corr:.2f} at best-lag alignment)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 0.05)  # 몇 주기만 확대해서 펄스 모양 비교

    freqs = np.fft.rfftfreq(4096, d=1 / sr)
    spec_speech = np.abs(np.fft.rfft(synth["speech"][: 4096] * np.hamming(4096)))
    axes[2].plot(freqs, 20 * np.log10(spec_speech + 1e-9), linewidth=0.8, label="speech spectrum")
    for true_f, _ in synth["formants"]:
        axes[2].axvline(true_f, color="k", linestyle="--", alpha=0.6)
    for est_f in formant_medians:
        if est_f is not None:
            axes[2].axvline(est_f, color="r", linestyle=":", alpha=0.8)
    axes[2].set_title("speech spectrum with true (black dashed) vs. IAIF-estimated (red dotted) formants")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_xlim(0, 4000)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "synthetic_validation.png"), dpi=110)
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    run_validation()
