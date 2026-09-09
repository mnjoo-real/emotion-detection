"""SP-V2-003: 실제 waveform에 대한 IAIF 정성적 시각화 (최소 요구사항).

합성 신호 검증(synthetic_validation.py)과 별개로, 브리핑이 요구한 대로 실제
발화 몇 개에 대해 "원 waveform / estimated glottal source / estimated
vocal-tract envelope"를 시각화한다 - 실제 데이터에는 ground truth가 없으므로
정량 검증이 아니라 정성적 확인(눈으로 formant 구조가 그럴듯하게 보이는지)이다.

감정 7개 각각에서 대표로 1개 파일씩 골라 확인한다.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.common.audio_io import load_waveform_float
from signal_v2.source_filter.iaif import iaif_windowed
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "source_filter")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

ANALYSIS_SR = 16000


def visualize_one(wav_id: str, situation: str) -> str:
    waveform, sr = load_waveform_float(get_audio_path(wav_id))
    import librosa
    y = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform

    result = iaif_windowed(y, ANALYSIS_SR, frame_sec=0.032, hop_sec=0.016)
    t = np.arange(len(y)) / ANALYSIS_SR

    # 발화 중 가장 에너지가 큰(=가장 뚜렷한 모음일 가능성이 높은) 프레임을 뽑아
    # VT 스펙트럼 포락선(spectral envelope) 확인 - 단순 중앙 프레임은 쉼/저에너지
    # 구간에 걸릴 수 있어 formant 구조 확인에 부적합할 수 있다.
    if not result["vt_lpc_frames"]:
        return ""
    frame_len = result["frame_len"]
    hop_len = result["hop_len"]
    frame_energies = [
        np.sum(y[i * hop_len: i * hop_len + frame_len] ** 2)
        for i in range(len(result["vt_lpc_frames"]))
    ]
    mid_frame_idx = int(np.argmax(frame_energies))
    a_vt = result["vt_lpc_frames"][mid_frame_idx]
    center = mid_frame_idx * hop_len + frame_len // 2
    frame = y[max(0, center - frame_len // 2): center + frame_len // 2]

    w, h = freqz([1.0], a_vt, worN=2048, fs=ANALYSIS_SR)
    frame_spec = np.abs(np.fft.rfft(frame * np.hamming(len(frame)), n=2048))
    freqs = np.fft.rfftfreq(2048, d=1 / ANALYSIS_SR)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(t, y, linewidth=0.6)
    axes[0].set_title(f"raw waveform — {situation} / {wav_id}")
    axes[0].set_xlabel("time (s)")

    axes[1].plot(t, result["glottal_flow_derivative"] / (np.max(np.abs(result["glottal_flow_derivative"])) + 1e-9), linewidth=0.6)
    axes[1].set_title("IAIF-estimated glottal flow derivative (normalized)")
    axes[1].set_xlabel("time (s)")

    spec_db = 20 * np.log10(frame_spec + 1e-9)
    env_db = 20 * np.log10(np.abs(h) + 1e-9)
    env_db_shifted = env_db + (np.max(spec_db) - np.max(env_db))  # 시각 비교용 스케일 정렬
    axes[2].plot(freqs, spec_db, linewidth=0.6, alpha=0.6, label="raw frame spectrum")
    axes[2].plot(w, env_db_shifted, linewidth=1.5, label="IAIF vocal-tract envelope")
    axes[2].set_title("mid-utterance frame: raw spectrum vs. estimated VT envelope")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_xlim(0, ANALYSIS_SR / 2)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"real_{situation}_{wav_id}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    groups = load_groups()
    for situation, wav_ids in groups.items():
        existing = [w for w in wav_ids if os.path.exists(get_audio_path(w))]
        if not existing:
            print(f"[건너뜀] {situation}: 사용 가능한 오디오 없음")
            continue
        wav_id = existing[0]
        path = visualize_one(wav_id, situation)
        print(f"[{situation}] {wav_id} -> {path}")


if __name__ == "__main__":
    main()
