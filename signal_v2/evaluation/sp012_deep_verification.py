"""SP-V2-012 overnight Tier-1 §6-8: 0.54Hz artifact 검증, common-envelope 제거,
harmonic-tracker-quality confound 검증. 세 가지 모두 harmonic trajectory의 raw
per-harmonic amplitude가 필요해서(요약 feature만으로는 부족) 1,400개 pilot 표본에서
새로 계산한다.

§6 (0.54Hz artifact): dominant_freq가 진짜 phrase-scale modulation인지, 아니면
utterance-duration에 따른 FFT bin 위치 artifact인지 확인한다.
  - peak_bin_index 자체를 저장(주파수가 아니라 "몇 번째 non-zero bin인지")
  - 1/duration과 dominant_freq의 상관
  - zero-padding을 여러 배로 바꿔가며 같은 trajectory의 peak가 이동하는지 확인
  - Welch PSD로 독립적으로 재확인

§7 (common-envelope removal): 모든 harmonic이 그냥 전체 음량과 같이 오르내리는
것뿐이라면, HMC는 "감정" 신호가 아니라 "발화가 얼마나 시끄러워졌다 조용해졌다
하는지"만 재는 것일 수 있다. 공유 envelope E(t)(harmonic 진폭들의 기하평균)를
빼고(log domain) relative amplitude로 다시 HMC를 계산해서 비교한다.

§8 (tracker-quality confound): HMC가 감정이 아니라 "이 발화에서 harmonic tracker가
얼마나 잘 작동했는지"를 재는 것일 수 있다. valid-frame 비율, 배음 검출 연속성과
HMC의 관계를 확인한다.
"""

import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harmony_features import FMAX, FMIN, PITCH_SR
from signal_v2.common.audio_io import load_waveform_float
from signal_v2.harmonic.harmonic_modulation import _modulation_spectrum, _significant_dominant_freq
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")
REAL_CORPUS_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus", "real_corpus_features.csv")
ANALYSIS_SR = 16000
N_HARMONICS = 5
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)  # 노트북 안정성 위해 여유 confirm


def common_envelope_relative_hmc(amp: np.ndarray) -> dict:
    """공유 envelope E(t) = 배음 진폭들의 기하평균(log domain 평균, 0/nan 안전 처리)을
    빼고(log A_k - log E) relative amplitude에서 HMC를 다시 계산한다."""
    n_harmonics = amp.shape[1]
    log_amp = np.full_like(amp, np.nan)
    valid_mask = amp > 0
    log_amp[valid_mask] = np.log(amp[valid_mask])

    # 각 시점에서 유효한 배음들의 평균(log domain) = 공유 envelope
    with np.errstate(invalid="ignore"):
        log_envelope = np.nanmean(log_amp, axis=1)
    relative_log_amp = log_amp - log_envelope[:, np.newaxis]

    correlations = []
    for i in range(n_harmonics):
        for j in range(i + 1, n_harmonics):
            a, b = relative_log_amp[:, i], relative_log_amp[:, j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() < 4 or a[valid].std() == 0 or b[valid].std() == 0:
                continue
            correlations.append(np.corrcoef(a[valid], b[valid])[0, 1])

    return {
        "relative_hmc": float(np.mean(correlations)) if correlations else np.nan,
        "n_valid_envelope_frames": int(np.sum(~np.isnan(log_envelope))),
    }


def peak_bin_and_zero_padding_check(trajectory: np.ndarray, hop_sec: float,
                                     nfft_multiples: tuple = (1, 2, 4, 8)) -> dict:
    """§6.2/6.3: dominant peak의 bin index와, zero-padding(FFT 길이)을 바꿨을 때
    peak 위치(Hz)가 실제로 안정적인지 확인. 진짜 물리적 peak라면 zero-padding을
    늘려도(주파수 grid만 촘촘해질 뿐) peak 주파수는 거의 그대로여야 한다."""
    valid = ~np.isnan(trajectory)
    if valid.sum() < 4:
        return {"peak_bin_index": np.nan, "zero_pad_freq_std": np.nan}

    t = np.arange(len(trajectory))
    filled = np.interp(t, t[valid], trajectory[valid])
    detrended = filled - filled.mean()
    base_n = len(detrended)

    freqs_at_mult = []
    base_bin_index = np.nan
    for mult in nfft_multiples:
        n_fft = base_n * mult
        windowed = detrended * np.hamming(len(detrended))
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=hop_sec)
        dom_freq = _significant_dominant_freq(freqs, spectrum)
        if dom_freq > 0:
            freqs_at_mult.append(dom_freq)
        if mult == 1:
            # non-zero-padded 기준의 정수 bin index(몇 번째 bin이 peak인지)
            nonzero = freqs > 0
            if np.any(nonzero) and dom_freq > 0:
                base_bin_index = int(round(dom_freq / (freqs[1] - freqs[0])))

    freq_std = float(np.std(freqs_at_mult)) if len(freqs_at_mult) >= 2 else np.nan
    return {"peak_bin_index": base_bin_index, "zero_pad_freq_std": freq_std,
            "zero_pad_freqs": freqs_at_mult}


def welch_peak_check(trajectory: np.ndarray, hop_sec: float) -> float:
    """§6.4: scipy.signal.welch로 독립적인 spectral estimate에서 dominant frequency
    재확인 - FFT peak-picking 한 가지 방법에만 의존하지 않기 위함."""
    from scipy.signal import welch

    valid = ~np.isnan(trajectory)
    if valid.sum() < 8:
        return np.nan
    t = np.arange(len(trajectory))
    filled = np.interp(t, t[valid], trajectory[valid])
    detrended = filled - filled.mean()

    fs = 1.0 / hop_sec
    nperseg = min(len(detrended), max(8, len(detrended) // 2))
    freqs, psd = welch(detrended, fs=fs, nperseg=nperseg)
    nonzero = freqs > 0
    if not np.any(nonzero) or psd[nonzero].max() <= 0:
        return np.nan
    return float(freqs[nonzero][np.argmax(psd[nonzero])])


def track_quality_features(traj) -> dict:
    """§8: harmonic tracker가 이 발화에서 얼마나 잘 작동했는지의 proxy."""
    freq = traj.freq  # (T, n_harmonics)
    valid_ratio_per_harmonic = np.mean(~np.isnan(freq), axis=0)
    overall_valid_ratio = float(np.mean(~np.isnan(freq)))
    # 프레임마다 몇 개 배음이 동시에 유효했는지(연속성 proxy)
    n_valid_per_frame = np.sum(~np.isnan(freq), axis=1)
    continuity = float(np.mean(n_valid_per_frame >= N_HARMONICS))  # 모든 배음이 잡힌 프레임 비율

    return {
        "track_overall_valid_ratio": overall_valid_ratio,
        "track_h1_valid_ratio": float(valid_ratio_per_harmonic[0]) if len(valid_ratio_per_harmonic) else np.nan,
        "track_h5_valid_ratio": float(valid_ratio_per_harmonic[-1]) if len(valid_ratio_per_harmonic) else np.nan,
        "track_full_continuity_ratio": continuity,
    }


def extract_one(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    y16k = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform
    y8k = librosa.resample(waveform, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else waveform

    f0, _, _ = librosa.pyin(y8k, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=2048, hop_length=512)
    frame_times = np.arange(len(f0)) * 512 / PITCH_SR

    traj = extract_harmonic_trajectory(y16k, ANALYSIS_SR, frame_times, f0, mode="adaptive", c=3, n_harmonics=N_HARMONICS)

    hop_sec = float(np.median(np.diff(traj.times))) if len(traj.times) > 1 else 0.01
    duration = len(y16k) / ANALYSIS_SR

    row = {"utterance_duration_sec": duration, "fft_resolution_hz": 1.0 / duration if duration > 0 else np.nan}

    # §7 common-envelope removal
    row.update(common_envelope_relative_hmc(traj.amp))

    # §6 0.54Hz artifact checks - H1과 H3(대표)에 대해서만(비용 절감, 전체 harmonic 다 할 필요 없음)
    for h_idx, h_name in [(0, "h1"), (2, "h3")]:
        pb = peak_bin_and_zero_padding_check(traj.amp[:, h_idx], hop_sec)
        row[f"{h_name}_peak_bin_index"] = pb["peak_bin_index"]
        row[f"{h_name}_zero_pad_freq_std"] = pb["zero_pad_freq_std"]
        row[f"{h_name}_welch_dominant_freq"] = welch_peak_check(traj.amp[:, h_idx], hop_sec)

    # §8 tracker quality
    row.update(track_quality_features(traj))

    return row


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        row = extract_one(audio_path)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"error": str(exc)}
    row["wav_id"] = wav_id
    row["situation"] = situation
    return row


def run(max_workers: int = None) -> str:
    workers = max_workers or MAX_WORKERS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REAL_CORPUS_CSV, encoding="utf-8-sig", newline="") as f:
        rows_in = list(csv.DictReader(f))
    load_groups()
    tasks = [(r["wav_id"], r["situation"], get_audio_path(r["wav_id"])) for r in rows_in]
    print(f"표본: {len(tasks)}개 파일 (deep verification, 기존 1,400개 pilot과 동일)")

    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_task, task) for task in tasks]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} 완료 ({done / (time.perf_counter() - start):.2f}개/초)")

    fieldnames = sorted({k for row in rows for k in row.keys() if k != "zero_pad_freqs"})
    out_path = os.path.join(OUTPUT_DIR, "deep_verification.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    n_errors = sum(1 for r in rows if r.get("error"))
    print(f"\n저장 완료: {out_path} (오류 {n_errors}/{len(rows)}개)")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    run(max_workers=args.max_workers)
