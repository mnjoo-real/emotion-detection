"""SP-V2-012: F008/F009/F010 (+ generic envelope-modulation baseline probe) 전용
추출. F006/F007은 이번 phase에서 freeze(§1) - 재추출하지 않는다.

기존 extract_real_corpus_features.py와 다른 점:
  - F006/F007 계산을 뺐다(불필요한 계산 비용 절감 - 이번 phase는 F008/F009/F010에만 집중).
  - "generic envelope modulation" 확률(§14)을 새로 추가한다 - 개별 배음이 아니라
    전체 waveform의 RMS 진폭 포락선 하나에 F010과 동일한 modulation-spectrum
    기계를 적용한 것. harmonic-specific(F010) vs generic 비교의 baseline이다.
  - duration/speaking-rate 관련 원시 quantity(발화 길이, 유성 구간 길이, 유성
    프레임 수)도 함께 저장해서 §12 confound 분석에 바로 쓸 수 있게 한다.
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harmony_features import FMAX, FMIN, PITCH_SR
from signal_v2.common.audio_io import load_waveform_float
from signal_v2.harmonic.harmonic_modulation import harmonic_modulation_features, _modulation_spectrum, _band_energy, _spectral_centroid_bandwidth, _spectral_entropy, _significant_dominant_freq
from signal_v2.harmonic.harmonic_phase import harmonic_phase_features
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory
from signal_v2.voice_quality.aperiodicity import extract_aperiodicity_features
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")
ANALYSIS_SR = 16000
N_HARMONICS = 5
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
ENV_HOP_SEC = 0.01


def generic_envelope_modulation_features(waveform: np.ndarray, sr: int) -> dict:
    """전체 waveform 하나의 RMS 진폭 포락선에서 harmonic_modulation.py와 동일한
    modulation-spectrum 통계를 계산한다 - "harmonic-specific" 주장을 검증하기
    위한 대조군(§14)."""
    hop = int(round(ENV_HOP_SEC * sr))
    frame = int(hop * 2)
    n_frames = 1 + max(0, (len(waveform) - frame) // hop)
    env = np.array([
        np.sqrt(np.mean(waveform[i * hop: i * hop + frame] ** 2))
        for i in range(n_frames)
    ])

    freqs, spectrum = _modulation_spectrum(env, ENV_HOP_SEC)
    if freqs.size == 0:
        return {
            "envmod_dominant_freq": 0.0, "envmod_low_rate_energy": 0.0, "envmod_high_rate_energy": 0.0,
            "envmod_centroid": 0.0, "envmod_bandwidth": 0.0, "envmod_entropy": 0.0,
        }

    dominant_freq = _significant_dominant_freq(freqs, spectrum)
    low_energy = _band_energy(freqs, spectrum, 0, 5.0)
    high_energy = _band_energy(freqs, spectrum, 5.0, 50.0)
    centroid, bandwidth = _spectral_centroid_bandwidth(freqs, spectrum)
    entropy = _spectral_entropy(spectrum)

    return {
        "envmod_dominant_freq": dominant_freq, "envmod_low_rate_energy": low_energy,
        "envmod_high_rate_energy": high_energy, "envmod_centroid": centroid,
        "envmod_bandwidth": bandwidth, "envmod_entropy": entropy,
    }


def duration_speaking_rate_features(waveform: np.ndarray, sr: int, f0: np.ndarray, frame_times: np.ndarray) -> dict:
    """duration/speaking-rate confound 분석(§12)에 바로 쓸 원시 quantity."""
    total_duration = len(waveform) / sr
    voiced = ~np.isnan(f0) & (f0 > 0)
    voiced_duration = float(voiced.sum() * np.median(np.diff(frame_times))) if voiced.sum() > 1 else 0.0
    n_voiced_frames = int(voiced.sum())
    speaking_rate_proxy = n_voiced_frames / total_duration if total_duration > 0 else 0.0
    return {
        "dur_total_sec": total_duration, "dur_voiced_sec": voiced_duration,
        "dur_n_voiced_frames": n_voiced_frames, "dur_speaking_rate_proxy": speaking_rate_proxy,
    }


def f0_dynamics_features(f0: np.ndarray, frame_times: np.ndarray) -> dict:
    """F0 dynamics confound 분석(§13)용 원시 quantity."""
    voiced = ~np.isnan(f0) & (f0 > 0)
    if voiced.sum() < 4:
        return {"f0dyn_mean": 0.0, "f0dyn_std": 0.0, "f0dyn_range": 0.0, "f0dyn_deriv_energy": 0.0}
    f0v = f0[voiced]
    ref = np.median(f0v)
    semitone = 12 * np.log2(f0v / ref)  # 이미 voiced만 걸러진 배열(len==voiced.sum()) - 다시 voiced로 인덱싱하면 안 됨
    interp = np.interp(frame_times, frame_times[voiced], semitone)
    dt = np.diff(frame_times)
    dt[dt == 0] = np.nan
    slope = np.diff(interp) / dt
    slope = slope[np.isfinite(slope)]
    return {
        "f0dyn_mean": float(np.mean(f0v)), "f0dyn_std": float(np.std(f0v)),
        "f0dyn_range": float(np.max(f0v) - np.min(f0v)),
        "f0dyn_deriv_energy": float(np.sqrt(np.mean(slope ** 2))) if slope.size else 0.0,
    }


def energy_dynamics_features(waveform: np.ndarray, sr: int) -> dict:
    """Energy dynamics confound 분석(§14)용 원시 quantity."""
    hop = int(round(ENV_HOP_SEC * sr))
    frame = int(hop * 2)
    n_frames = 1 + max(0, (len(waveform) - frame) // hop)
    rms = np.array([
        np.sqrt(np.mean(waveform[i * hop: i * hop + frame] ** 2))
        for i in range(n_frames)
    ])
    if rms.size < 2:
        return {"energy_mean": 0.0, "energy_std": 0.0, "energy_range": 0.0}
    return {
        "energy_mean": float(np.mean(rms)), "energy_std": float(np.std(rms)),
        "energy_range": float(np.max(rms) - np.min(rms)),
    }


def extract_all(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    y16k = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform
    y8k = librosa.resample(waveform, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else waveform

    f0, _, _ = librosa.pyin(y8k, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=2048, hop_length=512)
    frame_times = np.arange(len(f0)) * 512 / PITCH_SR

    row = {}
    row.update({f"f008_{k}": v for k, v in extract_aperiodicity_features(y16k, ANALYSIS_SR).items()})

    traj = extract_harmonic_trajectory(y16k, ANALYSIS_SR, frame_times, f0, mode="adaptive", c=3, n_harmonics=N_HARMONICS)
    row.update({f"f009_{k}": v for k, v in harmonic_phase_features(traj).items()})
    row.update({f"f010_{k}": v for k, v in harmonic_modulation_features(traj).items()})

    row.update(generic_envelope_modulation_features(y16k, ANALYSIS_SR))
    row.update(duration_speaking_rate_features(y16k, ANALYSIS_SR, f0, frame_times))
    row.update(f0_dynamics_features(f0, frame_times))
    row.update(energy_dynamics_features(y16k, ANALYSIS_SR))

    return row


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        row = extract_all(audio_path)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"error": str(exc)}
    row["wav_id"] = wav_id
    row["situation"] = situation
    return row


def nested_permutation(groups: dict, seed: int) -> dict:
    """situation별로 "존재하는 파일 전체"를 한 번만 무작위로 섞어서(permutation)
    반환한다. 이후 어떤 N에서든 이 순열의 앞 N/7개를 취하면 자동으로 nested
    subset이 된다(N1400 subset ⊂ N2800 subset ⊂ ...) - np.random.choice를
    N마다 다시 호출하면 같은 seed라도 결과가 subset 관계를 보장하지 않는다
    (요청 크기에 따라 내부 알고리즘이 달라짐)."""
    rng = np.random.RandomState(seed)
    permuted = {}
    for situation, wav_ids in groups.items():
        existing = [w for w in wav_ids if os.path.exists(get_audio_path(w))]
        permuted[situation] = list(rng.permutation(existing))
    return permuted


def sample_from_permutation(permuted: dict, n_per_emotion: int) -> list:
    sample = []
    for situation, ids in permuted.items():
        chosen = ids[:min(n_per_emotion, len(ids))]
        sample.extend((wid, situation, get_audio_path(wid)) for wid in chosen)
    return sample


def run_extraction(n_per_emotion: int, seed: int, out_name: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()
    permuted = nested_permutation(groups, seed)
    sample = sample_from_permutation(permuted, n_per_emotion)
    print(f"표본: {len(sample)}개 파일 ({n_per_emotion}개/감정 x {len(groups)}개 감정, seed={seed}, nested prefix)")

    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_process_task, task) for task in sample]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 200 == 0:
                elapsed = time.perf_counter() - start
                rate = done / elapsed
                print(f"  {done}/{len(sample)} 완료 ({rate:.2f}개/초, 남은 시간 {(len(sample) - done) / rate:.0f}초)")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    out_path = os.path.join(OUTPUT_DIR, out_name)
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
    parser.add_argument("--n-per-emotion", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-name", type=str, default="sp012_features_seed42.csv")
    args = parser.parse_args()
    run_extraction(args.n_per_emotion, args.seed, args.out_name)
