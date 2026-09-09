"""SP-V2-001: 여러 F0 추정기를 하나의 공통 인터페이스로 감싼다.

기존 파이프라인은 librosa.pyin 하나만 쓴다(harmony_features.py 등, 8kHz/65-2093Hz).
여기서는 그 pyin을 그대로 baseline으로 유지하면서, WORLD Harvest / REAPER / (설치되어
있다면) CREPE를 같은 합성 신호에 대해 나란히 돌릴 수 있게 감싼다.

공통 반환 형식: EstimatorResult(name, times, f0, voiced)
  - times: 프레임 중심 시각(초)
  - f0: 각 프레임의 추정 F0(Hz). unvoiced로 판단된 프레임은 np.nan.
  - voiced: 각 프레임의 voicing 여부(bool)
"""

from dataclasses import dataclass

import numpy as np

FMIN = 65.0
FMAX = 900.0  # 합성 조건 중 고음역 glide(최대 350Hz)까지 여유 있게 커버
FRAME_PERIOD_MS = 10.0  # 모든 추정기가 동일한 시간 해상도(10ms hop)로 비교되도록 통일


@dataclass
class EstimatorResult:
    name: str
    times: np.ndarray
    f0: np.ndarray
    voiced: np.ndarray
    error: str = ""


def estimate_pyin(waveform: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX) -> EstimatorResult:
    """기존 파이프라인의 F0 추정기(librosa.pyin)를 동일 파라미터 철학으로 감싼다."""
    import librosa

    hop_length = int(round(sr * FRAME_PERIOD_MS / 1000.0))
    frame_length = max(2048, 4 * int(sr / fmin))
    f0, voiced_flag, _ = librosa.pyin(
        waveform, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    f0 = np.where(voiced_flag, f0, np.nan)
    return EstimatorResult(name="pyin", times=times, f0=f0, voiced=np.asarray(voiced_flag, dtype=bool))


def estimate_world_harvest(waveform: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX) -> EstimatorResult:
    """WORLD vocoder의 Harvest F0 추정기 (Morise et al.) - continuous F0 tracking에
    강하다고 알려진 알고리즘. pyworld는 float64, 1D 입력을 기대한다."""
    import pyworld

    x = np.ascontiguousarray(waveform, dtype=np.float64)
    f0, t = pyworld.harvest(x, sr, f0_floor=fmin, f0_ceil=fmax, frame_period=FRAME_PERIOD_MS)
    voiced = f0 > 0
    f0_out = np.where(voiced, f0, np.nan)
    return EstimatorResult(name="world_harvest", times=t, f0=f0_out, voiced=voiced)


def estimate_reaper(waveform: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX) -> EstimatorResult:
    """REAPER (Robust Epoch And Pitch EstimatoR, Talkin) - pyreaper 바인딩.
    int16 PCM 입력을 기대하므로 float [-1,1] 파형을 스케일링해서 넘긴다."""
    import pyreaper

    x = np.ascontiguousarray(np.clip(waveform, -1.0, 1.0) * 32767.0, dtype=np.int16)
    frame_period_s = FRAME_PERIOD_MS / 1000.0
    _, _, f0_times, f0, _ = pyreaper.reaper(
        x, sr, minf0=fmin, maxf0=fmax, frame_period=frame_period_s,
    )
    voiced = f0 > 0
    f0_out = np.where(voiced, f0, np.nan)
    return EstimatorResult(name="reaper", times=f0_times, f0=f0_out, voiced=voiced)


def estimate_crepe(waveform: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX,
                    periodicity_threshold: float = 0.21) -> EstimatorResult:
    """CREPE(신경망 기반) F0 추정 - torchcrepe. 선택적 baseline: 미설치 시
    benchmark.py가 이 함수 호출을 건너뛰도록 ImportError를 그대로 전파한다."""
    import torch
    import torchcrepe

    hop_length = int(round(sr * FRAME_PERIOD_MS / 1000.0))
    audio = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)

    pitch, periodicity = torchcrepe.predict(
        audio, sr, hop_length, fmin, fmax, model="tiny",
        batch_size=512, device="cpu", return_periodicity=True,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    pitch = torchcrepe.filter.mean(pitch, 3)

    pitch = pitch.squeeze(0).numpy()
    periodicity = periodicity.squeeze(0).numpy()
    voiced = periodicity > periodicity_threshold
    f0_out = np.where(voiced, pitch, np.nan)
    times = np.arange(len(pitch)) * hop_length / sr
    return EstimatorResult(name="crepe_tiny", times=times, f0=f0_out, voiced=voiced)


ESTIMATORS = {
    "pyin": estimate_pyin,
    "world_harvest": estimate_world_harvest,
    "reaper": estimate_reaper,
    "crepe_tiny": estimate_crepe,
}


def run_estimator(name: str, waveform: np.ndarray, sr: int) -> EstimatorResult:
    fn = ESTIMATORS[name]
    try:
        return fn(waveform, sr)
    except Exception as exc:  # noqa: BLE001 - 벤치마크가 한 추정기 실패로 전체 중단되지 않게
        return EstimatorResult(name=name, times=np.array([]), f0=np.array([]), voiced=np.array([]), error=str(exc))
