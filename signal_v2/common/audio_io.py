"""signal_v2 전용 오디오 로딩 유틸리티.

경로 해석(wav_id -> 실제 파일 경로, 4차년도/5차년도_2차 폴더 매핑, situation 라벨
별칭 처리)은 전부 similarity.py의 load_groups()/get_audio_path()에 이미 구현되어
있으므로 재구현하지 않고 그대로 재사용한다. 이 파일은 그 위에 signal_v2 모듈들이
공통으로 쓰는 "wav_id -> (waveform, sr)" 한 단계만 얇게 덧붙인다.
"""

import numpy as np
from scipy.io import wavfile

from similarity import get_audio_path, load_groups  # noqa: F401  (재수출)


def load_waveform_float(audio_path: str) -> tuple:
    """WAV 파일을 (waveform, sr)로 읽고, 기존 *_features.py들과 동일한 방식으로
    모노 변환 + float64 + [-1, 1] 정규화한다 (harmony_features.py 등과 동일 관례)."""
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0
    return waveform, sr


def load_waveform_for_wav_id(wav_id: str) -> tuple:
    return load_waveform_float(get_audio_path(wav_id))
