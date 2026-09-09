"""SP-V2-001: F0 estimator ben치마크용 ground-truth 합성 신호.

실제 데이터에는 ground-truth F0가 없으므로(README에도 명시된 한계), 알려진 F0
궤적을 가진 합성 신호로 각 F0 추정기(pyin/WORLD Harvest/REAPER/CREPE)를 먼저
검증한다. 모든 생성 함수는 SyntheticF0Signal을 반환하며, ground truth는 1ms
간격의 촘촘한 시간축 위에서 제공한다 (평가 시 추정기의 프레임 시각으로 보간해서
비교).

harmonic pulse-train 합성은 위상을 F0(t)를 적분해서(instantaneous-phase 방식)
만들기 때문에 glide/vibrato 구간에서도 위상이 연속적이고 물리적으로 일관된다:
phi_k(t) = 2*pi*k*cumsum(f0(t))/sr, harmonic k의 순간 위상.
"""

from dataclasses import dataclass, field

import numpy as np


SR = 16000  # 모든 합성 신호 공통 샘플레이트 - 실제 파이프라인의 여러 SR(8k/12k/16k)을
            # 전부 다운샘플링 가능한 상한이라 어떤 추정기 설정과도 공정하게 비교 가능


@dataclass
class SyntheticF0Signal:
    name: str
    waveform: np.ndarray
    sr: int
    duration: float
    times: np.ndarray        # ground-truth 평가용 시간축 (1ms 간격)
    f0_true: np.ndarray      # times에서의 참 F0 (Hz), unvoiced 구간은 np.nan
    voiced_true: np.ndarray  # times에서의 참 voicing (bool)
    description: str = ""
    tags: dict = field(default_factory=dict)


def _gt_times(duration: float, gt_hop: float = 0.001) -> np.ndarray:
    return np.arange(0.0, duration, gt_hop)


def _synth_harmonic_stack(
    f0_per_sample: np.ndarray,
    voiced_per_sample: np.ndarray,
    sr: int,
    n_harmonics: int = 8,
    rolloff: float = 0.88,
    amplitude: np.ndarray | float = 1.0,
) -> np.ndarray:
    """순간 F0 궤적(샘플 단위)으로부터 배음 스택(글로탈 펄스열 근사)을 만든다.
    각 배음 진폭은 rolloff^(k-1)로 감쇠(스펙트럴 기울기 근사), Nyquist를 넘는
    배음은 자동으로 제외한다."""
    n = len(f0_per_sample)
    t_idx = np.arange(n)
    y = np.zeros(n)
    safe_f0 = np.where(voiced_per_sample, f0_per_sample, 0.0)
    cum_phase = 2 * np.pi * np.cumsum(safe_f0) / sr  # 기본 주파수의 순간 위상 적분

    if np.isscalar(amplitude):
        amp_env = np.full(n, float(amplitude))
    else:
        amp_env = amplitude

    for k in range(1, n_harmonics + 1):
        # 이 배음이 신호 전체에서 한 번이라도 Nyquist를 넘으면 스킵(에일리어싱 방지)
        if np.any(voiced_per_sample) and np.nanmax(safe_f0[voiced_per_sample] * k) >= sr / 2 - 50:
            continue
        y += (rolloff ** (k - 1)) * np.sin(k * cum_phase)

    y *= amp_env * voiced_per_sample.astype(np.float64)
    peak = np.max(np.abs(y)) if np.any(y) else 1.0
    return y / peak if peak > 0 else y


def _voiced_envelope(n: int, voiced_mask: np.ndarray, ramp_samples: int = 80) -> np.ndarray:
    """voiced/unvoiced 전환부에 급격한 클릭이 생기지 않도록 짧은 ramp를 적용."""
    env = voiced_mask.astype(np.float64)
    if ramp_samples <= 0:
        return env
    kernel = np.ones(ramp_samples) / ramp_samples
    smoothed = np.convolve(env, kernel, mode="same")
    return np.clip(smoothed, 0.0, 1.0)


def stationary_sinusoid(duration: float = 1.0, f0: float = 150.0, sr: int = SR) -> SyntheticF0Signal:
    """조건 A 계열 기반: 고정 F0의 순수 사인파 (배음 없음) - 가장 쉬운 케이스."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    y = np.sin(2 * np.pi * f0 * t)
    times = _gt_times(duration)
    return SyntheticF0Signal(
        name="stationary_sinusoid",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=np.full_like(times, f0), voiced_true=np.ones_like(times, dtype=bool),
        description=f"Pure {f0:.0f} Hz sinusoid, no harmonics, no noise.",
        tags={"f0": f0},
    )


def harmonic_stack(duration: float = 1.0, f0: float = 150.0, sr: int = SR,
                    n_harmonics: int = 8, rolloff: float = 0.88) -> SyntheticF0Signal:
    """정상상태(stationary) F0의 배음 스택 - glottal pulse train 근사."""
    n = int(duration * sr)
    f0_per_sample = np.full(n, f0)
    voiced = np.ones(n, dtype=bool)
    y = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)
    times = _gt_times(duration)
    return SyntheticF0Signal(
        name="harmonic_stack",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=np.full_like(times, f0), voiced_true=np.ones_like(times, dtype=bool),
        description=f"{n_harmonics}-harmonic stack at {f0:.0f} Hz, rolloff={rolloff}.",
        tags={"f0": f0, "n_harmonics": n_harmonics},
    )


def linear_glide(duration: float = 1.0, f0_start: float = 100.0, f0_end: float = 250.0,
                  sr: int = SR, n_harmonics: int = 6, rolloff: float = 0.88) -> SyntheticF0Signal:
    """선형 F0 glide - 성대 F0가 음표처럼 고정되지 않고 연속적으로 움직이는 조건."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    f0_per_sample = f0_start + (f0_end - f0_start) * (t / duration)
    voiced = np.ones(n, dtype=bool)
    y = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)
    times = _gt_times(duration)
    f0_true = f0_start + (f0_end - f0_start) * (times / duration)
    return SyntheticF0Signal(
        name="linear_glide",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=f0_true, voiced_true=np.ones_like(times, dtype=bool),
        description=f"Linear F0 glide {f0_start:.0f}->{f0_end:.0f} Hz over {duration:.2f}s.",
        tags={"f0_start": f0_start, "f0_end": f0_end},
    )


def exponential_glide(duration: float = 1.0, f0_start: float = 100.0, f0_end: float = 300.0,
                       sr: int = SR, n_harmonics: int = 6, rolloff: float = 0.88) -> SyntheticF0Signal:
    """지수(로그선형) F0 glide - semitone 축에서 선형인, 음악적으로 더 자연스러운 glide."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    log_ratio = np.log(f0_end / f0_start)
    f0_per_sample = f0_start * np.exp(log_ratio * t / duration)
    voiced = np.ones(n, dtype=bool)
    y = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)
    times = _gt_times(duration)
    f0_true = f0_start * np.exp(log_ratio * times / duration)
    return SyntheticF0Signal(
        name="exponential_glide",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=f0_true, voiced_true=np.ones_like(times, dtype=bool),
        description=f"Exponential F0 glide {f0_start:.0f}->{f0_end:.0f} Hz over {duration:.2f}s.",
        tags={"f0_start": f0_start, "f0_end": f0_end},
    )


def vibrato(duration: float = 1.0, f0_center: float = 180.0, vibrato_rate: float = 5.5,
            vibrato_depth_semitones: float = 0.5, sr: int = SR,
            n_harmonics: int = 6, rolloff: float = 0.88) -> SyntheticF0Signal:
    """비브라토 - 진짜 jitter 없이 F0 자체가 매끄럽게 주기적으로 흔들리는 조건.
    Phase G(glide-aware jitter)의 D/E 조건과 직접 연결된다: 좋은 jitter measure는
    이 조건에서 낮게 나와야 한다(진짜 cycle-to-cycle perturbation이 아니므로)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    semitone_mod = vibrato_depth_semitones * np.sin(2 * np.pi * vibrato_rate * t)
    f0_per_sample = f0_center * (2.0 ** (semitone_mod / 12.0))
    voiced = np.ones(n, dtype=bool)
    y = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)
    times = _gt_times(duration)
    semitone_mod_gt = vibrato_depth_semitones * np.sin(2 * np.pi * vibrato_rate * times)
    f0_true = f0_center * (2.0 ** (semitone_mod_gt / 12.0))
    return SyntheticF0Signal(
        name="vibrato",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=f0_true, voiced_true=np.ones_like(times, dtype=bool),
        description=(f"Vibrato at {f0_center:.0f} Hz center, {vibrato_rate:.1f} Hz rate, "
                     f"+/-{vibrato_depth_semitones:.2f} semitone depth, no true jitter."),
        tags={"f0_center": f0_center, "vibrato_rate": vibrato_rate},
    )


def abrupt_pitch_jump(duration: float = 1.0, f0_before: float = 120.0, f0_after: float = 220.0,
                       jump_at: float = 0.5, sr: int = SR,
                       n_harmonics: int = 6, rolloff: float = 0.88) -> SyntheticF0Signal:
    """돌발적 피치 도약 - 성대 등록(register) 전환이나 강한 억양 강세를 근사."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    f0_per_sample = np.where(t < jump_at, f0_before, f0_after)
    voiced = np.ones(n, dtype=bool)
    y = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)
    times = _gt_times(duration)
    f0_true = np.where(times < jump_at, f0_before, f0_after)
    return SyntheticF0Signal(
        name="abrupt_pitch_jump",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=f0_true, voiced_true=np.ones_like(times, dtype=bool),
        description=f"Abrupt jump {f0_before:.0f}->{f0_after:.0f} Hz at t={jump_at:.2f}s.",
        tags={"f0_before": f0_before, "f0_after": f0_after, "jump_at": jump_at},
    )


def additive_white_noise(duration: float = 1.0, f0: float = 150.0, snr_db: float = 10.0,
                          sr: int = SR, n_harmonics: int = 6, rolloff: float = 0.88) -> SyntheticF0Signal:
    """정상상태 배음 스택 + 가산성 백색잡음(지정 SNR) - 잡음 강건성 테스트."""
    n = int(duration * sr)
    f0_per_sample = np.full(n, f0)
    voiced = np.ones(n, dtype=bool)
    clean = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)

    rng = np.random.RandomState(42)
    noise = rng.normal(0, 1, n)
    sig_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power) if noise_power > 0 else noise
    y = clean + noise

    times = _gt_times(duration)
    return SyntheticF0Signal(
        name=f"additive_noise_snr{snr_db:g}db",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=np.full_like(times, f0), voiced_true=np.ones_like(times, dtype=bool),
        description=f"{f0:.0f} Hz harmonic stack + white noise at {snr_db:.0f} dB SNR.",
        tags={"f0": f0, "snr_db": snr_db},
    )


def breathy_noisy_harmonic(duration: float = 1.0, f0: float = 150.0, sr: int = SR,
                            n_harmonics: int = 6, rolloff: float = 0.88,
                            noise_mix: float = 0.4) -> SyntheticF0Signal:
    """호흡성(breathy)/쉰 목소리 근사 - 배음 스택에 완만한 고역강조(high-shelf) 필터링된
    잡음을 섞어 준주기성(quasi-periodicity)을 낮춘다(HNR이 낮은, aperiodicity가 높은
    조건). 실제 기식(aspiration) 잡음은 성문의 난류(turbulence)에서 나오며 저주파
    에너지가 약하고 중/고역에 걸쳐 있다는 특성을 반영해 1차 미분(간단한 +6dB/oct
    고역강조)으로 백색잡음을 정형화한다.

    이전 버전은 여기서 1차 저역통과(누설 적분기, b=0.02)를 썼는데, 그 필터의
    -3dB 컷오프가 계산해보면 약 51Hz(fc = -sr/(2*pi)*ln(1-b))로 사실상 거의 DC에
    가까운 매우 좁은 저역통과였다 - 100Hz대 F0 신호에 섞였을 때 "숨소리"가 아니라
    거의 감지되지 않는 완만한 드리프트만 추가하는 버그였다(SP-V2-008에서 이
    조건의 noise_mix를 0.2/0.4/0.6으로 바꿔도 결과가 거의 변하지 않는 것을 보고
    발견함 - 상세 내용은 docs/signal_processing_v2_log.md SP-V2-001/008 참고)."""
    n = int(duration * sr)
    f0_per_sample = np.full(n, f0)
    voiced = np.ones(n, dtype=bool)
    clean = _synth_harmonic_stack(f0_per_sample, voiced, sr, n_harmonics, rolloff)

    rng = np.random.RandomState(7)
    raw_noise = rng.normal(0, 1, n)
    noise = np.diff(raw_noise, prepend=raw_noise[0])  # 1차 미분 = 완만한 고역강조
    noise = noise / (np.max(np.abs(noise)) + 1e-9)

    y = (1 - noise_mix) * clean + noise_mix * noise
    peak = np.max(np.abs(y))
    y = y / peak if peak > 0 else y

    times = _gt_times(duration)
    return SyntheticF0Signal(
        name="breathy_noisy_harmonic",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=np.full_like(times, f0), voiced_true=np.ones_like(times, dtype=bool),
        description=f"{f0:.0f} Hz harmonic stack mixed {noise_mix:.0%} with low-passed noise (breathy).",
        tags={"f0": f0, "noise_mix": noise_mix},
    )


def voiced_unvoiced_transition(duration: float = 1.2, f0: float = 160.0, sr: int = SR,
                                n_harmonics: int = 6, rolloff: float = 0.88) -> SyntheticF0Signal:
    """유성/무성 전환 - voiced(harmonic) - unvoiced(noise) - voiced 세 구간을 순서대로
    배치해 voicing-decision 정확도를 별도로 측정한다."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    third = duration / 3.0
    voiced_mask = (t < third) | (t >= 2 * third)

    f0_per_sample = np.full(n, f0)
    y_voiced = _synth_harmonic_stack(f0_per_sample, voiced_mask, sr, n_harmonics, rolloff)

    rng = np.random.RandomState(3)
    unvoiced_noise = rng.normal(0, 1, n)
    unvoiced_noise = unvoiced_noise / np.max(np.abs(unvoiced_noise))
    unvoiced_mask = ~voiced_mask

    env = _voiced_envelope(n, voiced_mask.astype(np.float64))
    y = y_voiced + (1 - env) * unvoiced_noise * 0.3

    times = _gt_times(duration)
    voiced_true = (times < third) | (times >= 2 * third)
    f0_true = np.where(voiced_true, f0, np.nan)
    return SyntheticF0Signal(
        name="voiced_unvoiced_transition",
        waveform=y, sr=sr, duration=duration, times=times,
        f0_true=f0_true, voiced_true=voiced_true,
        description="voiced-unvoiced-voiced thirds, tests voicing-decision accuracy at transitions.",
        tags={"f0": f0},
    )


def all_conditions() -> list:
    """SP-V2-001 벤치마크가 도는 전체 합성 조건 목록."""
    return [
        stationary_sinusoid(),
        harmonic_stack(),
        linear_glide(),
        exponential_glide(),
        vibrato(),
        abrupt_pitch_jump(),
        additive_white_noise(snr_db=20.0),
        additive_white_noise(snr_db=10.0),
        additive_white_noise(snr_db=0.0),
        breathy_noisy_harmonic(),
        voiced_unvoiced_transition(),
        # 여성 화자대 음역대 반복(성별 F0 confound가 F0 추정 자체에도 영향을 주는지 확인)
        harmonic_stack(f0=220.0),
        linear_glide(f0_start=180.0, f0_end=350.0),
    ]
