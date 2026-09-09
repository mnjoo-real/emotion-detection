"""SP-V2-001/002 공통: 합성 신호 ground truth 대비 F0/harmonic 추정 품질 지표.

지표 정의 (사용자 브리핑 Phase A 그대로):
  - RMSE in cents: 두 F0가 모두 유성으로 판정된 프레임에서 cents 단위 오차의 RMS.
  - voiced frame error: 참 voicing과 추정 voicing이 불일치하는 프레임의 비율.
  - octave error rate: cents 오차가 매우 크면서(>200 cents) 옥타브 근방(600-1800 또는
    -1800~-600 cents, 즉 2배/0.5배 근방)인 프레임의 비율 - 일반 tracking 오차와
    옥타브 오류를 구분한다.
  - trajectory smoothness: 추정 F0 궤적의 2차 차분(가속도)의 RMS(cents 단위) - 매끄러운
    합성 glide/vibrato에 대해 값이 작을수록 추정기가 실제 궤적을 잘 따라간다는 뜻이고,
    값이 크면 프레임 단위로 들쭉날쭉한(jittery) 추정이라는 뜻.
"""

import numpy as np


def _interp_ground_truth(est_times: np.ndarray, gt_times: np.ndarray,
                          gt_f0: np.ndarray, gt_voiced: np.ndarray) -> tuple:
    """추정기의 프레임 시각으로 ground-truth F0/voicing을 보간한다.
    F0는 로그 영역에서 선형보간(주파수는 곱셈적 양이므로), voicing은 최근접."""
    if est_times.size == 0:
        return np.array([]), np.array([], dtype=bool)

    gt_voiced_at_est = np.interp(est_times, gt_times, gt_voiced.astype(np.float64)) >= 0.5

    log_f0 = np.full_like(gt_f0, np.nan)
    valid = gt_voiced & ~np.isnan(gt_f0)
    log_f0[valid] = np.log2(gt_f0[valid])
    # np.interp는 nan을 처리하지 못하므로 valid 구간만 보간하고 나머지는 nan으로 채운다
    if valid.sum() >= 2:
        interp_log_f0 = np.interp(est_times, gt_times[valid], log_f0[valid])
        f0_at_est = 2.0 ** interp_log_f0
    else:
        f0_at_est = np.full_like(est_times, np.nan)

    return f0_at_est, gt_voiced_at_est


def compute_metrics(est_times: np.ndarray, est_f0: np.ndarray, est_voiced: np.ndarray,
                     gt_times: np.ndarray, gt_f0: np.ndarray, gt_voiced: np.ndarray) -> dict:
    if est_times.size == 0:
        return {
            "rmse_cents": np.nan, "voiced_frame_error": np.nan,
            "octave_error_rate": np.nan, "trajectory_smoothness_cents": np.nan,
            "n_frames_compared": 0, "n_voiced_matched": 0,
        }

    f0_true_at_est, voiced_true_at_est = _interp_ground_truth(est_times, gt_times, gt_f0, gt_voiced)

    voiced_frame_error = float(np.mean(voiced_true_at_est != np.asarray(est_voiced, dtype=bool)))

    both_voiced = voiced_true_at_est & np.asarray(est_voiced, dtype=bool) & ~np.isnan(f0_true_at_est)
    valid_est_f0 = ~np.isnan(est_f0) & (est_f0 > 0)
    matched = both_voiced & valid_est_f0

    n_matched = int(matched.sum())
    if n_matched == 0:
        cents_err = np.array([])
    else:
        cents_err = 1200.0 * np.log2(est_f0[matched] / f0_true_at_est[matched])

    rmse_cents = float(np.sqrt(np.mean(cents_err ** 2))) if cents_err.size else np.nan

    octave_like = np.abs(np.abs(cents_err) - 1200.0) < 150.0  # 2배/0.5배 근방 +-150 cents
    gross_error = np.abs(cents_err) > 200.0
    octave_error_rate = float(np.mean(octave_like & gross_error)) if cents_err.size else np.nan

    # trajectory smoothness: 연속으로 매칭된(matched) 구간 안에서만 2차 차분
    smoothness_vals = []
    if n_matched >= 3:
        idx = np.where(matched)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs:
            if run.size < 3:
                continue
            log_f0_run = np.log2(est_f0[run])
            cents_run = 1200.0 * log_f0_run
            second_diff = np.diff(cents_run, n=2)
            smoothness_vals.extend(second_diff.tolist())
    trajectory_smoothness = float(np.sqrt(np.mean(np.square(smoothness_vals)))) if smoothness_vals else np.nan

    return {
        "rmse_cents": rmse_cents,
        "voiced_frame_error": voiced_frame_error,
        "octave_error_rate": octave_error_rate,
        "trajectory_smoothness_cents": trajectory_smoothness,
        "n_frames_compared": int(est_times.size),
        "n_voiced_matched": n_matched,
    }
