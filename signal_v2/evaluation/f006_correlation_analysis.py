"""SP-V2-006 심층 분석 2부: 브리핑 §11이 요구하는 상관/부분상관 분석.

  corr(legacy/Praat jitter, F0 motion energy)  vs  corr(residual jitter, F0 motion energy)

가 핵심 비교다. "좋은 결과"는 legacy/Praat는 F0 움직임과 강하게 상관되고
residual은 그 상관이 크게 줄어드는 것이다. 부분상관은 statsmodels 없이 numpy
최소자승으로 직접 residualize한다(gender dummy + f0_motion을 회귀로 제거한
뒤 남은 잔차와 emotion의 연관을 Kruskal-Wallis epsilon-squared로 본다) -
statsmodels가 설치되어 있지 않아 별도 의존성 추가 없이 구현.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dissonance_tonality_stats import epsilon_squared, effect_label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CORPUS_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus")


def load_merged() -> pd.DataFrame:
    main_df = pd.read_csv(os.path.join(REAL_CORPUS_DIR, "real_corpus_features.csv"), encoding="utf-8-sig")
    praat_df = pd.read_csv(os.path.join(REAL_CORPUS_DIR, "f006_praat_legacy.csv"), encoding="utf-8-sig")
    # "error" 칼럼의 빈 문자열이 pandas에서는 NaN으로 읽히므로 fillna로 보정한다
    # (run_family_evaluation.py의 build_merged_dataset()과 동일한 이유).
    if "error" in main_df.columns:
        main_df = main_df[main_df["error"].fillna("") == ""]
    if "error" in praat_df.columns:
        praat_df = praat_df[praat_df["error"].fillna("") == ""]

    merged = main_df.merge(praat_df, on=["wav_id", "situation"], how="inner", suffixes=("", "_praat"))

    from signal_v2.evaluation.run_family_evaluation import load_gender_map
    gender_map = load_gender_map()
    merged["gender"] = merged["wav_id"].map(gender_map)
    return merged


def spearman_corr(x: pd.Series, y: pd.Series) -> tuple:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[valid], y[valid])
    return float(r), float(p)


def residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """y를 covariates(절편 포함)로 최소자승 회귀한 뒤 잔차를 반환."""
    valid = np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)
    if valid.sum() < covariates.shape[1] + 5:
        return resid
    design = np.column_stack([np.ones(valid.sum()), covariates[valid]])
    coefs, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    pred = design @ coefs
    resid[valid] = y[valid] - pred
    return resid


def run() -> dict:
    df = load_merged()
    print(f"분석 대상: {len(df)}개 파일")

    results = {}

    jitter_metrics = {
        "legacy (existing_repo, real re-extraction)": "existing_repo_jitter",
        "praat": "praat_jitter",
        "conventional (this pilot's own reimplementation)": "f006_conv_jitter",
        "residual (glide-aware)": "f006_resid_jitter",
    }

    print("\n--- corr(jitter metric, F0 motion energy) ---")
    corr_results = {}
    for name, col in jitter_metrics.items():
        r, p = spearman_corr(df[col], df["f006_f0_motion_energy"])
        corr_results[name] = {"spearman_r": r, "p_value": p}
        print(f"  {name:<45} r={r:.3f} (p={p:.4g})")
    results["corr_with_f0_motion"] = corr_results

    print("\n--- corr(jitter metric, emotion) via Kruskal-Wallis eps^2 (pooled) ---")
    emotion_results = {}
    for name, col in jitter_metrics.items():
        sub = df[[col, "situation"]].dropna()
        groups = {e: sub.loc[sub["situation"] == e, col].values for e in sub["situation"].unique()}
        arrays = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 0]
        if len(arrays) < 2:
            continue
        n = sum(len(a) for a in arrays)
        k = len(arrays)
        h, p = stats.kruskal(*arrays)
        eps = epsilon_squared(h, n, k)
        emotion_results[name] = {"epsilon_sq": eps, "p_value": float(p), "effect_label": effect_label(eps)}
        print(f"  {name:<45} eps^2={eps:.4f} ({effect_label(eps)}) p={p:.4g}")
    results["emotion_association_pooled"] = emotion_results

    print("\n--- partial association with emotion, controlling for F0 motion + gender (residualized) ---")
    gender_dummy = (df["gender"] == "male").astype(float).values.reshape(-1, 1)
    f0_motion = df["f006_f0_motion_energy"].values.reshape(-1, 1)
    covariates = np.hstack([f0_motion, gender_dummy])

    partial_results = {}
    for name, col in jitter_metrics.items():
        y = df[col].values.astype(float)
        resid = residualize(y, covariates)
        valid = np.isfinite(resid)
        sub_situation = df["situation"].values[valid]
        sub_resid = resid[valid]
        groups = {e: sub_resid[sub_situation == e] for e in np.unique(sub_situation)}
        arrays = [a for a in groups.values() if len(a) > 0]
        if len(arrays) < 2:
            continue
        n = sum(len(a) for a in arrays)
        k = len(arrays)
        h, p = stats.kruskal(*arrays)
        eps = epsilon_squared(h, n, k)
        partial_results[name] = {"epsilon_sq_after_controlling": eps, "p_value": float(p), "effect_label": effect_label(eps)}
        print(f"  {name:<45} eps^2(residualized)={eps:.4f} ({effect_label(eps)}) p={p:.4g}")
    results["partial_emotion_association"] = partial_results

    out_path = os.path.join(REAL_CORPUS_DIR, "F006", "f006_correlation_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")
    return results


if __name__ == "__main__":
    run()
