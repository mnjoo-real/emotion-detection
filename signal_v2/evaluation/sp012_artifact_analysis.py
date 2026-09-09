"""SP-V2-012 overnight §6-8 결과 분석: deep_verification.csv + real_corpus_features.csv +
aux_features_original_1400.csv를 합쳐서 0.54Hz artifact / common-envelope / tracker-quality
세 가지 verdict를 낸다.
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
SP012_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")
REAL_CORPUS_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus", "real_corpus_features.csv")


def load_merged() -> pd.DataFrame:
    deep = pd.read_csv(os.path.join(SP012_DIR, "deep_verification.csv"), encoding="utf-8-sig")
    aux = pd.read_csv(os.path.join(SP012_DIR, "aux_features_original_1400.csv"), encoding="utf-8-sig")
    main = pd.read_csv(REAL_CORPUS_CSV, encoding="utf-8-sig")

    for df in (deep, aux, main):
        if "error" in df.columns:
            df.drop(df[df["error"].fillna("") != ""].index, inplace=True)

    merged = deep.merge(aux[["wav_id", "energy_std", "energy_mean"]], on="wav_id", how="inner")
    merged = merged.merge(
        main[["wav_id", "f010_inter_harmonic_modulation_coherence", "f010_h1_modulation_dominant_freq",
              "f010_h3_modulation_dominant_freq", "f008_low_band_periodicity_mean"]],
        on="wav_id", how="inner",
    )

    from signal_v2.evaluation.run_family_evaluation import load_gender_map
    gender_map = load_gender_map()
    merged["gender"] = merged["wav_id"].map(gender_map)
    return merged


def emotion_eps2(values: np.ndarray, labels: np.ndarray) -> dict:
    valid = np.isfinite(values)
    v, l = values[valid], labels[valid]
    groups = [v[l == e] for e in np.unique(l)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"epsilon_sq": np.nan, "p_value": np.nan}
    n = sum(len(g) for g in groups)
    k = len(groups)
    h, p = stats.kruskal(*groups)
    eps = epsilon_squared(h, n, k)
    return {"epsilon_sq": eps, "p_value": float(p)}


def residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)
    if valid.sum() < covariates.shape[1] + 5:
        return resid
    design = np.column_stack([np.ones(valid.sum()), covariates[valid]])
    coefs, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    resid[valid] = y[valid] - design @ coefs
    return resid


def section_6_artifact_check(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70 + "\n§6: 0.54Hz artifact 검증\n" + "=" * 70)
    out = {}

    # 6.1 dominant_freq vs 1/duration
    for h in ["h1", "h3"]:
        col = f"f010_{h}_modulation_dominant_freq"
        valid = np.isfinite(df[col]) & np.isfinite(df["fft_resolution_hz"]) & (df[col] > 0)
        r_s, p_s = stats.spearmanr(df[col][valid], df["fft_resolution_hz"][valid])
        r_p, p_p = stats.pearsonr(df[col][valid], df["fft_resolution_hz"][valid])
        out[f"{h}_freq_vs_1_over_duration"] = {"spearman_r": float(r_s), "spearman_p": float(p_s),
                                                "pearson_r": float(r_p), "pearson_p": float(p_p), "n": int(valid.sum())}
        print(f"  {h} dominant_freq vs 1/duration: spearman r={r_s:.3f} (p={p_s:.3g}), pearson r={r_p:.3f}")

    # 6.2 peak bin index distribution
    for h in ["h1", "h3"]:
        col = f"{h}_peak_bin_index"
        vals = df[col].dropna()
        bin_counts = vals.value_counts(normalize=True).sort_index()
        pct_bin1 = float(bin_counts.get(1.0, 0.0) * 100)
        pct_bin_le2 = float(bin_counts[bin_counts.index <= 2].sum() * 100) if len(bin_counts) else 0.0
        out[f"{h}_peak_bin_distribution"] = {"pct_at_bin1": pct_bin1, "pct_at_bin_le2": pct_bin_le2,
                                              "median_bin": float(vals.median()) if len(vals) else np.nan}
        print(f"  {h} peak bin index: {pct_bin1:.1f}% at bin=1, {pct_bin_le2:.1f}% at bin<=2, median={vals.median():.1f}")

    # 6.3 zero-padding stability
    for h in ["h1", "h3"]:
        col = f"{h}_zero_pad_freq_std"
        vals = df[col].dropna()
        out[f"{h}_zero_pad_stability"] = {"mean_std_hz": float(vals.mean()), "median_std_hz": float(vals.median())}
        print(f"  {h} zero-padding freq std: mean={vals.mean():.4f}Hz, median={vals.median():.4f}Hz "
              f"(작을수록 peak 위치가 zero-padding에 안정적)")

    # 6.4 FFT-peak vs Welch agreement
    for h in ["h1", "h3"]:
        fft_col = f"f010_{h}_modulation_dominant_freq"
        welch_col = f"{h}_welch_dominant_freq"
        valid = np.isfinite(df[fft_col]) & np.isfinite(df[welch_col]) & (df[fft_col] > 0) & (df[welch_col] > 0)
        if valid.sum() < 5:
            continue
        r, p = stats.spearmanr(df[fft_col][valid], df[welch_col][valid])
        median_ratio = float(np.median(df[welch_col][valid] / df[fft_col][valid]))
        out[f"{h}_fft_vs_welch"] = {"spearman_r": float(r), "p": float(p), "median_ratio_welch_over_fft": median_ratio, "n": int(valid.sum())}
        print(f"  {h} FFT-peak vs Welch: spearman r={r:.3f} (n={valid.sum()}), median(welch/fft)={median_ratio:.2f}")

    # 6.5 duration-matched subset: emotion effect on dominant_freq within a narrow duration band
    dur = df["utterance_duration_sec"]
    lo, hi = dur.quantile(0.4), dur.quantile(0.6)
    matched = df[(dur >= lo) & (dur <= hi)]
    print(f"  duration-matched subset: {lo:.2f}-{hi:.2f}s, n={len(matched)}")
    for h in ["h1", "h3"]:
        col = f"f010_{h}_modulation_dominant_freq"
        full_eps = emotion_eps2(df[col].values, df["situation"].values)
        matched_eps = emotion_eps2(matched[col].values, matched["situation"].values)
        out[f"{h}_duration_matched_emotion_effect"] = {"full_sample_eps2": full_eps["epsilon_sq"],
                                                         "duration_matched_eps2": matched_eps["epsilon_sq"],
                                                         "n_matched": len(matched)}
        print(f"  {h} emotion eps^2: full sample={full_eps['epsilon_sq']:.4f}, duration-matched={matched_eps['epsilon_sq']:.4f}")

    return out


def section_7_envelope_removal(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70 + "\n§7: common-envelope removal (raw HMC vs relative HMC)\n" + "=" * 70)
    out = {}

    raw_col = "f010_inter_harmonic_modulation_coherence"
    rel_col = "relative_hmc"

    valid = np.isfinite(df[raw_col]) & np.isfinite(df[rel_col])
    r, p = stats.spearmanr(df[raw_col][valid], df[rel_col][valid])
    out["raw_vs_relative_correlation"] = {"spearman_r": float(r), "p": float(p), "n": int(valid.sum())}
    print(f"  raw HMC vs relative(envelope-removed) HMC correlation: r={r:.3f} (n={valid.sum()})")

    raw_eps = emotion_eps2(df[raw_col].values, df["situation"].values)
    rel_eps = emotion_eps2(df[rel_col].values, df["situation"].values)
    out["emotion_effect"] = {"raw_eps2": raw_eps["epsilon_sq"], "raw_p": raw_eps["p_value"],
                              "relative_eps2": rel_eps["epsilon_sq"], "relative_p": rel_eps["p_value"]}
    print(f"  emotion eps^2: raw={raw_eps['epsilon_sq']:.4f} (p={raw_eps['p_value']:.3g}), "
          f"relative(envelope-removed)={rel_eps['epsilon_sq']:.4f} (p={rel_eps['p_value']:.3g})")

    # gender effect comparison
    raw_gender = emotion_eps2(df[raw_col].values, df["gender"].values)
    rel_gender = emotion_eps2(df[rel_col].values, df["gender"].values)
    out["gender_effect"] = {"raw_eps2": raw_gender["epsilon_sq"], "relative_eps2": rel_gender["epsilon_sq"]}
    print(f"  gender eps^2: raw={raw_gender['epsilon_sq']:.4f}, relative={rel_gender['epsilon_sq']:.4f}")

    # energy correlation comparison
    valid_e = np.isfinite(df[raw_col]) & np.isfinite(df["energy_std"])
    r_raw_e, _ = stats.spearmanr(df[raw_col][valid_e], df["energy_std"][valid_e])
    valid_e2 = np.isfinite(df[rel_col]) & np.isfinite(df["energy_std"])
    r_rel_e, _ = stats.spearmanr(df[rel_col][valid_e2], df["energy_std"][valid_e2])
    out["energy_correlation"] = {"raw_rho": float(r_raw_e), "relative_rho": float(r_rel_e)}
    print(f"  energy_std correlation: raw HMC rho={r_raw_e:.3f}, relative HMC rho={r_rel_e:.3f}")

    return out


def section_8_tracker_quality(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70 + "\n§8: harmonic-tracker-quality confound\n" + "=" * 70)
    out = {}

    hmc_col = "f010_inter_harmonic_modulation_coherence"
    quality_cols = ["track_overall_valid_ratio", "track_full_continuity_ratio", "f008_low_band_periodicity_mean"]

    print("  raw correlation with HMC:")
    for col in quality_cols:
        valid = np.isfinite(df[hmc_col]) & np.isfinite(df[col])
        r, p = stats.spearmanr(df[hmc_col][valid], df[col][valid])
        out[f"corr_{col}"] = {"spearman_r": float(r), "p": float(p)}
        print(f"    {col}: rho={r:.3f} (p={p:.3g})")

    gender_dummy = (df["gender"] == "male").astype(float).values.reshape(-1, 1)
    covariates = np.hstack([df[quality_cols].values, gender_dummy])
    y = df[hmc_col].values.astype(float)
    resid = residualize(y, covariates)

    raw_eps = emotion_eps2(y, df["situation"].values)
    controlled_eps = emotion_eps2(resid, df["situation"].values)
    out["emotion_effect_controlling_for_track_quality"] = {
        "raw_eps2": raw_eps["epsilon_sq"], "controlled_eps2": controlled_eps["epsilon_sq"],
    }
    print(f"  emotion eps^2: raw={raw_eps['epsilon_sq']:.4f} -> controlling for track-quality+gender="
          f"{controlled_eps['epsilon_sq']:.4f}")

    return out


def run() -> dict:
    df = load_merged()
    print(f"분석 대상: {len(df)}개 파일")

    results = {
        "section_6_artifact_check": section_6_artifact_check(df),
        "section_7_envelope_removal": section_7_envelope_removal(df),
        "section_8_tracker_quality": section_8_tracker_quality(df),
    }

    out_path = os.path.join(SP012_DIR, "artifact_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {out_path}")
    return results


if __name__ == "__main__":
    run()
