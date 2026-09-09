"""SP-V2-012 §11-16, §18: F010(및 F008/F009) mechanistic analysis.

기존 1,400개 real-corpus pilot 표본(F008/F009/F010 이미 추출됨) + 새로 추출한
보조 quantity(duration/F0 dynamics/energy dynamics/generic envelope modulation)를
합쳐서, F010이 실제로 어떤 물리적 mechanism을 포착하는지 검증한다:

  §12 duration/speaking-rate confound
  §13 F0 dynamics confound
  §14 energy dynamics confound + generic envelope vs harmonic-specific 비교
  §15 harmonic-order analysis (H1..H5별 effect size/redundancy)
  §16 modulation-rate analysis (관측된 dominant frequency의 실제 분포에 기반해
      band 경계를 사후적으로 정당화 - 임의 경계를 미리 정하지 않음)
  §18 partial association (emotion effect가 duration/F0/speaking-rate/gender를
      통제한 뒤에도 남는지)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dissonance_tonality_stats import epsilon_squared, effect_label
from signal_v2.evaluation.real_corpus_common import classifier_ablation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CORPUS_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus", "real_corpus_features.csv")
AUX_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012", "aux_features_original_1400.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

DURATION_COLS = ["dur_total_sec", "dur_voiced_sec", "dur_n_voiced_frames", "dur_speaking_rate_proxy"]
F0DYN_COLS = ["f0dyn_mean", "f0dyn_std", "f0dyn_range", "f0dyn_deriv_energy"]
ENERGY_COLS = ["energy_mean", "energy_std", "energy_range"]
ENVMOD_COLS = ["envmod_dominant_freq", "envmod_low_rate_energy", "envmod_high_rate_energy",
               "envmod_centroid", "envmod_bandwidth", "envmod_entropy"]


def load_merged() -> pd.DataFrame:
    main_df = pd.read_csv(REAL_CORPUS_CSV, encoding="utf-8-sig")
    aux_df = pd.read_csv(AUX_CSV, encoding="utf-8-sig")
    if "error" in main_df.columns:
        main_df = main_df[main_df["error"].fillna("") == ""]
    if "error" in aux_df.columns:
        aux_df = aux_df[aux_df["error"].fillna("") == ""]

    from signal_v2.evaluation.run_family_evaluation import load_gender_map, build_merged_dataset
    merged = main_df.merge(aux_df, on=["wav_id", "situation"], how="inner", suffixes=("", "_aux"))
    gender_map = load_gender_map()
    merged["gender"] = merged["wav_id"].map(gender_map)
    return merged


def f010_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("f010_") and pd.api.types.is_numeric_dtype(df[c])]


def confound_correlation_table(df: pd.DataFrame, feature_cols: list, confound_cols: list) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        row = {"feature": feat}
        for conf in confound_cols:
            valid = np.isfinite(df[feat]) & np.isfinite(df[conf])
            if valid.sum() < 5 or df[feat][valid].std() == 0 or df[conf][valid].std() == 0:
                row[conf] = np.nan
                continue
            r, _ = stats.spearmanr(df[feat][valid], df[conf][valid])
            row[conf] = r
        rows.append(row)
    return pd.DataFrame(rows)


def residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)
    if valid.sum() < covariates.shape[1] + 5:
        return resid
    design = np.column_stack([np.ones(valid.sum()), covariates[valid]])
    coefs, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    resid[valid] = y[valid] - design @ coefs
    return resid


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
    return {"epsilon_sq": eps, "p_value": float(p), "effect_label": effect_label(eps)}


def run() -> dict:
    df = load_merged()
    print(f"분석 대상: {len(df)}개 파일")
    results = {}

    f010_features = f010_cols(df)

    # --- SS12/13/14: duration / F0 dynamics / energy dynamics confound ---
    print("\n=== §12 duration/speaking-rate confound (max |rho| per F010 feature) ===")
    dur_table = confound_correlation_table(df, f010_features, DURATION_COLS)
    dur_table["max_abs_rho"] = dur_table[DURATION_COLS].abs().max(axis=1)
    dur_table = dur_table.sort_values("max_abs_rho", ascending=False)
    print(dur_table[["feature", "max_abs_rho"]].head(10).to_string(index=False))
    n_strong_dur = int((dur_table["max_abs_rho"] > 0.7).sum())
    print(f"  |rho|>0.7인 feature: {n_strong_dur}/{len(f010_features)}")
    results["duration_confound"] = {"table": dur_table.to_dict("records"), "n_strong_proxy": n_strong_dur}

    print("\n=== §13 F0 dynamics confound ===")
    f0_table = confound_correlation_table(df, f010_features, F0DYN_COLS)
    f0_table["max_abs_rho"] = f0_table[F0DYN_COLS].abs().max(axis=1)
    f0_table = f0_table.sort_values("max_abs_rho", ascending=False)
    print(f0_table[["feature", "max_abs_rho"]].head(10).to_string(index=False))
    n_strong_f0 = int((f0_table["max_abs_rho"] > 0.7).sum())
    print(f"  |rho|>0.7인 feature: {n_strong_f0}/{len(f010_features)}")
    results["f0_dynamics_confound"] = {"table": f0_table.to_dict("records"), "n_strong_proxy": n_strong_f0}

    print("\n=== §14 energy dynamics confound ===")
    energy_table = confound_correlation_table(df, f010_features, ENERGY_COLS)
    energy_table["max_abs_rho"] = energy_table[ENERGY_COLS].abs().max(axis=1)
    energy_table = energy_table.sort_values("max_abs_rho", ascending=False)
    print(energy_table[["feature", "max_abs_rho"]].head(10).to_string(index=False))
    n_strong_energy = int((energy_table["max_abs_rho"] > 0.7).sum())
    print(f"  |rho|>0.7인 feature: {n_strong_energy}/{len(f010_features)}")
    results["energy_dynamics_confound"] = {"table": energy_table.to_dict("records"), "n_strong_proxy": n_strong_energy}

    # --- §14 generic envelope vs harmonic-specific ---
    print("\n=== §14 generic envelope modulation vs harmonic-specific (F010) ===")
    from emotion_classifier import load_combined_features
    wav_ids_base, X_base_full, y_situation_full, feature_names = load_combined_features()
    base_lookup = {w: X_base_full[i] for i, w in enumerate(wav_ids_base)}
    situation_lookup = {w: y_situation_full[i] for i, w in enumerate(wav_ids_base)}
    common = [w for w in df["wav_id"] if w in base_lookup]
    df_c = df[df["wav_id"].isin(common)].set_index("wav_id").loc[common].reset_index()
    X_baseline = np.array([base_lookup[w] for w in df_c["wav_id"]])
    y_situation = np.array([situation_lookup[w] for w in df_c["wav_id"]])

    X_envmod = df_c[ENVMOD_COLS].fillna(0.0).values
    X_f010_all = df_c[f010_features].fillna(0.0).values
    h1_cols = [c for c in f010_features if c.startswith("f010_h1_")]
    X_f010_h1 = df_c[h1_cols].fillna(0.0).values

    env_result = classifier_ablation(X_baseline, X_envmod, y_situation)
    f010_all_result = classifier_ablation(X_baseline, X_f010_all, y_situation)
    f010_h1_result = classifier_ablation(X_baseline, X_f010_h1, y_situation)

    print(f"  generic envelope (6 dims):      delta={env_result['delta_mean']:+.4f} ({env_result['n_folds_improved']}/5 folds)")
    print(f"  F010 all (33 dims):             delta={f010_all_result['delta_mean']:+.4f} ({f010_all_result['n_folds_improved']}/5 folds)")
    print(f"  F010 H1-only ({len(h1_cols)} dims, dim-matched): delta={f010_h1_result['delta_mean']:+.4f} ({f010_h1_result['n_folds_improved']}/5 folds)")
    results["generic_vs_harmonic_specific"] = {
        "generic_envelope": env_result, "f010_all": f010_all_result, "f010_h1_only": f010_h1_result,
        "h1_cols": h1_cols,
    }

    # --- §15 harmonic-order analysis ---
    print("\n=== §15 harmonic-order analysis (F010, per H1-H5) ===")
    harmonic_order = {}
    for h in range(1, 6):
        h_cols = [c for c in f010_features if c.startswith(f"f010_h{h}_")]
        if not h_cols:
            continue
        eps_list = []
        for c in h_cols:
            r = emotion_eps2(df_c[c].values, df_c["situation"].values)
            eps_list.append((c, r["epsilon_sq"]))
        eps_list.sort(key=lambda kv: -(kv[1] if not np.isnan(kv[1]) else -999))
        top_feat, top_eps = eps_list[0]
        harmonic_order[f"H{h}"] = {"top_feature": top_feat, "top_epsilon_sq": top_eps,
                                    "all_features": eps_list}
        print(f"  H{h}: top={top_feat} eps^2={top_eps:.4f}" if not np.isnan(top_eps) else f"  H{h}: n/a")
    results["harmonic_order"] = harmonic_order

    # --- §16 modulation-rate analysis (실측 dominant_freq 분포 기반) ---
    print("\n=== §16 modulation-rate analysis ===")
    dom_freq_cols = [c for c in f010_features if c.endswith("_dominant_freq")]
    all_dom_freqs = df_c[dom_freq_cols].values.flatten()
    all_dom_freqs = all_dom_freqs[np.isfinite(all_dom_freqs) & (all_dom_freqs > 0)]
    if all_dom_freqs.size:
        print(f"  관측된 non-zero dominant_freq 분포: median={np.median(all_dom_freqs):.2f}Hz, "
              f"p25={np.percentile(all_dom_freqs, 25):.2f}Hz, p75={np.percentile(all_dom_freqs, 75):.2f}Hz, "
              f"max={np.max(all_dom_freqs):.2f}Hz")
    # 사후적으로: 관측 분포 자체의 사분위(p25/p75)를 band 경계로 사용 - 임의 경계 대신
    # 데이터가 실제로 어떻게 분포하는지에 기반. (말소리 리듬 문헌이 흔히 인용하는
    # ~4-5Hz 음절 속도 대와도 대략 일치하는지 참고용으로 비교.)
    if all_dom_freqs.size >= 10:
        slow_edge, fast_edge = np.percentile(all_dom_freqs, [33, 67])
    else:
        slow_edge, fast_edge = 2.0, 8.0
    print(f"  data-driven band 경계(관측 분포 33/67 백분위): slow<{slow_edge:.2f}Hz, fast>{fast_edge:.2f}Hz")

    band_by_emotion = {}
    for h in range(1, 6):
        col = f"f010_h{h}_modulation_dominant_freq"
        if col not in df_c.columns:
            continue
        vals = df_c[col].values
        situations = df_c["situation"].values
        bands = np.where(vals <= 0, "none", np.where(vals < slow_edge, "slow", np.where(vals < fast_edge, "medium", "fast")))
        for sit in np.unique(situations):
            mask = situations == sit
            counts = pd.Series(bands[mask]).value_counts(normalize=True).to_dict()
            band_by_emotion.setdefault(f"H{h}", {})[sit] = counts
    results["modulation_rate"] = {"slow_edge_hz": float(slow_edge), "fast_edge_hz": float(fast_edge),
                                   "band_distribution_by_emotion": band_by_emotion}

    # --- §18 partial association ---
    print("\n=== §18 partial association: emotion effect controlling for duration+speaking_rate+F0_std+gender ===")
    covariate_cols = ["dur_total_sec", "dur_speaking_rate_proxy", "f0dyn_std"]
    gender_dummy = (df_c["gender"] == "male").astype(float).values.reshape(-1, 1)
    covariates = np.hstack([df_c[covariate_cols].values, gender_dummy])

    top_f010 = sorted(
        [(c, emotion_eps2(df_c[c].values, df_c["situation"].values)["epsilon_sq"]) for c in f010_features],
        key=lambda kv: -(kv[1] if not np.isnan(kv[1]) else -999),
    )[:5]

    partial_results = {}
    for feat, raw_eps in top_f010:
        y = df_c[feat].values.astype(float)
        resid = residualize(y, covariates)
        controlled = emotion_eps2(resid, df_c["situation"].values)
        partial_results[feat] = {"raw_epsilon_sq": raw_eps, "controlled_epsilon_sq": controlled["epsilon_sq"],
                                  "controlled_p": controlled.get("p_value")}
        print(f"  {feat}: raw eps^2={raw_eps:.4f} -> controlled eps^2={controlled['epsilon_sq']:.4f}")
    results["partial_association"] = partial_results

    with open(os.path.join(OUTPUT_DIR, "mechanistic_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(OUTPUT_DIR, 'mechanistic_analysis.json')}")
    return results


if __name__ == "__main__":
    run()
