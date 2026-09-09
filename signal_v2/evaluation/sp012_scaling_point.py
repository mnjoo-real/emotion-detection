"""SP-V2-012 §4-10: 하나의 (N, seed, label) 지점에서 M0-M7 RF ablation grid +
random-feature negative control + 3-classifier 비교 + stability selection을
전부 수행한다. 여러 N/seed에 대해 이 함수를 반복 호출해서 learning curve를 만든다.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.evaluation.real_corpus_common import classifier_ablation
from signal_v2.evaluation.sp012_classifiers import classifier_comparison, stability_selection_frequency
from signal_v2.evaluation.sp012_negative_controls import negative_control_comparison
from signal_v2.evaluation.sp012_scaling_common import FAMILY_PREFIXES, load_level_dataset, numeric_cols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

M_MODELS = {
    "M0_baseline": [],
    "M1_F008": ["F008"],
    "M2_F009": ["F009"],
    "M3_F010": ["F010"],
    "M4_F008_F009": ["F008", "F009"],
    "M5_F008_F010": ["F008", "F010"],
    "M6_F009_F010": ["F009", "F010"],
    "M7_F008_F009_F010": ["F008", "F009", "F010"],
}


def run_one_point(csv_path: str, seed: int, n_per_emotion: int, label: str = "situation",
                   run_classifier_comparison: bool = True) -> dict:
    tag = f"N{n_per_emotion * 7}_seed{seed}_{label}"
    print(f"\n{'#' * 70}\n{tag}\n{'#' * 70}")

    data = load_level_dataset(csv_path, n_per_emotion=n_per_emotion, seed=seed)
    df = data["df"]
    if label == "majority_vote":
        df = df.dropna(subset=["majority_vote"])
    df = df.reset_index(drop=True)
    y = df[label].values

    n_actual = len(df)
    print(f"실제 표본 수: {n_actual}")

    result = {"tag": tag, "n": n_actual, "seed": seed, "n_per_emotion": n_per_emotion, "label": label}

    # --- M0-M7 RF ablation grid ---
    family_cols = {fam: numeric_cols(df, prefix) for fam, prefix in FAMILY_PREFIXES.items()}
    ablation_results = {}
    for m_name, fams in M_MODELS.items():
        cols = [c for fam in fams for c in family_cols[fam]]
        if not cols:
            continue
        X_new = df[cols].fillna(0.0).values
        res = classifier_ablation(_align_baseline(data, df), X_new, y)
        ablation_results[m_name] = res
        print(f"  {m_name:<20} delta={res['delta_mean']:+.4f} ({res['n_folds_improved']}/{res['n_folds']} folds)")
    result["ablation_grid"] = ablation_results

    # --- random-feature negative controls (F010 alone, and F008+F009+F010) ---
    neg_controls = {}
    for probe_name, fams in [("F010_alone", ["F010"]), ("F008_F009_F010", ["F008", "F009", "F010"])]:
        cols = [c for fam in fams for c in family_cols[fam]]
        X_new = df[cols].fillna(0.0).values
        nc = negative_control_comparison(_align_baseline(data, df), X_new, y, seed_for_controls=seed)
        neg_controls[probe_name] = nc
        print(f"  negctrl[{probe_name}] real={nc['real_v2']['mean']:.4f} gaussian={nc['gaussian_noise']['mean']:.4f} "
              f"permuted={nc['permuted_v2']['mean']:.4f} baseline={nc['baseline']['mean']:.4f}")
    result["negative_controls"] = neg_controls

    # --- 3-classifier comparison (RF/ElasticNet/SVM) for M3(F010) and M7 ---
    if run_classifier_comparison:
        clf_comparison = {}
        for probe_name, fams in [("F010_alone", ["F010"]), ("F008_F009_F010", ["F008", "F009", "F010"])]:
            cols = [c for fam in fams for c in family_cols[fam]]
            X_new = df[cols].fillna(0.0).values
            cc = classifier_comparison(_align_baseline(data, df), X_new, y)
            clf_comparison[probe_name] = cc["summary"]
            if probe_name == "F010_alone":
                stability = stability_selection_frequency(cc["elasticnet_new_feature_coefs"], cols)
                result["f010_stability_selection"] = stability
            for clf_name, s in cc["summary"].items():
                print(f"  clf[{probe_name}][{clf_name}] delta={s['delta_mean']:+.4f} ({s['n_folds_improved']}/{s['n_folds']})")
        result["classifier_comparison"] = clf_comparison

    return result


def _align_baseline(data: dict, df) -> np.ndarray:
    """df(잘라낸/재정렬된)에 맞춰 baseline X를 wav_id 기준으로 재정렬."""
    lookup = {w: i for i, w in enumerate(df["wav_id"].values)}
    # data["X_baseline"]는 load_level_dataset이 처음 만들 때의 df 순서와 일치했지만
    # 이후 dropna/reset_index로 df가 바뀌었을 수 있으므로 wav_id로 다시 매핑한다.
    from signal_v2.evaluation.sp012_scaling_common import _baseline
    cache = _baseline()
    return np.array([cache["base_lookup"][w] for w in df["wav_id"].values])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-per-emotion", type=int, required=True)
    parser.add_argument("--label", type=str, default="situation")
    parser.add_argument("--out-name", type=str, default=None)
    parser.add_argument("--skip-classifier-comparison", action="store_true")
    args = parser.parse_args()

    result = run_one_point(args.csv_path, args.seed, args.n_per_emotion, args.label,
                            run_classifier_comparison=not args.skip_classifier_comparison)
    out_name = args.out_name or f"point_N{args.n_per_emotion * 7}_seed{args.seed}_{args.label}.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(OUTPUT_DIR, out_name)}")
