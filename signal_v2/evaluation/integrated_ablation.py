"""Stage 7: family-level integrated ablation.

브리핑 §16이 요구하는 대로 stepwise(누적 추가)만 main comparison으로 쓰지 않는다
- 추가 순서에 따라 결과가 달라질 수 있으므로, 5개 family를 전부 더한 상태에서
  하나씩 빼는 leave-one-out으로 각 family의 marginal contribution도 별도로 측정한다.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.evaluation.real_corpus_common import classifier_ablation
from signal_v2.evaluation.run_family_evaluation import FAMILY_PREFIXES, build_merged_dataset, numeric_feature_cols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CORPUS_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus")

STEP_ORDER = ["F006", "F007", "F008", "F009", "F010"]


def run() -> dict:
    real_df, feature_names, baseline_X, common_ids = build_merged_dataset()
    family_cols = {fam: numeric_feature_cols(real_df, prefix) for fam, prefix in FAMILY_PREFIXES.items()}

    results = {"stepwise": {}, "leave_one_out": {}}

    for label_col, min_n in [("situation", 0), ("majority_vote", 50)]:
        sub_df = real_df.dropna(subset=[label_col]) if label_col == "majority_vote" else real_df
        if len(sub_df) < min_n:
            continue
        idx = [i for i, wid in enumerate(common_ids) if wid in sub_df.index]
        X_base_sub = baseline_X[idx]
        y = sub_df[label_col].values

        print(f"\n{'=' * 70}\nSTEPWISE ({label_col})\n{'=' * 70}")
        cumulative_cols = []
        step_results = {}
        for fam in STEP_ORDER:
            cumulative_cols += family_cols[fam]
            X_new = sub_df[cumulative_cols].fillna(0.0).values
            res = classifier_ablation(X_base_sub, X_new, y)
            step_results[fam] = res
            print(f"  +{fam:<6} (cumulative {len(cumulative_cols):>3} new feats) "
                  f"combined_f1={res['combined_f1_mean']:.4f} delta={res['delta_mean']:+.4f} "
                  f"({res['n_folds_improved']}/{res['n_folds']} folds)")
        results["stepwise"][label_col] = step_results

        print(f"\n{'=' * 70}\nLEAVE-ONE-OUT ({label_col}, all 5 families combined minus one)\n{'=' * 70}")
        all_cols = [c for fam in STEP_ORDER for c in family_cols[fam]]
        X_all = sub_df[all_cols].fillna(0.0).values
        res_all = classifier_ablation(X_base_sub, X_all, y)
        print(f"  ALL 5 families ({len(all_cols)} feats): combined_f1={res_all['combined_f1_mean']:.4f} "
              f"delta={res_all['delta_mean']:+.4f} ({res_all['n_folds_improved']}/{res_all['n_folds']} folds)")

        loo_results = {"all_families": res_all}
        for fam in STEP_ORDER:
            minus_cols = [c for f2 in STEP_ORDER if f2 != fam for c in family_cols[f2]]
            X_minus = sub_df[minus_cols].fillna(0.0).values
            res_minus = classifier_ablation(X_base_sub, X_minus, y)
            marginal = res_all["combined_f1_mean"] - res_minus["combined_f1_mean"]
            loo_results[fam] = {"without_this_family": res_minus, "marginal_contribution": marginal}
            print(f"  ALL - {fam:<6}: combined_f1={res_minus['combined_f1_mean']:.4f} "
                  f"(marginal contribution of {fam} = {marginal:+.4f})")
        results["leave_one_out"][label_col] = loo_results

    out_path = os.path.join(REAL_CORPUS_DIR, "integrated_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {out_path}")
    return results


if __name__ == "__main__":
    run()
