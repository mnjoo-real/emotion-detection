"""SP-V2-004 파일럿 결과 분석: effect size, 성별 민감도, 기존 feature와의 redundancy.

dissonance_tonality_stats.py의 epsilon_squared/effect_label을 그대로 재사용해서
기존 저장소가 이미 정착시킨 "p-value보다 effect size" 원칙을 새 feature에도 똑같이
적용한다(새 통계 방법을 따로 만들지 않는다).
"""

import csv
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dissonance_tonality_stats import epsilon_squared, effect_label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "source_roughness", "pilot_per_file.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "output", "signal_v2", "source_roughness", "pilot_analysis.json")

VARIANTS = ["raw_roughness_mean", "source_roughness_mean", "source_roughness_norm_mean"]
EXISTING = ["existing_partial_roughness_mean", "existing_formant_roughness_mean"]


def load_rows() -> list:
    with open(PILOT_CSV, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    return rows


def effect_size_by_situation(rows: list, feature: str) -> dict:
    groups: dict = {}
    for r in rows:
        val = r[feature]
        if val == "":
            continue
        groups.setdefault(r["situation"], []).append(float(val))
    arrays = [np.array(v) for v in groups.values() if len(v) > 0]
    if len(arrays) < 2:
        return {}
    n = sum(len(a) for a in arrays)
    k = len(arrays)
    h_stat, p_value = stats.kruskal(*arrays)
    eps = epsilon_squared(h_stat, n, k)
    means = {sit: float(np.mean(v)) for sit, v in groups.items()}
    ranked = sorted(means.items(), key=lambda kv: kv[1])
    return {
        "n": n, "k": k, "h_stat": float(h_stat), "p_value": float(p_value),
        "epsilon_sq": eps, "effect_label": effect_label(eps),
        "lowest": ranked[0], "highest": ranked[-1], "means": means,
    }


def gender_sensitivity(rows: list, feature: str) -> dict:
    by_gender: dict = {}
    for r in rows:
        val = r[feature]
        if val == "" or r["gender"] not in ("male", "female"):
            continue
        by_gender.setdefault(r["gender"], []).append(float(val))
    if len(by_gender) < 2:
        return {}
    male, female = np.array(by_gender.get("male", [])), np.array(by_gender.get("female", []))
    if male.size < 2 or female.size < 2:
        return {}
    h_stat, p_value = stats.kruskal(male, female)
    eps = epsilon_squared(h_stat, male.size + female.size, 2)
    return {
        "male_mean": float(male.mean()), "female_mean": float(female.mean()),
        "gender_epsilon_sq": eps, "gender_effect_label": effect_label(eps),
        "p_value": float(p_value),
    }


def redundancy(rows: list, feature: str, existing_feature: str) -> dict:
    xs, ys = [], []
    for r in rows:
        if r[feature] == "" or r[existing_feature] == "":
            continue
        xs.append(float(r[feature]))
        ys.append(float(r[existing_feature]))
    if len(xs) < 3:
        return {}
    xs, ys = np.array(xs), np.array(ys)
    if xs.std() == 0 or ys.std() == 0:
        return {"pearson_r": 0.0, "spearman_r": 0.0}
    pearson_r = float(np.corrcoef(xs, ys)[0, 1])
    spearman_r = float(stats.spearmanr(xs, ys).correlation)
    return {"pearson_r": pearson_r, "spearman_r": spearman_r}


def main() -> None:
    rows = load_rows()
    n_total = len(rows) + sum(1 for _ in [])
    print(f"분석 대상: {len(rows)}개 파일 (오류 제외)")

    results = {"n_files": len(rows), "variants": {}}

    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        sit_result = effect_size_by_situation(rows, variant)
        if sit_result:
            lo_sit, lo_val = sit_result["lowest"]
            hi_sit, hi_val = sit_result["highest"]
            print(
                f"  situation effect: H={sit_result['h_stat']:.1f} p={sit_result['p_value']:.4f} "
                f"eps^2={sit_result['epsilon_sq']:.4f} ({sit_result['effect_label']}) "
                f"{lo_sit}({lo_val:.2e}) <-> {hi_sit}({hi_val:.2e})"
            )
        gender_result = gender_sensitivity(rows, variant)
        if gender_result:
            print(
                f"  gender effect: male={gender_result['male_mean']:.2e} female={gender_result['female_mean']:.2e} "
                f"eps^2={gender_result['gender_epsilon_sq']:.4f} ({gender_result['gender_effect_label']})"
            )
        redund = {ex: redundancy(rows, variant, ex) for ex in EXISTING}
        for ex, r in redund.items():
            if r:
                print(f"  redundancy vs {ex}: pearson r={r['pearson_r']:.3f}, spearman r={r['spearman_r']:.3f}")

        results["variants"][variant] = {
            "situation_effect": sit_result, "gender_effect": gender_result, "redundancy": redund,
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
