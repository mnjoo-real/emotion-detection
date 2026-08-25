"""
harmony_features.py / partial_roughness.py / tonality_features.py / melody_rhythm_features.py가
뽑은 화성(dissonance)·음계(tonality)·선율/리듬(melody_rhythm) feature들이 감정 그룹 사이에서
통계적으로 유의미하게 다른지 확인한다.

두 가지 감정 라벨링 기준으로 각각 검정한다:
- situation: 원본 CSV의 '상황' 칼럼 (시나리오상 의도된 감정)
- majority_vote: emotion_classifier.load_majority_vote_labels() (평가자 5명의
  실제 인지 감정을 강도 가중 다수결로 합친 라벨, 동률인 애매한 파일은 제외)

지금까지 output/*/summary_by_emotion.csv는 그룹별 평균/표준편차만 보여줬을 뿐 실제
유의성 검정은 없었다. 표본 수가 그룹당 최소 1,700개 이상으로 매우 크기 때문에
정규성을 가정하기 어렵고(값들이 0 근처에 몰린 비대칭 분포), 극단값에도 민감하지 않은
Kruskal-Wallis H 검정(비모수, one-way ANOVA의 순위 기반 대응)을 쓴다.

표본이 크면 아주 작은 차이도 p<0.001로 나오기 쉬우므로, p-value와 별개로
효과크기(epsilon-squared, eta-squared의 순위 기반 버전)를 같이 보고한다:
  0.01 근처 = 미미, 0.04~ = 작음, 0.16~ = 중간, 0.36~ = 큼 (Cohen의 경험적 기준을 순위
  기반 통계에 준용).
"""

import csv
import os

import numpy as np
from scipy import stats

from emotion_classifier import load_majority_vote_labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "harmony": {
        "path": os.path.join(BASE_DIR, "output", "harmony", "per_file_harmony.csv"),
        "features": [
            "formant_roughness_mean",
            "melodic_dissonance_mean",
            "pitch_class_entropy",
        ],
    },
    "partial_roughness": {
        "path": os.path.join(BASE_DIR, "output", "partial_roughness", "per_file_partial_roughness.csv"),
        "features": [
            "partial_roughness_mean",
            "partial_roughness_std",
        ],
    },
    "tonality": {
        "path": os.path.join(BASE_DIR, "output", "tonality", "per_file_tonality.csv"),
        "features": [
            "key_major_corr",
            "key_minor_corr",
            "key_major_minor_diff",
        ],
    },
    "melody_rhythm": {
        "path": os.path.join(BASE_DIR, "output", "melody_rhythm", "per_file_melody_rhythm.csv"),
        "features": [
            "notes_per_sec",
            "note_dur_mean", "note_dur_cv",
            "ioi_mean", "ioi_cv",
            "note_interval_semitone_mean", "note_interval_semitone_std",
            "pct_notes_rising", "pct_notes_falling", "pct_notes_repeated",
            "duration_pitch_corr",
        ],
    },
}


def load_groups_for_feature(path: str, feature: str, label_map: dict = None) -> dict:
    groups: dict = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            val = row[feature]
            if val == "" or val is None:
                continue
            if label_map is None:
                label = row["situation"]
            else:
                label = label_map.get(row["wav_id"])
                if label is None:
                    continue
            groups.setdefault(label, []).append(float(val))
    return {k: np.array(v) for k, v in groups.items()}


def epsilon_squared(h_stat: float, n: int, k: int) -> float:
    # Kruskal-Wallis용 순위 기반 효과크기 (eta-squared의 비모수 대응).
    return float((h_stat - k + 1) / (n - k)) if n > k else float("nan")


def effect_label(eps: float) -> str:
    if eps < 0.01:
        return "무시할 수준"
    if eps < 0.04:
        return "작음"
    if eps < 0.16:
        return "중간"
    return "큼"


def run_feature_test(path: str, feature: str, label_map: dict = None) -> dict | None:
    groups = load_groups_for_feature(path, feature, label_map)
    if len(groups) < 2:
        return None
    arrays = list(groups.values())
    n = sum(len(a) for a in arrays)
    k = len(arrays)

    h_stat, p_value = stats.kruskal(*arrays)
    eps = epsilon_squared(h_stat, n, k)

    means = {sit: float(a.mean()) for sit, a in groups.items()}
    ranked = sorted(means.items(), key=lambda kv: kv[1])

    return {
        "feature": feature,
        "n": n,
        "k": k,
        "h_stat": h_stat,
        "p_value": p_value,
        "epsilon_sq": eps,
        "means": means,
        "lowest": ranked[0],
        "highest": ranked[-1],
    }


def run_labeling(label_name: str, label_map: dict = None) -> None:
    results = []
    for source_name, cfg in SOURCES.items():
        if not os.path.exists(cfg["path"]):
            print(f"[건너뜀] {source_name}: {cfg['path']} 없음 (아직 추출 중이거나 미실행)")
            continue
        for feature in cfg["features"]:
            r = run_feature_test(cfg["path"], feature, label_map)
            if r is None:
                continue
            r["source"] = source_name
            results.append(r)

    results.sort(key=lambda r: -r["epsilon_sq"])

    print("\n" + "#" * 100)
    print(f"# 라벨 기준: {label_name}")
    print("#" * 100)

    print("=" * 100)
    print(f"{'feature':<28}{'source':<18}{'H':>10}{'p':>12}{'eps^2':>10}  효과크기   최저↔최고 그룹 (평균)")
    print("=" * 100)
    for r in results:
        p_str = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        lo_sit, lo_val = r["lowest"]
        hi_sit, hi_val = r["highest"]
        print(
            f"{r['feature']:<28}{r['source']:<18}{r['h_stat']:>10.1f}{p_str:>12}{r['epsilon_sq']:>10.4f}"
            f"  {effect_label(r['epsilon_sq']):<8} {lo_sit}({lo_val:.4f}) ↔ {hi_sit}({hi_val:.4f})"
        )

    print("\n--- 그룹별 평균 상세 ---")
    for r in results:
        p_str = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        print(f"\n[{r['source']}] {r['feature']}  (H={r['h_stat']:.1f}, p={p_str}, eps^2={r['epsilon_sq']:.4f})")
        for sit, val in sorted(r["means"].items(), key=lambda kv: kv[1]):
            print(f"    {sit:<12} {val:.5f}")


def main() -> None:
    run_labeling("situation (상황 - 시나리오 의도 라벨)", label_map=None)

    majority_vote = load_majority_vote_labels()
    n_ambiguous = sum(1 for v in majority_vote.values() if v is None)
    print(f"\n(다수결 라벨: 전체 {len(majority_vote)}개 중 동률/애매 {n_ambiguous}개 제외)")
    run_labeling("majority_vote (평가자 5명 다수결 - 실제 인지 라벨)", label_map=majority_vote)


if __name__ == "__main__":
    main()
