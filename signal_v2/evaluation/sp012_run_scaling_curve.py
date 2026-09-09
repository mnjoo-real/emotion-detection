"""SP-V2-012: 전체 scaling curve를 돈다 - 3 seed x N{1400,2800,5600} x label{situation,
majority_vote}. classifier-comparison(RF/ElasticNet/SVM, 비용이 큼)은 §"Experimental
Design"에서 명시한 대로 N 범위의 양 끝(1400, 5600)에서만 실행한다.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.evaluation.sp012_scaling_point import run_one_point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SP012_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

SEEDS = [43, 44, 45]
N_PER_EMOTION_LEVELS = [200, 400, 800]  # -> N=1400, 2800, 5600


def main() -> None:
    os.makedirs(SP012_DIR, exist_ok=True)
    all_results = []

    for seed in SEEDS:
        csv_path = os.path.join(SP012_DIR, f"sp012_seed{seed}_n5600.csv")
        if not os.path.exists(csv_path):
            print(f"[건너뜀] {csv_path} 없음 - 아직 추출 안 됨")
            continue

        for n_per_emotion in N_PER_EMOTION_LEVELS:
            is_endpoint = n_per_emotion in (N_PER_EMOTION_LEVELS[0], N_PER_EMOTION_LEVELS[-1])
            for label in ["situation", "majority_vote"]:
                result = run_one_point(
                    csv_path, seed, n_per_emotion, label,
                    run_classifier_comparison=is_endpoint,
                )
                all_results.append(result)

                out_name = f"point_N{n_per_emotion * 7}_seed{seed}_{label}.json"
                with open(os.path.join(SP012_DIR, out_name), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(SP012_DIR, "scaling_curve_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(SP012_DIR, 'scaling_curve_all.json')} ({len(all_results)}개 지점)")


if __name__ == "__main__":
    main()
