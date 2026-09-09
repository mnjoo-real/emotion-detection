"""SP-V2-012 overnight: 나머지 RF-only(M0-M7 grid + negative control, classifier
비교 없음) 지점들을 순차적으로 채운다. 이미 있는 point_*.json은 건너뛴다(resume-safe -
중간에 중단돼도 다시 실행하면 이어서 진행).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.evaluation.sp012_scaling_point import run_one_point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SP012_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

SEEDS = [43, 44, 45]
N_LEVELS = [(200, 1400), (400, 2800), (800, 5600)]
LABELS = ["situation", "majority_vote"]


def main() -> None:
    os.makedirs(SP012_DIR, exist_ok=True)
    for seed in SEEDS:
        csv_path = os.path.join(SP012_DIR, f"sp012_seed{seed}_n5600.csv")
        if not os.path.exists(csv_path):
            print(f"[건너뜀] {csv_path} 없음")
            continue
        for n_per_emotion, n_total in N_LEVELS:
            if seed == 43 and n_total == 5600:
                # seed43/N5600는 별도의 classifier-comparison endpoint job(situation/
                # majority_vote 각각)이 이미 처리 중 - 여기서 건드리면 같은 파일에 race condition
                print(f"[건너뜀, classifier-comparison job이 처리 중] seed=43, N=5600")
                continue
            for label in LABELS:
                out_name = f"point_N{n_total}_seed{seed}_{label}.json"
                out_path = os.path.join(SP012_DIR, out_name)
                if os.path.exists(out_path):
                    print(f"[건너뜀, 이미 있음] {out_name}")
                    continue
                try:
                    result = run_one_point(csv_path, seed, n_per_emotion, label, run_classifier_comparison=False)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                    print(f"저장: {out_name}")
                except Exception as exc:  # noqa: BLE001 - 한 지점 실패로 전체 sweep이 멈추지 않게
                    print(f"[실패] {out_name}: {exc}")


if __name__ == "__main__":
    main()
