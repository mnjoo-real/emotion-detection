"""
사전학습된 wav2vec2 기반 감정 회귀 모델(audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)로
각 발화의 arousal(각성도)/dominance(지배성)/valence(정서가)를 예측해 feature로 뽑는다.

지금까지의 스크립트들은 전부 손으로 설계한 신호처리 feature(MFCC, 화성, 선율, 리듬, 음계)였는데,
이건 대규모 음성 코퍼스로 사전학습된 self-supervised 모델(wav2vec2-large)의 표현력을 대신
빌려오는 방식이다 - 감정과 관련된 훨씬 더 풍부한 정보를 담고 있을 것으로 기대.

모델 하나(약 300M 파라미터)를 한 번만 로드해서 재사용해야 하므로, 다른 추출 스크립트들과
달리 CPU 프로세스 풀 병렬화를 쓰지 않는다 - 대신 GPU(MPS, Apple Metal)가 있으면 그걸 쓰고,
단일 프로세스에서 순차적으로 처리한다. similarity.py의 load_groups()/get_audio_path()를
재사용해서 4차년도+5차년도 통합 데이터셋, 7개 감정 전부를 대상으로 하고, 재실행 시
이미 처리된 wav_id는 건너뛴다.
"""

import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "vad")

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TARGET_SR = 16000

PROGRESS_LOG_EVERY = 50

FIELDNAMES = ["wav_id", "situation", "arousal", "dominance", "valence"]


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return hidden_states, self.classifier(hidden_states)


def pick_device() -> torch.device:
    # 벤치마크 결과 이 하드웨어(Radeon Pro 560, 4GB)에서는 MPS가 배치=1 추론에서
    # CPU보다 오히려 느렸다(0.33개/초 vs 0.74개/초) - MPS 커널 디스패치/전송 오버헤드가
    # wav2vec2-large 배치=1 추론의 실제 연산량보다 커서 생기는 역전 현상으로 보임.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = EmotionModel.from_pretrained(MODEL_NAME)

    # 저장된 체크포인트가 weight-normalized conv로 직렬화되어 있는데, 최신 torch/transformers
    # 조합에서는 로드 후 파라미터화를 제거하고 정규화된 가중치를 직접 채워야 정상 동작한다.
    conv = model.wav2vec2.encoder.pos_conv_embed.conv
    if P.is_parametrized(conv, "weight"):
        P.remove_parametrizations(conv, "weight", leave_parametrized=True)
        state_dict = torch.load(hf_hub_download(MODEL_NAME, "pytorch_model.bin"), map_location="cpu")
        with torch.no_grad():
            conv.weight.copy_(
                torch._weight_norm(
                    state_dict["wav2vec2.encoder.pos_conv_embed.conv.weight_v"],
                    state_dict["wav2vec2.encoder.pos_conv_embed.conv.weight_g"],
                    dim=2,
                )
            )
            conv.bias.copy_(state_dict["wav2vec2.encoder.pos_conv_embed.conv.bias"])

    model.eval()
    model.to(device)
    return processor, model


def predict_vad(audio_path: str, processor, model, device: torch.device) -> tuple:
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    # 이 venv의 torch 빌드가 numpy 1.x C-API를 기대하는데 설치된 numpy는 2.x라
    # tensor.numpy()의 제로카피 브릿지가 깨져 있다. 리스트를 거쳐 우회한다.
    y = np.asarray(waveform.squeeze(0).tolist(), dtype=np.float32)

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        _, logits = model(input_values)
    arousal, dominance, valence = logits.squeeze().tolist()
    return float(arousal), float(dominance), float(valence)


def load_already_processed(per_file_path: str) -> set:
    processed = set()
    if os.path.exists(per_file_path):
        with open(per_file_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                processed.add(row["wav_id"])
    return processed


def build_tasks(groups: dict, processed: set) -> list:
    tasks = []
    for situation, wav_ids in groups.items():
        for wav_id in wav_ids:
            if wav_id in processed:
                continue
            audio_path = get_audio_path(wav_id)
            if not os.path.exists(audio_path):
                continue
            tasks.append((wav_id, situation, audio_path))
    return tasks


def extract_all(groups: dict, per_file_path: str, processor, model, device: torch.device) -> None:
    processed = load_already_processed(per_file_path)
    tasks = build_tasks(groups, processed)

    total = len(processed) + len(tasks)
    done = len(processed)
    if not tasks:
        print(f"모든 파일이 이미 처리되어 있습니다 ({done}/{total}).")
        return

    print(f"{len(tasks)}개 파일 처리 시작 (이미 완료 {done}개, 전체 {total}개, device={device})")

    mode = "a" if processed else "w"
    start = time.perf_counter()

    with open(per_file_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not processed:
            writer.writeheader()

        for wav_id, situation, audio_path in tasks:
            try:
                arousal, dominance, valence = predict_vad(audio_path, processor, model, device)
            except Exception as exc:  # noqa: BLE001
                print(f"경고: '{wav_id}' 처리 실패, 건너뜀 ({exc})")
                continue

            writer.writerow({
                "wav_id": wav_id, "situation": situation,
                "arousal": arousal, "dominance": dominance, "valence": valence,
            })

            done += 1
            if done % PROGRESS_LOG_EVERY == 0:
                f.flush()
                elapsed = time.perf_counter() - start
                rate = (done - len(processed)) / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else float("inf")
                print(f"{done}/{total} 완료 ({rate:.2f}개/초, 남은 시간 {remaining / 60:.1f}분)")


def save_summary_by_emotion(per_file_path: str, summary_path: str) -> None:
    numeric_cols = ["arousal", "dominance", "valence"]
    per_situation: dict = {}
    with open(per_file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            situation = row["situation"]
            per_situation.setdefault(situation, {col: [] for col in numeric_cols})
            for col in numeric_cols:
                per_situation[situation][col].append(float(row[col]))

    fieldnames = ["situation", "n_files"]
    for col in numeric_cols:
        fieldnames += [f"{col}_avg", f"{col}_std"]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for situation, columns in per_situation.items():
            row = {"situation": situation, "n_files": len(columns["arousal"])}
            for col in numeric_cols:
                values = np.array(columns[col])
                row[f"{col}_avg"] = float(values.mean())
                row[f"{col}_std"] = float(values.std())
            writer.writerow(row)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()

    device = pick_device()
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
    processor, model = load_model(device)

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_vad.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path, processor, model, device)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
