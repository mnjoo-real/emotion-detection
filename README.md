# 감정 분류를 위한 대화 음성 데이터셋 - 음향학적 분석

AI Hub "감정 분류를 위한 대화 음성 데이터셋"(4차년도 + 5차년도_2차, 33,964개 발화, 7개 감정: angry/disgust/fear/happiness/neutral/sadness/surprise)을 대상으로, 손으로 설계한 음향학적 feature들이 감정별로 유의미하게 다른지 확인하고 이를 바탕으로 감정 분류기를 만드는 프로젝트입니다.

음악의 3요소(화성·선율·리듬)를 음성 프로소디 분석에 번역해 적용하는 것이 핵심 아이디어입니다. 원본 오디오(4차년도/, 5차년도_2차/, 샘플/)는 라이선스 및 용량(20GB+) 문제로 이 저장소에 포함하지 않았습니다.

## 요약 성능 (최종 시점)

79개 feature(음향 기본 + 화성 + 선율음정 + 배음 불협화도 + 음계 + 선율/리듬)를 합쳐 Decision Tree / Random Forest로 학습한 결과입니다. (`emotion_classifier.py`)

| 라벨 기준 | 모델 | accuracy | macro F1 |
|---|---|---|---|
| situation (시나리오 의도 라벨) | Decision Tree (depth 6) | 0.219 | 0.192 |
| situation (시나리오 의도 라벨) | Random Forest (300 trees) | 0.379 | 0.295 |
| majority_vote (평가자 5명 다수결, 실제 인지 라벨) | Decision Tree (depth 6) | 0.230 | 0.211 |
| majority_vote (평가자 5명 다수결, 실제 인지 라벨) | **Random Forest (300 trees)** | **0.442** | **0.320** |

7-클래스 랜덤 베이스라인(~14%) 대비 확실히 개선되지만, **실제 청취 인지 라벨(majority_vote)이 시나리오 의도 라벨(situation)보다 음향 feature와 일관되게 더 잘 맞습니다.** sadness가 가장 잘 분류되고(F1 0.52~0.58), fear/disgust/surprise는 표본 수가 적고 음향적으로 겹치는 부분이 많아 여전히 어렵습니다.

> 주의: 원본 메타데이터에 화자 구분 ID가 없어 train/test를 화자 단위로 분리하지 못했습니다. 같은 화자의 다른 발화가 양쪽에 섞여 있을 수 있어 정확도가 실제보다 낙관적일 수 있습니다.

## 개발 단계 (Phase)

### Phase 0 — 파형 유사도 탐색
`similarity.py`, `similarity_vectorized.py`, `DTW_colab.ipynb`
같은 감정 그룹 내 파형들이 얼마나 비슷한지 cross-correlation/cosine similarity/MSE/DTW로 탐색. 이후 모든 스크립트가 재사용하는 `load_groups()`/`get_audio_path()`(4차년도+5차년도_2차 통합, 상황 라벨 통일)가 여기서 만들어짐. 파일 단위 결과물이 최대 2.18GB로 커서 이후 단계에서는 쓰이지 않음.

### Phase 1 — 기본 음향 feature
`extract_features.py`
MFCC(13개 평균/표준편차), 피치(F0, librosa.pyin) 평균/표준편차/중앙값/유성음 비율, 에너지(RMS) 평균/표준편차. 이후 모든 분류기의 기본 축.

### Phase 2 — 화성학적(harmony) feature
`harmony_features.py`
단성 음성엔 화음이 없지만 세 가지로 "화성" 개념을 번역:
- **포먼트 러프니스**: LPC로 뽑은 포먼트(F1~F4)들 사이 Plomp-Levelt/Sethares 감각적 불협화도
- **선율적 불협화도**: 인접 유성음 프레임 간 피치 진행에 같은 불협화도 모델 적용
- **조성 안정성**: 유성음 피치를 파일 자신의 중앙값 기준 12음계로 접은 분포의 엔트로피

### Phase 3 — 선율 음정 프로파일
`melodic_profile.py`
연속 음정을 반음(semitone) 단위로 측정해 화자 음역대와 무관하게 비교. 순차진행(step) vs 도약(leap), 상행/하행/유지 비율, 방향+크기별 7단계 히스토그램.

### Phase 4 — 부분음(순음) 배음 불협화도
`partial_roughness.py`
FFT로 프레임별 스펙트럼 피크(배음/부분음)를 분리하고, 그 사이의 Sethares 감각적 불협화도를 계산. 목소리 질감(맑음 vs 거침/creaky)을 화성학적 불협화도로 정량화.

### Phase 5 — 텍스트 feature (보조, 이후 제외)
`text_features.py`
발화문 TF-IDF + SVD. 코퍼스의 대본 어휘에 종속적이라 새 데이터셋에는 일반화되지 않아 최종 모델에서는 제외.

### Phase 6 — 1차 통합 분류기
`emotion_classifier.py` (초기 버전)
Phase 1~4 feature를 합쳐 Decision Tree/Random Forest 학습. 사람이 읽을 수 있는 규칙 추출이 목적인 DT와, feature 세트의 성능 상한을 보는 RF를 병행.

### Phase 7 — 통계적 유의성 검증 도입
`dissonance_tonality_stats.py`
그동안 감정별 비교가 평균/표준편차뿐이었다는 걸 인지하고, Kruskal-Wallis H 검정 + 효과크기(epsilon-squared)를 도입. 표본이 워낙 커서(그룹당 최소 1,700개+) p-value는 항상 유의하므로 효과크기로 실질적 크기를 판단.

| feature | 계열 | eps² (situation) | 최저 ↔ 최고 |
|---|---|---|---|
| pitch_class_entropy | 화성 | 0.041 (중간) | sadness ↔ surprise |
| melodic_dissonance_mean | 화성 | 0.027 (작음) | sadness ↔ surprise |
| partial_roughness_mean/std | 배음 불협화도 | 0.016/0.013 (작음) | sadness ↔ surprise |
| formant_roughness_mean | 화성 | 0.004 (무시할 수준) | fear ↔ happiness |

두 라벨링(situation vs majority_vote)을 모두 지원하도록 설계 — majority_vote 기준에서는 대부분 효과크기가 커짐(예: formant_roughness 0.004 → 0.017), 실제 인지 감정이 시나리오 의도보다 음향 feature와 더 잘 맞는다는 신호.

### Phase 8 — 음계(장조/단조) feature
`tonality_features.py`
유성음 F0를 12음계 분포로 접어 Krumhansl-Schmuckler 장조/단조 프로파일(모든 조옮김)과 상관비교. **결과: 모든 감정에서 eps² 0.0005~0.0026으로 전부 무시할 수준.** 이산적 음계 구조를 전제로 한 알고리즘이 연속적으로 미끄러지는 말소리 억양에는 잘 들어맞지 않는 것으로 해석 — 화성/리듬과 달리 감정 판별에 기여하지 못하는 부정적(negative) 결과.

### Phase 9 — 선율/리듬 feature
`melody_rhythm_features.py`
연속 유성음 프레임을 피치 안정 구간("음", note) 단위로 묶어 리듬 정보(음 길이, 음 사이 간격 IOI)를 선율 분석에 도입. **결과: `note_dur_mean`이 지금까지 뽑은 모든 feature 중 eps²=0.0513(중간)으로 가장 강한 단일 판별력을 보임** — sadness는 음을 길게 끌고(0.35초) 간격도 넓은(IOI 0.41초) 반면 surprise는 짧고(0.29초) 빽빽함(초당 2.58개). `duration_pitch_corr`(음높이-길이 상관)는 부정적 정서(fear/sadness/disgust/angry)에서 음수, 긍정적 정서(happiness/surprise)에서 양수로 갈리는 패턴도 확인.

### Phase 10 — 최종 통합 및 버그 수정
`emotion_classifier.py` (최종 버전)
Phase 1~4, 8~9의 모든 feature(79개, harmony/tonality/melody_rhythm 세 소스에 중복 존재하던 `n_voiced_pitch_frames` 컬럼 중복 제거)를 합쳐 재학습. 결과는 상단 요약 표 참고. Random Forest 피처 중요도 상위는 여전히 MFCC·피치 통계·partial_roughness가 지배적이며, `note_dur_mean`은 단독 효과크기는 가장 컸지만 다변량 모델에서는 기존 feature와 정보가 겹쳐 79개 중 54위에 그침 — 통계적 유의성과 모델 기여도가 반드시 일치하지 않는다는 걸 보여주는 사례.

## 저장소 구성

```
├── similarity.py, similarity_vectorized.py   # Phase 0
├── extract_features.py                       # Phase 1
├── harmony_features.py                       # Phase 2
├── melodic_profile.py                        # Phase 3
├── partial_roughness.py                      # Phase 4
├── text_features.py                          # Phase 5 (최종 모델에는 미사용)
├── emotion_classifier.py                     # Phase 6, 10 - 통합 학습/평가
├── dissonance_tonality_stats.py              # Phase 7 - 유의성 검정
├── tonality_features.py                      # Phase 8
├── melody_rhythm_features.py                 # Phase 9
├── output/
│   ├── features/, harmony/, melodic_profile/,
│   │   partial_roughness/, tonality/, melody_rhythm/
│   │       ├── per_file_*.csv         # 파일 단위 feature (34k rows)
│   │       └── summary_by_emotion.csv # 감정별 평균/표준편차
│   └── model/
│       ├── decision_tree*.joblib, feature_names*.json
│       └── random_forest*.joblib      # Git LFS
```

`output/similarity/`(초기 탐색 단계 산출물, 파일당 최대 2.18GB)와 원본 오디오/메타데이터는 저장소에 포함하지 않았습니다 (`.gitignore` 참고).

## 다음 단계 후보

- **사전학습 음성 임베딩(Wav2Vec2/HuBERT) 도입**: 손으로 설계한 feature의 성능 상한을 넘기 위한 가장 유력한 방향. 계산 비용 증가와 해석력 저하가 트레이드오프.
- **클래스 불균형 처리 강화**: fear/disgust/surprise recall이 낮음 — SMOTE, 클래스별 threshold 조정 검토.
- **화자 단위 train/test 분리**: 현재 성능 수치의 낙관 편향을 검증.
- **LightGBM/XGBoost**로 모델 교체 비교.
