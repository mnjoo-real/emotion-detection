# 감정 분류를 위한 대화 음성 데이터셋 - 음향학적 분석

AI Hub "감정 분류를 위한 대화 음성 데이터셋"(4차년도 + 5차년도_2차, 33,964개 발화, 7개 감정: angry/disgust/fear/happiness/neutral/sadness/surprise)을 대상으로, 손으로 설계한 음향학적 feature들이 감정별로 유의미하게 다른지 확인하고 이를 바탕으로 감정 분류기를 만드는 프로젝트입니다.

음악의 3요소(화성·선율·리듬)를 음성 프로소디 분석에 번역해 적용하는 것이 핵심 아이디어입니다. 원본 오디오(4차년도/, 5차년도_2차/, 샘플/)는 라이선스 및 용량(20GB+) 문제로 이 저장소에 포함하지 않았습니다.

## 요약 성능 (최종 시점)

117개 feature(음향 기본 + 화성 + 선율음정 + 배음 불협화도 + 음계 + 선율/리듬 + 화자/성별 정규화 + wav2vec2 VAD + 스펙트럼 모양 + 휴지 + 포먼트 대역폭 + 음질(jitter/shimmer/HNR))를 합쳐 Decision Tree / Random Forest로 학습한 결과입니다. (`emotion_classifier.py`)

| 라벨 기준 | 모델 | accuracy | macro F1 |
|---|---|---|---|
| situation (시나리오 의도 라벨) | Decision Tree (depth 6) | 0.219 | 0.192 |
| situation (시나리오 의도 라벨) | Random Forest (300 trees) | 0.409 | 0.324 |
| majority_vote (평가자 5명 다수결, 실제 인지 라벨) | Decision Tree (depth 6) | 0.230 | 0.211 |
| majority_vote (평가자 5명 다수결, 실제 인지 라벨) | **Random Forest (300 trees)** | **0.477** | **0.349** |

7-클래스 랜덤 베이스라인(macro F1 ~0.13, 자세한 값은 Phase 15) 대비 확실히 개선되고, **200회 순열 검정(permutation test)에서 p<0.005로 통계적 유의성을 확인**했습니다 — 라벨을 무작위로 섞어 200번 재학습해도 실제 성능의 절반에도 못 미칩니다(Phase 15). **실제 청취 인지 라벨(majority_vote)이 시나리오 의도 라벨(situation)보다 음향 feature와 일관되게 더 잘 맞습니다.** sadness가 가장 잘 분류되고(F1 0.58~0.61), fear/disgust/surprise는 표본 수가 적고 음향적으로 겹치는 부분이 많아 여전히 약합니다(F1 0.13~0.24).

> 주의: 원본 메타데이터에 화자 구분 고유 ID가 없어 train/test를 화자 단위로 분리하지 못했습니다. 같은 화자의 다른 발화가 양쪽에 섞여 있을 수 있어 정확도가 실제보다 낙관적일 수 있습니다. MFCC 기반 비지도 클러스터링으로 화자를 대신 추정해보려 했으나 실패했습니다 — 자세한 내용은 Phase 17.

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

### Phase 11 — 화자/성별 정규화
`speaker_norm_features.py`
원본 메타데이터에 화자 고유 ID는 없지만 `성별` 컬럼은 있고, 감정별 성별 비율이 16.5%~41.7%로 불균형하다는 걸 확인 — 성별이 `pitch_mean` 분산의 43%를 설명(상황 효과는 3%뿐). 두 가지로 대응:
- **포먼트 비율**(F2/F1, F3/F2, F4/F3): 절대 Hz 대신 음성학 표준 화자 정규화 기법 적용. `f3_f2_ratio`가 이후 Random Forest에서 기존 `f1~f4_mean` 4개를 전부 앞지름 — 확실한 개선.
- **성별 z-score**(`formant_roughness_gender_z`, `pitch_*_gender_z`): 성별 내부로 나눠보면 `formant_roughness` 효과크기가 남성 기준 거의 3배로 뛰는 걸 확인했지만, 정작 성별 z-score로 다시 합치면 개선되지 않음 — 남성은 angry가 최고, 여성은 happiness가 최고로 **감정 순위 자체가 성별마다 다르기 때문**. 반면 `pitch_mean`은 감정 순위가 두 성별 모두 동일해서 z-score가 일반화에는 도움되지만 이 데이터셋 자체의 예측력은 원본보다 약간 낮아짐 — 성능과 일반화의 트레이드오프 사례.

### Phase 12 — 사전학습 음성 임베딩(VAD) 도입
`vad_features.py`
Phase 10까지 손으로 설계한 feature의 성능 상한을 넘기 위해, audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim(사전학습 wav2vec2 기반 감정 회귀 모델)로 각 발화의 arousal/dominance/valence를 예측해 feature 3개로 추가. 이 모델(약 300M 파라미터) 하나만 재사용해야 해서 다른 스크립트와 달리 프로세스 풀 병렬화 대신 단일 프로세스로 순차 처리. MPS(Radeon Pro 560)가 배치=1 추론에서 오히려 CPU보다 느려서(0.33개/초 vs 0.74개/초, 커널 디스패치 오버헤드가 실제 연산량보다 큼) CPU 8스레드로 전환, 33,964개 처리에 약 13시간 소요. 이후 도입한 Phase 14 최종 모델에서 arousal/dominance가 나란히 피처 중요도 1·2위를 차지 — 가장 강력한 단일 feature 계열.

### Phase 13 — 추가 음향학적 feature 4종
CPU가 VAD 추출로 점유된 동안 자동 체이닝으로 순차 실행:
- `spectral_dynamics_features.py`: spectral centroid/rolloff/flatness, zero-crossing rate, spectral flux — MFCC가 놓치는 음색의 밝기·변화 속도. `spectral_flux_mean`이 최종 모델에서 피처 중요도 3위.
- `pause_features.py`: 휴지(pause) 개수/길이/비율 — 망설임·숨고르기 등 "소리가 끊기는 패턴".
- `formant_bandwidth_features.py`: harmony_features.py의 LPC 계산 과정에서 이미 계산되지만 버려지던 포먼트 대역폭을 저장(성대 긴장/breathiness 지표). `harmony_features.py`의 `extract_formants_from_frame()`에 `return_bandwidth` 옵션을 추가해 재사용.
- `voice_quality_features.py`: jitter(주기 길이 미세 변동)/shimmer(진폭 미세 변동)/HNR(조화 대 잡음비) — 성대 진동 자체의 안정성. Praat 수준의 정밀한 성문 폐쇄점 검출 대신, pyin의 프레임별 F0 추정치를 이용해 파형 국소 최댓값을 따라가는 간이 구현. `hnr_mean`이 최종 모델에서 피처 중요도 10위.

### Phase 14 — 최종 통합 재학습
`emotion_classifier.py` (117개 feature)
Phase 1~13의 모든 feature를 합쳐 재학습, `n_voiced_pitch_frames`가 harmony/tonality/melody_rhythm 세 소스에 중복 존재하던 문제를 다시 한번 dedupe. 결과는 상단 요약 표 참고 — 79개(Phase 10) → 86개(Phase 11) → 117개(Phase 14)로 늘면서 situation macro F1 0.295 → 0.306 → 0.324, majority_vote macro F1 0.320 → 0.325 → 0.349로 단계마다 실제 개선을 확인했다.

### Phase 15 — 성능의 통계적 유의성 검증
`significance_validation.py`
"macro F1 0.5가 넘어야 유의미하다"는 통설은 이진·균형 분류에서만 성립 — 7-클래스 불균형 문제에서 실제 chance-level macro F1은 0.06~0.14(전략에 따라)에 불과함을 더미 분류기로 먼저 확인. 이어서 두 가지로 엄밀하게 검증:
- **5-fold 교차검증**: accuracy 0.473±0.006, macro F1 0.339±0.006 — 특정 train/test 분할의 우연이 아님을 확인.
- **순열 검정(200회)**: 라벨을 무작위로 섞어 동일 절차로 200번 재학습한 귀무분포(macro F1 평균 0.133, 최댓값 0.142)와 실제 성능(macro F1 0.342)을 비교, p<0.005. 실제 성능이 귀무분포 평균보다 55.8 표준편차 위에 있어, 우연일 가능성은 사실상 0.

### Phase 16 — Russell 모델 확장(감정 영역을 타원으로) 검증 — 실패
`emotion_region_vad.py`
한의환·차형태(2017)의 "타원 방정식으로 감정을 점이 아닌 영역으로 표현 + 베이지안 결정 규칙" 방법론을 Phase 12의 VAD 값에 적용. 수학적으로 이 방법은 scikit-learn `QuadraticDiscriminantAnalysis`와 동일하다는 걸 이용해 구현. **결과: 실패.** 7개 감정의 1σ 타원이 거의 완전히 겹쳐(sadness만 근소하게 분리) macro F1이 0.126~0.133으로 무작위 baseline보다도 낮음. 원 논문은 사람이 의식적으로 응답한 ANEW 단어 설문 평균(이미 잘 분리된 데이터)에 적용해 92.86% 정확도를 얻었지만, 우리는 wav2vec2가 실제 음성에서 예측한 잡음 많은 값을 쓰기 때문에 재현되지 않음 — "좋은 feature도 다른 feature 없이 단독으로 쓰면 무너진다"는 반증 사례.

### Phase 17 — MFCC 기반 화자 클러스터링 시도 — 실패, 중단
화자 단위 train/test 분리(Phase 10 이후 계속 언급된 한계)를 위해 MFCC로 화자를 비지도 클러스터링해보려 했으나, 검증 결과 실패로 판단해 중단. 가장 쉬운 케이스(성별 2그룹, `pitch_mean` 효과크기 0.43으로 이미 알려진 정답)로 KMeans(k=2)를 테스트한 결과 성별과 무관하게 뒤섞임 — 발화문이 28,512종으로 거의 다 다르고 같은 화자가 같은 문장을 감정별로 반복한 구조도 아니라서, 클러스터링이 화자보다 발화 내용/감정 변동을 더 크게 잡는 것으로 판단. 개별 화자 식별은 전용 화자 임베딩 모델(ECAPA-TDNN 등)이 필요하며, 그래도 정답(진짜 화자 ID) 없이는 검증이 불가능하다는 근본적 한계는 남는다 — 추가 파이프라인 구축 없이 여기서 중단.

## 저장소 구성

```
├── similarity.py, similarity_vectorized.py   # Phase 0
├── extract_features.py                       # Phase 1
├── harmony_features.py                       # Phase 2 (Phase 13에서 return_bandwidth 옵션 추가)
├── melodic_profile.py                        # Phase 3
├── partial_roughness.py                      # Phase 4
├── text_features.py                          # Phase 5 (최종 모델에는 미사용)
├── emotion_classifier.py                     # Phase 6, 10, 14 - 통합 학습/평가
├── dissonance_tonality_stats.py              # Phase 7 - 유의성 검정
├── tonality_features.py                      # Phase 8
├── melody_rhythm_features.py                 # Phase 9
├── speaker_norm_features.py                  # Phase 11
├── vad_features.py                           # Phase 12
├── spectral_dynamics_features.py             # Phase 13
├── pause_features.py                         # Phase 13
├── formant_bandwidth_features.py             # Phase 13
├── voice_quality_features.py                 # Phase 13
├── significance_validation.py                # Phase 15
├── emotion_region_vad.py                     # Phase 16 (실패 사례)
├── output/
│   ├── features/, harmony/, melodic_profile/, partial_roughness/,
│   │   tonality/, melody_rhythm/, speaker_norm/, vad/, pause/,
│   │   spectral_dynamics/, formant_bandwidth/, voice_quality/
│   │       ├── per_file_*.csv         # 파일 단위 feature (34k rows)
│   │       └── summary_by_emotion.csv # 감정별 평균/표준편차
│   ├── vad_region/                    # Phase 16 시각화 (PNG)
│   └── model/
│       ├── decision_tree*.joblib, feature_names*.json
│       └── random_forest*.joblib      # Git LFS
```

`output/similarity/`(초기 탐색 단계 산출물, 파일당 최대 2.18GB)와 원본 오디오/메타데이터는 저장소에 포함하지 않았습니다 (`.gitignore` 참고).

## 다음 단계 후보

- **클래스 불균형 처리 강화**: fear/disgust/surprise recall이 여전히 낮음(0.08~0.18) — SMOTE, 클래스별 threshold 조정 검토.
- **LightGBM/XGBoost**로 모델 교체 비교.
- **화자 임베딩 기반 재시도**: MFCC 클러스터링은 실패(Phase 17)했지만, ECAPA-TDNN 등 화자 인식 전용 임베딩으로는 가능성이 남아있음 — 다만 검증 불가능하다는 근본적 한계는 여전함.
- **텍스트/문맥 정보 재도입**: valence는 음향만으로 예측하기 어렵다는 게 Phase 16에서 다시 확인됨 — 어휘·문맥 정보 결합이 필요할 수 있음(Phase 5에서 시도했으나 스크립트 종속적이라 폐기됐던 접근을 다른 방식으로 재검토).
