# Pediatric Chest X-ray Pneumonia Classification

소아 Chest X-ray 영상을 이용하여 **NORMAL / PNEUMONIA를 분류**하는 딥러닝 프로젝트입니다.

동일한 ResNet-50 architecture에서 **Scratch 학습과 ImageNet 사전학습 기반 Transfer Learning**을 비교하고, 최종적으로 **Grad-CAM++**을 적용하여 모델의 판단 영역을 시각화하는 것을 목표로 합니다.

> 현재 개발 및 실험 진행 중인 프로젝트입니다. 현재 공개된 성능은 Validation 기반의 preliminary result이며, 최종 Test 성능이 아닙니다.

---

## Key Highlights

- **Architecture:** ResNet-50 Scratch vs ImageNet Transfer Learning
- **Data Split:** Filename-derived group 기반 `StratifiedGroupKFold`
- **Data Integrity:** Group overlap 및 SHA/pHash 기반 cross-split duplicate 검사
- **Evaluation:** Accuracy, Recall, Precision, F1-score, AUROC, Confusion Matrix
- **Explainability:** Grad-CAM++ 기반 모델 판단 영역 시각화 예정

---

## 1. Project Overview

### Objectives

- 소아 Chest X-ray의 NORMAL / PNEUMONIA 이진 분류
- 동일한 ResNet-50 구조에서 Scratch와 Transfer Learning 비교
- 두 학습 방식의 분류 성능 및 수렴 특성 비교
- 클래스 불균형을 고려한 다양한 평가 지표 분석
- Grad-CAM++을 이용한 모델 판단 영역 확인
- 최종 모델을 사용자 UI와 연동

Scratch와 Transfer Learning 비교에서는 다음을 주요하게 확인합니다.

1. ImageNet pretrained weights가 분류 성능에 미치는 영향
2. 두 학습 방식이 높은 성능에 도달하는 데 필요한 epoch 수
3. 학습 과정에서의 Validation 성능 및 Loss 변화

---

## 2. Dataset & Split

**Chest X-Ray Images (Pneumonia)** 데이터셋을 사용합니다.

### Original Dataset

| Class | Images | Ratio |
| --- | ---: | ---: |
| NORMAL | 1,583 | 27.0% |
| PNEUMONIA | 4,273 | 73.0% |
| **Total** | **5,856** | **100%** |

원본 데이터셋은 다음과 같이 제공됩니다.

| Split | Images |
| --- | ---: |
| Train | 5,216 |
| Validation | 16 |
| Test | 624 |

Validation set이 16장으로 매우 작기 때문에 원본 split을 그대로 사용하지 않고 전체 데이터를 병합한 뒤 새로운 Train / Validation / Test split을 구성했습니다.

### Group-aware Split

파일명에서 반복되는 식별자 패턴을 추출하여 **filename-derived group ID**를 정의했습니다.

- PNEUMONIA → `personXX`
- NORMAL → `IM-XXXX`

동일 group이 여러 split에 포함되는 것을 방지하기 위해 `StratifiedGroupKFold`를 사용했습니다.

```text
n_splits = 10
shuffle = True
random_state = 42

Fold 0-7 → Train
Fold 8   → Validation
Fold 9   → Test
```

> 데이터셋에 명시적인 patient-level metadata가 제공되지 않으므로 해당 group ID를 검증된 patient ID로 해석하지 않습니다.

### Data Integrity Check

높은 Scratch Validation 성능이 관찰되어 데이터 누수 가능성을 추가로 점검했습니다.

- Filename-derived group 기준 cross-split overlap 검사
- SHA 기반 exact duplicate 검사
- pHash 기반 near-duplicate 후보 탐색 및 시각 검수
- 실제 cross-split duplicate 1건 확인
- 중복본 1장을 실험 대상에서 제외한 후 동일한 방법으로 split 재생성
- 재생성 후 group overlap 및 exact duplicate leakage가 없음을 확인

원본 데이터는 수정하지 않고 그대로 보존합니다.

### Current Split

중복본 1장을 제외한 **5,855장**을 현재 실험에 사용합니다.

| Split | Images | Groups | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,684 | 2,233 | 1,266 | 3,418 |
| Validation | 586 | 279 | 158 | 428 |
| Test | 585 | 278 | 158 | 427 |
| **Total** | **5,855** | **2,790** | **1,582** | **4,273** |

현재 split의 filename-derived group overlap:

```text
Train-Val overlap : 0
Train-Test overlap: 0
Val-Test overlap  : 0
```

SHA 기반 검사에서도 **cross-split exact duplicate 0건**을 확인했습니다.

---

## 3. Preprocessing

모든 입력 영상은 ResNet-50 입력을 위해 `224 × 224` 크기로 변환합니다.

### Resize & Padding

원본 영상의 aspect ratio를 유지하면서 resize한 뒤 부족한 영역에 padding을 적용합니다.

- Input size: `224 × 224`
- Aspect ratio 유지
- Padding value: `0`

이를 통해 영상을 단순히 `224 × 224`로 강제 변환할 때 발생할 수 있는 기하학적 왜곡을 줄입니다.

### Normalization

Scratch와 Transfer Learning의 입력 조건을 동일하게 유지하기 위해 두 모델 모두 ImageNet normalization을 적용합니다.

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Data Augmentation

Train dataset에만 다음 augmentation을 적용합니다.

```text
Random Rotation: ±10°
Rotation fill  : 0
```

Validation과 Test dataset에는 augmentation을 적용하지 않습니다.

Horizontal Flip은 영상의 좌우 해부학적 위치를 변경할 수 있어 현재 augmentation에서 제외했습니다.

---

## 4. Model & Training

### ResNet-50 Scratch

- ResNet-50 architecture
- Random initialization
- ImageNet pretrained weights 사용하지 않음
- 전체 network를 처음부터 학습
- Final classifier를 2-class classification에 맞게 변경

### ResNet-50 Transfer Learning

ImageNet pretrained ResNet-50을 이용한 Transfer Learning을 진행할 예정입니다.

초기에는 pretrained backbone을 freeze하고 classifier를 학습한 뒤, 이후 일부 또는 전체 layer를 fine-tuning하는 방식을 사용할 예정입니다.

Scratch와 Transfer Learning에는 동일한 dataset split, preprocessing 및 evaluation 기준을 적용합니다.

### Training Configuration

현재 Scratch 실험의 주요 설정은 다음과 같습니다.

| Setting | Value |
| --- | --- |
| Model | ResNet-50 Scratch |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 0.0005 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Epochs | 100 |
| Positive Class | PNEUMONIA |

각 epoch마다 학습 및 Validation 결과를 기록하며, **Validation F1-score가 가장 높은 checkpoint**를 저장합니다.

Test set은 hyperparameter 및 모델 선택에 사용하지 않고 최종 평가 단계까지 분리하여 유지합니다.

---

## 5. Experiments

### Initial Scratch Baseline

초기 Scratch baseline은 다음 조건에서 20 epoch 동안 수행했습니다.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Epochs | 20 |

결과:

| Metric | Result |
| --- | ---: |
| Best Validation F1 | 0.9671 |
| Best F1 Epoch | 12 |
| Minimum Validation Loss | 0.1163 |
| Minimum Val Loss Epoch | 19 |
| Validation AUROC | 약 0.97–0.99 |

### Extended Scratch Experiment

Learning Rate를 `0.0005`로 낮추고 학습을 100 epoch까지 확장하여 장기적인 학습 특성을 확인했습니다.

Train Loss는 장기 학습에서 매우 낮은 수준까지 감소했고 train-validation 간 generalization gap이 증가하여 후반부에 overfitting 신호가 관찰되었습니다.

반면 Validation F1-score는 높은 수준을 유지했으며 최고 약 **0.9825**가 관찰되었습니다.

예상보다 높은 Scratch Validation 성능이 확인되어 이후 data integrity 검사를 추가로 수행했습니다.

> 위 실험은 현재 정제된 split 이전의 preliminary experiment이므로 최종 성능으로 사용하지 않습니다.

### Current Experiment

Data integrity 검사 후 중복 이미지 1장을 제외하고 동일한 group-aware split 방법으로 데이터를 재구성했습니다.

현재 정제된 split에서 Scratch ResNet-50을 동일한 설정으로 재학습하여 Validation 성능의 재현 여부를 확인하고 있습니다.

---

## 6. Evaluation

PNEUMONIA를 positive class로 정의합니다.

다음 평가 지표를 사용합니다.

- Accuracy
- Recall
- Precision
- F1-score
- AUROC
- Confusion Matrix

데이터셋은 약 **NORMAL 27% : PNEUMONIA 73%**의 클래스 불균형을 가지고 있으므로 Accuracy만으로 모델 성능을 판단하지 않습니다.

특히 PNEUMONIA Recall과 함께 Precision, F1-score, AUROC 및 Confusion Matrix를 확인하여 클래스별 예측 특성을 분석합니다.

최종 모델과 학습 설정이 결정된 이후 독립된 Test set을 이용해 최종 성능을 평가할 예정입니다.

---

## 7. Explainability

최종 모델에는 **Grad-CAM++**을 적용할 예정입니다.

Chest X-ray 위에 Class Activation Map을 overlay하여 모델이 NORMAL / PNEUMONIA를 판단할 때 어떤 영역에 주목했는지 확인합니다.

이를 통해 모델이 실제 폐 영역과 관련된 특징을 활용하는지, 또는 배경이나 영상 marker와 같은 비의도적 특징에 의존하는지 분석할 예정입니다.

---

## 8. User Interface

최종 학습 모델은 사용자 UI와 연결할 예정입니다.

```text
Chest X-ray Upload
        ↓
Preprocessing
        ↓
Model Inference
        ↓
NORMAL / PNEUMONIA Probability
        ↓
Grad-CAM++ Visualization
        ↓
Result Display
```

> 본 프로젝트는 학습 및 연구 목적의 프로젝트이며 실제 의료 진단을 목적으로 하지 않습니다.

---

## 9. Experiment Logging

각 실험은 별도의 experiment directory에 저장합니다.

```text
experiments/
└── scratch/
    └── exp_XXX/
        ├── config.json
        ├── history.csv
        ├── loss_curve.png
        ├── validation_metrics.png
        └── best_model.pth
```

- `config.json` — 실험 설정
- `history.csv` — epoch별 Loss 및 Validation metrics
- `loss_curve.png` — Train / Validation Loss
- `validation_metrics.png` — Validation metric 변화
- `best_model.pth` — Validation F1 기준 best checkpoint

Model checkpoint와 Dataset 원본 등 대용량 파일은 GitHub repository에 업로드하지 않습니다.

---

## 10. Project Structure

```text
pneumonia-diagnosis/
│
├── data/
│   ├── raw/
│   └── splits/
│       ├── group_split.csv
│       └── near_duplicates.csv
│
├── experiments/
│   └── scratch/
│       └── exp_XXX/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── inspect_preprocess.py
│   ├── plot_history.py
│   ├── check_near_duplicates.py
│   └── visualize_near_duplicates.py
│
├── src/
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   ├── split_dataset.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   └── resnet.py
│   │
│   └── training/
│       └── optimizer.py
│
├── train.py
├── README.md
└── .gitignore
```

---

## 11. Progress

- [x] Dataset 구조 및 클래스 분포 분석
- [x] Filename-derived group ID 정의
- [x] Group-aware Train / Validation / Test split
- [x] Group overlap 검사
- [x] SHA/pHash 기반 duplicate 검사
- [x] Cross-split duplicate 처리 및 split 재생성
- [x] Preprocessing pipeline
- [x] ResNet-50 Scratch 구현
- [x] Training / Validation pipeline
- [x] Experiment logging
- [x] Scratch preliminary experiments
- [ ] 정제된 split에서 Scratch 재학습 및 분석
- [ ] Scratch baseline 확정
- [ ] ResNet-50 Transfer Learning
- [ ] Scratch vs Transfer Learning 비교
- [ ] Final Test evaluation
- [ ] Grad-CAM++
- [ ] UI implementation

---

## 12. Next Steps

현재 정제된 dataset split에서 Scratch 모델의 Validation 성능을 재검증하고 있습니다.

이후 진행 순서는 다음과 같습니다.

1. Scratch 재학습 결과 및 Confusion Matrix 분석
2. Scratch baseline 확정
3. ImageNet pretrained ResNet-50 Transfer Learning
4. Scratch vs Transfer Learning 성능 및 수렴 특성 비교
5. 최종 모델 선정 및 Test evaluation
6. Grad-CAM++ 적용
7. UI 연동

---

## 13. Limitations

현재 데이터셋에는 명시적인 patient-level metadata가 제공되지 않기 때문에 파일명에서 추출한 식별자를 group ID로 사용합니다.

따라서 filename-derived group을 검증된 patient ID로 해석하지 않습니다.

또한 pHash는 이미지의 의미적 동일성을 보장하는 방법이 아니므로 near-duplicate 후보 탐색 용도로 사용하고 후보 이미지는 시각적으로 검수합니다.

현재 공개된 실험 결과는 Validation 기반의 preliminary result이며 최종 Test 성능이 아닙니다.

향후 연구로 확장할 경우 명확한 patient-level metadata를 제공하는 데이터셋과 외부 Chest X-ray 데이터셋을 이용한 **external validation 및 domain shift 평가**를 고려합니다.