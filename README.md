# Pediatric Chest X-ray Pneumonia Classification

소아 Chest X-ray 영상을 이용하여 **NORMAL / PNEUMONIA를 분류**하는 딥러닝 프로젝트입니다.

동일한 ResNet-50 구조에서 **Scratch 학습과 ImageNet 사전학습 기반 Transfer Learning**을 비교하고, 최종적으로 **Grad-CAM++**을 이용해 모델의 판단 근거를 시각화하는 것을 목표로 합니다.

> 현재 개발 중인 프로젝트입니다. 아래 실험 결과는 최종 Test 성능이 아닌 Validation 기반의 preliminary experiment 결과입니다.

---

## 1. Project Overview

### 목표

- 소아 Chest X-ray 영상의 NORMAL / PNEUMONIA 이진 분류
- ResNet-50 Scratch와 Transfer Learning 비교
- Accuracy뿐만 아니라 Recall, Precision, F1-score, AUROC 등 다양한 지표를 이용한 평가
- 두 학습 방식의 최종 성능뿐만 아니라 수렴 속도와 학습 안정성 비교
- Grad-CAM++을 이용한 모델 판단 영역 시각화
- 최종 학습 모델을 사용자 UI와 연결

---

## 2. Dataset

**Chest X-Ray Images (Pneumonia)** 데이터셋을 사용합니다.

- Total: **5,856 images**
- NORMAL: **1,583 images (27.0%)**
- PNEUMONIA: **4,273 images (73.0%)**
- Pediatric Chest X-ray
- NORMAL / PNEUMONIA Binary Classification

### Original Split

| Split | Images |
| --- | ---: |
| Train | 5,216 |
| Validation | 16 |
| Test | 624 |

원본 데이터셋은 Validation 데이터가 16장으로 매우 적기 때문에 전체 데이터를 병합한 후 새로운 Train / Validation / Test split을 구성했습니다.

---

## 3. Group-aware Data Split

데이터셋의 파일명을 분석하는 과정에서 동일한 식별자 패턴을 공유하는 이미지가 존재하는 것을 확인했습니다.

예를 들어 PNEUMONIA 이미지에서는 `personXX`, NORMAL 이미지에서는 `IM-XXXX`와 같은 반복되는 식별자 패턴이 존재합니다.

따라서 이미지 단위로 무작위 분할할 경우 서로 연관되어 있을 가능성이 있는 이미지가 서로 다른 split에 포함될 수 있다고 판단했습니다.

이를 줄이기 위해 파일명에서 **filename-derived group ID**를 생성하고, 동일 group이 여러 split에 분산되지 않도록 group-aware split을 적용했습니다.

> 해당 group ID가 실제 patient ID임을 직접 확인할 수 있는 메타데이터는 제공되지 않으므로, 본 프로젝트에서는 이를 patient ID가 아닌 **filename-derived group identifier**로 정의합니다.

### Group ID

- PNEUMONIA → `personXX`
- NORMAL → `IM-XXXX`
- Label prefix를 추가하여 클래스 간 ID 충돌 방지

### Split Method

`StratifiedGroupKFold`를 사용하여 클래스 비율을 고려하면서 동일 group이 서로 다른 fold에 포함되지 않도록 분할했습니다.

- `n_splits=10`
- `shuffle=True`
- `random_state=42`
- Fold 0–7 → Train
- Fold 8 → Validation
- Fold 9 → Test

이를 통해 약 **8 : 1 : 1** 비율로 구성했습니다.

### Final Split

| Split | Images | Groups | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,686 | 2,232 | 1,267 | 3,419 |
| Validation | 585 | 279 | 158 | 427 |
| Test | 585 | 279 | 158 | 427 |
| **Total** | **5,856** | **2,790** | **1,583** | **4,273** |

### Leakage Check

Filename-derived group ID를 기준으로 각 split 간 group overlap을 확인했습니다.

```text
Train-Val overlap : 0
Train-Test overlap: 0
Val-Test overlap  : 0
```

따라서 **정의된 filename-derived group 기준으로 split 간 group overlap이 없음을 확인했습니다.**

---

## 4. Preprocessing

현재 적용된 preprocessing pipeline은 다음과 같습니다.

### Resize & Padding

- Input size: `224 × 224`
- 원본 영상의 aspect ratio 유지
- 부족한 영역은 padding 적용
- Padding pixel value: `0`

원본 영상을 단순히 224 × 224로 강제 변환하지 않고 aspect ratio를 유지하여 영상의 기하학적 왜곡을 줄입니다.

### Normalization

ImageNet pretrained model과 Scratch model의 입력 조건을 동일하게 유지하기 위해 두 모델 모두 **ImageNet normalization**을 적용합니다.

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Data Augmentation

Train dataset에만 다음 augmentation을 적용합니다.

- Random Rotation: `±10°`
- Rotation fill value: `0`

Validation과 Test dataset에는 augmentation을 적용하지 않습니다.

Horizontal Flip은 영상의 좌우 해부학적 위치를 변경할 수 있으므로 현재 augmentation에서 제외했습니다.

---

## 5. Model

### ResNet-50

동일한 ResNet-50 architecture를 사용하여 두 가지 학습 방식을 비교합니다.

#### Scratch

- Random initialization
- ImageNet pretrained weights 사용하지 않음
- 전체 network를 처음부터 학습
- Output classifier를 binary classification에 맞게 수정

#### Transfer Learning

- ImageNet pretrained ResNet-50 사용
- 초기에는 backbone을 freeze하고 classifier를 학습
- 이후 필요에 따라 일부 또는 전체 layer를 fine-tuning

동일한 dataset split, preprocessing 및 evaluation 기준을 사용하여 두 학습 방법을 비교할 예정입니다.

최종 성능뿐만 아니라 **최고 성능에 도달하는 데 필요한 epoch 수와 validation 성능의 안정성**도 비교 대상으로 확인할 예정입니다.

---

## 6. Training Pipeline

PyTorch 기반의 학습 및 검증 pipeline을 구성했습니다.

현재 Scratch baseline에서는 다음 설정을 사용하여 preliminary experiment를 진행했습니다.

| Setting | Value |
| --- | --- |
| Model | ResNet-50 Scratch |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Epochs | 20 |
| Positive Class | PNEUMONIA |

각 epoch마다 Train Loss와 Validation Loss 및 평가 지표를 기록합니다.

Validation F1-score를 기준으로 best checkpoint를 저장하며, Test dataset은 모델 및 hyperparameter 선택 과정에 사용하지 않고 최종 평가 단계에서 사용합니다.

### Experiment Logging

각 실험은 별도의 experiment directory에 기록합니다.

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

- `config.json`: 실험 설정
- `history.csv`: epoch별 loss 및 validation metrics
- `loss_curve.png`: Train / Validation Loss 변화
- `validation_metrics.png`: Validation F1 / Recall / AUROC 변화
- `best_model.pth`: Validation F1 기준 best checkpoint

> `.pth` checkpoint 파일은 GitHub repository에 업로드하지 않습니다.

---

## 7. Preliminary Scratch Experiment

학습 pipeline의 정상 동작을 확인하기 위해 먼저 **1 epoch smoke test**를 수행했습니다.

이를 통해 다음 전체 흐름이 정상적으로 동작하는 것을 확인했습니다.

```text
DataLoader
    ↓
GPU Training
    ↓
Validation
    ↓
Metric Calculation
    ↓
Experiment Logging
    ↓
Best Checkpoint Saving
```

이후 Scratch ResNet-50을 20 epoch 동안 학습하여 preliminary learning behavior를 확인했습니다.

### Preliminary Result

- Best Validation F1-score: **0.9671**
- Best F1 Epoch: **12**
- Minimum Validation Loss: **0.1163**
- Minimum Validation Loss Epoch: **19**
- Validation AUROC: 전반적으로 약 **0.97–0.99**

Train Loss는 학습이 진행됨에 따라 전반적으로 감소했습니다.

Validation Loss는 epoch별 변동성을 보였지만 지속적으로 상승하는 경향은 관찰되지 않았으며, 후반 epoch에서도 낮은 Validation Loss와 높은 Validation F1-score를 유지했습니다.

따라서 현재 **20 epoch 범위에서는 뚜렷한 지속적 overfitting pattern이 관찰되지 않았습니다.**

> 위 결과는 Validation set을 이용한 preliminary experiment 결과이며 최종 Test 성능이 아닙니다.

---

## 8. Evaluation

모델 평가는 다음 지표를 사용합니다.

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Confusion Matrix

PNEUMONIA를 positive class로 정의합니다.

데이터셋의 클래스 비율이 불균형하기 때문에 Accuracy만으로 모델을 평가하지 않고 Recall, Precision, F1-score 및 AUROC를 함께 확인합니다.

특히 의료 영상 분류 특성을 고려하여 **PNEUMONIA Recall**을 주요 평가 지표 중 하나로 확인합니다.

최종 모델 및 hyperparameter가 결정된 이후 독립된 Test set을 이용하여 최종 성능을 평가할 예정입니다.

---

## 9. Explainable AI

최종 모델의 판단 근거를 확인하기 위해 **Grad-CAM++**을 적용할 예정입니다.

Chest X-ray 원본 영상 위에 Class Activation Map을 overlay하여 모델이 PNEUMONIA / NORMAL을 판단할 때 어떤 영역에 주목했는지 시각적으로 확인합니다.

특히 모델이 실제 폐 영역 및 병변과 관련된 특징을 이용하는지, 배경이나 영상의 비의도적 특징에 의존하고 있지는 않은지 확인할 예정입니다.

---

## 10. User Interface

최종적으로 학습된 모델을 UI와 연결할 예정입니다.

예상 흐름:

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

---

## 11. Tech Stack

- Python
- PyTorch
- torchvision
- pandas
- scikit-learn
- Pillow
- Matplotlib
- Git / GitHub

향후 모델 inference 및 UI 구현 단계에서 필요한 기술을 추가할 예정입니다.

---

## 12. Project Structure

```text
pneumonia-diagnosis/
│
├── data/
│   ├── raw/
│   └── splits/
│       └── group_split.csv
│
├── experiments/
│   └── scratch/
│       └── exp_XXX/
│           ├── config.json
│           ├── history.csv
│           ├── loss_curve.png
│           └── validation_metrics.png
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── inspect_preprocess.py
│   └── plot_history.py
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

Dataset 원본 및 학습된 model checkpoint 등 대용량 파일은 GitHub repository에 직접 업로드하지 않습니다.

---

## 13. Progress

- [x] Dataset 구조 분석
- [x] 클래스 분포 확인
- [x] Filename-derived group ID 정의
- [x] Group-aware Train / Validation / Test split
- [x] Group overlap validation
- [x] Data preprocessing pipeline
- [x] ResNet-50 Scratch model implementation
- [x] Training / validation pipeline
- [x] Experiment logging
- [x] 1 epoch smoke test
- [x] Scratch preliminary 20 epoch experiment
- [x] Learning curve visualization
- [ ] DataLoader performance optimization
- [ ] Scratch hyperparameter experiments
- [ ] ResNet-50 Transfer Learning
- [ ] Scratch vs Transfer Learning comparison
- [ ] Final Test evaluation
- [ ] Grad-CAM++
- [ ] UI implementation
- [ ] Final comparison and analysis

---

## 14. Current Issue & Next Steps

Scratch preliminary training 과정에서 GPU가 정상적으로 사용되고 있음을 확인했지만, GPU utilization이 지속적으로 높게 유지되지 않고 변동하는 현상을 확인했습니다.

현재 DataLoader의 `num_workers=0` 설정에 따른 **data-loading bottleneck 가능성**을 확인하고 있습니다.

아직 원인이 확정된 것은 아니며, 다음 단계에서 worker 수 및 data loading 설정을 변경하여 epoch time과 GPU utilization 변화를 비교할 예정입니다.

이후 진행 계획은 다음과 같습니다.

1. DataLoader 성능 개선 및 학습 환경 점검
2. Scratch optimizer / learning rate 비교
3. Scratch baseline 확정
4. ImageNet pretrained ResNet-50 Transfer Learning 구현
5. Scratch vs Transfer Learning 성능 및 수렴 특성 비교
6. 최종 모델 선정 후 Test set 평가
7. Grad-CAM++ 적용
8. UI 연동

---

## 15. Limitations

현재 사용 중인 데이터셋에서는 명시적인 patient-level metadata를 직접 확인할 수 없기 때문에 파일명에서 추출한 식별자를 group ID로 사용했습니다.

따라서 본 프로젝트의 group ID를 검증된 patient ID로 해석하지 않습니다.

또한 현재 Scratch 실험 결과는 Validation set 기반의 preliminary result이며, 모델 선택 및 실험 설정이 완료되기 전까지 Test set은 최종 성능 평가에 사용하지 않습니다.

향후 연구로 확장할 경우 명확한 patient-level metadata를 제공하는 원본 데이터 또는 추가 데이터셋을 활용하여 보다 엄격한 patient-level evaluation 및 external validation을 수행하는 것을 고려합니다.