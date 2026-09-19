# Pediatric Chest X-ray Pneumonia Classification

소아 Chest X-ray 영상을 이용하여 **NORMAL / PNEUMONIA를 분류**하는 딥러닝 프로젝트입니다.

동일한 ResNet-50 architecture에서 **Scratch 학습과 ImageNet 사전학습 기반 Transfer Learning**을 비교하고, 최종적으로 **Grad-CAM++**을 적용하여 모델의 판단 영역을 시각화하는 것을 목표로 합니다.

> 현재 개발 및 실험 진행 중인 프로젝트입니다. 현재 공개된 성능은 Validation 기반 결과이며, 최종 Test 성능이 아닙니다.

---

## Key Highlights

- **Architecture:** ResNet-50 Scratch vs ImageNet Transfer Learning
- **Data Split:** Filename-derived group 기반 `StratifiedGroupKFold`
- **Data Integrity:** Group overlap 및 SHA-256 / pHash 기반 cross-split duplicate 검사
- **Training Analysis:** Validation instability, adaptive learning rate, regularization 분석
- **Transfer Learning:** FC-only warm-up → Full fine-tuning
- **Evaluation:** Accuracy, Recall, Precision, F1-score, AUROC, Confusion Matrix
- **Explainability:** Grad-CAM++ 기반 모델 판단 영역 시각화 예정

---

## 1. Project Overview

### Objectives

- 소아 Chest X-ray의 NORMAL / PNEUMONIA 이진 분류
- 동일한 ResNet-50 architecture에서 Scratch와 Transfer Learning 비교
- 두 학습 방식의 분류 성능 및 학습 수렴 특성 비교
- 클래스 불균형을 고려한 다양한 평가 지표 분석
- Grad-CAM++을 이용한 모델 판단 영역 확인
- 최종 모델을 사용자 UI와 연동

단순히 최고 성능만 비교하는 것이 아니라 데이터 무결성, 학습 안정성, generalization gap 및 수렴 과정까지 함께 분석합니다.

---

## 2. Research Question

### RQ1

> **Scratch 학습과 Transfer Learning은 분류 성능과 학습 수렴 속도에서 어떤 차이를 보이는가?**

### Hypotheses

**H1 — Classification Performance**

ImageNet pretrained Transfer Learning이 Scratch보다 높은 분류 성능을 보일 것으로 가정합니다.

**H2 — Convergence Speed**

Transfer Learning이 Scratch보다 더 적은 epoch에서 높은 Validation 성능 및 안정적인 수렴 구간에 도달할 것으로 가정합니다.

비교 실험에서는 동일한 dataset split, preprocessing, batch size 및 evaluation metrics를 사용합니다.

---

## 3. Dataset & Split

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

> 데이터셋에 명시적인 patient-level metadata가 제공되지 않으므로 filename-derived group ID를 검증된 patient ID로 해석하지 않습니다.

---

## 4. Data Integrity

초기 Scratch 실험에서 예상보다 높은 Validation 성능이 관찰되어 데이터 누수 가능성을 추가로 검증했습니다.

### Group Leakage Check

Filename-derived group 기준으로 Train / Validation / Test 간 overlap을 검사했습니다.

```text
Train-Val overlap : 0
Train-Test overlap: 0
Val-Test overlap  : 0
```

### Exact Duplicate Check

전체 이미지의 실제 파일 내용을 기반으로 SHA-256 hash를 계산했습니다.

검사 과정에서 서로 다른 split에 위치한 동일 영상 1쌍을 발견했습니다.

중복본 중 1장을 실험 대상에서 제외했으며 원본 데이터 자체는 삭제하거나 수정하지 않았습니다.

### Near-Duplicate Check

pHash를 이용하여 서로 다른 split 사이의 유사 이미지 후보를 탐색했습니다.

```text
Hamming Distance ≤ 4

Total candidates : 71 pairs
Distance 0       : 1 pair
Distance 2       : 4 pairs
Distance 4       : 66 pairs
```

후보 이미지를 직접 확인한 결과 Distance 0의 1쌍만 실제 동일 영상으로 확인했으며, 나머지 후보는 서로 다른 영상이었습니다.

중복 이미지 제거 후 동일한 방법으로 split을 다시 생성하고 SHA-256 및 group overlap 검사를 다시 수행했습니다.

### Current Clean Split

현재 실험에는 중복본 1장을 제외한 **5,855장**을 사용합니다.

| Split | Images | Groups | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,684 | 2,233 | 1,266 | 3,418 |
| Validation | 586 | 279 | 158 | 428 |
| Test | 585 | 278 | 158 | 427 |
| **Total** | **5,855** | **2,790** | **1,582** | **4,273** |

재검증 결과:

```text
Cross-split exact duplicate hash groups: 0
Train-Val group overlap                : 0
Train-Test group overlap               : 0
Val-Test group overlap                 : 0
```

현재 split에서는 정의된 filename-derived group 및 exact duplicate 기준의 cross-split leakage가 확인되지 않았습니다.

---

## 5. Preprocessing

모든 입력 영상은 ResNet-50 입력을 위해 `224 × 224` 크기로 변환합니다.

### Resize & Padding

원본 영상의 aspect ratio를 유지하면서 resize한 뒤 부족한 영역에 padding을 적용합니다.

```text
Input size    : 224 × 224
Aspect ratio  : Preserved
Padding value : 0
```

이를 통해 영상을 단순히 `224 × 224`로 강제 변환할 때 발생할 수 있는 기하학적 왜곡을 줄입니다.

### Channel Conversion

원본 Chest X-ray를 grayscale로 처리한 후 ResNet-50 입력 형식에 맞게 3-channel로 변환합니다.

```text
Grayscale (L)
      ↓
1 Channel
      ↓
Repeat
      ↓
3 Channels
```

### Normalization

Scratch와 Transfer Learning의 입력 조건을 동일하게 유지하기 위해 두 모델 모두 ImageNet normalization을 적용합니다.

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Data Augmentation

Train dataset에만 다음 augmentation을 적용합니다.

```text
Random Rotation : ±10°
Rotation fill   : 0
```

Validation과 Test dataset에는 augmentation을 적용하지 않습니다.

Horizontal Flip은 영상의 좌우 해부학적 위치를 변경할 수 있어 현재 augmentation에서 제외했습니다.

---

## 6. Model

### ResNet-50 Scratch

- ResNet-50 architecture
- `weights=None`
- Random initialization
- 전체 network를 처음부터 학습
- Final classifier를 2-class classification에 맞게 변경

### ResNet-50 Transfer Learning

Transfer Learning에서는 torchvision ResNet-50의 **ImageNet pretrained weights (`IMAGENET1K_V2`)**를 사용합니다.

학습은 두 단계로 구성합니다.

#### Phase 1 — FC-only Training

```text
Epoch 1-10
Backbone       : Frozen
Classifier     : Trainable
Learning Rate  : 0.001
```

Pretrained backbone을 먼저 고정하고 새롭게 초기화된 final classifier를 학습합니다.

#### Phase 2 — Full Fine-tuning

```text
Epoch 11-100
Backbone       : Unfrozen
Entire Network : Trainable
Initial LR     : 0.0001
```

Epoch 11부터 전체 network를 unfreeze하여 Chest X-ray domain에 맞게 fine-tuning합니다.

Freeze 구간과 Fine-tuning 구간의 Learning Rate는 각각 독립적으로 설정할 수 있도록 training pipeline을 구성했습니다.

또한 Fine-tuning 단계에서는 필요에 따라 `ReduceLROnPlateau`를 적용할 수 있도록 구현했으며, 첫 번째 본 Transfer Learning 실험에서는 고정 Learning Rate를 사용합니다.

---

## 7. Scratch Experiments

### Initial Baseline

초기 Scratch baseline은 다음 조건에서 수행했습니다.

```text
Optimizer     : AdamW
Learning Rate : 0.001
Weight Decay  : 0.0001
Batch Size    : 32
Epochs        : 20
```

초기 실험에서 높은 Validation 성능이 관찰되었으며, 이를 계기로 data integrity 및 학습 안정성에 대한 추가 검증을 진행했습니다.

> 초기 실험은 현재 Clean Split 이전의 preliminary experiment이므로 최종 비교 성능으로 사용하지 않습니다.

### Clean Split & Reproducibility

중복 이미지 제거 및 split 재생성 이후에도 높은 Scratch Validation 성능이 유지되었습니다.

이후 실험 간 비교의 재현성을 높이기 위해 다음 random seed를 `42`로 고정했습니다.

- Python
- NumPy
- PyTorch
- CUDA

Seed 고정 이후에도 높은 Validation 성능이 재현되었습니다.

### Validation Instability Analysis

일부 epoch에서 Validation Loss와 실제 prediction이 크게 변하는 현상이 관찰되었습니다.

Validation sample별 prediction과 loss를 기록하여 분석한 결과, Loss spike가 발생한 epoch에서 다수 sample의 class prediction이 동시에 변화하는 것을 확인했습니다.

또한 일부 high-confidence 오분류가 CrossEntropyLoss 증가를 추가로 증폭시키는 현상도 관찰되었습니다.

Learning Rate를 `0.0005 → 0.0001`로 감소시킨 이후에도 이러한 변동성이 지속되어 높은 고정 Learning Rate만이 해당 현상의 주된 원인은 아닌 것으로 판단했습니다.

---

## 8. Scratch Finalization

### exp_007 — Adaptive Learning Rate

고정 Learning Rate 대신 Validation Loss에 따라 Learning Rate를 감소시키기 위해 `ReduceLROnPlateau`를 적용했습니다.

```text
Initial LR       : 0.0001
Weight Decay     : 0.0001
Scheduler        : ReduceLROnPlateau
Monitor          : Validation Loss
Factor           : 0.1
Patience         : 10
Minimum LR       : 0.000001
Epochs           : 100
Seed             : 42
```

Learning Rate 감소 이후 초기 학습 구간에서 나타났던 큰 prediction instability가 상당히 완화되었으며 높은 Validation 성능을 안정적으로 유지하는 구간이 형성되었습니다.

| Metric | Result |
| --- | ---: |
| Best Validation F1 | 0.9882 |
| Best Epoch | 47 |
| Accuracy | 0.9829 |
| Recall | 0.9813 |
| Precision | 0.9953 |
| AUROC | 0.9976 |

다만 Train Loss가 거의 0까지 감소한 반면 Validation Loss는 더 이상 감소하지 않고 후반부 일부 상승하는 경향을 보여 generalization gap이 남아 있었습니다.

### exp_008 — Stronger Weight Decay

exp_007의 설정을 기반으로 Weight Decay를 증가시켜 regularization을 강화했습니다.

```text
Initial LR       : 0.0001
Weight Decay     : 0.001
Scheduler        : ReduceLROnPlateau
Monitor          : Validation Loss
Factor           : 0.1
Patience         : 10
Minimum LR       : 0.000001
Epochs           : 100
Seed             : 42
```

| Metric | Result |
| --- | ---: |
| Best Validation F1 | 0.9884 |
| Best Epoch | 68 |
| Accuracy | 0.9829 |
| Recall | 0.9953 |
| Precision | 0.9816 |
| AUROC | 0.9982 |

exp_007과 exp_008의 최고 Validation F1 차이는 매우 작았습니다.

다만 exp_008에서는 Learning Rate 감소 이후 Validation Loss가 후반부 약 `0.05` 수준에서 비교적 안정적으로 유지되었으며, exp_007에서 관찰되었던 후반 Validation Loss의 상승 경향은 두드러지지 않았습니다.

Train Loss는 계속해서 거의 0까지 감소했기 때문에 Train-Validation generalization gap 자체는 여전히 존재했습니다.

현재 **exp_008을 Scratch vs Transfer Learning 비교를 위한 Scratch 기준선으로 사용합니다.**

---

## 9. Transfer Learning Pilot

본 Transfer Learning 실험의 freeze 기간을 결정하기 위해 pretrained backbone을 고정하고 classifier만 학습하는 FC-only pilot experiment를 수행했습니다.

### Pilot Configuration

```text
Model              : ResNet-50
Pretrained Weights : IMAGENET1K_V2
Backbone            : Frozen
Trainable Layer     : Final FC
Learning Rate       : 0.001
Epochs              : 20
Seed                : 42
```

### Pilot Result

초기에는 Validation F1이 빠르게 상승했습니다.

```text
Epoch 1  F1 : 0.9278
Epoch 7  F1 : 0.9517
Epoch 10 F1 : 0.9512
Epoch 12 F1 : 0.9549
```

Epoch 10 이후 Validation F1은 약 `0.95` 부근에서 변동했으며 Validation Loss와 AUROC의 개선폭도 감소했습니다.

이를 바탕으로 본 Transfer Learning 실험에서는 **FC-only freeze 기간을 10 epochs로 설정**했습니다.

본 실험은 다음과 같이 진행합니다.

```text
Epoch 1-10
FC-only Training
LR = 0.001

        ↓ Unfreeze

Epoch 11-100
Full Fine-tuning
Initial LR = 0.0001
```

Freeze 구간도 task-specific training 과정의 일부이므로 Scratch와 Transfer Learning의 수렴 속도를 비교할 때 **Epoch 1부터 global epoch 기준으로 계산**합니다.

---

## 10. Evaluation

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

모델 및 학습 설정이 결정되기 전까지 Test set은 모델 선택이나 hyperparameter 조정에 사용하지 않습니다.

최종 모델과 학습 설정을 확정한 이후 독립된 Test set을 이용하여 최종 성능을 평가할 예정입니다.

---

## 11. Explainability

최종 모델에는 **Grad-CAM++**을 적용할 예정입니다.

Chest X-ray 위에 Class Activation Map을 overlay하여 모델이 NORMAL / PNEUMONIA를 판단할 때 어떤 영역에 주목했는지 확인합니다.

이를 통해 모델이 실제 폐 영역과 관련된 특징을 활용하는지, 또는 배경이나 영상 marker와 같은 비의도적 특징에 의존하는지 분석할 예정입니다.

---

## 12. User Interface

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

## 13. Experiment Logging

각 실험은 별도의 experiment directory에 저장합니다.

```text
experiments/
│
├── scratch/
│   └── exp_XXX/
│       ├── config.json
│       ├── history.csv
│       ├── val_predictions.csv
│       ├── loss_curve.png
│       └── validation_metrics.png
│
└── transfer/
    │
    ├── pilot_fc_freeze/
    │   └── exp_XXX/
    │
    └── exp_XXX/
        ├── config.json
        ├── history.csv
        ├── val_predictions.csv
        ├── loss_curve.png
        └── validation_metrics.png
```

각 experiment에는 다음 정보를 기록합니다.

- `config.json` — 실험 설정
- `history.csv` — epoch별 Loss, Validation metrics, Learning Rate
- `val_predictions.csv` — Validation sample별 prediction, probability 및 loss
- `loss_curve.png` — Train / Validation Loss
- `validation_metrics.png` — Validation metric 변화

Transfer Learning의 `history.csv`에는 학습 phase를 함께 기록하여 FC-only와 Fine-tuning 구간을 구분합니다.

```text
epoch,phase,train_loss,val_loss,accuracy,recall,precision,f1,auroc,lr
```

Model checkpoint와 Dataset 원본 등 대용량 파일은 GitHub repository에 업로드하지 않습니다.

---

## 14. Project Structure

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
│   ├── scratch/
│   │   └── exp_XXX/
│   │
│   └── transfer/
│       ├── pilot_fc_freeze/
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

## 15. Progress

- [x] Dataset 구조 및 클래스 분포 분석
- [x] Filename-derived group ID 정의
- [x] Group-aware Train / Validation / Test split
- [x] Group overlap 검사
- [x] SHA-256 / pHash 기반 duplicate 검사
- [x] Cross-split duplicate 처리 및 split 재생성
- [x] Preprocessing pipeline
- [x] ResNet-50 Scratch 구현
- [x] Training / Validation pipeline
- [x] Experiment logging
- [x] Clean Split 기반 Scratch 재학습
- [x] Random seed 고정 및 재현성 확인
- [x] Validation prediction instability 분석
- [x] Adaptive Learning Rate 실험
- [x] Weight Decay regularization 실험
- [x] Scratch baseline 확정
- [x] ImageNet pretrained ResNet-50 구현
- [x] Transfer Learning FC-only pilot
- [x] Freeze → Full Fine-tuning pipeline 구현
- [ ] Scratch vs Transfer Learning 비교
- [ ] Final Test evaluation
- [ ] Grad-CAM++
- [ ] UI implementation

---

## 16. Next Steps

1. Transfer Learning 100-epoch experiment
2. Scratch vs Transfer Learning 분류 성능 및 수렴 특성 비교
3. 필요 시 Transfer Learning optimization 및 adaptive learning rate 실험
4. 최종 학습 설정 확정
5. 독립 Test set을 이용한 최종 평가
6. Grad-CAM++ 분석
7. UI 연동

---

## 17. Limitations

현재 데이터셋에는 명시적인 patient-level metadata가 제공되지 않기 때문에 파일명에서 추출한 식별자를 group ID로 사용합니다.

따라서 filename-derived group을 검증된 patient ID로 해석하지 않습니다.

pHash는 이미지의 의미적 동일성을 보장하는 방법이 아니므로 near-duplicate 후보 탐색 용도로만 사용했으며 후보 이미지는 직접 시각적으로 검수했습니다.

현재 공개된 실험 결과는 Validation 기반 결과이며 최종 Test 성능이 아닙니다.

또한 현재 주요 실험은 고정된 random seed를 기반으로 수행되므로, 향후 보다 강한 재현성 검증을 위해 multiple random seeds를 이용한 반복 실험 및 평균·표준편차 분석을 고려할 수 있습니다.

향후 연구로 확장할 경우 명확한 patient-level metadata를 제공하는 데이터셋과 외부 Chest X-ray 데이터셋을 이용한 **external validation 및 domain shift 평가**를 고려합니다.