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
- **Fair Comparison:** Scratch / Transfer Learning에 동일한 규모의 2×2 hyperparameter search space 적용
- **Convergence Analysis:** Best Validation F1의 99% 도달 및 10-epoch stability window 기반 수렴 정의
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

비교 실험에서는 동일한 dataset split, preprocessing, batch size, optimizer, training epochs, random seed 및 evaluation metrics를 사용합니다.

Scratch와 Transfer Learning은 학습 방식의 특성이 서로 다르므로 모든 hyperparameter를 동일한 값으로 강제하지 않습니다. 대신 두 방식에 동일한 규모의 hyperparameter 탐색 기회를 제공하고, 사전에 정의한 동일한 기준을 이용하여 각 방식의 대표 설정을 선택합니다.

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

- `NORMAL2-IM-0096-0001.jpeg`
- `NORMAL2-IM-0095-0001.jpeg`

후보를 직접 시각적으로 비교하여 동일한 이미지임을 확인했습니다.

중복본 중 `NORMAL2-IM-0095-0001.jpeg` 한 장을 실험 대상에서 제외했으며 원본 데이터 자체는 삭제하거나 수정하지 않았습니다.

### Near-Duplicate Check

pHash를 이용하여 서로 다른 split 사이의 유사 이미지 후보를 탐색했습니다.

초기 split에서는 다음 후보가 탐지되었습니다.

```text
Hamming Distance ≤ 4

Total candidates : 71 pairs
Distance 0       : 1 pair
Distance 2       : 4 pairs
Distance 4       : 66 pairs
```

후보 이미지를 직접 확인한 결과 Distance 0의 1쌍만 실제 동일 영상으로 확인했으며, 나머지 후보는 서로 다른 영상이었습니다.

### Current Clean Split

중복 이미지 한 장을 제외한 뒤 동일한 방법으로 전체 split을 다시 생성했습니다.

현재 실험에는 **5,855장**을 사용합니다.

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

Clean Split을 대상으로 pHash audit도 다시 수행했습니다.

```text
Total cross-split candidates : 76 pairs
Distance 0                   : 0
Distance 2                   : 2
Distance 4                   : 74
```

Distance 2 및 Distance 4 후보를 직접 시각적으로 확인했으며 동일 이미지로 판단되는 추가 사례는 발견되지 않았습니다.

현재 split에서는 정의된 filename-derived group 및 확인 가능한 exact duplicate 기준의 cross-split leakage가 추가로 확인되지 않았습니다.

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
Epoch 1-7

Backbone      : Frozen
Classifier    : Trainable
Learning Rate : 0.001
```

Pretrained backbone을 먼저 고정하고 새롭게 초기화된 final classifier를 학습합니다.

#### Phase 2 — Full Fine-tuning

```text
Epoch 8-100

Backbone       : Unfrozen
Entire Network : Trainable
```

Epoch 8부터 전체 network를 unfreeze하여 Chest X-ray domain에 맞게 fine-tuning합니다.

Fine-tuning 단계의 initial Learning Rate는 최종 hyperparameter comparison protocol에 따라 `0.0001`과 `0.00005`를 비교합니다.

Freeze 구간과 Fine-tuning 구간의 Learning Rate는 각각 독립적으로 설정합니다.

Fine-tuning 단계에서는 Validation Loss를 기준으로 `ReduceLROnPlateau`를 적용합니다.

---

## 7. Scratch Exploratory Experiments

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

### Adaptive Learning Rate

Validation Loss에 따라 Learning Rate를 감소시키기 위해 `ReduceLROnPlateau`를 적용했습니다.

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

Learning Rate 감소 이후 초기 학습 구간에서 나타났던 큰 prediction instability가 상당히 완화되었으며 높은 Validation 성능을 유지하는 구간이 형성되었습니다.

### Stronger Weight Decay

Adaptive Learning Rate 설정을 유지하면서 Weight Decay를 `0.0001 → 0.001`로 증가시켜 regularization의 영향을 확인했습니다.

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

해당 실험에서 Best Validation F1은 약 `0.9884`였으며, Learning Rate 감소 이후 Validation Loss가 후반부 약 `0.05` 수준에서 비교적 안정적으로 유지되었습니다.

Train Loss는 계속해서 거의 0까지 감소했기 때문에 Train-Validation generalization gap 자체는 여전히 존재했습니다.

이러한 Scratch 실험들은 최종 비교 protocol을 설계하기 위한 exploratory experiments로 사용합니다.

---

## 8. Transfer Learning Pilot

본 Transfer Learning 실험의 FC-only freeze 기간을 결정하기 위해 pretrained backbone을 고정하고 classifier만 학습하는 pilot experiment를 수행했습니다.

### Pilot Configuration

```text
Model              : ResNet-50
Pretrained Weights : IMAGENET1K_V2
Backbone           : Frozen
Trainable Layer    : Final FC
Learning Rate      : 0.001
Epochs             : 20
Seed               : 42
```

### Pilot Result

초기에는 Validation F1이 빠르게 상승했습니다.

```text
Epoch 1  F1 : 0.9278
Epoch 7  F1 : 0.9517
Epoch 10 F1 : 0.9512
Epoch 12 F1 : 0.9549
```

Epoch 7까지 Validation F1이 빠르게 증가한 이후 Epoch 8-10에서는 약 `0.94-0.95` 수준에서 변동했습니다.

Pilot 결과를 바탕으로 최종 비교 protocol에서 사용할 **FC-only freeze 기간을 7 epochs로 사전에 고정**합니다. Freeze duration 자체는 최종 2×2 hyperparameter search의 탐색 변수에 포함하지 않습니다.

```text
Epoch 1-7
FC-only Training
LR = 0.001
        ↓ Unfreeze
Epoch 8-100
Full Fine-tuning
```

Freeze 구간도 task-specific training 과정의 일부이므로 Scratch와 Transfer Learning의 수렴 속도를 비교할 때 **Epoch 1부터 global epoch 기준으로 계산**합니다.

---

## 9. Transfer Learning Exploratory Experiments

### Fixed Learning Rate

초기 Transfer Learning 실험에서는 FC-only training 이후 전체 network를 unfreeze하고 고정된 Fine-tuning Learning Rate를 사용했습니다.

Fine-tuning 직후 Validation F1이 빠르게 상승하여 pretrained representation을 이용한 빠른 task adaptation이 관찰되었습니다.

반면 전체 backbone을 고정 Learning Rate로 지속적으로 업데이트하는 과정에서 Validation Loss와 F1의 큰 변동성이 나타났습니다.

후반부에는 다시 높은 Validation 성능 영역에 도달했지만 Train Loss와 Validation Loss 사이의 generalization gap은 지속되었습니다.

### Adaptive Learning Rate

이후 `ReduceLROnPlateau`를 적용하여 Validation Loss가 개선되지 않는 경우 Fine-tuning Learning Rate를 단계적으로 감소시켰습니다.

```text
Scheduler  : ReduceLROnPlateau
Monitor    : Validation Loss
Factor     : 0.1
Patience   : 10
Minimum LR : 0.000001
```

Learning Rate 감소 이후 Fine-tuning 과정의 Validation fluctuation이 완화되는 경향을 확인했습니다.

다만 Train Loss는 계속해서 매우 낮은 값으로 감소하여 generalization gap 자체가 제거되지는 않았습니다.

### Stronger Weight Decay

Adaptive Learning Rate를 유지한 상태에서 Weight Decay를 `0.001 → 0.01`로 증가시켜 regularization의 영향을 확인했습니다.

강한 Weight Decay에서도 Fine-tuning 초기의 Validation instability와 Train-Validation generalization gap은 완전히 제거되지 않았습니다.

이러한 exploratory experiments를 바탕으로 최종 Scratch vs Transfer Learning 비교를 위한 hyperparameter search space를 고정했습니다.

---

## 10. Final Comparison Protocol

Scratch와 Transfer Learning의 비교에서 한쪽 학습 방식에 더 많은 hyperparameter tuning 기회를 제공하여 비교 결과가 편향되는 것을 줄이기 위해, 최종 비교에서는 두 방식에 **동일한 규모의 2×2 hyperparameter search space**를 적용합니다.

본 비교는 각 방식의 이론적인 최대 성능을 찾기 위한 exhaustive hyperparameter optimization이 아니라, 사전에 정의한 제한된 search space 내에서 두 학습 전략의 성능과 수렴 특성을 비교하는 것을 목적으로 합니다.

### Common Training Configuration

```text
Architecture      : ResNet-50
Optimizer         : AdamW
Batch Size        : 32
Epochs            : 100
Seed              : 42
Scheduler         : ReduceLROnPlateau
Scheduler Monitor : Validation Loss
Factor            : 0.1
Patience          : 10
Minimum LR        : 0.000001
```

Dataset split, preprocessing, augmentation, evaluation metrics 및 총 training epoch도 동일하게 유지합니다.

### Scratch Search Space

| | Weight Decay = 0.001 | Weight Decay = 0.01 |
| --- | ---: | ---: |
| **LR = 0.0001** | exp_008 | Experiment |
| **LR = 0.00005** | Experiment | Experiment |

### Transfer Learning Search Space

Transfer Learning에서는 FC-only phase를 다음과 같이 고정합니다.

```text
Freeze Epochs : 7
Freeze LR     : 0.001
```

Fine-tuning 단계에 대해서만 Scratch와 동일한 규모의 Learning Rate × Weight Decay search space를 적용합니다.

| | Weight Decay = 0.001 | Weight Decay = 0.01 |
| --- | ---: | ---: |
| **Fine-tune LR = 0.0001** | exp_002 | exp_003 |
| **Fine-tune LR = 0.00005** | Experiment | exp_004 |

현재 표의 `Experiment` 항목은 아직 수행되지 않은 실험을 의미합니다.

최종 비교용 Grid의 experiment ID와 진행 상태는 `experiments/final_2x2_experiment_grid.csv`에서도 별도로 관리합니다.

현재까지 최종 Grid에 포함되는 완료 실험은 다음과 같습니다.

```text
Scratch
0.0001 / WD 0.001  → exp_008

Transfer
0.0001  / WD 0.001 → exp_002
0.0001  / WD 0.01  → exp_003
0.00005 / WD 0.01  → exp_004
```

본 README는 프로젝트 진행 상황에 따라 업데이트되는 문서이며, 최종 비교 시점에는 모든 2×2 Grid 항목을 실제 experiment ID로 기록합니다.

### Model Selection

각 학습 방식의 4개 설정 중 **Best Validation F1이 가장 높은 설정**을 해당 방식의 대표 configuration으로 선택합니다.

대표 configuration이 결정된 이후 Scratch와 Transfer Learning의 다음 항목을 비교합니다.

- Best Validation F1
- Accuracy
- Recall
- Precision
- AUROC
- Confusion Matrix
- Convergence Epoch
- Validation stability
- Train-Validation generalization gap

최종 search space는 비교 결과를 확인한 뒤 추가 후보를 도입하지 않고 현재 정의된 2×2 Grid로 고정합니다.

---

## 11. Convergence Definition

단순히 Best Validation F1이 발생한 epoch를 수렴 시점으로 해석하지 않습니다.

각 모델의 학습 과정에서 도달한 Best Validation F1을 기준으로 상대적인 performance threshold를 정의합니다.

### Primary Convergence Threshold

각 모델에 대해 다음 threshold를 계산합니다.

```text
T99 = 0.99 × Best Validation F1
```

Epoch `t`가 다음 조건을 만족하는 가장 이른 epoch인 경우 해당 epoch를 **Convergence Epoch**으로 정의합니다.

1. Epoch `t`에서 Validation F1이 `T99` 이상에 도달
2. Epoch `t`부터 시작하는 연속 10 epochs 중 최소 8 epochs에서 Validation F1 ≥ `T99`

즉,

```text
F1(t) ≥ T99

and

Count(F1 ≥ T99 in epochs t ... t+9) ≥ 8
```

를 만족하는 가장 이른 `t`를 찾습니다.

이 정의는 일시적인 Validation F1 spike를 수렴으로 판단하는 것을 줄이고, 일정 기간 높은 성능이 유지되는지를 함께 평가하기 위한 것입니다.

### Sensitivity Analysis

99% threshold 선택에 따른 결과 의존성을 확인하기 위해 동일한 방법으로 다음 threshold도 계산합니다.

```text
T95 = 0.95 × Best Validation F1
T97 = 0.97 × Best Validation F1
T99 = 0.99 × Best Validation F1
```

각 threshold에 대해 동일한 **8-of-10 stability criterion**을 적용합니다.

`T99`를 primary convergence metric으로 사용하며 `T95`와 `T97`은 sensitivity analysis로 사용합니다.

현재 2×2 hyperparameter Grid가 진행 중이므로 실제 T95 / T97 / T99 convergence 결과는 아직 보고하지 않습니다. 대표 configuration이 결정된 이후 해당 결과를 계산하여 최종 비교에 추가합니다.

### Transfer Learning Global Epoch

Transfer Learning의 FC-only freeze phase도 실제 task-specific training 과정에 포함되므로 수렴 속도 비교에서는 전체 학습 과정을 포함한 **global epoch**을 사용합니다.

예를 들어 Freeze Epochs가 7인 경우:

```text
Global Epoch 1-7 : FC-only
Global Epoch 8   : Fine-tuning Epoch 1
Global Epoch 9   : Fine-tuning Epoch 2
...
```

따라서 Scratch와 Transfer Learning 모두 Epoch 1부터 동일하게 100 epochs의 training trajectory를 기준으로 수렴 시점을 계산합니다.

### Interpretation

본 수렴 기준은 **post-hoc analysis metric**이며 training을 중단하기 위한 early stopping rule이 아닙니다.

또한 본 수렴 정의는 Scratch와 Transfer Learning 모델 모두 동일하게 100 epochs 동안 학습하는 본 실험 설정을 전제로 합니다.

따라서 총 학습 epoch 수가 변경되는 경우 stability window의 크기와 유지 조건은 재검토할 필요가 있습니다.

---

## 12. Evaluation

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

### Validation

Validation set은 다음 목적으로 사용합니다.

- Hyperparameter configuration 비교
- 대표 configuration 선택
- 학습 안정성 분석
- Convergence analysis

### Test

Test set은 hyperparameter 선택이나 모델 설정 변경에 사용하지 않습니다.

Scratch와 Transfer Learning 각각의 대표 configuration이 Validation 결과를 기반으로 완전히 결정된 이후 독립된 Test set을 이용하여 최종 성능을 평가합니다.

---

## 13. Explainability

최종 모델에는 **Grad-CAM++**을 적용할 예정입니다.

Chest X-ray 위에 Class Activation Map을 overlay하여 모델이 NORMAL / PNEUMONIA를 판단할 때 어떤 영역에 주목했는지 확인합니다.

이를 통해 모델이 실제 폐 영역과 관련된 특징을 활용하는지, 또는 배경이나 영상 marker와 같은 비의도적 특징에 의존하는지 분석할 예정입니다.

Grad-CAM++ 결과는 모델의 attention pattern을 정성적으로 분석하기 위한 도구로 사용하며 임상적 타당성을 입증하는 근거로 해석하지 않습니다.

---

## 14. User Interface

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

## 15. Experiment Logging

각 실험은 별도의 experiment directory에 저장합니다.

```text
experiments/
│
├── final_2x2_experiment_grid.csv
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

`final_2x2_experiment_grid.csv`에는 최종 Scratch vs Transfer Learning 비교에 사용되는 2×2 hyperparameter Grid의 experiment ID와 진행 상태를 기록합니다.

Transfer Learning의 `history.csv`에는 학습 phase를 함께 기록하여 FC-only와 Fine-tuning 구간을 구분합니다.

```text
epoch,phase,train_loss,val_loss,accuracy,recall,precision,f1,auroc,lr
```

Model checkpoint와 Dataset 원본 등 대용량 파일은 GitHub repository에 업로드하지 않습니다.

---

## 16. Project Structure

```text
pneumonia-diagnosis/
│
├── data/
│   ├── raw/
│   └── splits/
│       ├── group_split.csv
│       └── near_duplicates.csv
│
├── docs/
│   └── data_integrity_audit.md
│
├── experiments/
│   ├── final_2x2_experiment_grid.csv
│   ├── scratch/
│   │   └── exp_XXX/
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

## 17. Progress

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
- [x] ImageNet pretrained ResNet-50 구현
- [x] Transfer Learning FC-only pilot
- [x] Freeze → Full Fine-tuning pipeline 구현
- [x] Transfer Learning exploratory experiments
- [x] Scratch vs Transfer Learning 최종 비교 protocol 정의
- [x] Convergence criterion 정의
- [ ] Scratch 2×2 hyperparameter Grid 완료
- [ ] Transfer Learning 2×2 hyperparameter Grid 완료
- [ ] 대표 Scratch / Transfer configuration 선택
- [ ] Scratch vs Transfer Learning 성능 및 수렴 비교
- [ ] Final Test evaluation
- [ ] Grad-CAM++
- [ ] UI implementation

---

## 18. Next Steps

1. Transfer Learning의 남은 `Fine-tune LR = 0.00005 / Weight Decay = 0.001` 실험 수행
2. Scratch의 남은 3개 2×2 Grid 실험 수행
3. Best Validation F1을 기준으로 각 학습 방식의 대표 configuration 선택
4. Scratch vs Transfer Learning Validation 성능 비교
5. T95 / T97 / T99 기반 convergence analysis
6. 독립 Test set을 이용한 최종 평가
7. Grad-CAM++ 분석
8. UI 연동

---

## 19. Limitations

현재 데이터셋에는 명시적인 patient-level metadata가 제공되지 않기 때문에 파일명에서 추출한 식별자를 group ID로 사용합니다.

따라서 filename-derived group을 검증된 patient ID로 해석하지 않습니다.

pHash는 이미지의 의미적 동일성을 보장하는 방법이 아니므로 near-duplicate 후보 탐색 용도로만 사용했으며 후보 이미지는 직접 시각적으로 검수했습니다.

현재 공개된 실험 결과는 Validation 기반 결과이며 최종 Test 성능이 아닙니다.

본 실험의 hyperparameter comparison은 사전에 정의한 제한된 2×2 search space를 사용하며 exhaustive hyperparameter optimization을 수행하지 않습니다. 따라서 실험 결과를 Scratch 또는 Transfer Learning이 달성할 수 있는 절대적인 최대 성능으로 해석하지 않습니다.

Transfer Learning의 FC-only freeze 기간은 별도의 pilot experiment를 통해 7 epochs로 결정했으며, 최종 2×2 hyperparameter search space의 각 Learning Rate × Weight Decay 조합에 대해 freeze epoch 수를 별도로 재탐색하지 않았습니다. 따라서 7 epochs를 모든 Transfer Learning configuration에서의 최적 freeze duration으로 해석하지 않습니다.

또한 현재 주요 실험은 고정된 random seed `42`를 기반으로 수행하므로, 향후 보다 강한 재현성 검증을 위해 multiple random seeds를 이용한 반복 실험 및 평균·표준편차 분석을 고려할 수 있습니다.

Validation set은 hyperparameter configuration 선택 및 반복적인 exploratory analysis에 사용되므로 Validation 결과 자체에도 selection bias가 발생할 가능성이 있습니다. 최종 Test set은 모든 모델 및 학습 설정이 확정된 이후에만 사용하여 이러한 영향을 최종 평가에서 분리합니다.

향후 연구로 확장할 경우 명확한 patient-level metadata를 제공하는 데이터셋과 외부 Chest X-ray 데이터셋을 이용한 **external validation 및 domain shift 평가**를 고려합니다.