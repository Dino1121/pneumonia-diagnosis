# Pediatric Chest X-ray Pneumonia Classification

소아 Chest X-ray 영상을 이용하여 **NORMAL / PNEUMONIA를 분류**하는 딥러닝 프로젝트입니다.

동일한 ResNet-50 architecture에서 **Scratch 학습과 ImageNet pretrained Transfer Learning**을 비교하고, 분류 성능뿐만 아니라 **학습 수렴 속도**의 차이도 함께 분석합니다.

추후 **Grad-CAM++**을 적용하여 모델의 판단 영역을 시각화하고 간단한 사용자 UI와 연동할 예정입니다.

> 본 프로젝트는 학습 및 연구 목적으로 수행되며 실제 의료 진단을 목적으로 하지 않습니다.

---

## 1. Project Overview

### Research Question

> **Scratch 학습과 Transfer Learning은 분류 성능과 학습 수렴 속도에서 어떤 차이를 보이는가?**

### Hypotheses

**H1 — Classification Performance**

ImageNet pretrained Transfer Learning이 Scratch보다 높은 분류 성능을 보일 것으로 가정했습니다.

**H2 — Convergence Speed**

Transfer Learning이 Scratch보다 더 적은 epoch에서 높은 Validation 성능의 안정적인 구간에 도달할 것으로 가정했습니다.

### Key Features

- ResNet-50 Scratch vs ImageNet Transfer Learning
- Filename-derived group 기반 Train / Validation / Test split
- SHA-256 / pHash 기반 cross-split duplicate audit
- Aspect-ratio preserving resize & padding
- 동일 규모의 2×2 hyperparameter search
- Validation F1 기반 representative model selection
- T95 / T97 / T99 기반 convergence analysis
- Independent Test set evaluation
- Grad-CAM++ explainability 예정
- PyQt 기반 UI 연동 예정

---

## 2. Dataset

**Chest X-Ray Images (Pneumonia)** 데이터셋을 사용합니다.

원본 데이터셋은 총 **5,856장**으로 구성되어 있습니다.

| Class | Images | Ratio |
| --- | ---: | ---: |
| NORMAL | 1,583 | 27.0% |
| PNEUMONIA | 4,273 | 73.0% |
| **Total** | **5,856** | **100%** |

원본 데이터셋의 Validation set은 16장으로 매우 작기 때문에 기존 split을 그대로 사용하지 않고 전체 데이터를 기반으로 새로운 Train / Validation / Test split을 구성했습니다.

### Group-aware Split

동일한 대상에서 생성된 것으로 추정되는 영상이 서로 다른 split에 포함되는 것을 줄이기 위해 파일명에서 반복되는 식별자 패턴을 추출하여 **filename-derived group ID**를 정의했습니다.

```text
PNEUMONIA → personXX
NORMAL    → IM-XXXX
```

이후 `StratifiedGroupKFold`를 사용했습니다.

```text
n_splits     = 10
shuffle      = True
random_state = 42

Fold 0-7 → Train
Fold 8   → Validation
Fold 9   → Test
```

> 데이터셋에 명시적인 patient-level metadata가 제공되지 않으므로 filename-derived group ID를 검증된 patient ID로 해석하지 않습니다.

---

## 3. Data Integrity

초기 실험에서 높은 Validation 성능이 관찰되어 split leakage 및 duplicate 가능성을 추가로 점검했습니다.

### Exact Duplicate Audit

전체 이미지에 대해 SHA-256 hash를 계산한 결과 서로 다른 split에 존재하는 동일 이미지 1쌍을 확인했습니다.

중복 이미지 한 장을 실험 대상에서 제외한 뒤 전체 split을 다시 생성했습니다.

### Near-Duplicate Audit

pHash를 이용하여 Hamming Distance ≤ 4인 cross-split 후보를 탐색하고 후보 이미지를 직접 확인했습니다.

Clean split에서의 결과는 다음과 같습니다.

```text
Cross-split pHash candidates : 76 pairs

Distance 0 : 0
Distance 2 : 2
Distance 4 : 74
```

Distance 2 및 4 후보를 직접 확인했으며 추가적인 동일 이미지는 발견되지 않았습니다.

### Final Clean Split

최종 실험에는 **5,855장**의 이미지를 사용합니다.

| Split | Images | Groups | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,684 | 2,233 | 1,266 | 3,418 |
| Validation | 586 | 279 | 158 | 428 |
| Test | 585 | 278 | 158 | 427 |
| **Total** | **5,855** | **2,790** | **1,582** | **4,273** |

최종 split에서 확인한 결과:

```text
Cross-split exact duplicate hash groups : 0

Train-Val group overlap  : 0
Train-Test group overlap : 0
Val-Test group overlap   : 0
```

보다 자세한 data integrity 과정은 `docs/data_integrity_audit.md`에 기록합니다.

---

## 4. Preprocessing

모든 입력 영상은 ResNet-50 입력을 위해 `224 × 224` 크기로 변환합니다.

### Resize & Padding

원본 영상의 aspect ratio를 유지하면서 resize한 뒤 부족한 영역에 padding을 적용합니다.

```text
Input size    : 224 × 224
Aspect ratio  : Preserved
Padding value : 0
```

단순 강제 resize로 인해 발생할 수 있는 해부학적 구조의 기하학적 왜곡을 줄이는 것을 목적으로 합니다.

### Channel Conversion

Chest X-ray를 grayscale로 처리한 뒤 ResNet-50 입력 형식에 맞게 3-channel로 변환합니다.

```text
Grayscale
    ↓
1 Channel
    ↓
Repeat
    ↓
3 Channels
```

### Normalization

Scratch와 Transfer Learning 모두 동일한 ImageNet normalization을 사용합니다.

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Data Augmentation

Train set에만 다음 augmentation을 적용합니다.

```text
Random Rotation : ±10°
Rotation Fill   : 0
```

Validation과 Test에는 augmentation을 적용하지 않습니다.

Horizontal Flip은 영상의 좌우 해부학적 위치를 변경할 수 있어 제외했습니다.

---

## 5. Models

### ResNet-50 Scratch

```text
Architecture : ResNet-50
Weights      : None
Output       : 2 classes
```

모든 parameter를 random initialization 상태에서 처음부터 학습합니다.

### ResNet-50 Transfer Learning

ImageNet pretrained ResNet-50의 `IMAGENET1K_V2` weights를 사용합니다.

학습은 두 단계로 진행합니다.

#### Phase 1 — FC-only

```text
Epoch 1-7

Backbone      : Frozen
Classifier    : Trainable
Learning Rate : 0.001
```

#### Phase 2 — Full Fine-tuning

```text
Epoch 8-100

Backbone       : Unfrozen
Entire Network : Trainable
```

FC-only 기간은 별도의 pilot experiment를 통해 **7 epochs**로 고정했습니다.

---

## 6. Experimental Protocol

Scratch와 Transfer Learning에 한쪽 방식만 더 많은 tuning 기회를 제공하는 것을 줄이기 위해 두 방식 모두 동일한 규모의 **2×2 hyperparameter search**를 수행했습니다.

### Common Configuration

```text
Architecture      : ResNet-50
Optimizer         : AdamW
Batch Size        : 32
Epochs            : 100
Seed              : 42
Scheduler         : ReduceLROnPlateau
Monitor           : Validation Loss
Factor            : 0.1
Patience          : 10
Minimum LR        : 1e-6
Loss              : CrossEntropyLoss
Positive Class    : PNEUMONIA
```

Dataset split, preprocessing, augmentation 및 evaluation metrics도 동일하게 유지했습니다.

### Scratch Grid

| Learning Rate | Weight Decay | Experiment | Best Val F1 | Best Epoch |
| ---: | ---: | --- | ---: | ---: |
| 0.0001 | 0.001 | **exp_008** | **0.9884** | 68 |
| 0.0001 | 0.01 | exp_009 | 0.9837 | 92 |
| 0.00005 | 0.001 | exp_010 | 0.9789 | 83 |
| 0.00005 | 0.01 | exp_011 | 0.9813 | 64 |

Scratch representative:

```text
Experiment   : exp_008
Learning Rate: 0.0001
Weight Decay : 0.001
Best Val F1  : 0.9884
Best Epoch   : 68
```

### Transfer Learning Grid

FC-only phase는 모든 configuration에서 다음과 같이 고정합니다.

```text
Freeze Epochs : 7
Freeze LR     : 0.001
```

| Fine-tune LR | Weight Decay | Experiment | Best Val F1 | Best Epoch |
| ---: | ---: | --- | ---: | ---: |
| 0.0001 | 0.001 | exp_002 | 0.9907 | 14 |
| 0.0001 | 0.01 | **exp_003** | **0.9919** | 18 |
| 0.00005 | 0.001 | exp_005 | 0.9907 | 30 |
| 0.00005 | 0.01 | exp_004 | 0.9860 | 18 |

Transfer Learning representative:

```text
Experiment       : exp_003
Fine-tune LR     : 0.0001
Weight Decay     : 0.01
Freeze Epochs    : 7
Best Val F1      : 0.9919
Best Global Epoch: 18
```

각 학습 방식에서 **Best Validation F1이 가장 높은 configuration**을 representative로 선택했습니다.

---

## 7. Validation Results

최종 representative configuration의 Validation 결과는 다음과 같습니다.

| Model | Experiment | Best Validation F1 | Best F1 Epoch |
| --- | --- | ---: | ---: |
| Scratch | exp_008 | 0.9884 | 68 |
| Transfer Learning | exp_003 | **0.9919** | **18** |

Transfer Learning이 Scratch보다 약 **0.35 percentage points 높은 Best Validation F1**을 기록했습니다.

Best F1이 발생한 epoch 자체는 공식적인 convergence metric으로 사용하지 않고 별도의 stability criterion을 적용했습니다.

---

## 8. Convergence Analysis

단일 epoch에서 발생한 높은 Validation F1을 수렴으로 판단하지 않기 위해 각 모델의 Best Validation F1을 기준으로 상대적인 threshold를 정의했습니다.

```text
T95 = 0.95 × Best Validation F1
T97 = 0.97 × Best Validation F1
T99 = 0.99 × Best Validation F1
```

Epoch `t`가 다음 조건을 만족하는 가장 이른 시점을 convergence epoch으로 정의합니다.

```text
F1(t) ≥ Threshold

and

epochs t ... t+9 중
F1 ≥ Threshold인 epoch가 최소 8개
```

`T99`를 primary convergence metric으로 사용하며 `T95`, `T97`은 sensitivity analysis로 사용합니다.

### Results

| Model | T95 | T97 | T99 |
| --- | ---: | ---: | ---: |
| Scratch exp_008 | **4** | 12 | 51 |
| Transfer exp_003 | 5 | **9** | **12** |

Primary metric인 **T99에서 Transfer Learning은 global epoch 12**, Scratch는 **epoch 51**에 수렴했습니다.

Transfer Learning의 global epoch에는 FC-only phase가 포함됩니다.

```text
Global Epoch 1-7 : FC-only
Global Epoch 8   : Fine-tuning Epoch 1
...
Global Epoch 12  : Fine-tuning Epoch 5
```

따라서 Transfer Learning은 본 실험에서 Scratch보다 훨씬 이른 시점에 자신의 최고 Validation F1에 근접한 안정적인 성능 영역에 도달했습니다.

> 본 convergence definition은 post-hoc analysis metric이며 early stopping criterion이 아닙니다.

---

## 9. Independent Test Evaluation

각 방식의 representative configuration을 **Validation 결과만으로 먼저 확정한 뒤**, 독립 Test set 585장을 이용하여 최종 평가했습니다.

Test 결과는 model 또는 hyperparameter를 다시 선택하는 데 사용하지 않습니다.

### Test Results

| Metric | Scratch exp_008 | Transfer exp_003 |
| --- | ---: | ---: |
| Accuracy | **0.9761** | 0.9607 |
| Recall | **0.9836** | 0.9719 |
| Precision | **0.9836** | 0.9742 |
| F1-score | **0.9836** | 0.9730 |
| AUROC | 0.9951 | **0.9956** |

### Confusion Matrix

**Scratch exp_008**

```text
             Predicted
             NORMAL  PNEUMONIA

Actual NORMAL   151       7
Actual PNEUMONIA  7     420

TN = 151
FP = 7
FN = 7
TP = 420
```

**Transfer exp_003**

```text
             Predicted
             NORMAL  PNEUMONIA

Actual NORMAL   147      11
Actual PNEUMONIA 12     415

TN = 147
FP = 11
FN = 12
TP = 415
```

Test F1은 Scratch가 Transfer Learning보다 약 **1.06 percentage points 높게** 나타났습니다.

반면 AUROC는 두 모델 모두 약 `0.995`로 매우 유사했으며 Transfer Learning이 소폭 높은 값을 기록했습니다.

---

## 10. Result Summary

본 실험에서 두 가설은 서로 다른 결과를 보였습니다.

### H1 — Classification Performance

> Transfer Learning이 Scratch보다 높은 분류 성능을 보일 것이다.

Validation에서는 Transfer Learning이 더 높은 Best F1을 기록했지만, 독립 Test set에서는 Scratch가 더 높은 Accuracy, Recall, Precision 및 F1을 기록했습니다.

따라서 **Test F1을 기준으로 H1은 지지되지 않았습니다.**

### H2 — Convergence Speed

> Transfer Learning이 Scratch보다 더 빠르게 높은 성능 영역에 수렴할 것이다.

Primary convergence metric인 T99 결과:

```text
Scratch  : Epoch 51
Transfer : Epoch 12
```

따라서 본 실험에서는 **Transfer Learning이 Scratch보다 훨씬 빠르게 높은 Validation 성능 영역에 도달했습니다.**

### Overall Observation

```text
                 Scratch      Transfer Learning
------------------------------------------------
Best Val F1       0.9884          0.9919
T99 Epoch             51              12
Test F1           0.9836          0.9730
Test AUROC        0.9951          0.9956
```

Transfer Learning은 **수렴 속도에서 뚜렷한 이점**을 보였지만, 최종 Test F1에서는 Scratch가 더 높은 값을 기록했습니다.

따라서 본 실험에서는 ImageNet pretraining의 효과가 최종 분류 성능 향상보다는 **빠른 task adaptation 및 convergence**에서 더 명확하게 관찰되었습니다.

Validation F1 기준으로 사전에 선택된 configuration은 **Transfer Learning exp_003**이며, Test 결과를 확인한 이후 representative model을 다시 선택하지 않았습니다.

---

## 11. Evaluation Metrics

PNEUMONIA를 positive class로 정의합니다.

다음 지표를 사용합니다.

- Accuracy
- Recall
- Precision
- F1-score
- AUROC
- Confusion Matrix

데이터셋은 약 `NORMAL 27% : PNEUMONIA 73%`의 클래스 불균형을 가지고 있기 때문에 Accuracy만으로 모델을 평가하지 않습니다.

Recall, Precision, F1, AUROC 및 Confusion Matrix를 함께 확인합니다.

---

## 12. Explainability — In Progress

최종 모델에는 **Grad-CAM++**을 적용할 예정입니다.

Chest X-ray 위에 Class Activation Map을 overlay하여 모델이 NORMAL / PNEUMONIA를 판단할 때 어떤 영역에 주목했는지 확인합니다.

특히 다음을 정성적으로 분석할 예정입니다.

- 폐 영역과 관련된 특징에 주목하는지
- 배경 영역에 과도하게 의존하는지
- 영상 marker 등 비의도적 특징을 활용하는지
- Correct / Incorrect prediction 사이의 attention pattern 차이

Grad-CAM++ 결과는 모델의 attention pattern을 분석하기 위한 도구로 사용하며 임상적 타당성을 입증하는 근거로 해석하지 않습니다.

---

## 13. User Interface — Planned

Grad-CAM++ 적용 이후 간단한 PyQt 기반 UI와 모델 inference pipeline을 연결할 예정입니다.

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

## 14. Experiment Logging

각 experiment는 별도의 directory에 저장합니다.

```text
experiments/
├── final_2x2_experiment_grid.csv
├── scratch/
│   └── exp_XXX/
│       ├── config.json
│       ├── history.csv
│       ├── val_predictions.csv
│       ├── convergence.json
│       ├── test_results.json
│       └── test_predictions.csv
│
└── transfer/
    ├── pilot_fc_freeze/
    └── exp_XXX/
        ├── config.json
        ├── history.csv
        ├── val_predictions.csv
        ├── convergence.json
        ├── test_results.json
        └── test_predictions.csv
```

주요 기록:

- `config.json` — experiment configuration
- `history.csv` — epoch별 training / validation history
- `val_predictions.csv` — Validation sample-level predictions
- `convergence.json` — T95 / T97 / T99 convergence result
- `test_results.json` — final Test metrics
- `test_predictions.csv` — Test sample-level predictions

Model checkpoint 및 원본 dataset과 같은 대용량 파일은 GitHub repository에 업로드하지 않습니다.

---

## 15. Project Structure

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
│   └── transfer/
│
├── results/
│   └── comparison.csv
│
├── scripts/
│   ├── analyze_convergence.py
│   ├── compare_models.py
│   ├── evaluate_test.py
│   ├── check_duplicates.py
│   ├── check_near_duplicates.py
│   ├── inspect_dataset.py
│   ├── inspect_dataloader.py
│   ├── inspect_preprocess.py
│   ├── plot_history.py
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

## 16. Current Progress

- [x] Dataset analysis
- [x] Group-aware Train / Validation / Test split
- [x] SHA-256 / pHash data integrity audit
- [x] Preprocessing & augmentation pipeline
- [x] ResNet-50 Scratch training
- [x] ImageNet Transfer Learning
- [x] FC-only → Full Fine-tuning pipeline
- [x] Scratch 2×2 hyperparameter search
- [x] Transfer Learning 2×2 hyperparameter search
- [x] Representative configuration selection
- [x] T95 / T97 / T99 convergence analysis
- [x] Independent Test evaluation
- [ ] Grad-CAM++
- [ ] PyQt UI

---

## 17. Limitations

### Patient-level Grouping

데이터셋에 명시적인 patient-level metadata가 제공되지 않아 filename-derived identifier를 group ID로 사용했습니다.

따라서 현재 grouping을 검증된 patient ID와 동일하게 해석할 수 없습니다.

### Near-Duplicate Detection

pHash는 이미지의 의미적 동일성을 보장하지 않으므로 near-duplicate 후보 탐색 용도로만 사용했습니다.

후보 이미지는 직접 시각적으로 확인했습니다.

### Hyperparameter Search

Scratch와 Transfer Learning 모두 동일한 규모의 제한된 2×2 search space를 사용했습니다.

Exhaustive hyperparameter optimization을 수행하지 않았으므로 본 결과를 각 학습 방식이 달성할 수 있는 절대적인 최대 성능으로 해석하지 않습니다.

### Freeze Duration

Transfer Learning의 FC-only freeze 기간은 pilot experiment를 기반으로 7 epochs로 고정했습니다.

각 Learning Rate × Weight Decay configuration마다 freeze duration을 별도로 탐색하지 않았으므로 7 epochs가 모든 Transfer Learning configuration에서 최적인지는 확인하지 않았습니다.

### Random Seed

주요 비교 실험은 `seed=42`의 단일 random seed를 사용했습니다.

따라서 Scratch와 Transfer Learning 사이의 작은 성능 차이에 대해서는 신중한 해석이 필요하며, 향후 multiple random seeds를 이용한 반복 실험과 평균 및 표준편차 분석이 필요합니다.

### Validation Selection Bias

Validation set은 hyperparameter configuration 선택과 exploratory analysis에 반복적으로 사용되었습니다.

따라서 Validation 성능에는 selection bias가 존재할 가능성이 있습니다.

독립 Test set은 representative configuration이 확정된 이후에만 평가했으며 Test 결과를 이용한 model reselection은 수행하지 않았습니다.

### External Generalization

현재 평가는 단일 pediatric Chest X-ray dataset 내부의 held-out Test split을 기반으로 합니다.

향후 연구로 확장할 경우 명확한 patient-level metadata를 제공하는 데이터셋 및 외부 Chest X-ray dataset을 이용한 external validation과 domain shift 평가가 필요합니다.