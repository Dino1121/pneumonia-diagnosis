# Pediatric Chest X-ray Pneumonia Classification

소아 Chest X-ray 영상을 이용해 **NORMAL / PNEUMONIA를 분류**하고, 동일한 ResNet-50 architecture에서 **Scratch 학습과 ImageNet pretrained Transfer Learning**의 차이를 비교하는 프로젝트입니다.

비교의 핵심은 다음 세 가지입니다.

- 최종 분류 성능
- 높은 Validation 성능 영역에 도달하는 수렴 속도
- Grad-CAM++ 기반 모델 판단 근거

> 본 프로젝트는 학습 및 연구 목적으로 수행되며 실제 의료 진단을 목적으로 하지 않습니다.

---

## 1. 프로젝트 요약

### 연구 질문

> **Scratch 학습과 Transfer Learning은 분류 성능, 수렴 속도, 모델 판단 근거에서 어떤 차이를 보이는가?**

### 핵심 결과

| 항목 | Scratch | Transfer Learning |
| --- | ---: | ---: |
| Representative Experiment | exp_008 | exp_003 |
| Best Validation F1 | 0.9884 | **0.9919** |
| T99 Convergence | 51 epoch | **12 epoch** |
| Test Accuracy | **0.9761** | 0.9607 |
| Test Recall | **0.9836** | 0.9719 |
| Test Precision | **0.9836** | 0.9742 |
| Test F1 | **0.9836** | 0.9730 |
| Test AUROC | 0.9951 | **0.9956** |

핵심적으로 다음과 같은 차이가 관찰되었습니다.

- Transfer Learning은 T99 기준 Scratch보다 **39 epoch 빠르게** 높은 Validation 성능 영역에 도달했습니다.
- 독립 Test set에서는 Scratch가 Accuracy, Recall, Precision, F1에서 더 높은 값을 기록했습니다.
- Test AUROC는 두 모델 모두 약 `0.995`로 매우 유사했습니다.
- Grad-CAM++ 비교에서는 Scratch가 일부 sample에서 영상 경계, padding 인접 영역 및 폐야 외부까지 넓게 반응한 반면, Transfer Learning은 상대적으로 폐야 내부의 제한된 영역에 집중하는 사례가 더 자주 관찰되었습니다.
- 따라서 본 실험에서는 ImageNet pretraining의 효과가 최종 Test F1 향상보다는 **빠른 task adaptation 및 convergence**, 그리고 모델이 활용하는 시각적 특징의 차이에서 더 뚜렷하게 나타났습니다.

---

## 2. 연구 가설

### H1 — 분류 성능

> ImageNet pretrained Transfer Learning이 Scratch보다 높은 분류 성능을 보일 것이다.

Validation에서는 Transfer Learning이 더 높은 Best F1을 기록했지만, 독립 Test set에서는 Scratch가 더 높은 Accuracy, Recall, Precision 및 F1을 기록했습니다.

따라서 **Test F1을 기준으로 H1은 지지되지 않았습니다.**

### H2 — 수렴 속도

> Transfer Learning이 Scratch보다 더 적은 epoch에서 높은 Validation 성능의 안정적인 구간에 도달할 것이다.

T99 결과:

```text
Scratch  : Epoch 51
Transfer : Epoch 12
```

따라서 본 실험에서는 **Transfer Learning이 Scratch보다 훨씬 빠르게 높은 Validation 성능 영역에 도달했습니다.**

---

## 3. 데이터셋 및 분할

**Chest X-Ray Images (Pneumonia)** 데이터셋을 사용합니다.

원본 데이터셋은 총 **5,856장**입니다.

| Class | Images | Ratio |
| --- | ---: | ---: |
| NORMAL | 1,583 | 27.0% |
| PNEUMONIA | 4,273 | 73.0% |
| **Total** | **5,856** | **100%** |

원본 Validation set은 16장으로 매우 작기 때문에 기존 split을 그대로 사용하지 않고 전체 데이터를 기반으로 새로운 Train / Validation / Test split을 구성했습니다.

### 그룹 기반 데이터 분할

파일명에서 반복되는 식별자 패턴을 추출해 **filename-derived group ID**를 정의했습니다.

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

> 데이터셋에 공식 patient-level metadata가 제공되지 않으므로 filename-derived group ID를 실제 patient ID와 동일하게 해석하지 않습니다.

### 데이터 무결성 검증

초기 실험에서 높은 Validation 성능이 관찰되어 leakage 및 duplicate 가능성을 추가로 점검했습니다.

- SHA-256 exact duplicate 검사
- 서로 다른 split에 존재하는 동일 이미지 1쌍 확인
- 중복 이미지 1장 제외 후 전체 split 재생성
- pHash Hamming Distance ≤ 4인 cross-split 후보 검사
- 최종 split에서 cross-split SHA-256 exact duplicate 0
- 최종 split에서 filename-derived group overlap 0

최종 실험에는 **5,855장**을 사용합니다.

| Split | Images | Groups | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,684 | 2,233 | 1,266 | 3,418 |
| Validation | 586 | 279 | 158 | 428 |
| Test | 585 | 278 | 158 | 427 |
| **Total** | **5,855** | **2,790** | **1,582** | **4,273** |

자세한 검증 과정은 [`docs/data_integrity_audit.md`](docs/data_integrity_audit.md)에 기록합니다.

---

## 4. 전처리

모든 영상은 ResNet-50 입력을 위해 `224 × 224` 크기로 변환합니다.

### Resize & Padding

원본 영상의 aspect ratio를 유지하면서 resize한 뒤 부족한 영역에 padding을 적용합니다.

```text
Input size    : 224 × 224
Aspect ratio  : Preserved
Padding value : 0
```

### 채널 변환

Chest X-ray를 grayscale로 처리한 뒤 3-channel로 변환합니다.

```text
Grayscale
    ↓
1 Channel
    ↓
Repeat
    ↓
3 Channels
```

### 정규화

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

Horizontal Flip은 영상의 좌우 해부학적 위치를 바꿀 수 있어 제외했습니다.

---

## 5. 모델

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

```text
Epoch 1-7
Backbone      : Frozen
Classifier    : Trainable
Learning Rate : 0.001

Epoch 8-100
Backbone       : Unfrozen
Entire Network : Trainable
```

FC-only 기간은 pilot experiment 결과를 바탕으로 **7 epochs**로 고정했습니다.

---

## 6. 실험 설계

Scratch와 Transfer Learning 모두 동일한 규모의 **2×2 Hyperparameter Search**를 수행했습니다.

### 공통 설정

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

### Scratch 탐색 결과

| Learning Rate | Weight Decay | Experiment | Best Val F1 | Best Epoch |
| ---: | ---: | --- | ---: | ---: |
| 0.0001 | 0.001 | **exp_008** | **0.9884** | 68 |
| 0.0001 | 0.01 | exp_009 | 0.9837 | 92 |
| 0.00005 | 0.001 | exp_010 | 0.9789 | 83 |
| 0.00005 | 0.01 | exp_011 | 0.9813 | 64 |

### Transfer Learning 탐색 결과

FC-only phase:

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

각 학습 방식에서 **Best Validation F1이 가장 높은 configuration**을 representative model로 선택했습니다.

```text
Scratch           : exp_008
Transfer Learning : exp_003
```

---

## 7. 수렴 속도 분석

단일 epoch에서 발생한 높은 Validation F1을 수렴으로 판단하지 않기 위해 상대 threshold와 stability condition을 함께 사용했습니다.

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

`T99`를 primary convergence metric으로 사용하고 `T95`, `T97`을 sensitivity analysis로 사용합니다.

| Model | T95 | T97 | T99 |
| --- | ---: | ---: | ---: |
| Scratch exp_008 | **4** | 12 | 51 |
| Transfer exp_003 | 5 | **9** | **12** |

Transfer Learning의 global epoch에는 FC-only phase가 포함됩니다.

```text
Global Epoch 1-7 : FC-only
Global Epoch 8   : Fine-tuning Epoch 1
...
Global Epoch 12  : Fine-tuning Epoch 5
```

> 본 convergence definition은 post-hoc analysis metric이며 early stopping criterion이 아닙니다.

---

## 8. 독립 Test 평가

Representative model은 Validation 결과만으로 먼저 확정한 뒤 독립 Test set 585장으로 최종 평가했습니다.

Test 결과는 model 또는 hyperparameter를 다시 선택하는 데 사용하지 않았습니다.

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

Actual NORMAL       151       7
Actual PNEUMONIA      7     420

TN = 151
FP = 7
FN = 7
TP = 420
```

**Transfer exp_003**

```text
             Predicted
             NORMAL  PNEUMONIA

Actual NORMAL       147      11
Actual PNEUMONIA     12     415

TN = 147
FP = 11
FN = 12
TP = 415
```

---

## 9. Grad-CAM++ 분석

대표 모델인 Scratch `exp_008`과 Transfer Learning `exp_003`에 Grad-CAM++을 적용해 동일한 Test 이미지에서 모델의 판단 근거를 비교합니다.

### Predicted-class Target

Grad-CAM++ target은 고정된 PNEUMONIA class가 아니라 **각 모델이 실제로 예측한 class**를 사용합니다.

```text
Predicted NORMAL
→ Grad-CAM++ target: NORMAL

Predicted PNEUMONIA
→ Grad-CAM++ target: PNEUMONIA
```

따라서 각 heatmap은 모델이 자신의 최종 예측을 형성하는 데 어떤 영역을 활용했는지 보여줍니다.

Evaluation metric에서는 기존과 동일하게 `PNEUMONIA`를 positive class로 정의하며, 이는 Grad-CAM++ target과 별개의 설정입니다.

### 비교 설정

```text
Scratch           : exp_008
Transfer Learning : exp_003
Target Layer      : layer4[-1]
CAM Method        : Grad-CAM++
Target Score      : pre-softmax predicted-class logit
```

Grad-CAM++ 비교용 sample은 Test split에서 class-balanced random sampling으로 선택합니다.

기본값:

```text
NORMAL      : 5 images
PNEUMONIA   : 5 images
Total       : 10 images
Random Seed : 42
```

### Grad-CAM++ 비교 결과

현재 확인한 여러 Test sample에서는 다음과 같은 차이가 관찰되었습니다.

- Scratch는 일부 sample에서 폐야 내부뿐 아니라 영상 경계, padding 인접 영역 및 폐야 외부까지 넓게 반응했습니다.
- Transfer Learning은 상대적으로 폐야 내부의 제한된 영역에 더 집중하는 사례가 자주 나타났습니다.
- 두 모델이 동일한 정답을 예측하더라도 heatmap의 spatial pattern은 다를 수 있었습니다.

이 결과는 Scratch 모델이 일부 sample에서 **비병변적 또는 shortcut-like visual feature에 더 의존했을 가능성**을 시사합니다.

다만 Grad-CAM++는 post-hoc attribution method이므로 이 결과만으로 shortcut learning 또는 병변 localization을 확정하지 않습니다.

자세한 내용은 [`docs/gradcam_analysis.md`](docs/gradcam_analysis.md)에 기록합니다.

---

## 10. 실행 방법

### Grad-CAM++ 기본 실행

```bash
python scripts/compare_gradcam.py
```

### 다른 Random Seed 사용

```bash
python scripts/compare_gradcam.py --seed 123
```

### 이미지 수 변경

```bash
python scripts/compare_gradcam.py --num_images 20 --seed 123
```

자동 생성되는 Grad-CAM++ 결과는 `results/gradcam/` 아래에 저장되며 Git repository에는 포함하지 않습니다.

---

## 11. 프로젝트 구조

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
│   ├── data_integrity_audit.md
│   └── gradcam_analysis.md
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
│   ├── compare_gradcam.py
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

## 12. 진행 상황

- [x] Dataset analysis
- [x] Group-aware Train / Validation / Test split
- [x] SHA-256 / pHash data integrity verification
- [x] Preprocessing & augmentation pipeline
- [x] ResNet-50 Scratch training
- [x] ImageNet Transfer Learning
- [x] FC-only → Full Fine-tuning pipeline
- [x] Scratch 2×2 Hyperparameter Search
- [x] Transfer Learning 2×2 Hyperparameter Search
- [x] Representative model selection
- [x] T95 / T97 / T99 convergence analysis
- [x] Independent Test evaluation
- [x] Grad-CAM++ Scratch / Transfer Learning comparison
- [ ] PyQt UI

---

## 13. 한계

### Patient-level Grouping

데이터셋에 공식 patient-level metadata가 제공되지 않아 filename-derived identifier를 group ID로 사용했습니다.

따라서 현재 grouping을 검증된 patient ID와 동일하게 해석할 수 없습니다.

### Near-duplicate Detection

pHash는 이미지의 의미적 동일성을 보장하지 않으므로 near-duplicate candidate screening 용도로 사용했습니다.

탐지된 후보 이미지는 직접 시각적으로 확인했습니다.

### 제한된 Hyperparameter Search

Scratch와 Transfer Learning 모두 동일한 규모의 제한된 2×2 search space를 사용했습니다.

본 결과를 각 학습 방식이 달성할 수 있는 절대적인 최대 성능으로 해석하지 않습니다.

### Freeze 기간

Transfer Learning의 FC-only freeze 기간은 pilot experiment를 기반으로 7 epochs로 고정했습니다.

각 configuration마다 freeze duration을 별도로 탐색하지 않았습니다.

### 단일 Random Seed

주요 비교 실험은 `seed=42`의 단일 random seed를 사용했습니다.

따라서 Scratch와 Transfer Learning 사이의 작은 성능 차이는 신중하게 해석해야 하며, 보다 강한 비교를 위해서는 multiple random seeds를 이용한 반복 실험이 필요합니다.

### Validation Selection Bias

Validation set은 hyperparameter configuration 선택과 exploratory analysis에 반복적으로 사용되었습니다.

독립 Test set은 representative model을 확정한 이후에만 평가했으며 Test 결과를 이용한 model reselection은 수행하지 않았습니다.

### Grad-CAM++ 해석의 한계

Grad-CAM++는 모델이 어떤 영역을 활용했는지 확인하기 위한 post-hoc explainability method입니다.

따라서 heatmap을 실제 병변 위치, 임상적 타당성 또는 causal explanation으로 직접 해석하지 않습니다.

### 외부 데이터 일반화

현재 평가는 단일 pediatric Chest X-ray dataset 내부의 held-out Test split을 기반으로 합니다.

향후에는 명확한 patient-level metadata를 제공하는 데이터셋 및 외부 Chest X-ray dataset을 이용한 external validation과 domain shift 평가가 필요합니다.
