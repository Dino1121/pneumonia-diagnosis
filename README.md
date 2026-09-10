# Pediatric Chest X-ray Pneumonia Classification

소아 Chest X-ray 영상을 이용하여 **NORMAL / PNEUMONIA를 분류**하는 딥러닝 프로젝트입니다.

동일한 ResNet-50 구조에서 **Scratch 학습과 ImageNet 사전학습 기반 Transfer Learning**의 성능을 비교하고, 최종적으로 **Grad-CAM++**을 이용해 모델의 판단 근거를 시각화하는 것을 목표로 합니다.

> 현재 개발 중인 프로젝트입니다.

---

## 1. Project Overview

### 목표

* 소아 Chest X-ray 영상의 NORMAL / PNEUMONIA 이진 분류
* ResNet-50 Scratch와 Transfer Learning 성능 비교
* Accuracy뿐만 아니라 Recall, Precision, F1-score 등 다양한 지표를 이용한 평가
* Grad-CAM++을 이용한 모델 판단 영역 시각화
* 최종 학습 모델을 사용자 UI와 연결

---

## 2. Dataset

**Chest X-Ray Images (Pneumonia)** 데이터셋을 사용합니다.

* Total: **5,856 images**
* NORMAL: **1,583 images (27.0%)**
* PNEUMONIA: **4,273 images (73.0%)**
* Pediatric Chest X-ray
* NORMAL / PNEUMONIA Binary Classification

### Original Split

| Split      | Images |
| ---------- | -----: |
| Train      |  5,216 |
| Validation |     16 |
| Test       |    624 |

원본 데이터셋은 Validation 데이터가 16장으로 매우 적기 때문에 프로젝트에서는 전체 데이터를 병합한 후 새로운 Train / Validation / Test split을 구성합니다.

---

## 3. Group-aware Data Split

데이터셋의 파일명을 분석하는 과정에서 동일한 식별자 패턴을 공유하는 이미지가 존재하는 것을 확인했습니다.

예를 들어 PNEUMONIA 이미지에서는 `personXX`, NORMAL 이미지에서는 `IM-XXXX`와 같은 반복되는 식별자 패턴이 존재합니다.

따라서 이미지 단위로 무작위 분할할 경우 서로 연관되어 있을 가능성이 있는 이미지가 서로 다른 split에 포함될 수 있다고 판단했습니다.

이를 줄이기 위해 파일명에서 **filename-derived group ID**를 생성하고, 동일 group이 여러 split에 분산되지 않도록 group-aware split을 적용했습니다.

> 해당 group ID가 실제 patient ID임을 직접 확인할 수 있는 메타데이터는 제공되지 않으므로, 본 프로젝트에서는 이를 patient ID가 아닌 **filename-derived group identifier**로 정의합니다.

### Group ID

* PNEUMONIA → `personXX`
* NORMAL → `IM-XXXX`
* Label prefix를 추가하여 클래스 간 ID 충돌 방지

### Split Method

`StratifiedGroupKFold`를 사용하여 클래스 비율을 고려하면서 동일 group이 서로 다른 fold에 포함되지 않도록 분할했습니다.

최종적으로 10개 fold 중:

* Fold 0–7 → Train
* Fold 8 → Validation
* Fold 9 → Test

로 구성하여 약 **8 : 1 : 1** 비율로 분할했습니다.

### Final Split

| Split      |    Images |    Groups |    NORMAL | PNEUMONIA |
| ---------- | --------: | --------: | --------: | --------: |
| Train      |     4,686 |     2,232 |     1,267 |     3,419 |
| Validation |       585 |       279 |       158 |       427 |
| Test       |       585 |       279 |       158 |       427 |
| **Total**  | **5,856** | **2,790** | **1,583** | **4,273** |

---

## 4. Model

### ResNet-50

동일한 ResNet-50 architecture를 사용하여 두 가지 학습 방식을 비교합니다.

#### Scratch

* Random initialization
* ImageNet pretrained weights 사용하지 않음
* 전체 network를 처음부터 학습

#### Transfer Learning

* ImageNet pretrained ResNet-50 사용
* 초기에는 backbone을 freeze하고 classifier를 학습
* 이후 필요에 따라 일부 또는 전체 layer를 fine-tuning

동일한 데이터 split과 평가 기준을 사용하여 두 학습 방법의 성능을 비교합니다.

---

## 5. Preprocessing

현재 계획:

* Input size: `224 × 224`
* ImageNet normalization
* Data augmentation 적용
* Train / Validation / Test에 동일한 기본 preprocessing 기준 적용
* Augmentation은 Train dataset에만 적용

세부 augmentation 방법은 baseline 실험을 진행하면서 확정할 예정입니다.

---

## 6. Evaluation

모델 평가는 다음 지표를 사용할 예정입니다.

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve / ROC-AUC
* PR Curve / PR-AUC

의료 영상 분류 특성을 고려하여 **PNEUMONIA Recall**을 주요 평가 지표 중 하나로 확인합니다.

---

## 7. Explainable AI

최종 모델의 판단 근거를 확인하기 위해 **Grad-CAM++**을 적용할 예정입니다.

Chest X-ray 원본 영상 위에 Class Activation Map을 overlay하여 모델이 PNEUMONIA / NORMAL을 판단할 때 어떤 영역에 주목했는지 시각적으로 확인합니다.

---

## 8. User Interface

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

## 9. Tech Stack

* Python
* PyTorch
* torchvision
* OpenCV
* scikit-learn
* PyQt
* Git / GitHub

---

## 10. Project Structure

```text
pediatric-cxr-pneumonia/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│
├── results/
│
├── README.md
└── .gitignore
```

> Dataset 및 학습된 model 파일 등 대용량 파일은 GitHub repository에 직접 업로드하지 않습니다.

---

## 11. Progress

* [x] Dataset 구조 분석
* [x] 클래스 분포 확인
* [x] Filename-derived group ID 정의
* [x] Group-aware Train / Validation / Test split
* [ ] Data preprocessing pipeline
* [ ] ResNet-50 Scratch baseline
* [ ] ResNet-50 Transfer Learning
* [ ] Model evaluation
* [ ] Grad-CAM++
* [ ] UI implementation
* [ ] Final comparison and analysis

---

## 12. Limitations

현재 사용 중인 데이터셋에서는 명시적인 patient-level metadata를 직접 확인할 수 없기 때문에 파일명에서 추출한 식별자를 group ID로 사용했습니다.

따라서 본 프로젝트의 group ID를 검증된 patient ID로 해석하지 않습니다.

향후 연구로 확장할 경우 명확한 patient-level metadata를 제공하는 원본 데이터 또는 추가 데이터셋을 활용하여 보다 엄격한 patient-level evaluation 및 external validation을 수행하는 것을 고려합니다.
