# Data Integrity Audit

## 1. 목적

본 프로젝트는 Pediatric Chest X-ray 데이터를 이용하여
NORMAL과 PNEUMONIA를 분류하는 모델을 구축한다.

초기 Scratch ResNet-50 실험에서 예상보다 높은 Validation 성능이 관찰되었다.
이에 따라 단순히 높은 성능을 모델의 학습 결과로 받아들이기보다,
데이터 분할 과정에서 leakage가 발생했을 가능성을 먼저 검토할 필요가 있다고 판단했다.

특히 의료영상 데이터는 동일 환자의 여러 이미지 또는 동일 이미지의 복사본이
Train과 Validation/Test에 동시에 포함될 경우 실제 일반화 성능보다
높은 평가 결과가 나타날 수 있다.

따라서 본 프로젝트에서는 모델 비교에 앞서 데이터 무결성을 검증하였다.


## 2. Original Dataset

사용한 데이터셋은 Pediatric Chest X-ray 데이터로,
전체 5,856장의 이미지로 구성되어 있다.

클래스 분포는 다음과 같다.

- NORMAL: 1,583 images (27.0%)
- PNEUMONIA: 4,273 images (73.0%)

원본 데이터셋은 다음과 같이 분할되어 있었다.

- Train: 5,216 images
- Validation: 16 images
- Test: 624 images

Validation 데이터가 16장으로 매우 적어 안정적인 모델 평가에 적합하지 않다고 판단했다.

따라서 기존 Train/Validation/Test 구성을 그대로 사용하지 않고,
전체 데이터를 통합한 뒤 새로운 Train/Validation/Test split을 생성하기로 했다.


## 3. Group-aware Split

### 3.1 문제 인식

단순한 image-level random split을 사용할 경우,
서로 관련된 이미지가 서로 다른 split으로 들어갈 가능성이 있다.

특히 PNEUMONIA 이미지의 파일명에는 다음과 같은 패턴이 존재했다.

`personXX_...`

NORMAL 이미지에서도 다음과 같은 파일명 패턴이 존재했다.

`IM-XXXX-...`

따라서 파일명에서 공통 식별자를 추출하여 동일 group에 속하는 이미지가
서로 다른 split으로 분리되지 않도록 하는 방법을 사용했다.


### 3.2 Group ID 정의

PNEUMONIA:

`(person\d+)`

NORMAL:

`(IM-\d+)`

추출된 값에는 클래스 prefix를 추가하여 group ID를 생성했다.

예:

- `PNEUMONIA_person123`
- `NORMAL_IM-0123`

단, 데이터셋에서 공식적인 patient metadata가 제공되는 것은 아니므로
이 값을 실제 patient ID라고 단정하지 않는다.

본 프로젝트에서는 이를 filename-derived group ID로 정의한다.


### 3.3 Split 방법

다음과 같이 StratifiedGroupKFold를 사용했다.

- `n_splits = 10`
- `shuffle = True`
- `random_state = 42`

10개의 fold 중

- Fold 0–7 → Train
- Fold 8 → Validation
- Fold 9 → Test

로 사용하여 약 8:1:1 비율로 구성하였다.

이 방법을 통해 클래스 비율을 가능한 한 유지하면서,
동일한 filename-derived group이 여러 split에 포함되지 않도록 했다.


## 4. 높은 Scratch 성능에 대한 의문

초기 Scratch ResNet-50 실험에서 Validation F1-score가
예상보다 높은 수준까지 상승하였다.

Scratch 모델은 ImageNet pretrained weight를 사용하지 않고

`weights=None`

으로 초기화했기 때문에,
높은 성능이 pretrained weight의 영향으로 발생한 것은 아니었다.

따라서 다음 가능성을 우선적으로 의심했다.

1. Train/Validation 간 동일 이미지 존재
2. 파일명이 다른 동일 이미지 존재
3. 매우 유사한 이미지가 서로 다른 split에 존재
4. filename-derived grouping으로 포착하지 못한 데이터 관계

이에 따라 파일명 기반 group overlap 검사만으로는 충분하지 않다고 판단하고,
이미지 자체를 기준으로 추가적인 duplicate audit을 수행했다.


## 5. SHA 기반 Exact Duplicate Audit

### 5.1 목적

파일명이 다르더라도 이미지 내용이 완전히 동일할 수 있다.

따라서 이미지 파일의 SHA hash를 계산하여
Train/Validation/Test 사이에 동일 이미지가 존재하는지 검사했다.


### 5.2 발견

검사 과정에서 서로 다른 filename-derived group ID를 가지고 있지만
실제로 동일한 이미지인 한 쌍을 발견했다.

- `NORMAL2-IM-0096-0001.jpeg`
- `NORMAL2-IM-0095-0001.jpeg`

두 이미지는 직접 시각적으로 비교하였으며,
동일한 이미지임을 확인했다.

이는 filename-derived grouping만으로는
모든 duplicate를 탐지할 수 없다는 것을 보여준다.


## 6. Duplicate 처리

원본 데이터셋 자체는 수정하지 않고 보존하기로 했다.

대신 split 생성 과정에서 다음 파일 한 장만 제외했다.

`NORMAL2-IM-0095-0001.jpeg`

이를 통해 raw data의 원본 상태를 유지하면서
실험용 split에서 duplicate가 사용되지 않도록 했다.


## 7. Dataset 재분할

Duplicate 한 장을 제외한 뒤
동일한 group-aware split 방법을 사용하여 데이터셋을 다시 분할했다.

최종 데이터 수는 5,855장이다.

### Train

- Images: 4,684
- Groups: 2,233
- NORMAL: 1,266
- PNEUMONIA: 3,418

### Validation

- Images: 586
- Groups: 279
- NORMAL: 158
- PNEUMONIA: 428

### Test

- Images: 585
- Groups: 278
- NORMAL: 158
- PNEUMONIA: 427

전체 클래스 분포:

- NORMAL: 1,582 (27.02%)
- PNEUMONIA: 4,273 (72.98%)

Group overlap 검사 결과:

- Train ↔ Validation: 0
- Train ↔ Test: 0
- Validation ↔ Test: 0


## 8. 재분할 후 SHA Audit

새로운 split을 대상으로 다시 exact duplicate 검사를 수행했다.

결과:

- Total images: 5,855
- Unique hashes: 5,824
- Duplicate image rows: 60
- Cross-split duplicate hash groups: 0

동일한 이미지가 데이터셋 내부에 일부 존재할 수는 있지만,
SHA 기준으로 동일한 이미지가 서로 다른 split에 걸쳐 존재하는 경우는 발견되지 않았다.

따라서 확인 가능한 exact duplicate에 의한
cross-split leakage는 제거된 것으로 판단했다.


## 9. Perceptual Hash Audit

### 9.1 목적

SHA hash는 이미지가 완전히 동일한 경우를 찾는 데 적합하지만,
resize, compression 또는 작은 픽셀 변화가 발생한 이미지는
서로 다른 hash를 갖게 된다.

따라서 perceptual hash(pHash)를 추가로 사용하여
시각적으로 매우 유사한 이미지가 서로 다른 split에 존재하는지 검사했다.

Cross-split image pair만 비교하였으며,
Hamming distance가 4 이하인 경우를 후보로 추출했다.


### 9.2 Duplicate 제거 이전

초기 검사에서는 총 71개의 cross-split candidate가 발견되었다.

- Distance 0: 1
- Distance 2: 4
- Distance 4: 66

Distance 0 후보를 확인한 결과,
앞에서 발견한 실제 duplicate 이미지였다.

Distance 2 및 Distance 4 후보도 시각적으로 확인하였다.

그러나 대부분 서로 다른 Chest X-ray 이미지였다.

Chest X-ray는 전체적인 해부학적 구조가 유사하며
촬영 영역, 배경, radiographic marker 등의 공통적인 시각 요소가 존재하기 때문에
pHash가 서로 다른 이미지도 유사 이미지로 탐지할 수 있다고 판단했다.


## 10. Duplicate 제거 이후 pHash 재검사

Duplicate 제거 및 재분할 후 다시 pHash audit을 수행했다.

결과:

- Total cross-split candidates: 76
- Distance 0: 0
- Distance 2: 2
- Distance 4: 74

Distance 2 후보:

- Train NORMAL `IM-0286-0001.jpeg`
  ↔ Validation NORMAL `IM-0307-0001.jpeg`

- Train NORMAL `IM-0401-0001.jpeg`
  ↔ Validation NORMAL `IM-0544-0001.jpeg`

후보 이미지들을 시각적으로 확인한 결과,
동일 이미지로 판단되는 사례는 발견되지 않았다.

특히 exact perceptual duplicate에 해당하는
Distance 0 후보가 더 이상 존재하지 않았다.


## 11. Audit 이후 Scratch 재실험

확인된 cross-split duplicate를 제거하고 새롭게 생성한 Clean Split에서도 높은 Scratch 성능이 유지되는지 확인하기 위해 ResNet-50을 다시 학습했다.

Clean Split 기반 재학습에서도 Validation F1-score는 약 98% 후반까지 도달했다.

또한 특정 random initialization에 의해 우연히 높은 성능이 나타났을 가능성을 확인하기 위해 Python, NumPy, PyTorch 및 CUDA의 random seed를 42로 고정하고 동일한 조건에서 추가 실험을 수행했다.

Seed 고정 이후에도 높은 Validation 성능이 다시 관찰되었다.

따라서 처음 발견된 cross-split duplicate 한 장이 초기 Scratch 모델의 높은 Validation 성능을 설명하는 주된 원인일 가능성은 낮다고 판단했다.

다만 높은 Validation 성능과 별개로 일부 epoch에서 Validation Loss와 class prediction이 크게 변하는 현상이 관찰되었다.

이에 따라 Validation sample별 prediction과 loss를 추가로 기록하여 분석한 결과, 특정 epoch에서 다수 sample의 class prediction이 동시에 변화하는 prediction instability가 존재함을 확인했다.

이후 Learning Rate 조정, adaptive learning rate 및 regularization 실험을 통해 Scratch 모델의 학습 안정성을 별도로 분석하였다.

이러한 결과는 현재 Clean Split에서 높은 Validation 성능이 재현되고 있음을 보여주지만, 모델이 실제 pathology-related feature를 학습했는지 또는 dataset-specific shortcut feature를 활용하고 있는지는 별도의 검증이 필요하다.


## 12. 현재 판단

현재까지 다음 항목을 확인하였다.

- Filename-derived group 기준 Train / Validation / Test overlap 없음
- SHA-256 기반 cross-split exact duplicate audit 수행
- 발견된 cross-split exact duplicate 1건 처리
- Duplicate 제거 후 전체 split 재생성
- 재생성된 Clean Split에서 SHA-256 기준 cross-split exact duplicate 없음
- pHash 기반 cross-split near-duplicate candidate 탐색
- pHash Hamming Distance 2/4 후보 시각적 검사
- 현재 Clean Split에서 pHash Distance 0 cross-split candidate 없음
- Clean Split에서도 높은 Scratch Validation 성능 재현
- Random seed 고정 이후에도 높은 Scratch Validation 성능 재현

따라서 초기 audit에서 실제 cross-split duplicate 한 쌍이 발견되었지만, 이를 제거하고 split을 재생성한 현재 Clean Split에서는 확인 가능한 group leakage 및 duplicate-image leakage가 추가로 발견되지 않았다.

또한 duplicate 제거 이후에도 높은 Scratch Validation 성능이 재현되었기 때문에, 발견된 duplicate가 초기 높은 성능의 주된 원인은 아닌 것으로 판단한다.

그러나 현재 결과가 모든 형태의 data leakage 가능성을 배제한다는 의미는 아니다.

특히 filename-derived group ID는 공식 patient metadata가 아니기 때문에 실제 patient-level independence를 완전히 검증할 수 없다.


## 13. 남아 있는 한계

### 13.1 Patient-level Identity

현재 grouping은 공식 patient metadata가 아니라 filename에서 추출한 identifier에 의존한다.

따라서 동일한 filename-derived group이 여러 split에 포함되지 않는 것은 확인했지만, 서로 다른 group ID가 실제로 서로 다른 환자를 의미하는지는 검증할 수 없다.

즉 현재 방법은 patient-level leakage의 위험을 줄이기 위한 방법이지만, 공식 patient identifier를 이용한 분할과 동일한 수준의 보장을 제공하지 않는다.


### 13.2 Perceptual Hash의 한계

pHash는 이미지의 의미적 동일성을 직접 판단하는 방법이 아니다.

Chest X-ray처럼 전체적인 구조가 유사한 영상에서는 서로 다른 이미지도 작은 Hamming distance를 가질 수 있다.

따라서 본 프로젝트에서는 pHash를 duplicate 판정 기준으로 직접 사용하지 않고 near-duplicate candidate를 탐색하기 위한 screening 방법으로 사용했으며, 탐지된 후보는 직접 시각적으로 검수했다.


### 13.3 Shortcut Learning 가능성

높은 Validation 성능이 데이터 누수에 의해 발생하지 않았더라도 모델이 실제 폐렴 관련 특징 대신 dataset-specific shortcut을 학습했을 가능성은 남아 있다.

예를 들어 다음과 같은 특징이 모델 예측에 영향을 줄 가능성이 있다.

- Radiographic markers
- Text
- Image borders
- Acquisition/device-related artifacts
- Crop 및 positioning 차이
- Resolution 차이
- Dataset-specific visual patterns

따라서 높은 Validation 성능만으로 모델이 실제 pathology-related feature를 학습했다고 결론 내리지 않는다.


### 13.4 Internal Validation의 한계

현재 성능 분석은 동일한 원본 데이터셋에서 생성한 Train / Validation split을 기반으로 한다.

따라서 현재 Clean Split에서 높은 성능이 재현되더라도 다른 병원, 장비, 환자 집단 또는 촬영 환경에서도 동일한 성능이 유지된다고 판단할 수 없다.

향후 보다 강한 일반화 성능 검증을 위해 외부 Chest X-ray dataset을 이용한 external validation 및 domain shift 평가를 고려할 수 있다.


## 14. 이후 검증 계획

Data Integrity Audit 이후에는 데이터 누수 여부와 별개로 모델의 학습 특성과 일반화 가능성을 추가로 검증한다.

현재 진행 중인 주요 검증은 다음과 같다.

### Scratch vs Transfer Learning

동일한 ResNet-50 architecture와 Clean Split을 사용하여 Scratch 학습과 ImageNet pretrained Transfer Learning의 분류 성능 및 수렴 특성을 비교한다.

이를 통해 현재 데이터셋에서 높은 Scratch 성능이 관찰되는 상황에서도 pretrained representation이 추가적인 성능 또는 수렴상의 이점을 제공하는지 분석한다.


### Grad-CAM++

최종 모델에는 Grad-CAM++을 적용하여 예측 과정에서 모델이 주목하는 영역을 시각화할 예정이다.

특히 모델이 실제 lung region 및 pathology-related region에 집중하는지, 또는 marker, border 및 기타 비의도적 특징에 의존하는지를 분석한다.

Grad-CAM++ 결과는 모델의 판단 근거에 대한 정성적 분석으로 사용하며, 그 자체를 모델의 임상적 타당성을 증명하는 근거로 해석하지 않는다.


### Final Test Evaluation

모델 및 학습 설정이 확정될 때까지 Test set은 모델 선택이나 hyperparameter 조정에 사용하지 않는다.

Scratch와 Transfer Learning 비교 및 최종 모델 선택이 완료된 이후 독립적으로 유지한 Test set에서 최종 성능을 평가한다.


## 15. 결론

초기 Scratch ResNet-50에서 예상보다 높은 Validation 성능이 관찰되어 data leakage 가능성을 우선적으로 검증하였다.

이를 위해 filename-derived group 기반 split 검증, SHA-256 exact duplicate audit, pHash near-duplicate screening 및 후보 이미지의 시각적 검수를 수행했다.

초기 split에서 서로 다른 filename-derived group에 속하지만 실제로 동일한 cross-split 이미지 한 쌍을 발견하였다.

이에 따라 원본 데이터는 그대로 보존하면서 중복본 한 장을 실험 대상에서 제외하고 전체 dataset split을 다시 생성하였다.

재생성된 Clean Split에서는 다음을 확인했다.

- Filename-derived group overlap 0
- SHA-256 기준 cross-split exact duplicate 0
- pHash Distance 0 cross-split candidate 0

또한 Clean Split 및 seed 고정 조건에서도 높은 Scratch Validation 성능이 재현되었다.

따라서 현재까지 확인된 결과에서는 발견된 cross-split duplicate가 초기 높은 Validation 성능의 주된 원인이었다고 보기 어렵다.

그러나 현재 grouping은 공식 patient metadata에 기반하지 않으며, duplicate audit만으로 모든 형태의 leakage 또는 shortcut learning 가능성을 배제할 수 없다.

따라서 현재 결과를 실제 임상 환경에서의 일반화 성능으로 해석하지 않으며, 이후 Scratch vs Transfer Learning 비교, 독립 Test evaluation, Grad-CAM++ 분석 및 필요시 external validation을 통해 모델의 학습 특성과 일반화 가능성을 추가적으로 검증할 예정이다.