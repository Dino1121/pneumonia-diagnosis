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

데이터 leakage가 높은 Scratch 성능의 원인일 가능성을 확인하기 위해
정리된 새로운 split으로 Scratch ResNet-50을 다시 학습했다.

그러나 데이터 정리 이후에도 Validation F1-score가
약 98% 후반까지 도달하였다.

따라서 처음 발견된 duplicate 한 장만으로
기존의 높은 Validation 성능을 설명하기는 어렵다고 판단했다.

이는 높은 성능 자체가 반드시 leakage를 의미하지는 않는다는 점을 보여주지만,
동시에 현재 데이터셋에서 모델이 실제 폐렴 병변을 학습한 것인지
다른 shortcut feature를 이용하고 있는지는 별도의 검증이 필요하다.


## 12. 현재 판단

현재까지 다음 항목을 확인하였다.

- Filename-derived group 기준 split overlap 없음
- SHA 기반 cross-split exact duplicate 없음
- pHash distance 0 cross-split candidate 없음
- pHash distance 2/4 후보 시각적 검사
- 확인된 duplicate 한 장 제거 후 데이터 재분할
- 정리된 split에서도 높은 Scratch 성능 재현

따라서 현재까지 확인 가능한
명백한 duplicate-image leakage는 발견되지 않았다.

다만 이것이 모든 형태의 data leakage가 존재하지 않는다는 것을
증명하는 것은 아니다.


## 13. 남아 있는 한계

현재 grouping은 공식 patient metadata가 아니라
filename에서 추출한 identifier에 의존한다.

따라서 실제 patient identity를 완전히 검증할 수 없다.

또한 모델이 다음과 같은 shortcut feature를 사용할 가능성이 존재한다.

- Radiographic markers
- Text
- Image borders
- Acquisition/device-related artifacts
- Resolution 또는 padding과 관련된 특징
- Dataset-specific visual patterns

향후 Grad-CAM++ 등을 이용하여 모델이 실제 lung region과
pathology-related feature에 집중하는지 확인할 필요가 있다.

외부 Pediatric Chest X-ray 데이터셋을 이용한 external validation 역시
현재 데이터셋에 특화된 특징을 학습했는지 평가하는 데 중요한 후속 실험이 될 수 있다.


## 14. 결론

초기 Scratch 모델에서 예상보다 높은 Validation 성능이 관찰되어
data leakage 가능성을 의심하였다.

이에 따라 group-aware split 검증,
SHA exact duplicate audit,
pHash near-duplicate audit,
시각적 검증을 순차적으로 수행하였다.

실제 duplicate 한 쌍을 발견하여 한 장을 split에서 제외하고
전체 데이터를 다시 분할하였다.

재검증 결과 SHA 기준 cross-split exact duplicate와
pHash distance 0 후보는 발견되지 않았다.

그럼에도 Scratch 모델의 높은 Validation 성능이 유지되었기 때문에,
현재까지 발견된 duplicate가 높은 성능의 주된 원인은 아닌 것으로 판단한다.

향후에는 반복 실험, Transfer Learning 비교,
Grad-CAM++ 기반 설명 가능성 분석 및 필요시 외부 데이터 검증을 통해
모델의 일반화 성능과 학습 특성을 추가적으로 분석할 예정이다.