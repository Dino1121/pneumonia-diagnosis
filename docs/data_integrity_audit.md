# 데이터 무결성 검증

## 1. 목적

본 프로젝트는 Pediatric Chest X-ray 데이터를 이용하여 NORMAL과 PNEUMONIA를 분류한다.

초기 Scratch ResNet-50 실험에서 예상보다 높은 Validation 성능이 관찰되어, 이를 단순히 모델의 학습 결과로 받아들이기 전에 데이터 분할 과정에서 leakage가 발생했을 가능성을 먼저 점검했다.

특히 의료영상 데이터에서는 동일 환자의 여러 이미지 또는 동일 이미지의 복사본이 Train과 Validation/Test에 동시에 포함될 경우 실제 일반화 성능보다 높은 평가 결과가 나타날 수 있다.

따라서 모델 비교에 앞서 filename-derived grouping, SHA-256 exact duplicate 검사, pHash near-duplicate screening을 이용해 데이터 무결성을 검증했다.

---

## 2. 원본 데이터셋

전체 이미지 수는 5,856장이다.

- NORMAL: 1,583 images (27.0%)
- PNEUMONIA: 4,273 images (73.0%)

원본 분할:

- Train: 5,216 images
- Validation: 16 images
- Test: 624 images

Validation 데이터가 16장으로 매우 적어 안정적인 모델 평가에 적합하지 않다고 판단했다.

따라서 전체 데이터를 기반으로 새로운 Train / Validation / Test split을 생성했다.

---

## 3. 그룹 기반 데이터 분할

### 3.1 문제 인식

단순 image-level random split을 사용할 경우 서로 관련된 이미지가 여러 split에 분리될 가능성이 있다.

PNEUMONIA 파일명에는 다음과 같은 패턴이 존재한다.

```text
personXX_...
```

NORMAL 파일명에는 다음과 같은 패턴이 존재한다.

```text
IM-XXXX-...
```

따라서 파일명에서 공통 식별자를 추출해 filename-derived group ID를 생성했다.

### 3.2 Group ID 정의

PNEUMONIA:

```text
(person\d+)
```

NORMAL:

```text
(IM-\d+)
```

예:

```text
PNEUMONIA_person123
NORMAL_IM-0123
```

이 값은 공식 patient metadata가 아니므로 실제 patient ID라고 단정하지 않는다.

### 3.3 Split 방법

`StratifiedGroupKFold`를 사용했다.

```text
n_splits     = 10
shuffle      = True
random_state = 42

Fold 0-7 → Train
Fold 8   → Validation
Fold 9   → Test
```

이 방법을 통해 클래스 비율을 가능한 한 유지하면서 동일한 filename-derived group이 여러 split에 포함되지 않도록 했다.

---

## 4. 높은 Scratch 성능에 대한 의문

초기 Scratch ResNet-50 실험에서 Validation F1-score가 예상보다 높은 수준까지 상승했다.

Scratch 모델은 ImageNet pretrained weight를 사용하지 않고 `weights=None`으로 초기화했기 때문에 높은 성능이 pretrained weight의 직접적인 영향으로 발생한 것은 아니었다.

따라서 다음 가능성을 우선적으로 점검했다.

1. Train / Validation 간 동일 이미지 존재
2. 파일명이 다른 동일 이미지 존재
3. 매우 유사한 이미지가 서로 다른 split에 존재
4. filename-derived grouping으로 포착하지 못한 데이터 관계

---

## 5. SHA-256 Exact Duplicate 검사

### 5.1 목적

파일명이 다르더라도 이미지 내용이 완전히 동일할 수 있으므로 전체 이미지의 SHA-256 hash를 계산했다.

### 5.2 발견 결과

서로 다른 filename-derived group ID를 가지고 있지만 실제로 동일한 이미지인 한 쌍을 발견했다.

```text
NORMAL2-IM-0096-0001.jpeg
NORMAL2-IM-0095-0001.jpeg
```

두 이미지는 직접 시각적으로 비교해 동일한 이미지임을 확인했다.

이는 filename-derived grouping만으로 모든 duplicate를 탐지할 수 없음을 보여준다.

---

## 6. 중복 이미지 처리

원본 데이터셋 자체는 수정하지 않고 보존했다.

대신 split 생성 과정에서 다음 파일 한 장을 제외했다.

```text
NORMAL2-IM-0095-0001.jpeg
```

이를 통해 raw data를 보존하면서 실험용 split에서 해당 duplicate가 사용되지 않도록 했다.

---

## 7. 데이터셋 재분할

Duplicate 한 장을 제외한 뒤 동일한 group-aware split 방법으로 다시 분할했다.

최종 데이터 수는 5,855장이다.

| Split | Images | Groups | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,684 | 2,233 | 1,266 | 3,418 |
| Validation | 586 | 279 | 158 | 428 |
| Test | 585 | 278 | 158 | 427 |
| **Total** | **5,855** | **2,790** | **1,582** | **4,273** |

Group overlap:

```text
Train ↔ Validation : 0
Train ↔ Test       : 0
Validation ↔ Test  : 0
```

---

## 8. 재분할 후 SHA-256 검사

새로운 split을 대상으로 exact duplicate 검사를 다시 수행했다.

```text
Total images                      : 5,855
Unique hashes                     : 5,824
Duplicate image rows              : 60
Cross-split duplicate hash groups : 0
```

데이터셋 내부에 동일한 이미지가 일부 존재할 수는 있지만, SHA-256 기준으로 동일 이미지가 서로 다른 split에 걸쳐 존재하는 경우는 발견되지 않았다.

따라서 확인 가능한 exact duplicate에 의한 cross-split leakage는 제거된 것으로 판단했다.

---

## 9. pHash Near-duplicate 검사

### 9.1 목적

SHA-256은 파일 내용이 완전히 동일한 경우를 찾는 데 적합하지만 resize, compression 또는 작은 픽셀 변화가 발생하면 서로 다른 hash를 갖는다.

따라서 pHash를 추가로 사용해 시각적으로 매우 유사한 cross-split image pair를 탐색했다.

Hamming Distance ≤ 4를 후보 기준으로 사용했다.

### 9.2 Duplicate 제거 이전

초기 검사 결과:

```text
Cross-split candidates : 71

Distance 0 : 1
Distance 2 : 4
Distance 4 : 66
```

Distance 0 후보는 앞에서 발견한 실제 duplicate였다.

Distance 2와 4 후보도 직접 확인했으나 대부분 서로 다른 Chest X-ray 이미지였다.

Chest X-ray는 전체적인 해부학적 구조, 촬영 영역, 배경 및 radiographic marker 등이 유사하기 때문에 서로 다른 이미지도 작은 pHash distance를 가질 수 있다.

---

## 10. Duplicate 제거 이후 pHash 재검사

재분할 후 결과:

```text
Total cross-split candidates : 76
Distance 0                   : 0
Distance 2                   : 2
Distance 4                   : 74
```

Distance 2 후보:

```text
Train NORMAL IM-0286-0001.jpeg
↔ Validation NORMAL IM-0307-0001.jpeg

Train NORMAL IM-0401-0001.jpeg
↔ Validation NORMAL IM-0544-0001.jpeg
```

후보 이미지를 직접 확인한 결과 동일 이미지로 판단되는 사례는 발견되지 않았다.

특히 pHash 값이 동일한 Distance 0 후보가 더 이상 발견되지 않았다. 같은 pHash 값이 이미지의 동일성을 보장하는 것은 아니다.

---

## 11. 검증 이후 Scratch 재실험

Clean Split에서도 높은 Scratch 성능이 유지되는지 확인하기 위해 ResNet-50을 다시 학습했다.

Clean Split 기반 재학습에서도 Validation F1-score는 약 98% 후반까지 도달했다.

또한 Python, NumPy, PyTorch 및 CUDA의 random seed를 42로 고정한 조건에서도 높은 Validation 성능이 다시 관찰되었다.

따라서 처음 발견된 cross-split duplicate 한 장이 초기 Scratch 모델의 높은 Validation 성능을 설명하는 주된 원인일 가능성은 낮다고 판단했다.

다만 일부 epoch에서는 Validation Loss와 class prediction이 크게 변하는 prediction instability가 관찰되었다.

이후 Learning Rate 조정, adaptive learning rate 및 regularization 실험을 통해 Scratch 모델의 학습 안정성을 별도로 분석했다.

---

## 12. 현재 판단

현재까지 다음을 확인했다.

- Filename-derived group 기준 Train / Validation / Test overlap 없음
- SHA-256 기반 cross-split exact duplicate 검사 수행
- 발견된 cross-split exact duplicate 1건 처리
- Duplicate 제거 후 전체 split 재생성
- 재생성된 Clean Split에서 SHA-256 기준 cross-split exact duplicate 없음
- pHash 기반 cross-split near-duplicate candidate 탐색
- pHash Distance 2/4 후보 직접 확인
- 현재 Clean Split에서 pHash Distance 0 cross-split candidate 없음
- Clean Split에서도 높은 Scratch Validation 성능 재현
- Random seed 고정 이후에도 높은 Scratch Validation 성능 재현

따라서 초기 검사에서 실제 cross-split duplicate 한 쌍이 발견되었지만, 이를 처리하고 split을 재생성한 현재 Clean Split에서는 확인 가능한 group leakage 및 duplicate-image leakage가 추가로 발견되지 않았다.

그러나 이 결과가 모든 형태의 data leakage 가능성을 배제한다는 의미는 아니다.

특히 filename-derived group ID는 공식 patient metadata가 아니기 때문에 실제 patient-level independence를 완전히 검증할 수 없다.

---

## 13. 남아 있는 한계

### 13.1 Patient-level Identity

현재 grouping은 공식 patient metadata가 아니라 filename-derived identifier에 의존한다.

따라서 동일한 filename-derived group이 여러 split에 포함되지 않는 것은 확인했지만, 서로 다른 group ID가 실제로 서로 다른 환자를 의미하는지는 검증할 수 없다.

### 13.2 pHash의 한계

pHash는 이미지의 의미적 동일성을 직접 판단하는 방법이 아니다.

Chest X-ray처럼 전체적인 구조가 유사한 영상에서는 서로 다른 이미지도 작은 Hamming distance를 가질 수 있다.

따라서 pHash는 duplicate 판정 기준이 아니라 near-duplicate candidate screening 용도로 사용했고, 후보 이미지는 직접 시각적으로 확인했다.

### 13.3 Shortcut Learning 가능성

Duplicate 및 group leakage를 점검한 이후에도 높은 Scratch 성능이 유지되었으므로, 높은 성능이 실제 pathology-related feature 학습을 의미하는지는 별도로 확인할 필요가 있다.

이를 위해 대표 Scratch 및 Transfer Learning 모델에 Grad-CAM++을 적용해 두 모델의 heatmap을 비교했다.

자세한 분석은 [`gradcam_analysis.md`](gradcam_analysis.md)를 참고한다.

### 13.4 내부 데이터셋 평가의 한계

현재 성능 분석은 동일한 원본 데이터셋에서 생성한 Train / Validation / Test split을 기반으로 한다.

따라서 다른 병원, 장비, 환자 집단 또는 촬영 환경에서도 동일한 성능이 유지된다고 판단할 수 없다.

본 프로젝트에서는 external validation 및 domain shift 평가를 수행하지 않았다.

---

## 14. 결론

초기 Scratch ResNet-50에서 예상보다 높은 Validation 성능이 관찰되어 data leakage 가능성을 우선적으로 검증했다.

수행한 검사는 다음과 같다.

- Filename-derived group 기반 split 검증
- SHA-256 exact duplicate 검사
- pHash near-duplicate screening
- pHash 후보 이미지의 직접 확인

초기 split에서는 서로 다른 filename-derived group에 속하지만 실제로 동일한 cross-split 이미지 한 쌍을 발견했다.

원본 데이터는 그대로 보존하면서 중복본 한 장을 실험 대상에서 제외하고 전체 dataset split을 다시 생성했다.

재생성된 Clean Split에서는 다음을 확인했다.

```text
Filename-derived group overlap       : 0
Cross-split SHA-256 exact duplicates : 0
Cross-split pHash Distance 0         : 0
```

Clean Split을 사용한 Scratch 재학습과 random seed 고정 실험에서도 높은 Validation 성능이 다시 관찰되었다.

따라서 발견된 cross-split duplicate 한 쌍만으로 초기의 높은 Validation 성능을 설명하기는 어렵다고 판단했다.

다만 filename-derived group ID는 공식 patient identifier가 아니므로 실제 patient-level independence를 완전히 검증할 수 없으며, duplicate 검사는 dataset-specific shortcut learning 가능성을 직접 평가하는 방법도 아니다.

따라서 현재 split을 확인 가능한 cross-split group overlap 및 duplicate 문제를 줄인 **Clean Split**으로 사용하되, 실제 pathology-related feature 학습 여부와 외부 데이터에 대한 일반화 가능성은 별개의 문제로 구분한다.

모델 비교와 최종 Test 결과는 프로젝트 `README.md`, Grad-CAM++ 분석은 `docs/gradcam_analysis.md`에서 별도로 관리한다.
