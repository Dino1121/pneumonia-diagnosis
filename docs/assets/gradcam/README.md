# Grad-CAM++ 비교 이미지 및 실행 기록

이 폴더에는 기존 Grad-CAM++ 실험으로 생성한 비교 이미지 중 **3개 사례**를 수록했다. 각 그림은 원본 입력, Scratch `exp_008`, Transfer `exp_003`의 CAM을 함께 보여준다.

| 저장한 그림 | 원본 영상 파일명 | 정답 | Scratch `exp_008` 예측/CAM target | Transfer `exp_003` 예측/CAM target | 분류 결과 |
|---|---|---|---|---|---|
| `IM-0223-0001.png` | `IM-0223-0001.jpeg` | NORMAL | NORMAL / NORMAL | NORMAL / NORMAL | both_correct |
| `IM-0209-0001.png` | `IM-0209-0001.jpeg` | NORMAL | NORMAL / NORMAL | NORMAL / NORMAL | both_correct |
| `person3_virus_16.png` | `person3_virus_16.jpeg` | PNEUMONIA | PNEUMONIA / PNEUMONIA | PNEUMONIA / PNEUMONIA | both_correct |

세 원본 영상의 정답 클래스와 `test` 배정은 저장소의 `data/splits/group_split.csv`에서 확인했다. 두 모델의 예측 클래스 역시 각각 `experiments/scratch/exp_008/test_predictions.csv`, `experiments/transfer/exp_003/test_predictions.csv`에서 일치함을 확인했다. 구체적으로 저장된 `p_pneumonia` 값은 다음과 같다.

| 원본 filename | Scratch P(PNEUMONIA) | Transfer P(PNEUMONIA) |
|---|---:|---:|
| `IM-0223-0001.jpeg` | 0.0000066010 | 0.0000027442 |
| `IM-0209-0001.jpeg` | 0.0000001257 | 0.0000026079 |
| `person3_virus_16.jpeg` | 0.9999996424 | 0.9999763966 |

이 예측 기록과 그림의 예측 방향 및 반올림된 probability 표기는 서로 모순되지 않는다. 체크포인트 식별 정보는 아래 manifest의 SHA-256 기록을 사용한다.

모든 그림에는 `P(예측 class): 1.000`으로 표시되어 있다. 이는 **출력 시 소수 셋째 자리까지 반올림한 값**이지 확률이 수학적으로 1임을 뜻하지 않는다.

### Grad-CAM++ 실행 manifest

[공개용 manifest](manifest.json)은 실제 Grad-CAM++ 실행 기록을 바탕으로 **로컬 절대경로만 저장소 상대경로로 변환**한 사본이다. 메서드, 선택 조건, 모델 체크포인트 해시, 20개 표본별 예측 결과는 원본 기록을 유지했다.

| 실행 항목 | 확인 내용 |
|---|---|
| 데이터 | 내부 held-out Test split |
| 표본 선택 | Class-balanced random, **20장 (NORMAL 10, PNEUMONIA 10), seed 123** |
| CAM target / layer | 각 모델의 predicted-class pre-softmax logit / `layer4[-1]` |
| 장치 | CUDA |
| Checkpoint (Scratch) | `experiments/scratch/exp_008/best_model.pth` · SHA-256 `3a25b53f706500a33fb1ebe4f5029c4e3dbab5cdaa51d643ed5bbd7c11e2be42` |
| Checkpoint (Transfer) | `experiments/transfer/exp_003/best_model.pth` · SHA-256 `002ca5d21f73240c15ba0454f1b91ab5b04b49902e2a4e9151377b296426b288` |
| `group_split.csv` 해시(원본 manifest 기록) | `f27c2ff95f82de31a04d9414a9f2da2d9bf44bca3e2255a7428840e635b2c926` |

Grad-CAM++ 실행에 사용한 Scratch `exp_008`과 Transfer `exp_003` 체크포인트는 위 SHA-256 값으로 구분한다.

원본 manifest의 `samples` 20개 중 **19개는 `both_correct`, 1개는 `scratch_only_correct`**로 기록돼 있다. 제공된 세 PNG는 모두 `both_correct`이고 아래와 같이 연결된다.

| 공개 파일 | 원본 manifest의 `figure` 값 | 원본 manifest 표본 순서 |
|---|---|---:|
| `IM-0209-0001.png` | `0000_IM-0209-0001.png` | 1/20 |
| `IM-0223-0001.png` | `0003_IM-0223-0001.png` | 4/20 |
| `person3_virus_16.png` | `0011_person3_virus_16.png` | 12/20 |

세 표본의 실제 클래스, 두 모델의 예측 클래스와 CAM target은 manifest 및 `test_predictions.csv`와 일치한다. 확률의 끝자리 차이는 부동소수점 계산과 반올림 과정에서 발생할 수 있다.

### 선택 및 해석의 한계

- 공개 README에는 NORMAL과 PNEUMONIA를 각각 하나씩 보여주기 위해 `IM-0223-0001`과 `person3_virus_16`을 선택했다. `IM-0209-0001`까지 제공된 3개 이미지 모두 상세 문서에 수록했다. 이는 **실제 20개 무작위 표본 중 문서에 수록한 사례**이며 전체 Test의 통계적 대표성을 주장하는 표본은 아니다.
- 전체 20개 표본의 예측 결과와 실행 설정은 manifest에 보존하고, 이 문서에는 그중 3개 사례의 PNG만 제시한다.
- 서로 다른 예측 class의 Grad-CAM은 서로 다른 target logit의 attribution이므로 동일 class 기반 CAM 비교가 아니다. 이 세 공개 이미지는 양쪽 모델의 CAM target이 동일한 `NORMAL` 또는 `PNEUMONIA`이다.
- CAM은 모델/입력마다 개별 정규화됐다. 색 강도는 confidence·병변 중증도·임상적으로 유효한 localization을 입증하지 않는다.
- PNG는 소아 흉부 X-ray 영상을 포함한다. 공개 시 데이터셋 이용 조건·이미지 재배포 조건은 별도 확인해야 한다.

### 그림 파일 SHA-256 (현재 공개 이미지)

```text
69d1aebacd1d4780c16f4deab6bb5b545f08349efb3aab79ce9d9fc9535e8015  IM-0223-0001.png
f24eef56cf48308c3bd860c2a63df07cd1f3c88869a50dd15c78e32dde25021b  IM-0209-0001.png
6c82d04c6b4bc31435a09d117924b8f0ba035d31529e6f50b752580ccd4aeeaf  person3_virus_16.png
```

이 해시는 공개된 PNG를 추적하기 위한 것이며, **원래 Grad-CAM++ 실행의 manifest/checkpoint 해시를 대신하지 않는다.**
