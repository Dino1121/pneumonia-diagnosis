# 실행 및 재현 안내 (Reproducibility)

이 문서는 `README.md`, `train.py`, `scripts/`, 실험별 `config.json`, `history.csv`, `convergence.json`, `test_results.json`을 기준으로 실행 방법을 정리한다. 동일한 수치를 재현하려면 원본 데이터 분할과 실행 환경을 유지해야 한다.

## 1. 데이터 및 저장소 상태

- 사용 데이터셋 이름: `Chest X-Ray Images (Pneumonia)` (README에 기재된 이름)
- 저장소에 원본 이미지 파일 `data/raw/chest_xray/`는 포함되지 않는다.
- 데이터셋 URL: [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). 실제 이용 조건 및 이미지 재배포 요건은 해당 페이지에서 확인해야 한다.
- 저장소에는 `data/splits/group_split.csv`가 포함돼 있으며, 최종 비교에 실제 사용된 Split이므로 그대로 이용한다.
- `*.pth`는 `.gitignore`에 포함돼 있어 최종 체크포인트를 clone만으로 얻을 수 없다. 대표 모델 체크포인트는 로컬에서 별도로 관리하며 이 저장소에는 포함하지 않는다.

예상 입력 경로:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

`group_split.csv`의 `filepath`는 위 원본 구조를 전제로 한 **저장소 상대경로**다. 원본 이미지는 기록된 경로와 일치하도록 배치해야 한다.

> 주의: `python src/data/split_dataset.py`를 실행하면 `data/splits/group_split.csv`를 덮어쓴다. 코드의 `folder.glob('*')` 탐색 순서는 OS/파일 시스템에 따라 변할 수 있으므로, `random_state=42`만으로 예전과 동일한 Split을 보장할 수 없다. 원본 CSV를 보존하고 재생성하지 않는 것이 이 프로젝트 결과를 해석하는 올바른 방법이다.

## 2. 설치 환경

프로젝트 루트에서:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt`는 프로젝트의 주요 패키지 목록이다. **Grad-CAM++ 실행 당시 확인된 라이브러리 버전은 7절과 [`manifest.json`](assets/gradcam/manifest.json)에 기록했다.** 이 버전 기록은 모델 학습 환경 전체에 대한 lockfile이 아니다. CUDA용 PyTorch 설치는 실행할 GPU와 운영체제에 맞게 구성한다.

작업 디렉터리를 저장소 루트로 두고 명령을 실행한다. 예제는 Unix/Windows 공통의 `python` 명령을 사용하며 환경에 따라 `py` 또는 가상환경 Python으로 대체할 수 있다.

## 3. Scratch `exp_008` 설정에 대응하는 학습 명령

원본 [`experiments/scratch/exp_008/config.json`](../experiments/scratch/exp_008/config.json) 기준:

```bash
python train.py \
  --model scratch \
  --optimizer adamw \
  --lr 0.0001 \
  --weight_decay 0.001 \
  --batch_size 32 \
  --epochs 100 \
  --scheduler plateau \
  --scheduler_factor 0.1 \
  --scheduler_patience 10 \
  --min_lr 0.000001
```

위 줄바꿈 문법은 Bash 기준이다. Windows PowerShell에서는 한 줄로 입력하거나 PowerShell의 줄 연결 문법을 사용한다.

Python 코드에서 seed는 `42`로 고정돼 있다(`train.py`의 `set_seed(42)`). 현재 CLI에는 `--seed` 인자가 없다. 동일 seed여도 환경과 비결정적 연산 등으로 결과가 완전히 일치한다고 보장할 수 없다.

## 4. Transfer Learning `exp_003` 설정에 대응하는 학습 명령

원본 [`experiments/transfer/exp_003/config.json`](../experiments/transfer/exp_003/config.json) 기준:

```bash
python train.py \
  --model transfer \
  --optimizer adamw \
  --freeze_epochs 7 \
  --freeze_lr 0.001 \
  --finetune_lr 0.0001 \
  --weight_decay 0.01 \
  --batch_size 32 \
  --epochs 100 \
  --scheduler plateau \
  --scheduler_factor 0.1 \
  --scheduler_patience 10 \
  --min_lr 0.000001
```

1~7 epoch은 FC-only, 8~100 epoch은 full fine-tuning이며 `IMAGENET1K_V2` 가중치를 사용한다. Torchvision이 필요에 따라 사전학습 가중치를 다운로드할 수 있다.

> **중요:** `train.py`는 각 실행에서 현재 폴더 번호를 확인해 새 `exp_NNN` 폴더를 생성한다. 위 명령으로 기존 `exp_008` 또는 `exp_003`이 그대로 만들어지는 것은 아니며, 기존 실험과 숫자까지 똑같이 재현됐다는 뜻도 아니다. 새 폴더의 `config.json`, `history.csv`, `val_predictions.csv`, `best_model.pth` 등을 확인해야 한다.

## 5. 모델 선택과 수렴 지표

각 방식의 최종 2×2 탐색 기록은 다음과 같다.

| 방식 | 최종 탐색 폴더 | 대표 모델 | 대표 선정 기준 |
| --- | --- | --- | --- |
| Scratch | `exp_008`, `exp_009`, `exp_010`, `exp_011` | `exp_008` | Best Validation F1 |
| Transfer | `exp_002`, `exp_003`, `exp_005`, `exp_004` | `exp_003` | Best Validation F1 |

`experiments/scratch/exp_001`~`exp_007`, `experiments/transfer/pilot_fc_freeze/` 및 `experiments/transfer/exp_001`은 초기 탐색·pilot·이전 실험의 기록이다. **실험 폴더를 이름 순서대로 모두 최종 비교 대상으로 간주하지 않는다.**

대표 실험 로그에서 수렴 지표를 계산한 기존 분석의 명령 형식:

```bash
python scripts/analyze_convergence.py \
  --history experiments/scratch/exp_008/history.csv \
  --name "Scratch exp_008" \
  --output results/reproduced_scratch_convergence.json

python scripts/analyze_convergence.py \
  --history experiments/transfer/exp_003/history.csv \
  --name "Transfer exp_003" \
  --output results/reproduced_transfer_convergence.json

python scripts/compare_models.py \
  --scratch results/reproduced_scratch_convergence.json \
  --transfer results/reproduced_transfer_convergence.json \
  --output results/reproduced_comparison.csv
```

이 명령은 **이미 커밋된 `convergence.json`과 `results/comparison.csv`를 보존**하도록 별도 출력 경로를 사용한다.

T99는 모델별 Best Validation F1의 99% 상대 임계값을 만족하고, 해당 시작 epoch 포함 10 epoch 중 8 epoch 이상 임계값을 유지하는 가장 이른 시점이다. 사후 분석 지표이며 wall-clock 시간 비교가 아니다.

## 6. 독립 Test 평가

`evaluate_test.py`는 선택한 실험 폴더의 `best_model.pth`를 읽고 결과를 **같은 실험 폴더의 `test_results.json`과 `test_predictions.csv`에 저장(덮어쓰기)** 한다.

```bash
python scripts/evaluate_test.py --model scratch --experiment exp_008
python scripts/evaluate_test.py --model transfer --experiment exp_003
```

위는 **기존 체크포인트를 보유하고 있고 저장된 평가 파일을 백업한 경우에만 실행해야 하는 명령 형식**이다. 공개 저장소에는 체크포인트가 없으므로 그대로 실행하면 실패한다. 새로 학습한 모델은 새로 생성된 실험 번호를 지정해야 한다. 새로 학습한 모델의 Test 평가는 기존 보고된 0.9836/0.9730과 같다는 보장이 없다.

기존 결과는 다음에 보존돼 있다.

- [`Scratch test_results.json`](../experiments/scratch/exp_008/test_results.json)
- [`Transfer test_results.json`](../experiments/transfer/exp_003/test_results.json)
- 각 폴더의 `test_predictions.csv`

## 7. Grad-CAM++

`compare_gradcam.py`의 기본 체크포인트 위치:

```text
experiments/scratch/exp_008/best_model.pth
experiments/transfer/exp_003/best_model.pth
```

원본 Dataset 및 해당 체크포인트가 준비돼 있을 때:

```bash
python scripts/compare_gradcam.py --seed 42 --num_images 10
```

필요하면 `--scratch_checkpoint`, `--transfer_checkpoint`, `--csv`, `--device`, `--output_dir`, `--filename` 옵션으로 입력을 바꿀 수 있다. 기본 출력은 `results/gradcam/<UTC timestamp>/`이며 그림 PNG, raw CAM NPZ 및 `manifest.json`을 저장한다.

Grad-CAM++ 실행은 **seed 123, 20개 Test 표본(NORMAL 10, PNEUMONIA 10)**을 대상으로 수행했다. README와 상세 분석 문서에는 이 실행의 비교 PNG 3개를 대표 사례로 수록했다([그림별 기록](assets/gradcam/README.md)). 20개 표본의 예측 결과 및 실행 메타데이터는 [`manifest.json`](assets/gradcam/manifest.json)에 보존했다.

```text
실제 기록된 실행 인자: --seed 123 --num_images 20
Sampling             : class-balanced random (NORMAL 10 / PNEUMONIA 10)
Split                : test (재분할한 내부 Test)
Device               : cuda
Target               : predicted-class pre-softmax logit
Target layer         : layer4[-1]
Torch                : 2.14.0+cu126
Torchvision          : 0.29.0+cu126
grad-cam             : 1.5.7
NumPy                : 2.4.6
Matplotlib           : 3.11.2
```

**위 버전은 Grad-CAM++ 실행 환경 기준**이며 모델 학습 환경 전체의 버전 명세는 아니다. 공개용 [`manifest.json`](assets/gradcam/manifest.json)은 로컬 절대경로를 저장소 상대경로로 바꾼 사본이며, seed·모델 해시·표본 정보는 유지했다. 동일한 표본 선택을 위한 실행 명령은 다음과 같다.

```bash
python scripts/compare_gradcam.py --seed 123 --num_images 20
```



## 8. 대표 모델 체크포인트 식별 및 배치

대표 모델의 `best_model.pth` 두 개는 Git에 추가하지 않고 각 모델의 실험 폴더에서 관리한다. 아래 SHA-256 값은 Grad-CAM++ 실행 manifest의 기록과 일치한다.

| 로컬 보관 파일명 예시 | 저장할 경로 | SHA-256 | 모델 식별 |
| --- | --- | --- | --- |
| `best_model.pth` | `experiments/scratch/exp_008/best_model.pth` | `3a25b53f706500a33fb1ebe4f5029c4e3dbab5cdaa51d643ed5bbd7c11e2be42` | Scratch `exp_008` |
| `best_model(1).pth` | `experiments/transfer/exp_003/best_model.pth` | `002ca5d21f73240c15ba0454f1b91ab5b04b49902e2a4e9151377b296426b288` | Transfer `exp_003` |

두 체크포인트의 SHA-256은 Grad-CAM++ 실행 manifest에 기록한 값과 일치한다. 실험 번호 및 가중치 파일의 식별에 이 해시를 사용한다.

Windows PowerShell에서 각 파일을 별도 위치에 보관한 경우, 파일명을 구분해 위 표의 경로로 직접 복사한다. 프로젝트의 `.gitignore`에 `*.pth`가 포함되어 있으므로 `git add -f`로 강제로 커밋하지 않는다. 원본 CSV와 기존 결과 JSON도 건드리지 않는다. 기존 결과 파일의 덮어쓰기를 피하려면 평가 스크립트 대신 먼저 원본과 파일 해시만 확인하는 것이 안전하다.

체크포인트는 파일명만으로 식별하지 않고 SHA-256과 모델 실험 경로를 함께 확인한다.

## 9. 코드·기록 위치 및 재현성 한계

| 확인 대상 | 경로 |
| --- | --- |
| 대표 설정 | `experiments/scratch/exp_008/config.json`, `experiments/transfer/exp_003/config.json` |
| Epoch별 원본 기록 | 위 실험 폴더의 `history.csv` |
| Validation 예측 로그 | 위 실험 폴더의 `val_predictions.csv` |
| 수렴 결과 | 위 실험 폴더의 `convergence.json` 및 `results/comparison.csv` |
| 최종 내부 Test 기록 | 위 실험 폴더의 `test_results.json`, `test_predictions.csv` |
| 데이터 중복·분할 근거 | `docs/data_integrity_audit.md`, `data/splits/group_split.csv` |
| Grad-CAM++ 분석 | `docs/gradcam_analysis.md`, `scripts/compare_gradcam.py` |

이 문서의 명령은 현재 코드와 대표 실험 설정을 기준으로 작성했다. 정확한 재현을 위해서는 동일한 데이터 분할, 체크포인트 및 실행 환경을 사용해야 하며, 라이브러리나 하드웨어 차이로 결과가 달라질 수 있다.
