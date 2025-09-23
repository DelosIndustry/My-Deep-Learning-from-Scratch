---

# Deep Learning from Scratch (DLFS)

딥러닝을 **NumPy만으로 직접 구현**하며 핵심 개념과 구현 감각을 익히는 학습용 리포지토리입니다.
목표: 자동미분 → 최적화 → MLP/CNN/간단한 Transformer까지 “바닥부터” 만들어 보기.

---

## 1) 폴더 구조 (최소형)

```
.
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ dlfs/
│  │  ├─ autograd.py        # 간단 역전파 엔진
│  │  ├─ nn.py              # Linear/Conv/ReLU/Softmax 등
│  │  ├─ optim.py           # SGD/Adam
│  │  ├─ data.py            # MNIST/CIFAR-10 로더(간단)
│  │  └─ trainer.py         # 훈련/평가 루프
│  └─ experiments/
│     ├─ mlp_mnist.py       # MLP로 MNIST 분류
│     └─ cnn_cifar10.py     # CNN으로 CIFAR-10 분류
└─ tests/
   └─ test_autograd.py      # 기초 단위 테스트 예시
```

> 필요에 따라 `notebooks/`를 추가하거나 `transformer_toy.py`를 더해 확장하세요.

---

## 2) 설치

```bash
# (권장) 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필수 패키지
pip install -r requirements.txt
```

**requirements.txt 예시**

```
numpy
matplotlib
tqdm
pytest
requests   # (데이터 다운로드용, 선택)
```

---

## 3) 빠른 시작

### MNIST (MLP)

```bash
python -m src.experiments.mlp_mnist --epochs 5 --batch-size 128 --lr 1e-3 --seed 42
```

### CIFAR-10 (CNN)

```bash
python -m src.experiments.cnn_cifar10 --epochs 20 --batch-size 128 --lr 3e-4 --augment --seed 42
```

**공통 옵션(예시)**

* `--seed 42` 재현성 고정
* `--save-dir runs/exp001` 체크포인트/로그 저장
* `--device cpu` (NumPy 기반이라 기본 CPU 사용)

---

## 4) 학습 순서(체크리스트)

* [ ] **수학 기초**: 미분/연쇄법칙, 행렬곱
* [ ] **Autograd**: 계산 그래프 / `backward()` / 브로드캐스팅
* [ ] **레이어·함수**: Linear, Conv, ReLU, Softmax, CE Loss
* [ ] **최적화**: SGD(+Momentum), Adam
* [ ] **훈련 루프**: 에폭/배치, 로그/체크포인트
* [ ] **모델**: MLP → CNN → (선택) Transformer toy
* [ ] **데이터**: MNIST/CIFAR-10 로더·정규화·간단 증강
* [ ] **테스트**: Autograd 수치미분 대조, 레이어 출력형상 검증

---

## 5) 핵심 파일 간단 설명

* `dlfs/autograd.py`

  * 텐서 노드에 연산 기록 → `backward()`로 그래프 역전파
  * 수치미분과의 오차로 기본 검증

* `dlfs/nn.py`

  * `Linear`, `Conv2d`, `ReLU`, `Softmax`, `CrossEntropyLoss` 등 최소 구성
  * 파라미터 수집용 간단 인터페이스 제공

* `dlfs/optim.py`

  * `SGD`, `Adam` 구현 (lr, weight\_decay, grad\_clip 등 기본 옵션)

* `dlfs/data.py`

  * 간단한 **MNIST/CIFAR-10** 다운로드 & numpy 배열 반환
  * 정규화·(선택) 수평 뒤집기 등 가벼운 증강

* `dlfs/trainer.py`

  * 훈련/평가 루프, 체크포인트 저장/로드, 로그 프린트

---

## 6) 기준 실험값(참고)

| Task     | Model   | Epochs | Batch | LR   | Val(대략) |
| -------- | ------- | ------ | ----- | ---- | ------- |
| MNIST    | MLP(2층) | 5      | 128   | 1e-3 | 97–98%  |
| CIFAR-10 | CNN(소형) | 20     | 128   | 3e-4 | 65–70%  |

> CPU/시드/버전에 따라 달라질 수 있습니다.

---

## 7) 테스트

```bash
pytest -q
# 또는 특정 테스트만
pytest tests/test_autograd.py::TestBackward -q
```

---

## 8) 팁

* **시드 고정**으로 실험 비교를 공정하게 (`numpy.random.seed(seed)`).
* **작게 시작**: 오버핏이 나더라도 일단 학습이 되는지 확인 → 점진 개선.
* **가독성 우선**: 성능 최적화(Numba/JAX 등)는 별도 브랜치로.

---

## 9) 로드맵(선택)

* [ ] BatchNorm/LayerNorm/Dropout 추가
* [ ] 스케줄러(Warmup/Cosine) 및 AdamW
* [ ] Tiny Transformer 예제(`transformer_toy.py`)
* [ ] 간단 시각화(손실곡선 저장)

---

## 10) 라이선스 & 기여

* **License:** MIT
* **Contributing:** 이슈/PR 환영합니다. 작은 단위로 변경하고 간단한 테스트를 포함해 주세요.

---

---
