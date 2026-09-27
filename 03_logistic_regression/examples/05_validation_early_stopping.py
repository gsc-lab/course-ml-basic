"""
Logistic Regression - Validation 과 Early Stopping
04번의 문제: "언제 멈출까" 를 test 데이터를 보고 정했다.
그러면 test 는 더 이상 처음 보는 데이터가 아니고, test 점수는 부풀려진 점수다.

해결: 데이터를 세 묶음으로 나눈다
  train       학습용 — gradient 를 계산한다
  validation  학습 중 들여다보는 용도 — 언제 멈출지 같은 "설정" 을 고른다
  test        최종 채점용 — 모든 결정이 끝난 뒤 딱 한 번만 본다

Early Stopping
  매 epoch validation loss 를 재고, 가장 낮았던 시점의 w, b 를 따로 저장해 둔다.
  validation loss 가 patience 번의 epoch 동안 나아지지 않으면 학습을 멈춘다.
  최종 모델은 멈춘 시점의 w 가 아니라 "저장해 둔 가장 좋았던 w" 다.

평가 지표 — 정확도만으로는 부족하다
  "암을 놓친 것" 과 "정상을 암이라 한 것" 은 무게가 다른 오답인데,
  정확도는 둘을 구별하지 못한다.
  sklearn 의 classification_report 가 클래스별로 두 숫자를 보여준다.
    precision: 그 클래스라고 예측한 것 중 실제로 그 클래스인 비율
    recall:    실제 그 클래스 중 그 클래스라고 맞힌 비율
  이 데이터에서는 malignant(0) 행의 recall 이 "암 환자 중 암이라고 맞힌 비율" 이다.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 1. 데이터 준비 — train / validation / test = 60% / 20% / 20%
#    train_test_split 을 두 번 쓴다. 두 번 다 stratify 로 클래스 비율을 유지한다.
#    표준화 통계는 train 에서만 구해 세 묶음 모두에 적용한다.
# ============================================================
X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=10   # 0.8 × 0.25 = 0.2
)

x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
X_train = (X_train - x_mean) / x_std
X_val = (X_val - x_mean) / x_std
X_test = (X_test - x_mean) / x_std

N, D = X_train.shape
print(f"train: {N}건, validation: {len(X_val)}건, test: {len(X_test)}건")


# ============================================================
# 2. 함수
#    sigmoid, bce_loss 는 04번과 같다.
#    이번에는 학습 루프를 두 번 돌리므로(early stopping → 비교 실험)
#    "한 걸음 학습" 과 "표 한 줄 출력" 도 함수로 뺀다.
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce_loss(H, y, eps=1e-15):
    H = np.clip(H, eps, 1 - eps)
    return -np.mean(y * np.log(H) + (1 - y) * np.log(1 - H))


def gradient_step(w, b):
    """Gradient Descent 한 걸음 — 01~04번 학습 루프의 몸통과 같다."""
    H = sigmoid(X_train @ w + b)
    w -= lr * X_train.T @ (H - y_train) / N
    b -= lr * np.mean(H - y_train)
    return w, b


def print_row(epoch, w, b):
    train_loss = bce_loss(sigmoid(X_train @ w + b), y_train)
    val_loss = bce_loss(sigmoid(X_val @ w + b), y_val)
    print(f"{epoch:>7} | {train_loss:10.4f} {val_loss:9.4f}")


# ============================================================
# 3. 하이퍼파라미터 & 초기화
# ============================================================
lr = 1.0
epochs = 30_000
patience = 500      # validation loss 가 이만큼의 epoch 동안 나아지지 않으면 멈춘다
checkpoints = [30, 100, 300, 1000, 3000, 10_000, 30_000]

w = np.zeros(D)
b = 0.0

# 지금까지 validation loss 가 가장 낮았던 시점을 기억해 둘 변수
best_val_loss = np.inf
best_epoch = 0
best_w, best_b = None, None

# ============================================================
# 4. 학습 루프 — Early Stopping
#    매 epoch: 한 걸음 학습 → validation loss 측정
#             → 최저 기록이면 w, b 저장
#             → patience 동안 개선이 없으면 중단
#    test 데이터는 이 루프 안에서 한 번도 쓰지 않는다.
# ============================================================
print(f"\n{'epoch':>7} | {'train_loss':>10} {'val_loss':>9}")
print("-" * 32)

for epoch in range(1, epochs + 1):
    w, b = gradient_step(w, b)

    val_loss = bce_loss(sigmoid(X_val @ w + b), y_val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # 주의: best_w = w 라고 쓰면 같은 배열을 가리켜서 이후 w 가 바뀔 때 같이 바뀐다.
        #       반드시 복사본을 저장한다.
        best_w, best_b = w.copy(), b
    elif epoch - best_epoch >= patience:
        print(f"   [early stopping] epoch {epoch}: {patience} epoch 동안 개선 없음 → 중단")
        break

    if epoch in checkpoints:
        print_row(epoch, w, b)

stop_epoch = epoch
print(f"   validation loss 최저: {best_val_loss:.4f} (epoch {best_epoch}) → 이 시점의 w 를 사용")

# ============================================================
# 5. 비교 실험 — 멈추지 않았다면?
#    early stopping 이 없었을 때를 보기 위해, 멈춘 지점부터 끝까지 계속 학습한다.
#    표가 이어지므로 val_loss 가 계속 오르는 것을 그대로 볼 수 있다.
# ============================================================
for epoch in range(stop_epoch + 1, epochs + 1):
    w, b = gradient_step(w, b)
    if epoch in checkpoints:
        print_row(epoch, w, b)
print("-" * 32)

# ============================================================
# 6. 최종 평가 — test 데이터는 여기서 딱 한 번
#    early stopping 으로 고른 w 와, 끝까지 돌린 w 를 나란히 비교한다.
# ============================================================
target_names = ["malignant(0)", "benign(1)"]   # 라벨 순서 (0, 1) 대로 이름을 붙인다


def evaluate(title, w, b):
    H_test = sigmoid(X_test @ w + b)
    y_pred = (H_test >= 0.5).astype(int)          # 확률 → 0/1 라벨
    print(f"\n[{title}]  test loss: {bce_loss(H_test, y_test):.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


evaluate(f"Early stopping — epoch {best_epoch} 의 w", best_w, best_b)
evaluate(f"끝까지 학습 — epoch {epochs} 의 w", w, b)

# ============================================================
# 7. 결과 읽는 법
#    - malignant(0) 행의 recall   : 실제 암 환자 중 암이라고 맞힌 비율 → 가장 중요한 숫자
#    - malignant(0) 행의 precision: 암이라고 한 사람 중 진짜 암인 비율
#                                   → 낮으면 불필요한 정밀검사가 늘어난다
#    - support   : 각 클래스의 실제 건수
#    - macro avg : 두 클래스를 같은 비중으로 평균한 값. 건수가 적은 클래스도 똑같이 반영된다.
#
#    최저 epoch 은 몇 번이 "정답" 인가?
#    이번 실행에서는 43 이었지만 04번에서는 500 근처였다. 이 숫자는 어떤 환자가
#    train 과 validation 에 들어갔느냐에 따라 크게 달라진다 (random_state 만 바꿔도 수십~수백).
#    validation 의 목적은 정확한 epoch 숫자를 찾는 것이 아니라,
#    test 를 건드리지 않고 "과적합이 시작되기 전의 w" 를 건지는 것이다.
#    (이 흔들림을 줄이는 방법이 교차 검증(cross-validation) 이다.)
#
#    한 걸음 더
#    early stopping 은 과적합을 "감지해서 멈추는" 방법이다.
#    w 가 커지는 것 자체를 막는 방법도 있다 — loss 에 λ·|w|² 을 더하는 L2 정규화.
#    sklearn 의 LogisticRegression 은 기본으로 L2 정규화가 켜져 있어서(C=1.0),
#    NumPy 로 만든 이 모델과는 결과가 조금 다를 수 있다.
# ============================================================
