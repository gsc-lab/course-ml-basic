"""
Logistic Regression - Sonar 데이터로 Early Stopping 다시 보기
05번의 코드를 07번에서 살펴본 sonar 데이터에 그대로 적용한다.
유방암 데이터에서는 미미하게 나타났던 과적합이 이 데이터에서는 뚜렷하게 관찰된다.

달라지는 것
  데이터만 바뀐다. 모델(sigmoid), 손실(BCE), 옵티마이저(GD), early stopping 로직은 05번과 같다.
  train 124건에 파라미터 61개로, 파라미터 하나당 데이터가 2건에 불과하다.
  이 조건에서는 모델이 일반적인 규칙을 찾기보다 train 데이터 자체를 기억하는 방향으로 학습된다.

관찰 포인트
  1) validation loss 는 100 epoch 이내에 최저점에 도달한 뒤 다시 상승하며, 이후 감소하지 않는다.
  2) 학습을 끝까지 진행하면 train loss 는 0 에 가까워지지만 validation loss 는 최저점의 2~3배가 된다.
  3) early stopping 시점의 w 와 끝까지 학습한 w 를 test 데이터로 비교하면 성능 차이가 분명하다.
  4) patience 는 05번의 500 이 아니라 50 이다. best epoch 이 수십 단위이면 patience 도 같은 자릿수가 적절하다.
     full-batch GD 는 validation 곡선이 매끄럽게 변하므로 patience 를 작게 두어도 된다.
     큰 값이 필요한 경우는 mini-batch 학습처럼 곡선이 흔들릴 때다.

평가 지표
  Mine(1) 이 Positive 다. 기뢰를 바위로 판정하는 오류(FN)의 비용이 크므로 MINE 행의 recall 을 확인한다.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 1. 데이터 준비 — 07번과 동일 (Mine = 1, 60/20/20 분할, train 통계로 표준화)
# ============================================================
ds = fetch_openml("sonar", version=1, as_frame=True)
X = ds.data.to_numpy(dtype=float)
y = (ds.target == "Mine").astype(int).to_numpy()

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
print(f"train: {N}건, validation: {len(X_val)}건, test: {len(X_test)}건, 파라미터: {D + 1}개")


# ============================================================
# 2. 함수 — 05번과 같다
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, w, b):
    return sigmoid(x @ w + b)


def bce_loss(H, y, eps=1e-15):
    H = np.clip(H, eps, 1 - eps)                # log(0) 방지
    return -np.mean(y * np.log(H) + (1 - y) * np.log(1 - H))


def gradient_step(w, b):
    """Gradient Descent 한 걸음"""
    H = predict(X_train, w, b)
    w -= lr * X_train.T @ (H - y_train) / N
    b -= lr * np.mean(H - y_train)
    return w, b


def print_row(epoch, w, b):
    train_loss = bce_loss(predict(X_train, w, b), y_train)
    val_loss = bce_loss(predict(X_val, w, b), y_val)
    train_acc = np.mean((predict(X_train, w, b) >= 0.5) == y_train)
    print(f"{epoch:>7} | {train_loss:10.4f} {val_loss:9.4f} | {train_acc:9.3f}")


# ============================================================
# 3. 하이퍼파라미터 & 초기화
#    lr 은 05번의 1.0 대신 0.1 을 사용한다. 학습 속도를 늦춰 validation loss 가
#    최저점을 지나 상승으로 돌아서는 구간이 표에 나타나도록 한다.
# ============================================================
lr = 0.1
epochs = 10_000
patience = 50
checkpoints = [10, 30, 50, 100, 150, 300, 1000, 3000, 10_000]

w = np.zeros(D)
b = 0.0

best_val_loss = np.inf
best_epoch = 0
best_w, best_b = None, None

# ============================================================
# 4. 학습 루프 — Early Stopping (05번과 동일)
# ============================================================
print(f"\n{'epoch':>7} | {'train_loss':>10} {'val_loss':>9} | {'train_acc':>9}")
print("-" * 44)

for epoch in range(1, epochs + 1):
    w, b = gradient_step(w, b)

    val_loss = bce_loss(predict(X_val, w, b), y_val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_w, best_b = w.copy(), b             # 복사본을 저장한다. w 는 이후에도 계속 갱신된다
    elif epoch - best_epoch >= patience:
        print(f"   [early stopping] epoch {epoch}: {patience} epoch 동안 개선 없음 → 중단")
        break

    if epoch in checkpoints:
        print_row(epoch, w, b)

stop_epoch = epoch
print(f"   validation loss 최저: {best_val_loss:.4f} (epoch {best_epoch}) → 이 시점의 w 를 사용")

# ============================================================
# 5. 비교 실험 — early stopping 을 하지 않은 경우
#    중단 지점부터 끝까지 학습을 이어 간다. train_acc 는 1.000 에 도달하고
#    val_loss 는 계속 상승한다. 모델이 train 124건을 그대로 기억한 상태다.
# ============================================================
for epoch in range(stop_epoch + 1, epochs + 1):
    w, b = gradient_step(w, b)
    if epoch in checkpoints:
        print_row(epoch, w, b)
print("-" * 44)

# ============================================================
# 6. 최종 평가 — test 데이터는 여기서 한 번만 사용한다
# ============================================================
target_names = ["ROCK(0)", "MINE(1)"]


def evaluate(title, w, b):
    H_test = predict(X_test, w, b)
    y_pred = (H_test >= 0.5).astype(int)
    print(f"\n[{title}]  test loss: {bce_loss(H_test, y_test):.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


evaluate(f"Early stopping — epoch {best_epoch} 의 w", best_w, best_b)
evaluate(f"끝까지 학습 — epoch {epochs} 의 w", w, b)

# ============================================================
# 7. 결과 읽는 법
#    - 유방암(05번)에서는 두 모델의 test 결과가 거의 같았지만, 여기서는 차이가 크다.
#      끝까지 학습한 모델은 test loss 가 몇 배로 커지고, MINE recall 도 낮아진다.
#    - test 가 42건이므로 패턴 하나가 정확도 2.4%p 에 해당한다. 한 번의 결과만으로
#      판단하기보다 random_state 를 바꿔 가며 여러 분할에서 확인하는 것이 좋다.
#      어느 분할에서든 val_loss 는 최저점 대비 2배 이상 상승한다.
#    - 이 데이터에서 선형 모델의 정확도는 70~85% 수준에 머문다. 결정 경계가 직선 하나인
#      로지스틱 회귀로는 그 이상 분리하기 어렵다. 원래 신경망 연구를 위해 만들어진
#      데이터이며, 신경망 단원에서 다시 다룬다.
# ============================================================
