"""
Logistic Regression - 과적합(Overfitting) 발견하기
03번과 같은 모델을 훨씬 오래 학습시키면서, 학습 데이터와 테스트 데이터의 loss 를
나란히 찍어 본다. 두 loss 가 언제부터 반대로 움직이는지 관찰한다.

과적합이란?
  학습 데이터에는 점점 더 잘 맞는데, 처음 보는 데이터에는 오히려 나빠지는 현상.
  train loss 는 계속 내려가는데 test loss 는 어느 순간부터 올라간다.

왜 생기나?
  이 데이터는 악성/양성종양이 거의 직선으로 나뉜다. 학습을 계속하면 train loss 를
  0 에 더 가깝게 만들려고 w 가 계속 커진다. w 가 크면 sigmoid 가 계단처럼 가팔라져서
  경계 근처의 테스트 환자에게 0.9999 같은 극단적인 확률을 내놓는다.
  그런 예측이 틀리면 log loss 가 크게 튄다.

관찰 포인트
  1) test loss 는 최저점을 찍고 다시 오른다.
  2) test accuracy 는 loss 만큼 나빠지지 않는다.
     loss 는 "얼마나 확신했나" 를 재고, accuracy 는 "0.5 를 넘었나" 만 재기 때문이다.
     과적합이 진행돼도 결정 경계는 크게 안 움직이고, 틀린 예측의 확신만 커진다.
  3) |w| 가 계속 커진다.

※ 미리 알아 둘 것
  이 예제는 학습 도중 test 데이터를 계속 들여다보고, 그걸로 "언제 멈출지" 를 정한다.
  사실 이것은 해서는 안 되는 절차다. 왜 그런지는 마지막 6번에서 설명하고,
  05번에서 올바른 방법으로 고친다.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ============================================================
# 1. 데이터 준비 — 03번과 동일 (분할 + 표준화)
# ============================================================
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)

x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
X_train = (X_train - x_mean) / x_std
X_test = (X_test - x_mean) / x_std

N, D = X_train.shape


# ============================================================
# 2. 함수 — sigmoid, loss, accuracy
#    train 과 test 두 데이터에 같은 계산을 반복하므로 함수로 뺀다.
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce_loss(H, y, eps=1e-15):
    H = np.clip(H, eps, 1 - eps)   # log(0) 방지 (03번 참고)
    return -np.mean(y * np.log(H) + (1 - y) * np.log(1 - H))


def accuracy(H, y):
    return np.mean((H >= 0.5).astype(int) == y)


# ============================================================
# 3. 하이퍼파라미터 & 초기화
#    과적합은 "너무 오래" 학습할 때 생긴다. 멀리까지 가 보려고
#    lr 을 03번의 10배(1.0), epochs 를 30배(10만) 로 잡았다.
#    (표준화된 데이터라 lr=1.0 에서도 발산하지 않는다.)
# ============================================================
lr = 1.0
epochs = 100_000
checkpoints = [30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]   # 표에 찍을 epoch

w = np.zeros(D)
b = 0.0

# ============================================================
# 4. 학습 루프 — train loss 와 test loss 를 나란히 기록
#    매 epoch test loss 를 재서 최저점을 찾고, 체크포인트에서만 표로 찍는다.
#    test 데이터는 loss 를 "보기만" 한다. gradient 계산에는 쓰지 않는다.
# ============================================================
print(f"{'epoch':>7} | {'train_loss':>10} {'test_loss':>10} | {'train_acc':>9} {'test_acc':>8} | {'|w|max':>6}")
print("-" * 66)

best_test_loss = np.inf
best_epoch = 0

for epoch in range(1, epochs + 1):
    H = sigmoid(X_train @ w + b)
    grad_w = X_train.T @ (H - y_train) / N
    grad_b = np.mean(H - y_train)
    w -= lr * grad_w
    b -= lr * grad_b

    H_test = sigmoid(X_test @ w + b)
    test_loss = bce_loss(H_test, y_test)
    if test_loss < best_test_loss:
        best_test_loss, best_epoch = test_loss, epoch

    if epoch in checkpoints:
        H = sigmoid(X_train @ w + b)   # 갱신된 w 로 다시 예측 (test 와 같은 시점의 모델)
        print(f"{epoch:>7} | {bce_loss(H, y_train):10.4f} {test_loss:10.4f} | "
              f"{accuracy(H, y_train):9.4f} {accuracy(H_test, y_test):8.4f} | {np.abs(w).max():6.2f}")

# ============================================================
# 5. 관찰 정리
# ============================================================
print("-" * 66)
print(f"test loss 최저: {best_test_loss:.4f} (epoch {best_epoch})")
print("→ 그 이후로 train loss 는 계속 내려가지만 test loss 는 올라간다 = 과적합")
print("→ test accuracy 는 loss 만큼 나빠지지 않는다: 결정 경계보다 '확신' 이 먼저 망가진다")

# ============================================================
# 6. 그런데 — 이 방법에는 문제가 있다
#    "epoch 500 근처가 최적" 이라고 했다. 무엇을 보고 정했나? test loss 를 보고 정했다.
#    학습 횟수라는 설정값(하이퍼파라미터)을 고르는 데 test 를 써 버렸으니,
#    test 는 더 이상 "처음 보는 데이터" 가 아니다.
#    이 상태에서 나온 test accuracy 는 실제보다 좋게 나온 점수다.
#
#    → 해결: 학습 중 들여다보는 데이터(validation)와 최종 채점용 데이터(test)를
#      따로 둔다. 05번에서 다룬다.
# ============================================================
