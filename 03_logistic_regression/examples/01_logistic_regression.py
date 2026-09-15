"""
Logistic Regression - 기본 예제
NumPy 로 이진 분류(binary classification)를 구현하여
가설, 손실함수, 옵티마이저의 관계를 이해한다.

핵심 개념:
  가설(Hypothesis):  H(x) = sigmoid(x·w + b) = 1 / (1 + e^-(x·w + b))
                     → 출력이 0~1 사이의 "확률" 이 된다.
  손실함수(Loss):    Binary Cross Entropy
                     BCE = -(1/N) Σ [ y·log(H) + (1-y)·log(1-H) ]
  옵티마이저:        Gradient Descent
    ∂Loss/∂w = (1/N) Σ (H(x) - y) · x
    ∂Loss/∂b = (1/N) Σ (H(x) - y)

  Linear Regression 과 비교:
    - 가설에 sigmoid 를 씌워 출력을 확률로 바꾼 것 외에는 구조가 같다.
    - 손실은 MSE 대신 BCE 를 쓰지만, gradient 식은
      (예측 - 정답) × 입력 의 평균으로 형태가 동일하다.
"""
import numpy as np

# ============================================================
# 1. 데이터셋
#    공부 시간(x) → 합격 여부(y)   (1: 합격, 0: 불합격)
#    공부를 많이 할수록 합격 확률이 높아진다는 직관적인 데이터.
#    5~6 시간 부근이 경계이고, 그 근처에는 예외도 섞여 있다.
# ============================================================
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)   # (N, 1)
y_train = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])                    # (N,)

N = len(y_train)
D = X_train.shape[1]


# ============================================================
# 2. 가설에 쓰이는 sigmoid 함수
#    어떤 실수든 0~1 사이로 눌러 준다. z=0 일 때 0.5.
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 3. 하이퍼파라미터 & 파라미터 초기화
# ============================================================
lr = 0.1
epochs = 3000

w = np.zeros(D)   # (D,)
b = 0.0

# ============================================================
# 4. 학습 루프 (Batch Gradient Descent)
# ============================================================
for epoch in range(1, epochs + 1):
    # ① Predict: H = sigmoid(X·w + b)  → 합격 확률
    H = sigmoid(X_train @ w + b)                    # (N,)

    # ② Gradient — (예측-정답)×입력, 전체 평균
    grad_w = X_train.T @ (H - y_train) / N          # (D,)
    grad_b = np.mean(H - y_train)                   # scalar

    # ③ 동시 업데이트
    w -= lr * grad_w
    b -= lr * grad_b

    # 손실 (Binary Cross Entropy)
    if epoch == 1 or epoch % 300 == 0:
        loss = -np.mean(y_train * np.log(H) + (1 - y_train) * np.log(1 - H))
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: {w[0]:.4f}, b: {b:.4f}")

# ============================================================
# 5. 학습 결과
#    확률 0.5 를 기준으로 합격(1) / 불합격(0) 을 결정한다.
# ============================================================
H = sigmoid(X_train @ w + b)
y_pred = (H >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_train)

print("\n" + "-" * 10)
print(f"학습된 w: {w[0]:.4f}, b: {b:.4f}")
print(f"결정 경계(확률 0.5): x = {-b / w[0]:.2f} 시간")
print(f"정확도: {accuracy:.2f}")
print()
print("예측 결과:")
for x, p, pred, y in zip(X_train[:, 0], H, y_pred, y_train):
    print(f"  공부 {x:2d}시간 → 합격 확률: {p:.3f}, 예측: {pred}, 정답: {y}")
