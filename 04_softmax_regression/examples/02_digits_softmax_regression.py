"""
Softmax Regression - 손글씨 숫자(digits) 다중 분류
01번에서 살펴본 데이터를 softmax 회귀로 학습한다.

핵심 개념:
  가설(Hypothesis):  H(x) = softmax(x·W + b)          (K=10 개 확률, 합은 1)
  손실함수(Loss):    Cross Entropy = -(1/N) Σ Σ_k Y_k·log(H_k)   (Y 는 원-핫 정답)
  옵티마이저:        Gradient Descent
    ∂Loss/∂W = (1/N) Xᵀ (H - Y)
    ∂Loss/∂b = (1/N) Σ (H - Y)
  03장의 로지스틱 회귀와 구조가 같다. sigmoid → softmax, w (D,) → W (D, K) 로 바뀔 뿐이다.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ============================================================
# 1. 데이터 준비 — 픽셀 값(0~16)을 16으로 나누어 0~1 로 맞춘다
# ============================================================
ds = datasets.load_digits()
X, y = ds.data / 16, ds.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=40
)

N, D = X_train.shape            # 글씨 수, 픽셀 수(64)
K = len(ds.target_names)        # 클래스 수(10)
print(f"학습: {N}장 × {D}픽셀, 테스트: {len(X_test)}장, 클래스: {K}개")

# 원-핫 인코딩 — 정답 정수 y 를 길이 K 인 벡터로 바꾼다 (01번 5. 참고)
Y_train = np.eye(K)[y_train]    # (N, K)


# ============================================================
# 2. softmax 함수 — 점수(logit) K 개를 합이 1 인 확률 K 개로 바꾼다
#    axis=-1 : 글씨 한 장마다(행마다) 따로 확률을 구한다
# ============================================================
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


# ============================================================
# 3. 하이퍼파라미터 & 파라미터 초기화
#    W : (D, K) — 클래스마다 픽셀 가중치 한 벌씩, 열 벌
#    b : (K,)   — 클래스마다 편향 하나씩
# ============================================================
lr = 0.1
epochs = 3000

W = np.zeros((D, K))
b = np.zeros(K)

# ============================================================
# 4. 학습 (Gradient Descent)
# ============================================================
for epoch in range(1, epochs + 1):
    H = softmax(X_train @ W + b)            # (N, K) 가설: 글씨마다 확률 K 개

    error = H - Y_train                     # (N, K)
    W_grad = X_train.T @ error / N          # (D, K)
    b_grad = error.mean(axis=0)             # (K,)  — 샘플 축으로만 평균

    W -= lr * W_grad
    b -= lr * b_grad

    if epoch % 100 == 0:
        loss = -np.mean(np.sum(Y_train * np.log(H), axis=1))
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

# ============================================================
# 5. 테스트 — 확률이 가장 큰 자리(argmax)를 예측 클래스로 삼는다
# ============================================================
H_test = softmax(X_test @ W + b)            # (N_test, K)
y_pred = H_test.argmax(axis=1)              # (N_test,)
accuracy = np.mean(y_pred == y_test)
print(f"\n테스트 정확도: {accuracy:.4f}")
