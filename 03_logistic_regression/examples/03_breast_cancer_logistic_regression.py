"""
Logistic Regression - 실제 데이터로 학습하기 (위스콘신 유방암 데이터셋)
02번에서 살펴본 데이터를 01번의 로지스틱 회귀로 학습한다.
수식과 코드 구조는 01번과 같다. 달라지는 것은 입력이 1개에서 30개(D=30)로 늘어난 것뿐이다.

핵심 개념 (01번과 동일):
  가설(Hypothesis):  H(x) = sigmoid(x·w + b)
  손실함수(Loss):    BCE = -(1/N) Σ [ y·log(H) + (1-y)·log(1-H) ]
  옵티마이저:        Gradient Descent
    ∂Loss/∂w = (1/N) Σ (H(x) - y) · x
    ∂Loss/∂b = (1/N) Σ (H(x) - y)

실제 데이터라서 새로 필요한 것 두 가지:
  1) 표준화 (Feature Standardization)
     특성마다 크기가 제각각이다 (area 는 수천, smoothness 는 0.1 안팎).
     그대로 학습하면 x·w 가 너무 커져서 sigmoid 가 0 또는 1 에 딱 붙고,
     log(0) 때문에 loss 가 NaN 이 된다. 3번에서 직접 확인하고 4번에서 고친다.
  2) log(0) 방지
     학습이 잘 될수록 H 가 0 이나 1 에 아주 가까워진다. 컴퓨터의 실수 계산에는
     한계가 있어서 정확히 0.0 이나 1.0 이 되기도 한다.
     log 를 취하기 전에 H 를 [eps, 1-eps] 범위로 잘라 둔다.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ============================================================
# 1. 데이터 준비 — 02번과 똑같이 나눈다
# ============================================================
ds = datasets.load_breast_cancer()
X, y = ds.data, ds.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)

N, D = X_train.shape
print(f"학습: {N}건 × {D}특성, 테스트: {len(X_test)}건")


# ============================================================
# 2. sigmoid 함수 (01번과 동일)
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 3. 먼저 문제를 본다 — 표준화 없이 학습하면?
#    Epoch 1: w=0 이라 모든 예측이 0.5, loss = 0.6931. 아직 괜찮다.
#    Epoch 2: 한 번 업데이트된 w 로 x·w 가 수만이 된다
#             → sigmoid 가 0/1 에 붙음 → log(0) → loss 가 NaN
#    (RuntimeWarning 이 뜨는 것이 정상이다. 4번에서 고친다.)
# ============================================================
print("\n[표준화 없이 학습]")
w = np.zeros(D)
b = 0.0
for epoch in range(1, 4):
    H = sigmoid(X_train @ w + b)
    w -= 0.1 * X_train.T @ (H - y_train) / N
    b -= 0.1 * np.mean(H - y_train)
    loss = -np.mean(y_train * np.log(H) + (1 - y_train) * np.log(1 - H))
    print(f"  Epoch {epoch} | Loss: {loss} | x·w 최댓값: {np.abs(X_train @ w).max():.0f}")

# ============================================================
# 4. 표준화 — 특성마다 평균 0, 표준편차 1 로 맞춘다
#    axis=0 : 특성(열)마다 따로 계산한다. 전체를 하나의 숫자로 평균내면 안 된다.
#    테스트 데이터도 "학습 데이터의" 평균·표준편차로 변환한다.
#    테스트 데이터로 통계를 내면 시험 문제를 미리 보는 셈이다.
# ============================================================
x_mean = X_train.mean(axis=0)   # (D,)
x_std = X_train.std(axis=0)     # (D,)

X_train = (X_train - x_mean) / x_std
X_test = (X_test - x_mean) / x_std

# ============================================================
# 5. 하이퍼파라미터 & 파라미터 초기화 (01번과 동일)
# ============================================================
lr = 0.1
epochs = 3000
eps = 1e-15         # log(0) 방지용 아주 작은 수

w = np.zeros(D)     # (D,)
b = 0.0

# ============================================================
# 6. 학습 루프 (Batch Gradient Descent) — 01번과 같은 구조
# ============================================================
print("\n[표준화 후 학습]")
for epoch in range(1, epochs + 1):
    # ① Predict: H = sigmoid(X·w + b)  → 양성종양(1)일 확률
    H = sigmoid(X_train @ w + b)                    # (N,)

    # ② Gradient — (예측-정답)×입력, 전체 평균
    grad_w = X_train.T @ (H - y_train) / N          # (D,)
    grad_b = np.mean(H - y_train)                   # scalar

    # ③ 동시 업데이트
    w -= lr * grad_w
    b -= lr * grad_b

    # 손실 출력 — 방금 갱신한 w 로 다시 예측하고, log(0) 이 안 나오게 H 를 잘라서 계산
    if epoch == 1 or epoch % 300 == 0:
        H = sigmoid(X_train @ w + b)
        H_safe = np.clip(H, eps, 1 - eps)
        loss = -np.mean(y_train * np.log(H_safe) + (1 - y_train) * np.log(1 - H_safe))
        print(f"  Epoch {epoch:4d} | Loss: {loss:.4f}")

# ============================================================
# 7. 학습 데이터 정확도
#    학습에 쓴 데이터를 얼마나 맞히는가. (01번과 동일)
#    처음 보는 환자(X_test)도 맞히는지는 04번에서 확인한다.
# ============================================================
H = sigmoid(X_train @ w + b)
y_pred = (H >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_train)

print("\n" + "-" * 10)
print(f"학습 정확도: {accuracy:.4f}  ({np.sum(y_pred == y_train)}/{N})")

# ============================================================
# 8. 학습된 w 살펴보기 — 어떤 특성이 판단을 좌우하나
#    표준화했으므로 |w| 가 클수록 영향이 큰 특성이다.
#    w < 0 : 값이 클수록 악성(0) 쪽,  w > 0 : 값이 클수록 양성종양(1) 쪽
#    02번에서 본 "암세포 핵이 더 크다" 가 w 의 부호로 나타난다.
#    (비슷한 정보를 가진 특성끼리는 w 를 나눠 가지므로 순위는 참고만 한다.)
# ============================================================
top = np.argsort(np.abs(w))[::-1][:5]   # |w| 큰 순서로 인덱스 5개
print("\n영향력 큰 특성 5개:")
for i in top:
    print(f"  {ds.feature_names[i]:24s} w = {w[i]:+.3f}")
