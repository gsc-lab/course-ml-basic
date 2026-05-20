"""
Multiple Linear Regression - numpy 벡터화 + Feature Standardization
===================================================================
10 (pure Python MLR + Mini-batch SGD) 의 진화판.

핵심 변화 (10 대비):
  1. numpy 벡터화
     - w1, w2, w3, w4 (개별 변수) → w (shape (D,) 배열)
     - sample 단위 for 루프 제거 → 행렬곱 한 줄로 batch 전체 처리
     - 코드량 1/3, 속도 100배+

  2. Feature Standardization
     - feature 마다 평균 0, 표준편차 1 로 정규화
     - 학습 안정성 ↑, lr 을 크게 사용 가능
     - 10 의 lr=0.001 / epochs=10000  →  여기선 lr=0.05 / epochs=200
     - 학습 후 결과를 원본 공간으로 역변환해서 ground truth 와 비교

가설 (Hypothesis):
  H(X) = X @ w + b           ← (N, D) @ (D,) + scalar → (N,)

손실 함수 (Loss):
  MSE = (1/B) Σ (H(X) - y)²

Gradient (벡터화, factor 2 포함):
  ∂L/∂w = (2/B) · Xᵀ @ error   ← (D, B) @ (B,) → (D,)
  ∂L/∂b = (2/B) · Σ error      ← scalar
"""
import numpy as np

# ============================================================
# 1. 데이터셋
#    정답 식: y = 1.0·x1 + 0.4·x2 + 0.2·x3 + 1.2·x4 + 1.0 + ε
#    x 의 값 범위 [1, 20] 으로 일부러 크게 잡아서 standardization
#    필요성을 보이기 위함.
# ============================================================
np.random.seed(0)

n_samples = 32
true_w = np.array([1.0, 0.4, 0.2, 1.2])
true_b = 1.0

x_data = np.random.uniform(1, 20, (n_samples, 4))
y_data = x_data @ true_w + true_b + np.random.randn(n_samples) * 0.5   # noise sigma=0.5

# ============================================================
# 2. Feature Standardization
#    x 의 값 범위가 [1, 20] 으로 크면 gradient 가 폭주해서
#    lr 을 매우 작게 잡아야 함 (불안정). 평균 0, std 1 로 정규화하면
#    gradient 가 안정되어 큰 lr 사용 가능 → 빠른 수렴.
# ============================================================
x_mean = x_data.mean(axis=0)
x_std  = x_data.std(axis=0)
x_std_data = (x_data - x_mean) / x_std    # 원본 x_data 는 보존, 별도 변수 사용

# ============================================================
# 3. 하이퍼파라미터
#    batch_size 는 일부러 num_features (=4) 와 다르게 잡음.
#    같으면 (B, D) 가 정사각이라 shape 버그가 ValueError 로
#    드러나지 않고 silent bug 가 됨 (gradient 식이 틀려도 동작).
# ============================================================
epochs = 200
batch_size = 8
lr = 0.05

# ============================================================
# 4. 파라미터 초기화
# ============================================================
num_samples, num_features = x_std_data.shape

w = np.random.randn(num_features)   # (D,)
b = np.random.randn()                # scalar

print("-" * 10)
print("Multiple Linear Regression (numpy + Standardization)")
print("-" * 10)
print(f"데이터 수: {num_samples}개,  feature 수: {num_features},  batch_size: {batch_size}")
print(f"epoch당 업데이트 횟수: {int(np.ceil(num_samples / batch_size))}회")
print(f"초기 w: {w}")
print(f"초기 b: {b:.4f}")
print()

# ============================================================
# 5. 학습 루프 (Mini-batch SGD, 벡터화)
#    구조: epoch → batch 의 2중 루프.
#    sample 단위 for 가 사라지고 batch 전체가 행렬곱 한 줄로 처리됨.
# ============================================================
# Loop: epoch
for epoch in range(1, epochs + 1):
    # Shuffle sample indices
    perm = np.random.permutation(num_samples)
    shuffle_x = x_std_data[perm]
    shuffle_y = y_data[perm]
    epoch_loss = 0.0

    # Loop: batch
    for start_idx in range(0, num_samples, batch_size):
        batch_x = shuffle_x[start_idx:start_idx + batch_size]   # (B, D)
        batch_y = shuffle_y[start_idx:start_idx + batch_size]   # (B,)
        cur_batch_size = len(batch_y)

        # Prediction (한 줄로 batch 전체)
        y_pred = batch_x @ w + b                                 # (B,)

        # Error (모든 sample 의 오차 벡터)
        error = y_pred - batch_y                                 # (B,)

        # Loss accumulation
        epoch_loss += (error ** 2).sum()

        # Gradient (벡터화, factor 2 포함)
        #   w_grad[j] = (2/B) · Σᵢ error[i] · batch_x[i, j]
        #             = (2/B) · (batch_xᵀ @ error)
        w_grad = (2 / cur_batch_size) * (batch_x.T @ error)      # (D,)
        b_grad = (2 / cur_batch_size) * error.sum()              # scalar

        # Parameter update
        w -= lr * w_grad
        b -= lr * b_grad

    # Average loss over epoch
    epoch_loss /= num_samples

    # Print progress every 20 epochs (+ 첫 epoch)
    if epoch == 1 or epoch % 20 == 0:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.4f} | w: {w}")

# ============================================================
# 6. 결과 — 원본 공간으로 역변환 후 비교
#    학습은 standardized 공간 ((X - μ) / σ) 에서 진행됐으므로,
#    학습된 (w, b) 를 원본 X 공간에 적용하려면 다음 변환 필요:
#
#      y = ((X - μ) / σ) · w_std + b_std
#        = X · (w_std / σ) + (b_std - μ · (w_std / σ))
#        ↓
#      w_orig = w_std / σ
#      b_orig = b_std - μ · w_orig
# ============================================================
w_orig = w / x_std
b_orig = b - x_mean @ w_orig

print()
print("-" * 10)
print("학습 완료")
print(f"  학습된 w (정규화 공간): {w}")
print(f"  학습된 b (정규화 공간): {b:.4f}")
print()
print("원본 공간 역변환 후 ground truth 와 비교:")
print(f"  학습된 w_orig: {w_orig}")
print(f"  정답   true_w: {true_w}")
print(f"  학습된 b_orig: {b_orig:.4f}")
print(f"  정답   true_b: {true_b}")
print()
print("예측 결과 (앞 5개) - 원본 x 로 예측:")
for i in range(5):
    xi = x_data[i]
    yi = y_data[i]
    y_pred = xi @ w_orig + b_orig
    print(f"  x={np.round(xi, 1)} → 예측: {y_pred:.2f}, 정답: {yi:.2f}")
