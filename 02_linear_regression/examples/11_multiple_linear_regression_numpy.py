import numpy as np

# Data
np.random.seed(0)
N = 32
D = 4

b_true = 1.0
w_true = np.array([1.0, 0.4, 0.2, 1.2]).reshape(D, 1)            # (D, 1)

x_raw = np.random.uniform(1, 20, (N, D))                          # (N, D)
y_data = x_raw @ w_true + b_true + np.random.randn(N, 1) * 0.5    # (N, 1)

# Standardization (x 범위 [1, 20] 이라 lr 안정성을 위해 정규화)
x_mean = x_raw.mean(axis=0)                                       # (D,)
x_std  = x_raw.std(axis=0)                                        # (D,)
x_data = (x_raw - x_mean) / x_std                                 # (N, D)

# Option
LOG = False

# Hyperparameters
# batch_size 는 일부러 D(=4) 와 다르게 잡음 — 같으면 (B, D) 가 정사각이라
# gradient 식의 shape 버그가 silent 하게 통과될 수 있음.
epochs = 200
batch_size = 8
lr = 0.05

# Model parameters
w = np.random.randn(D, 1) * 0.01                                  # (D, 1)
b = 0.0

if LOG:
    print(f"w shape: {w.shape}, value: \n{w}")
    print(f"b: {b}")

# Model training (Mini-batch SGD)
for epoch in range(1, epochs + 1):
    # shuffle
    perm = np.random.permutation(N)
    x_shuffled = x_data[perm]                                     # (N, D)
    y_shuffled = y_data[perm]                                     # (N, 1)
    epoch_loss = 0.0

    for start in range(0, N, batch_size):
        end = start + batch_size
        x_batch = x_shuffled[start:end]                           # (B, D)
        y_batch = y_shuffled[start:end]                           # (B, 1)
        B = len(y_batch)

        # prediction
        y_pred = x_batch @ w + b                                  # (B, 1)

        # error
        error = y_pred - y_batch                                  # (B, 1)

        # gradient
        w_grad = (2 / B) * (x_batch.T @ error)                    # (D, 1)
        b_grad = (2 / B) * error.sum()                            # scalar

        # parameter update
        w = w - lr * w_grad
        b = b - lr * b_grad

        # loss 누적 (epoch 평균용)
        epoch_loss += (error ** 2).sum()

    epoch_loss /= N

    # print epoch loss
    if epoch == 1 or epoch % 20 == 0:
        print(f"epoch {epoch:4d} | loss: {epoch_loss:.4f}")

# 원본 공간 역변환
# y = ((X - μ) / σ) @ w + b = X @ (w/σ) + (b - μ·(w/σ))
w_orig = w / x_std.reshape(D, 1)                                  # (D, 1)
b_orig = b - (x_mean @ w_orig).item()                             # scalar

print()
print(f"w_true: {w_true.flatten()}")
print(f"w:      {w_orig.flatten()}")
print(f"b_true: {b_true}, b: {b_orig:.4f}")
