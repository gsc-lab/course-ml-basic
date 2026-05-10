"""정답: SGD with Momentum (per-step)"""
import random

# ============================================================
# 데이터셋 (변경 금지) -- 정답: H(x) = 0.5x + 2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 하이퍼파라미터 + 파라미터/velocity 초기화
# ============================================================
n = len(x_data)
epochs = 500
batch_size = 4
lr = 0.001
beta = 0.9              # momentum 계수 (0=일반 SGD, 클수록 관성 ↑, 보통 0.9)
w, b = 0.0, 0.0
vw, vb = 0.0, 0.0       # velocity: 직전 gradient 들의 누적 (학습 시작 시 0, 끝까지 유지)

print(f"SGD with Momentum -- lr={lr}, epochs={epochs}, "
      f"batch_size={batch_size}, beta={beta}\n")


# ============================================================
# 학습 루프
# ============================================================
random.seed(42)
for epoch in range(1, epochs + 1):
    indices = list(range(n))
    random.shuffle(indices)

    loss_sum = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        # ---- 배치 내 gradient + loss 누적 (MSE) ----
        batch_indices = indices[start:start + batch_size]
        m = len(batch_indices)
        dw, db, batch_loss = 0.0, 0.0, 0.0
        for i in batch_indices:
            x, y = x_data[i], y_data[i]
            error = (w * x + b) - y
            dw += error * x
            db += error
            batch_loss += error ** 2

        dw = (2.0 / m) * dw
        db = (2.0 / m) * db

        # ---- Momentum 업데이트 (PyTorch SGD convention) ----
        #   velocity   : 직전 gradient 들의 지수 가중 평균 (관성)
        #                v ← β·v + grad
        #   파라미터   : gradient 대신 velocity 방향으로 이동
        #                → 진동 ↓, 수렴 ↑
        vw = beta * vw + dw
        vb = beta * vb + db
        w -= lr * vw
        b -= lr * vb

        loss_sum += batch_loss / m
        n_batches += 1

    if epoch == 1 or epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss_sum / n_batches:.4f} | "
              f"w: {w:.2f}, b: {b:.2f} | vw: {vw:.4f}, vb: {vb:.4f}")

print(f"\n최종 w = {w:.2f}, b = {b:.2f}  (정답 w=0.5, b=2.0)")
