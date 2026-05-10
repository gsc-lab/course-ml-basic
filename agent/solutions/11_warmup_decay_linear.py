"""정답: Linear Warmup + Linear Decay (삼각형, per-step)"""
import random

# ============================================================
# 데이터셋 (변경 금지) -- 정답: H(x) = 0.5x + 2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 하이퍼파라미터 + 파라미터 초기화
# ============================================================
n = len(x_data)
epochs = 500
batch_size = 4
base_lr = 0.005
warmup_steps = 250
# 학습 전체 길이 = warmup_steps + decay_steps = epochs × (n / batch_size)
decay_steps = epochs * ((n + batch_size - 1) // batch_size) - warmup_steps
w, b = 0.0, 0.0

print(f"Linear Warmup + Linear Decay (삼각형, per-step) -- base_lr={base_lr}, "
      f"epochs={epochs}, batch_size={batch_size}")
print(f"  warmup_steps={warmup_steps}, decay_steps={decay_steps}\n")


# ============================================================
# 학습 루프
# ============================================================
random.seed(42)
global_step = 0
for epoch in range(1, epochs + 1):
    indices = list(range(n))
    random.shuffle(indices)

    loss_sum = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        global_step += 1

        # ---- 삼각형 스케줄: 0 → base_lr (warmup) → 0 (decay) ----
        if global_step < warmup_steps:
            # Phase 1 — Warmup: 0 → base_lr 선형 증가
            lr = base_lr * (global_step / warmup_steps)
        else:
            # Phase 2 — Linear Decay: base_lr → 0 선형 감소
            #   step 이 (warmup_steps + decay_steps) 를 초과하면 (1 - …) 가
            #   음수 → max(0, ..) 로 0 으로 clip (음수 lr 방지)
            lr = base_lr * max(0.0, 1 - (global_step - warmup_steps) / decay_steps)

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
        w -= lr * dw
        b -= lr * db

        loss_sum += batch_loss / m
        n_batches += 1

    if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
        print(f"Epoch {epoch:4d} | step {global_step:5d} | "
              f"lr: {lr:.6f} | Loss: {loss_sum / n_batches:.4f} | "
              f"w: {w:.2f}, b: {b:.2f}")

print(f"\n최종 w = {w:.2f}, b = {b:.2f}  (정답 w=0.5, b=2.0)")
