"""정답: Warmup -> Constant -> Step Decay (3단계 스케줄, per-step)"""
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
warmup_steps = 150          # ≈ 30 epoch  : 초기 안정화 구간
decay_start_step = 1000     # ≈ 200 epoch : decay 시작 시점
decay_steps = 500           # 100 epoch period : decay_rate 적용 주기
decay_rate = 0.5            # 1 주기마다 lr × 0.5
w, b = 0.0, 0.0

print(f"3단계 스케줄 (per-step) -- base_lr={base_lr}, epochs={epochs}, "
      f"batch_size={batch_size}")
print(f"  warmup_steps={warmup_steps}, decay_start_step={decay_start_step}, "
      f"decay_steps={decay_steps}, decay_rate={decay_rate}\n")


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

        # ---- 3단계 스케줄: Warmup → Constant → Step Decay ----
        if global_step <= warmup_steps:
            # Phase 1 — Warmup: 0 → base_lr 선형 증가 (초기 안정화)
            lr = base_lr * global_step / warmup_steps
        elif global_step <= decay_start_step:
            # Phase 2 — Constant: base_lr 유지 (본격 학습 구간)
            lr = base_lr
        else:
            # Phase 3 — Step Decay: decay_steps 마다 lr × decay_rate
            #   times = (현재 step - decay 시작점 - 1) // decay_steps + 1
            #   step = decay_start_step + 1     → times=1 → lr = base_lr × 0.5
            #   step = decay_start_step + 500   → times=1 (같은 주기)
            #   step = decay_start_step + 501   → times=2 → lr = base_lr × 0.25
            times = (global_step - decay_start_step - 1) // decay_steps + 1
            lr = base_lr * (decay_rate ** times)

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

    if epoch == 1 or epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | step {global_step:5d} | "
              f"lr: {lr:.6f} | Loss: {loss_sum / n_batches:.4f} | "
              f"w: {w:.2f}, b: {b:.2f}")

print(f"\n최종 w = {w:.2f}, b = {b:.2f}  (정답 w=0.5, b=2.0)")
