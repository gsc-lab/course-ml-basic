"""정답: Linear Warmup + Linear Decay 파라미터 튜닝 (per-step, 함수 정의 없이)"""
import random

# ============================================================
# 데이터셋 (변경 금지)
#   x: 일일 평균 카페인 섭취량 (잔)
#   y: 야간 수면 부족 (시간) — 카페인이 많을수록 수면 시간 감소
# ============================================================
random.seed(0)
N = 32
x_data = [round(0.2 + 4.8 * i / (N - 1), 3) for i in range(N)]   # 0.2 ~ 5.0
y_data = [round(2.5 * x + 1.0 + random.uniform(-2.0, 2.0), 3) for x in x_data]


# ============================================================
# 공통 하이퍼파라미터
#   전체 step 수 = epochs × ceil(n/batch_size) = 75 × 8 = 600
# ============================================================
n = len(x_data)
epochs = 75
batch_size = 4
base_lr = 0.03
total_steps = epochs * ((n + batch_size - 1) // batch_size)   # 600


# ============================================================
# 조합 A : warmup_steps = 500  (전체 600 의 ~83%)
# ============================================================
print("=" * 56)
print(f"[A] warmup_steps = 500, decay_steps = 100  (총 {total_steps})")
print("=" * 56)

warmup_steps = 500
decay_steps = total_steps - warmup_steps   # 100
w, b = 0.0, 0.0
random.seed(42)
global_step = 0

for epoch in range(1, epochs + 1):
    indices = list(range(n))
    random.shuffle(indices)
    loss_sum = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        global_step += 1

        if global_step < warmup_steps:
            lr = base_lr * (global_step / warmup_steps)
        else:
            lr = base_lr * max(0.0, 1 - (global_step - warmup_steps) / decay_steps)

        batch_idx = indices[start:start + batch_size]
        m = len(batch_idx)
        dw, db, batch_loss = 0.0, 0.0, 0.0
        for i in batch_idx:
            x, y = x_data[i], y_data[i]
            err = (w * x + b) - y
            dw += err * x
            db += err
            batch_loss += err ** 2
        dw = (2.0 / m) * dw
        db = (2.0 / m) * db
        w -= lr * dw
        b -= lr * db

        loss_sum += batch_loss / m
        n_batches += 1

    if epoch == 1 or epoch % 15 == 0 or epoch == epochs:
        print(f"  epoch {epoch:3d} | step {global_step:4d} | lr {lr:.5f} | "
              f"avg-batch-loss {loss_sum / n_batches:7.4f} | "
              f"w {w:6.3f} b {b:6.3f}")

final_loss_A = (1.0 / n) * sum((w * x + b - y) ** 2 for x, y in zip(x_data, y_data))
w_A, b_A = w, b


# ============================================================
# 조합 B : warmup_steps = 100  (전체 600 의 ~17%)
# ============================================================
print()
print("=" * 56)
print(f"[B] warmup_steps = 100, decay_steps = 500  (총 {total_steps})")
print("=" * 56)

warmup_steps = 100
decay_steps = total_steps - warmup_steps   # 500
w, b = 0.0, 0.0
random.seed(42)
global_step = 0

for epoch in range(1, epochs + 1):
    indices = list(range(n))
    random.shuffle(indices)
    loss_sum = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        global_step += 1

        if global_step < warmup_steps:
            lr = base_lr * (global_step / warmup_steps)
        else:
            lr = base_lr * max(0.0, 1 - (global_step - warmup_steps) / decay_steps)

        batch_idx = indices[start:start + batch_size]
        m = len(batch_idx)
        dw, db, batch_loss = 0.0, 0.0, 0.0
        for i in batch_idx:
            x, y = x_data[i], y_data[i]
            err = (w * x + b) - y
            dw += err * x
            db += err
            batch_loss += err ** 2
        dw = (2.0 / m) * dw
        db = (2.0 / m) * db
        w -= lr * dw
        b -= lr * db

        loss_sum += batch_loss / m
        n_batches += 1

    if epoch == 1 or epoch % 15 == 0 or epoch == epochs:
        print(f"  epoch {epoch:3d} | step {global_step:4d} | lr {lr:.5f} | "
              f"avg-batch-loss {loss_sum / n_batches:7.4f} | "
              f"w {w:6.3f} b {b:6.3f}")

final_loss_B = (1.0 / n) * sum((w * x + b - y) ** 2 for x, y in zip(x_data, y_data))
w_B, b_B = w, b


# ============================================================
# 비교 + 선택 + 근거
# ============================================================
print()
print("=" * 56)
print("최종 비교")
print("=" * 56)
print(f"  A (warmup_steps=500, ~83%): final_loss = {final_loss_A:.4f} | "
      f"w = {w_A:.3f}, b = {b_A:.3f}")
print(f"  B (warmup_steps=100, ~17%): final_loss = {final_loss_B:.4f} | "
      f"w = {w_B:.3f}, b = {b_B:.3f}")
print()
print("[선택]   B (warmup_steps = 100)")
print("[근거]   A 는 warmup 비율이 ~83% 라 lr 이 base_lr 에 도달하자마자 decay 로")
print("        0 에 수렴, 본 학습 구간(decay 구간)이 부족. B 는 ~17% warmup 으로")
print("        안정적 ramp-up 후 충분한 본학습 — warmup 비율은 보통 전체의 5~20% 권장.")
