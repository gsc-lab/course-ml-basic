"""
정답: Step Decay 구현 (YJU Agent:Eval 형식)
============================================
"""
import random

# ============================================================
# 데이터셋 (20개) - 변경 금지
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# Helper: gradient / loss (변경 금지)
# ============================================================
def compute_mse_gradient(batch_x, batch_y, w, b):
    m = len(batch_x)
    dw = (2.0 / m) * sum((w * x + b - y) * x for x, y in zip(batch_x, batch_y))
    db = (2.0 / m) * sum(w * x + b - y for x, y in zip(batch_x, batch_y))
    return dw, db


def compute_mse_loss(batch_x, batch_y, w, b):
    m = len(batch_x)
    return (1.0 / m) * sum((w * x + b - y) ** 2 for x, y in zip(batch_x, batch_y))


# ============================================================
# 학생 TODO 1: Step Decay 학습률
#   times = (epoch - 1) // decay_step
#   return base_lr * (decay_rate ** times)
# ============================================================
def get_step_decay_lr(epoch, base_lr, decay_step, decay_rate):
    times = (epoch - 1) // decay_step
    return base_lr * (decay_rate ** times)


# ============================================================
# 학습 함수 (출제자 영역)
# ============================================================
def train_with_step_decay(x_data, y_data, base_lr, epochs, batch_size,
                          decay_step, decay_rate, w_init, b_init):
    w, b = w_init, b_init
    n = len(x_data)

    for epoch in range(1, epochs + 1):
        current_lr = get_step_decay_lr(epoch, base_lr, decay_step, decay_rate)

        indices = list(range(n))
        random.shuffle(indices)

        epoch_loss_sum = 0.0
        batch_count = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_x = [x_data[i] for i in batch_idx]
            batch_y = [y_data[i] for i in batch_idx]

            dw, db = compute_mse_gradient(batch_x, batch_y, w, b)
            batch_loss = compute_mse_loss(batch_x, batch_y, w, b)

            # 학생 TODO 2: 파라미터 업데이트
            w = w - current_lr * dw
            b = b - current_lr * db

            epoch_loss_sum += batch_loss
            batch_count += 1

        avg_loss = epoch_loss_sum / batch_count

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | lr: {current_lr:.6f} | "
                  f"Loss: {avg_loss:.4f} | w: {w:.2f}, b: {b:.2f}")

    return w, b


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    w_init, b_init = 0.0, 0.0
    base_lr = 0.005
    epochs = 500
    batch_size = 4
    decay_step = 100
    decay_rate = 0.5

    n = len(x_data)
    print("-" * 10)
    print("Step Decay Mini-batch GD 학습")
    print("-" * 10)
    print(f"데이터 수: {n}개, epochs: {epochs}, batch_size: {batch_size}")
    print(f"base_lr: {base_lr}, decay_step: {decay_step}, decay_rate: {decay_rate}")
    print(f"  → {decay_step} epoch 마다 lr 에 {decay_rate} 를 곱한다")
    print()

    random.seed(42)
    w, b = train_with_step_decay(
        x_data, y_data, base_lr, epochs, batch_size,
        decay_step, decay_rate, w_init, b_init
    )

    print()
    print(f"최종 파라미터: w = {w:.2f}, b = {b:.2f}")
    print(f"정답과 비교: w(정답=0.5), b(정답=2.0)")
