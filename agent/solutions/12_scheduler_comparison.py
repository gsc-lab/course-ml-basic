"""
정답: 4가지 스케줄 종합 벤치마크 (YJU 형식)
============================================
"""
import random

# ============================================================
# 데이터셋 (노이즈 있는 20개) - 변경 금지
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-1.0, 1.0) for x in x_data]


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
# 학생 TODO 1: 4가지 스케줄러
# ============================================================
def lr_constant(epoch, base_lr, warmup, total):
    return base_lr


def lr_warmup_only(epoch, base_lr, warmup, total):
    if epoch <= warmup:
        return base_lr * epoch / warmup
    return base_lr


def lr_decay_only(epoch, base_lr, warmup, total):
    progress = (total - epoch) / total
    return base_lr * max(progress, 0.0)


def lr_warmup_decay(epoch, base_lr, warmup, total):
    if epoch <= warmup:
        return base_lr * epoch / warmup
    progress = (total - epoch) / (total - warmup)
    return base_lr * max(progress, 0.0)


# ============================================================
# 학습 함수 (출제자 영역)
# ============================================================
def train(x_data, y_data, scheduler, base_lr, epochs, batch_size,
          warmup, w_init, b_init):
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        current_lr = scheduler(epoch, base_lr, warmup, epochs)

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

            w = w - current_lr * dw
            b = b - current_lr * db

            epoch_loss_sum += batch_loss
            batch_count += 1

        loss_history.append(epoch_loss_sum / batch_count)

    return w, b, loss_history


# ============================================================
# 학생 TODO 2: 메트릭 - 처음으로 threshold 미만이 되는 epoch
# ============================================================
def find_first_below(loss_history, threshold):
    for i, loss in enumerate(loss_history, 1):
        if loss < threshold:
            return i
    return -1


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    w_init, b_init = 0.0, 0.0
    base_lr = 0.005
    epochs = 500
    batch_size = 4
    warmup = 50
    threshold = 1.0
    tail = 50

    schedulers = [
        ("고정 lr", lr_constant),
        ("warmup 만", lr_warmup_only),
        ("decay 만", lr_decay_only),
        ("warmup+decay", lr_warmup_decay),
    ]

    print("-" * 10)
    print("4가지 lr 스케줄 종합 벤치마크")
    print("-" * 10)
    print(f"base_lr: {base_lr}, epochs: {epochs}, batch_size: {batch_size}")
    print(f"warmup: {warmup}, threshold: {threshold}, tail: {tail}")
    print()
    print(f"{'스케줄':>14s} | {'≤ '+str(threshold)+' 도달':>10s} | "
          f"{'tail 평균':>10s} | {'tail 스윙폭':>11s} | {'final loss':>10s}")
    print("-" * 75)

    for label, scheduler in schedulers:
        random.seed(42)
        w, b, loss_history = train(
            x_data, y_data, scheduler, base_lr, epochs, batch_size,
            warmup, w_init, b_init
        )

        first_ep = find_first_below(loss_history, threshold)
        first_str = f"{first_ep}" if first_ep > 0 else "도달X"

        tail_losses = loss_history[-tail:]
        tail_avg = sum(tail_losses) / len(tail_losses)
        tail_swing = max(tail_losses) - min(tail_losses)
        final_loss = loss_history[-1]

        print(f"{label:>14s} | {first_str:>10s} | "
              f"{tail_avg:>10.4f} | {tail_swing:>11.4f} | {final_loss:>10.4f}")

    print()
    print("관찰:")
    print("  · 수렴 속도 : 고정/decay 만은 빠르고, warmup 있는 쪽은 느린 편")
    print("  · 안정성   : decay 들어간 쪽은 tail 스윙폭이 100배 작다")
    print("  · 최종 loss : decay 들어간 쪽이 더 낮게 안착한다")
