"""
정답: Decay 활용 - 진동 데이터 안정성 비교 (YJU Agent:Eval 형식)
================================================================
"""
import random

# ============================================================
# 데이터셋 (노이즈 큰 20개) - 변경 금지
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-2.0, 2.0) for x in x_data]


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
# ============================================================
def get_step_decay_lr(epoch, base_lr, decay_step, decay_rate):
    times = (epoch - 1) // decay_step
    return base_lr * (decay_rate ** times)


# ============================================================
# 학습 함수 (출제자 영역)
# ============================================================
def train(x_data, y_data, base_lr, epochs, batch_size, w_init, b_init,
          use_decay, decay_step, decay_rate):
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        if use_decay:
            current_lr = get_step_decay_lr(epoch, base_lr, decay_step, decay_rate)
        else:
            current_lr = base_lr

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
# 학생 TODO 2: tail 메트릭 추출
# ============================================================
def extract_tail_metrics(loss_history, tail):
    tail_losses = loss_history[-tail:]
    avg = sum(tail_losses) / len(tail_losses)
    mx = max(tail_losses)
    mn = min(tail_losses)
    swing = mx - mn
    return avg, mx, mn, swing


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
    tail = 50

    print("-" * 10)
    print("Decay 활용 실험 (노이즈 큰 데이터)")
    print("-" * 10)
    print(f"base_lr: {base_lr}, epochs: {epochs}, batch_size: {batch_size}")
    print(f"마지막 {tail} epoch 의 평균/최대/최소 loss 로 안정성 평가")
    print()
    print(f"{'설정':>14s} | {'tail 평균':>10s} | {'tail 최대':>10s} | {'tail 최소':>10s} | {'스윙폭':>8s}")
    print("-" * 70)

    cases = [
        ("고정 lr", False),
        ("Step Decay", True),
    ]

    for label, use_decay in cases:
        random.seed(42)
        w, b, loss_history = train(
            x_data, y_data, base_lr, epochs, batch_size,
            w_init, b_init, use_decay, decay_step, decay_rate
        )

        avg, mx, mn, swing = extract_tail_metrics(loss_history, tail)

        print(f"{label:>14s} | {avg:>10.4f} | {mx:>10.4f} | "
              f"{mn:>10.4f} | {swing:>8.4f}")

    print()
    print("결론: 고정 lr 은 노이즈 때문에 마지막까지 진동한다.")
    print("      Step Decay 는 lr 이 줄어들면서 진동폭이 작아져 안정적으로 안착한다.")
