"""
정답: Warmup 활용 - 큰 학습률 안정화 비교 (YJU Agent:Eval 형식)
==============================================================
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
# 학생 TODO 1: warmup 학습률
#   warmup_epochs == 0 이면 항상 base_lr 반환
# ============================================================
def get_warmup_lr(epoch, base_lr, warmup_epochs):
    if warmup_epochs == 0 or epoch > warmup_epochs:
        return base_lr
    return base_lr * epoch / warmup_epochs


# ============================================================
# 학습 함수 (출제자 영역)
# ============================================================
def train(x_data, y_data, base_lr, epochs, batch_size,
          warmup_epochs, w_init, b_init):
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        current_lr = get_warmup_lr(epoch, base_lr, warmup_epochs)

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
# 학생 TODO 2: peak_loss / final_loss 추출
# ============================================================
def extract_metrics(loss_history, head_window):
    peak_loss = max(loss_history[:head_window])
    final_loss = loss_history[-1]
    return peak_loss, final_loss


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    w_init, b_init = 0.0, 0.0
    base_lr = 0.0075
    epochs = 500
    batch_size = 4
    head_window = 30

    cases = [
        ("warmup 없음", 0),
        ("warmup 5 epoch", 5),
        ("warmup 50 epoch", 50),
    ]

    print("-" * 10)
    print("Warmup 활용 실험")
    print("-" * 10)
    print(f"base_lr: {base_lr}, epochs: {epochs}, batch_size: {batch_size}")
    print()
    print(f"{'설정':>16s} | {'초기 peak loss':>15s} | {'최종 loss':>10s} | {'w':>6s} | {'b':>6s}")
    print("-" * 70)

    for label, warmup_epochs in cases:
        random.seed(42)
        w, b, loss_history = train(
            x_data, y_data, base_lr, epochs, batch_size, warmup_epochs,
            w_init, b_init
        )
        peak_loss, final_loss = extract_metrics(loss_history, head_window)

        print(f"{label:>16s} | {peak_loss:>15.2f} | "
              f"{final_loss:>10.4f} | {w:>6.2f} | {b:>6.2f}")

    print()
    print("결론: warmup이 있으면 초기 peak loss가 크게 줄어든다.")
    print("      → 학습 초반의 불안정한 큰 발걸음을 방지한다.")
