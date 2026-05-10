"""정답: Warmup 활용 - 큰 학습률 안정화 비교 (per-step)"""
import random

# 데이터셋 (변경 금지) -- 정답: H(x) = 0.5x + 2  (w=0.5, b=2)
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# 하이퍼파라미터
n = len(x_data)
epochs = 500
batch_size = 4
base_lr = 0.0075       # 의도적으로 큰 lr (warmup 효과 부각)
head_window = 30       # 초기 30 epoch 의 peak loss 만 측정

cases = [
    ("warmup 없음", 0),
    ("warmup 25 step", 25),
    ("warmup 250 step", 250),
]

print(f"Warmup 활용 실험 -- base_lr={base_lr}, epochs={epochs}, batch_size={batch_size}\n")
print(f"{'설정':>16s} | {'초기 peak loss':>15s} | {'최종 loss':>10s} | {'w':>6s} | {'b':>6s}")
print("-" * 70)

# 케이스별로 학습 반복
for label, warmup_steps in cases:
    # 케이스마다 학습 상태 초기화
    random.seed(42)
    w, b = 0.0, 0.0
    global_step = 0
    loss_history = []

    for epoch in range(1, epochs + 1):
        indices = list(range(n))
        random.shuffle(indices)

        loss_sum = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            global_step += 1
            # Linear Warmup (warmup_steps == 0 면 항상 base_lr)
            if warmup_steps == 0:
                lr = base_lr
            else:
                lr = base_lr * min(1.0, global_step / warmup_steps)

            # 배치 내 gradient / loss 누적
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

        loss_history.append(loss_sum / n_batches)

    peak_loss = max(loss_history[:head_window])
    print(f"{label:>16s} | {peak_loss:>15.2f} | "
          f"{loss_history[-1]:>10.4f} | {w:>6.2f} | {b:>6.2f}")

print("\n결론: warmup_steps 가 클수록 초기 peak loss 가 작아진다.")
