"""정답: Decay 활용 - 진동 데이터 안정성 비교 (per-step)"""
import random

# 데이터셋 (변경 금지, 노이즈 큼) -- 정답: H(x) = 0.5x + 2  (w=0.5, b=2)
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-2.0, 2.0) for x in x_data]


# 하이퍼파라미터
n = len(x_data)
epochs = 500
batch_size = 4
base_lr = 0.005
decay_steps = epochs * ((n + batch_size - 1) // batch_size)
tail = 50  # 마지막 N epoch 로 안정성 평가

cases = [
    ("고정 lr",      False),
    ("Linear Decay", True),
]

print(f"Decay 활용 실험 (노이즈 큰 데이터) -- base_lr={base_lr}, "
      f"epochs={epochs}, batch_size={batch_size}")
print(f"마지막 {tail} epoch 의 평균/최대/최소 loss 로 안정성 평가\n")
print(f"{'설정':>14s} | {'tail 평균':>10s} | {'tail 최대':>10s} | "
      f"{'tail 최소':>10s} | {'스윙폭':>8s}")
print("-" * 70)

# 케이스별로 학습 반복
for label, use_decay in cases:
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
            # decay 사용 여부에 따라 lr 결정
            if use_decay:
                lr = base_lr * max(0.0, 1 - global_step / decay_steps)
            else:
                lr = base_lr

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

    tail_losses = loss_history[-tail:]
    avg = sum(tail_losses) / len(tail_losses)
    mx, mn = max(tail_losses), min(tail_losses)
    print(f"{label:>14s} | {avg:>10.4f} | {mx:>10.4f} | "
          f"{mn:>10.4f} | {mx - mn:>8.4f}")

print("\n결론: Decay 는 lr 이 줄어들면서 진동폭이 작아져 안정적으로 안착한다.")
