"""정답: 4가지 스케줄 종합 벤치마크 (per-step)"""
import random

# 데이터셋 (변경 금지, 적당한 노이즈) -- 정답: H(x) = 0.5x + 2  (w=0.5, b=2)
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-1.0, 1.0) for x in x_data]


# 하이퍼파라미터
n = len(x_data)
epochs = 500
batch_size = 4
base_lr = 0.005
warmup_steps = 250
decay_steps = 2250        # warmup_steps + decay_steps == epochs × (n / batch_size)
total_steps = warmup_steps + decay_steps
threshold = 1.0
tail = 50

# 비교할 4가지 스케줄
cases = ["constant", "warmup_only", "decay_only", "warmup_decay"]
labels = {
    "constant":     "고정 lr",
    "warmup_only":  "warmup 만",
    "decay_only":   "decay 만",
    "warmup_decay": "warmup+decay",
}

print(f"4가지 lr 스케줄 종합 벤치마크 (per-step) -- base_lr={base_lr}, "
      f"epochs={epochs}, batch_size={batch_size}")
print(f"  warmup_steps={warmup_steps}, decay_steps={decay_steps}, "
      f"threshold={threshold}, tail={tail} epoch\n")
print(f"{'스케줄':>14s} | {'<= '+str(threshold)+' 도달':>10s} | "
      f"{'tail 평균':>10s} | {'tail 스윙폭':>11s} | {'final loss':>10s}")
print("-" * 75)

# 케이스별로 학습 반복
for case in cases:
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

            # 스케줄별 lr 계산
            if case == "constant":
                lr = base_lr
            elif case == "warmup_only":
                lr = base_lr * min(1.0, global_step / warmup_steps)
            elif case == "decay_only":
                # warmup 없이 학습 전체에 걸쳐 decay (마지막 step 에서 lr=0)
                lr = base_lr * max(0.0, 1 - global_step / total_steps)
            else:  # "warmup_decay" (삼각형)
                if global_step < warmup_steps:
                    lr = base_lr * (global_step / warmup_steps)
                else:
                    lr = base_lr * max(0.0, 1 - (global_step - warmup_steps) / decay_steps)

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

    # 메트릭: threshold 처음 도달한 epoch
    first_ep = -1
    for i, loss in enumerate(loss_history, 1):
        if loss < threshold:
            first_ep = i
            break
    first_str = str(first_ep) if first_ep > 0 else "도달X"

    # tail 메트릭
    tail_losses = loss_history[-tail:]
    tail_avg = sum(tail_losses) / len(tail_losses)
    tail_swing = max(tail_losses) - min(tail_losses)

    print(f"{labels[case]:>14s} | {first_str:>10s} | "
          f"{tail_avg:>10.4f} | {tail_swing:>11.4f} | {loss_history[-1]:>10.4f}")

print("\n관찰:")
print("  - 수렴 속도 : warmup 있는 쪽은 초반 느림")
print("  - 안정성   : decay 들어간 쪽은 tail 스윙폭이 작다")
print("  - 최종 loss : decay 들어간 쪽이 더 낮게 안착한다")
