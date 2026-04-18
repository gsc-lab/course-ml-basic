"""
Batch Size가 학습에 미치는 영향 - 배달 소요시간 예측
====================================================
같은 Mini-batch GD에서 batch_size만 바꿔가며
학습 안정성과 수렴 속도가 어떻게 달라지는지 관찰한다.

데이터셋:
  배달 거리(km)에 따른 소요시간(분) 100건.
  실제 관계: 시간 ≈ 3분/km × 거리 + 10분(조리·포장)
  교통 상황, 신호, 날씨 등으로 인한 노이즈가 포함되어 있다.

핵심 개념:
  batch_size가 작으면
    - epoch당 업데이트가 많아 빠르게 움직인다
    - 하지만 소수 샘플의 기울기만 보므로 방향이 흔들린다 (noise ↑)
    - 초반 탐색에는 유리할 수 있다

  batch_size가 크면
    - 많은 샘플의 평균 기울기를 사용하므로 방향이 안정적이다
    - 하지만 epoch당 업데이트 횟수가 줄어 수렴이 느려질 수 있다

  ┌──────────────┬───────────┬───────────────┬────────────┐
  │  batch_size   │ 업데이트/epoch │  안정성     │  수렴 속도  │
  ├──────────────┼───────────┼───────────────┼────────────┤
  │  1 (SGD)     │  100회     │  매우 불안정   │  빠르지만 진동 │
  │  4           │  25회      │  불안정       │  빠름       │
  │  16          │  6회       │  적당        │  적당       │
  │  50          │  2회       │  안정적      │  느림       │
  │  100 (BGD)   │  1회       │  매우 안정    │  매우 느림   │
  └──────────────┴───────────┴───────────────┴────────────┘
"""
import random

# ============================================================
# 1. 데이터셋: 배달 거리(km) → 소요시간(분)
#    정답: H(x) = 3x + 10  →  w=3, b=10
#    노이즈: ±5분 (교통, 신호, 날씨 등)
# ============================================================
random.seed(0)
n = 100
x_data = [random.uniform(0.5, 10.0) for _ in range(n)]
y_data = [3 * x + 10 + random.gauss(0, 2.5) for x in x_data]

print("=" * 70)
print("배달 소요시간 예측 — Batch Size가 학습에 미치는 영향")
print("=" * 70)
print(f"데이터: 배달 {n}건 (거리 0.5~10km, 소요시간 ≈ 3×거리 + 10분)")
print()


# ============================================================
# 2. Mini-batch GD 학습 함수
#    04번 예제의 학습 루프를 함수로 분리
# ============================================================
def train_minibatch(x_data, y_data, learning_rate, epochs, batch_size,
                    w_init, b_init):
    """Mini-batch GD로 학습하고, 매 epoch의 loss를 기록한다."""
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        indices = list(range(n))
        random.shuffle(indices)

        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            bs = len(batch_indices)

            w_grad = 0.0
            b_grad = 0.0
            batch_loss = 0.0

            for idx in batch_indices:
                x = x_data[idx]
                y = y_data[idx]
                predict = w * x + b
                error = predict - y

                w_grad += 2 * x * error
                b_grad += 2 * error
                batch_loss += error ** 2

            w_grad /= bs
            b_grad /= bs

            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad

            epoch_loss += batch_loss

        loss_history.append(epoch_loss / n)

    return w, b, loss_history


# ============================================================
# 3. 다양한 batch_size로 학습 실행
# ============================================================
batch_sizes = [1, 4, 16, 50, 100]
epochs = 200
learning_rate = 0.001

# 모든 실험에서 동일한 초기값 사용
random.seed(42)
w_init = random.random()
b_init = random.random()

results = []

for bs in batch_sizes:
    # batch_size별로 동일한 shuffle 시드 사용
    random.seed(99)
    w, b, loss_history = train_minibatch(
        x_data, y_data, learning_rate, epochs, bs, w_init, b_init
    )
    results.append((bs, w, b, loss_history))

    updates_per_epoch = (n + bs - 1) // bs  # 올림 나눗셈
    label = ""
    if bs == 1:
        label = " (= SGD)"
    elif bs == n:
        label = " (= BGD)"

    print(f"[ batch_size = {bs:3d}{label} ]")
    print(f"  epoch당 업데이트: {updates_per_epoch}회"
          f"  |  총 업데이트: {updates_per_epoch * epochs}회")
    print(f"  최종 w: {w:.4f} (정답: 3.0)")
    print(f"  최종 b: {b:.4f} (정답: 10.0)")
    print(f"  최종 loss: {loss_history[-1]:.4f}")
    print()


# ============================================================
# 4. Loss 변화 비교 (epoch 구간별 평균으로 안정성 확인)
# ============================================================
print("=" * 70)
print("Loss 변화 비교 (매 50 epoch 시점)")
print("=" * 70)

# 헤더
header = f"{'Epoch':>8s}"
for bs, _, _, _ in results:
    header += f"  {'bs=' + str(bs):>10s}"
print(header)
print("-" * (8 + 12 * len(results)))

# 각 epoch 시점별 loss
for ep in [1, 10, 50, 100, 150, 200]:
    row = f"{ep:8d}"
    for _, _, _, loss_history in results:
        row += f"  {loss_history[ep - 1]:10.4f}"
    print(row)


# ============================================================
# 5. Loss 진동 폭 분석 (마지막 50 epoch)
#    loss가 얼마나 흔들리는지 = 안정성의 지표
# ============================================================
print()
print("=" * 70)
print("안정성 분석 — 마지막 50 epoch의 loss 진동 폭")
print("=" * 70)

for bs, w, b, loss_history in results:
    last_50 = loss_history[-50:]
    avg_loss = sum(last_50) / len(last_50)
    min_loss = min(last_50)
    max_loss = max(last_50)
    swing = max_loss - min_loss

    # 진동 크기에 따라 시각적 바 표시
    bar_len = min(int(swing * 2), 40)
    bar = "█" * bar_len if bar_len > 0 else "▏"

    label = ""
    if bs == 1:
        label = "(SGD) "
    elif bs == n:
        label = "(BGD) "

    print(f"  batch_size = {bs:3d} {label}"
          f"| 평균: {avg_loss:7.4f} "
          f"| 진동 폭: {swing:7.4f} {bar}")


# ============================================================
# 6. 결론
# ============================================================
print()
print("=" * 70)
print("정리")
print("=" * 70)
print("""
  batch_size가 작을수록 (→ SGD에 가까움)
    ✓ epoch당 업데이트 횟수가 많다 → 빠르게 움직인다
    ✗ 소수 샘플만 보고 업데이트 → gradient noise가 크다 (loss 진동)

  batch_size가 클수록 (→ BGD에 가까움)
    ✓ 많은 샘플의 평균 → gradient가 안정적이다 (loss 매끄러움)
    ✗ epoch당 업데이트 횟수가 적다 → 수렴이 느려질 수 있다

  실무에서는?
    - 보통 batch_size = 32, 64, 128을 사용한다
    - 안정성과 수렴 속도의 균형점을 찾는 것이 핵심이다
    - GPU 메모리에 맞는 최대 batch_size를 쓰는 것도 중요한 기준이다
""")
