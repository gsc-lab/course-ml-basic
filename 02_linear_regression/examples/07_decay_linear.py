"""
Linear Decay - 학습률 선형 감쇠
====================================
04 (Mini-batch GD, constant lr) 의 다음 단계.
학습 시작부터 lr 을 base_lr 에서 **선형으로 감소**시키며,
학습 끝에는 0 에 도달한다.

핵심 개념:
  학습 초반에는 큰 lr 로 빠르게 손실을 줄이는 게 효율적이지만,
  최적점 근처에서는 큰 lr 이 오히려 진동을 일으켜 수렴을 방해한다.
  → 학습이 진행될수록 lr 을 작게 줄여 정밀한 수렴을 유도 = decay.

  결과:
    - 초반: 큰 step 으로 빠른 수렴
    - 후반: 작은 step 으로 안정적 미세조정

수식 (per-step 적용):
  lr = base_lr × max(0, 1 - global_step / total_steps)

  - step 1                 : lr ≈ base_lr (거의 최대)
  - step total_steps / 2   : lr = base_lr × 0.5
  - step total_steps       : lr = 0

용어:
  - global_step  : 학습 전체에서 누적된 update 횟수
  - total_steps  : 전체 학습의 총 update 횟수 = epochs × (n / batch_size)
  - base_lr      : 학습 시작 시점의 최대 lr
  - lr           : 현재 step 의 실제 lr (감소 중)

특징:
  - warmup 과 마찬가지로 **매 batch (= 매 step)** 마다 lr 갱신.
  - 학습 끝 무렵에는 lr ≈ 0 이라 파라미터가 거의 안 움직임.
  - 노이즈 있는 데이터에서 안정적 수렴을 도와줌.
  - PyTorch `LinearLR`, HuggingFace decay phase 와 동일한 식.

warmup 과의 차이:
  - warmup : 0 → base_lr  (학습 초반 안정화)
  - decay  : base_lr → 0  (학습 후반 미세조정)
  - 둘은 보완 관계 → 다음 예제 (08) 에서 결합.
"""
import random
import math

# ============================================================
# 1. 데이터셋 (04 와 동일)
# ============================================================
random.seed(0)
x_data = [x for x in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]
n_samples = len(x_data)

# ============================================================
# 2. 하이퍼파라미터
# ============================================================
epochs = 3000   # decay 가 lr 을 줄이므로 06 (warmup) 보다 더 많은 epoch 필요
batch_size = 4
base_lr = 0.001

# ============================================================
# 3. 파생 파라미터
# ============================================================
total_steps = math.ceil(n_samples / batch_size) * epochs

# ============================================================
# 4. 파라미터 초기화
# ============================================================
random.seed(42)
w = random.random()
b = random.random()

print("-" * 10)
print("Linear Decay")
print("-" * 10)
print(f"데이터 수: {n_samples}개, batch_size: {batch_size}")
print(f"전체 step: {total_steps}, base_lr: {base_lr}")
print(f"초기 w: {w:.4f}, 초기 b: {b:.4f}")
print()

# ============================================================
# 5. 학습 루프 (Mini-batch SGD + Linear Decay)
# ============================================================
global_step = 0
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0

    sample_indices = list(range(n_samples))
    random.shuffle(sample_indices)

    for batch_start in range(0, n_samples, batch_size):
        global_step += 1

        # Linear decay: base_lr 에서 0 으로 선형 감소
        lr = base_lr * max(0, 1 - global_step / total_steps)

        batch_indices = sample_indices[batch_start: batch_start + batch_size]
        cur_batch_size = len(batch_indices)

        w_grad = 0.0
        b_grad = 0.0

        for sample_idx in batch_indices:
            x = x_data[sample_idx]
            y = y_data[sample_idx]

            # Prediction
            y_pred = x * w + b

            # Error
            error = y_pred - y

            # Loss
            epoch_loss += error ** 2

            # Gradient
            w_grad += error * x * 2
            b_grad += error * 2

        # Average gradient over batch
        w_grad /= cur_batch_size
        b_grad /= cur_batch_size

        # Parameter update
        w -= lr * w_grad
        b -= lr * b_grad

    epoch_loss /= n_samples

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | step {global_step:5d} | lr {lr:.6f} | "
              f"loss {epoch_loss:.4f} | w {w:.4f}, b {b:.4f}")

# ============================================================
# 6. 결과
# ============================================================
print()
print("-" * 10)
print("학습 완료")
print(f"  학습된 w: {w:.4f}  (정답: 0.5)")
print(f"  학습된 b: {b:.4f}  (정답: 2.0)")
print()
print("decay 효과:")
print(f"  step 1   : lr ≈ {base_lr}")
print(f"  step {total_steps // 2}: lr ≈ {base_lr / 2}")
print(f"  step {total_steps}: lr ≈ 0")
