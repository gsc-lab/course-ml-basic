"""
Linear Warmup - 학습률 선형 워밍업
====================================
04 (Mini-batch GD, constant lr) 의 다음 단계.
학습 초반에 lr 을 0 에서 base_lr 까지 **선형으로 증가**시킨 뒤,
그 이후로는 base_lr 을 유지한다.

핵심 개념:
  학습 초반에는 파라미터가 무작위 초기값이라 gradient 가 불안정하다.
  큰 lr 로 첫 step 부터 업데이트하면 발산하거나 나쁜 local minimum 으로
  빠질 위험이 있다.
  → 처음 일부 step 동안 lr 을 작게 시작해서 천천히 키운다 = warmup.

  warmup 후에는 base_lr 로 안정적으로 학습.

수식 (per-step 적용):
  step ≤ warmup_steps 인 경우:
    lr = base_lr × (step / warmup_steps)
  step > warmup_steps 인 경우:
    lr = base_lr
  → min(1, step / warmup_steps) 로 한 줄에 표현 가능.

용어:
  - global_step  : 학습 전체에서 누적된 update 횟수 (= 처리한 batch 수)
  - total_steps  : 전체 학습의 총 update 횟수 = epochs × (n / batch_size)
  - warmup_steps : warmup 구간의 step 수 = total_steps × warmup_ratio
  - base_lr      : warmup 후 도달할 목표 lr (상수)
  - lr           : 현재 step 의 실제 lr (시점마다 다름)

특징:
  - lr 은 **매 batch (= 매 step) 마다** 업데이트된다.
  - epoch 단위가 아니라 step 단위로 스케줄링하는 게 표준 (per-step convention).
  - PyTorch / HuggingFace 모두 동일한 컨벤션.
"""
import random
import math

# ============================================================
# 1. 데이터셋 (04 와 동일: 1-feature, y = 0.5x + 2)
# ============================================================
random.seed(0)
x_data = [x for x in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]
n_samples = len(x_data)

# ============================================================
# 2. 하이퍼파라미터
#    base_lr  : warmup 후 도달할 목표 lr
#    warmup_ratio : 전체 학습 중 warmup 에 할당할 비율 (예: 0.3 = 30%)
# ============================================================
epochs = 1000
batch_size = 4
base_lr = 0.001
warmup_ratio = 0.3

# ============================================================
# 3. 파생 파라미터 (per-step 스케줄링용)
#    total_steps  = epochs × (epoch 당 batch 수)
#    warmup_steps = total_steps × warmup_ratio
# ============================================================
total_steps = math.ceil(n_samples / batch_size) * epochs
warmup_steps = math.ceil(total_steps * warmup_ratio)

# ============================================================
# 4. 파라미터 초기화
# ============================================================
random.seed(42)
w = random.random()
b = random.random()

print("-" * 10)
print("Linear Warmup")
print("-" * 10)
print(f"데이터 수: {n_samples}개, batch_size: {batch_size}")
print(f"전체 step: {total_steps}, warmup step: {warmup_steps} ({warmup_ratio*100:.0f}%)")
print(f"초기 w: {w:.4f}, 초기 b: {b:.4f}")
print()

# ============================================================
# 5. 학습 루프 (Mini-batch SGD + Linear Warmup)
#    핵심: 매 batch (= step) 마다 lr 을 다시 계산.
# ============================================================
global_step = 0
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0

    # Shuffle sample indices
    sample_indices = list(range(n_samples))
    random.shuffle(sample_indices)

    for batch_start in range(0, n_samples, batch_size):
        global_step += 1

        # Linear warmup: lr 을 0 에서 base_lr 까지 선형 증가, 이후 유지
        lr = base_lr * min(1, global_step / warmup_steps)

        batch_indices = sample_indices[batch_start: batch_start + batch_size]
        cur_batch_size = len(batch_indices)

        w_grad = 0.0
        b_grad = 0.0

        # Loop: sample
        for sample_idx in batch_indices:
            x = x_data[sample_idx]
            y = y_data[sample_idx]

            # Prediction
            y_pred = x * w + b

            # Error
            error = y_pred - y

            # Loss
            epoch_loss += error ** 2

            # Gradient (factor 2 포함, (2/m) Σ 컨벤션)
            w_grad += error * x * 2
            b_grad += error * 2

        # Average gradient over batch
        w_grad /= cur_batch_size
        b_grad /= cur_batch_size

        # Parameter update (warmup 이 반영된 lr 사용)
        w -= lr * w_grad
        b -= lr * b_grad

    # Average loss over epoch
    epoch_loss /= n_samples

    # Print progress every 100 epochs
    if epoch % 100 == 0:
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
print("warmup 효과:")
print(f"  처음 {warmup_steps} step 동안 lr 이 0 → {base_lr} 로 선형 증가")
print(f"  이후 {total_steps - warmup_steps} step 동안 lr = {base_lr} 유지")
