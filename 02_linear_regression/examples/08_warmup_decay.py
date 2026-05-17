"""
Linear Warmup + Linear Decay - 실무 표준 lr 스케줄
=====================================================
06 (Warmup) 와 07 (Decay) 의 결합. 학습 초반엔 warmup 으로 안정화,
이후엔 decay 로 미세조정. 실무에서 가장 흔히 쓰이는 lr 스케줄.

핵심 개념:
  warmup 과 decay 는 보완 관계.
    - warmup : 학습 초반 발산 방지 (0 → base_lr)
    - decay  : 학습 후반 미세조정  (base_lr → 0)
  둘을 합치면 lr 이 삼각형 모양의 궤적을 그린다:
    0  →  base_lr  →  0

  PyTorch `OneCycleLR`, HuggingFace `get_linear_schedule_with_warmup`,
  BERT/GPT/T5 등 대형 모델 학습이 모두 이 스케줄을 채택.

수식 (per-step 적용, HuggingFace 컨벤션):
  global_step ≤ warmup_steps:
    lr = base_lr × (global_step / warmup_steps)              ← 선형 증가
  global_step > warmup_steps:
    decay_progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr × max(0, 1 - decay_progress)                ← 선형 감소

  - step 1                       : lr ≈ 0
  - step warmup_steps            : lr = base_lr (peak)
  - step total_steps             : lr = 0

용어:
  - global_step  : 학습 전체에서 누적된 update 횟수
  - total_steps  : 전체 학습의 총 update 횟수
  - warmup_steps : warmup 구간의 step 수 = total_steps × warmup_ratio
  - decay_steps  : decay 구간의 step 수 = total_steps - warmup_steps
  - base_lr      : 최대 lr (warmup 후 도달, decay 시작점)

특징:
  - 06 + 07 의 자연스러운 합성. 각 단계가 독립적으로 동작.
  - warmup_ratio 가 클수록 warmup 구간 길어지고 decay 구간 짧아짐.
  - 실무에서 warmup_ratio = 0.06 ~ 0.1 가 일반적 (Transformer 학습).
  - 본 예제는 학습 효과 가시화를 위해 0.3 사용.
"""
import random
import math

# ============================================================
# 1. 데이터셋 (04, 06, 07 과 동일)
# ============================================================
random.seed(0)
x_data = [x for x in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]
n_samples = len(x_data)

# ============================================================
# 2. 하이퍼파라미터
# ============================================================
epochs = 3000   # decay 구간이 lr 을 줄이므로 06 (warmup) 보다 더 많은 epoch 필요
batch_size = 4
base_lr = 0.001
warmup_ratio = 0.3   # 전체 학습 중 warmup 비율 (나머지는 decay)

# ============================================================
# 3. 파생 파라미터
# ============================================================
total_steps = math.ceil(n_samples / batch_size) * epochs
warmup_steps = math.ceil(total_steps * warmup_ratio)
decay_steps = total_steps - warmup_steps   # warmup 이후 남은 step

# ============================================================
# 4. 파라미터 초기화
# ============================================================
random.seed(42)
w = random.random()
b = random.random()

print("-" * 10)
print("Linear Warmup + Linear Decay")
print("-" * 10)
print(f"데이터 수: {n_samples}개, batch_size: {batch_size}")
print(f"전체 step: {total_steps}")
print(f"  warmup: {warmup_steps} ({warmup_ratio*100:.0f}%)")
print(f"  decay : {decay_steps} ({(1-warmup_ratio)*100:.0f}%)")
print(f"base_lr: {base_lr}")
print(f"초기 w: {w:.4f}, 초기 b: {b:.4f}")
print()

# ============================================================
# 5. 학습 루프 (Mini-batch SGD + Warmup + Decay)
#    lr 스케줄:
#      step ≤ warmup_steps  → 선형 증가 (0 → base_lr)
#      step > warmup_steps  → 선형 감소 (base_lr → 0)
# ============================================================
global_step = 0
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0

    sample_indices = list(range(n_samples))
    random.shuffle(sample_indices)

    for batch_start in range(0, n_samples, batch_size):
        global_step += 1

        # Warmup + Decay schedule (per-step)
        if global_step <= warmup_steps:
            # Linear warmup phase
            lr = base_lr * (global_step / warmup_steps)
        else:
            # Linear decay phase
            decay_progress = (global_step - warmup_steps) / decay_steps
            lr = base_lr * max(0, 1 - decay_progress)

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
        phase = "warmup" if global_step <= warmup_steps else "decay"
        print(f"Epoch {epoch:4d} | step {global_step:5d} | {phase:6s} | "
              f"lr {lr:.6f} | loss {epoch_loss:.4f} | w {w:.4f}, b {b:.4f}")

# ============================================================
# 6. 결과
# ============================================================
print()
print("-" * 10)
print("학습 완료")
print(f"  학습된 w: {w:.4f}  (정답: 0.5)")
print(f"  학습된 b: {b:.4f}  (정답: 2.0)")
print()
print("lr 궤적 (삼각형):")
print(f"  step 1              : lr ≈ 0")
print(f"  step {warmup_steps:5d} (warmup 끝): lr = {base_lr} (peak)")
print(f"  step {total_steps:5d} (학습 끝)  : lr ≈ 0")
