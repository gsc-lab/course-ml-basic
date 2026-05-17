"""
Multiple Linear Regression + Mini-batch SGD
================================================
04 (Mini-batch GD, 1 feature) 와 06 (MLR, BGD) 를 결합한 종합편.

이전 단계 정리:
  04: 1-feature 데이터를 batch_size 로 묶어 Mini-batch GD 적용
  06: 4-feature 데이터를 전체 평균 (BGD) 으로 MLR 학습
  → 07: 4-feature 데이터를 Mini-batch SGD 로 학습 (실무에서 가장 흔한 형태)

핵심 변화 (06 BGD 대비):
  - epoch 마다 sample 순서 shuffle
  - 전체 평균 1회 업데이트 → batch 단위 (n_samples / batch_size) 회 업데이트
  - 같은 가설 / 손실 / gradient 식 (factor 2 포함)

가설 (Hypothesis):
  H(x) = w1·x1 + w2·x2 + w3·x3 + w4·x4 + b

손실 함수 (Loss):
  MSE = (1/n) Σ (H(x) - y)²

Gradient (per-sample, factor 2 포함):
  ∂L/∂wk = 2 · (H(x) - y) · xk
  ∂L/∂b  = 2 · (H(x) - y)
  → batch 안에서 누적 후 batch_size 로 나눠 평균
"""
import random

# ============================================================
# 1. 데이터셋 (06 과 동일: 학생 성적 30명)
#    feature 4개 (공부시간 / 출석률 / 수면시간 / 이전 성적)
#    정답 식: y = 2·x1 + 1.5·x2 + 1·x3 + 1.5·x4 + 5
# ============================================================
random.seed(0)
n_samples = 30
x1_data = [random.randint(1, 10) for _ in range(n_samples)]   # 공부시간
x2_data = [random.randint(5, 10) for _ in range(n_samples)]   # 출석률
x3_data = [random.randint(4, 9)  for _ in range(n_samples)]   # 수면시간
x4_data = [random.randint(5, 10) for _ in range(n_samples)]   # 이전 성적

y_data = [
    2 * x1 + 1.5 * x2 + 1 * x3 + 1.5 * x4 + 5 + random.uniform(-0.5, 0.5)
    for x1, x2, x3, x4 in zip(x1_data, x2_data, x3_data, x4_data)
]

# ============================================================
# 2. 하이퍼파라미터
#    batch_size 가 06 (BGD) 대비 추가된 핵심 파라미터.
#    batch_size = n_samples 이면 BGD, batch_size = 1 이면 SGD.
# ============================================================
lr = 0.001
epochs = 10000
batch_size = 6   # 30개를 6개씩 5묶음

# ============================================================
# 3. 파라미터 초기화 (06 과 동일한 초기값을 위해 동일 seed)
# ============================================================
random.seed(42)
w1 = random.random()
w2 = random.random()
w3 = random.random()
w4 = random.random()
b  = random.random()

print("-" * 10)
print("Multiple Linear Regression (Mini-batch SGD)")
print("-" * 10)
print(f"데이터 수: {n_samples}개,  feature 수: 4,  batch_size: {batch_size}")
print(f"epoch당 업데이트 횟수: {n_samples // batch_size}회")
print(f"초기 w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f}, b={b:.4f}")
print()

# ============================================================
# 4. 학습 루프 (Mini-batch SGD)
#    구조: epoch → batch → sample 의 3중 루프
#      - epoch:  전체 데이터 1바퀴
#      - batch:  batch_size 묶음 단위로 gradient 평균 후 업데이트
#      - sample: 배치 안의 각 샘플에서 gradient 누적
# ============================================================
# Loop: epoch
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0

    # Shuffle sample indices
    sample_indices = list(range(n_samples))
    random.shuffle(sample_indices)

    # Loop: batch
    for batch_start in range(0, n_samples, batch_size):
        w1_grad = 0.0
        w2_grad = 0.0
        w3_grad = 0.0
        w4_grad = 0.0
        b_grad  = 0.0

        batch_indices = sample_indices[batch_start: batch_start + batch_size]
        cur_batch_size = len(batch_indices)

        # Loop: sample
        for sample_idx in batch_indices:
            x1 = x1_data[sample_idx]
            x2 = x2_data[sample_idx]
            x3 = x3_data[sample_idx]
            x4 = x4_data[sample_idx]
            y  = y_data[sample_idx]

            # Prediction
            y_pred = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + b

            # Error (모든 파라미터 gradient 에 공통으로 들어감)
            error = y_pred - y

            # Loss
            epoch_loss += error ** 2

            # Gradient (per-sample, factor 2 포함)
            # 각 weight 의 gradient = 오차 × 자기 자신의 입력
            w1_grad += error * x1 * 2
            w2_grad += error * x2 * 2
            w3_grad += error * x3 * 2
            w4_grad += error * x4 * 2
            # bias 는 입력과 곱해지지 않으므로 오차 자체
            b_grad  += error * 2

        # Gradient average (배치 평균)
        w1_grad /= cur_batch_size
        w2_grad /= cur_batch_size
        w3_grad /= cur_batch_size
        w4_grad /= cur_batch_size
        b_grad  /= cur_batch_size

        # Parameter update (5개 파라미터 동시 갱신 = 동기 업데이트)
        w1 -= lr * w1_grad
        w2 -= lr * w2_grad
        w3 -= lr * w3_grad
        w4 -= lr * w4_grad
        b  -= lr * b_grad

    # Average loss over epoch
    epoch_loss /= n_samples

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {epoch_loss:.4f} "
              f"| w1={w1:.3f} w2={w2:.3f} w3={w3:.3f} w4={w4:.3f} b={b:.3f}")

# ============================================================
# 5. 결과
# ============================================================
print()
print("-" * 10)
print("학습 완료")
print(f"  학습된 w1: {w1:.4f}  (정답: 2.0)   ← 공부시간")
print(f"  학습된 w2: {w2:.4f}  (정답: 1.5)   ← 출석률")
print(f"  학습된 w3: {w3:.4f}  (정답: 1.0)   ← 수면시간")
print(f"  학습된 w4: {w4:.4f}  (정답: 1.5)   ← 이전 성적")
print(f"  학습된 b : {b:.4f}  (정답: 5.0)")
print()
print("예측 결과 (앞 5개):")
for x1, x2, x3, x4, y in zip(x1_data[:5], x2_data[:5], x3_data[:5], x4_data[:5], y_data[:5]):
    y_pred = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + b
    print(f"  (x1={x1}, x2={x2}, x3={x3}, x4={x4}) → 예측: {y_pred:.2f}, 정답: {y:.2f}")
