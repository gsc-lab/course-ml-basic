"""
Mini-batch Gradient Descent - 미니배치 경사하강법
=================================================
BGD와 SGD의 장점을 결합한 방식이다.
전체 데이터를 batch_size 크기로 나누어, 배치 단위로 업데이트한다.

핵심 개념:
  BGD, SGD, Mini-batch GD는 모두 같은 경사하강법이다.
  차이는 오직 **한 번에 몇 개의 데이터를 보고 업데이트하느냐**이다.

  ┌──────────────┬──────────────┬──────────────────────┐
  │    방식       │  batch_size  │  epoch당 업데이트 횟수  │
  ├──────────────┼──────────────┼──────────────────────┤
  │  BGD         │  n (전체)     │  1회                  │
  │  Mini-batch  │  k (예: 4)   │  n/k회 (예: 5회)      │
  │  SGD         │  1           │  n회                  │
  └──────────────┴──────────────┴──────────────────────┘

  장점: BGD의 안정성 + SGD의 빈번한 업데이트
       → 실무에서 가장 많이 사용되는 방식이다.

특징 정리:
  - batch_size개씩 묶어서 평균 기울기 계산 → epoch당 (n/batch_size)번 업데이트
  - BGD보다 업데이트가 빈번하여 빠르게 수렴
  - SGD보다 기울기가 안정적 (여러 샘플의 평균)
  - 실무에서 가장 널리 사용되는 방식 (보통 batch_size = 32, 64, 128)

batch_size에 따른 스펙트럼:
  batch_size = n (전체) → BGD
  batch_size = k        → Mini-batch GD
  batch_size = 1        → SGD
"""
import random

# ============================================================
# 1. 데이터셋 (BGD, SGD 예제와 동일)
#    정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]

# ============================================================
# 2. 하이퍼파라미터
#    batch_size가 새로 추가된다.
#    batch_size = 20이면 BGD, batch_size = 1이면 SGD와 동일하다.
# ============================================================
learning_rate = 0.001
epochs = 300
batch_size = 4  # 20개 데이터를 4개씩 5묶음으로 나눈다
n = len(x_data)

# ============================================================
# 3. 파라미터 초기화 (BGD, SGD 예제와 동일한 초기값)
# ============================================================
random.seed(42)
w = random.random()
b = random.random()

print("-" * 10)
print("Mini-batch Gradient Descent")
print("-" * 10)
print(f"데이터 수: {n}개, batch_size: {batch_size}")
print(f"epoch당 업데이트 횟수: {n // batch_size}회")
print(f"초기 w: {w:.4f}, 초기 b: {b:.4f}")
print()

# ============================================================
# 4. 학습 루프 (Mini-batch GD)
#    핵심: 데이터를 batch_size만큼 묶어서 처리한다.
#          배치 내에서는 BGD처럼 평균 기울기를 구하고,
#          배치마다 업데이트하므로 SGD처럼 빈번하게 움직인다.
# ============================================================
for epoch in range(1, epochs + 1):
    # --- 매 epoch마다 데이터 순서를 섞는다 ---
    indices = list(range(n))
    random.shuffle(indices)

    epoch_loss = 0.0

    # --- 배치 단위로 처리 ---
    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        bs = len(batch_indices)  # 마지막 배치는 batch_size보다 작을 수 있다

        # 배치 내 기울기 누적 (BGD와 동일한 방식)
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

        # 배치 평균
        w_grad /= bs
        b_grad /= bs

        # --- 배치마다 업데이트 (SGD처럼 빈번하게) ---
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        epoch_loss += batch_loss

    epoch_loss /= n

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.6f} | w: {w:.4f}, b: {b:.4f}")

# ============================================================
# 5. 결과
# ============================================================
print()
print("-" * 10)
print("학습 완료")
print(f"  학습된 w: {w:.4f}  (정답: 0.5)")
print(f"  학습된 b: {b:.4f}  (정답: 2.0)")
