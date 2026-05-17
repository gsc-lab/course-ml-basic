"""
Multiple Linear Regression (MLR) - 다중선형회귀
==============================================
입력 변수가 1개 → 여러 개로 확장된 형태.
"공부시간"만으로 성적을 예측하는 게 아니라,
"공부시간·출석률·수면시간·이전 성적" 4가지를 동시에 보고 예측한다.

핵심 개념:
  가설(Hypothesis):
    H(x) = w1·x1 + w2·x2 + w3·x3 + w4·x4 + b
    → 단일 변수와 구조는 같다. 단지 (w·x) 항이 4개로 늘어났을 뿐.

  손실함수(Loss):
    MSE = (1/n) Σ (H(x) - y)²    (단일 변수와 동일)

  옵티마이저(Gradient Descent):
    각 가중치는 "자기 자신과 곱해진 입력"에 대해서만 미분된다.
      ∂Loss/∂w1 = (2/n) Σ (H(x) - y) · x1
      ∂Loss/∂w2 = (2/n) Σ (H(x) - y) · x2
      ∂Loss/∂w3 = (2/n) Σ (H(x) - y) · x3
      ∂Loss/∂w4 = (2/n) Σ (H(x) - y) · x4
      ∂Loss/∂b  = (2/n) Σ (H(x) - y)

특징 정리:
  - 오차 (error = H(x) - y) 는 모든 파라미터의 기울기에 공통으로 들어간다.
  - 각 가중치 w_k 의 기울기는 "오차 × 자기 입력 x_k" 형태.
    → 입력값이 큰 feature 일수록 그 가중치의 기울기도 커진다.
  - bias b 는 입력과 곱해지지 않으므로 "오차 자체" 가 기울기.
  - 5개 파라미터(w1, w2, w3, w4, b) 는 같은 epoch 의 (w, b) 로 계산된
    기울기를 사용해 동시에 업데이트된다 (= 동기 업데이트).
"""
import random

# ============================================================
# 1. 데이터셋 (가상의 학생 성적 데이터, 30명)
#    feature 4개 (모두 비슷한 스케일로 단순화):
#      x1 공부시간 (h)
#      x2 출석률   (10점 만점, 9.0 == 90%)
#      x3 수면시간 (h)
#      x4 이전 성적 (10점 만점, 8.0 == 80점)
#    정답:
#      y = 2·x1 + 1.5·x2 + 1·x3 + 1.5·x4 + 5
#
#    각 feature 는 서로 독립적으로 무작위 생성한다.
#    (feature 들이 강하게 상관되어 있으면 각 weight 가 유일하게
#     결정되지 않고 여러 조합이 같은 예측값을 낼 수 있다.)
# ============================================================
random.seed(0)
n = 30
x1_data = [random.randint(1, 10) for _ in range(n)]   # 공부시간
x2_data = [random.randint(5, 10) for _ in range(n)]   # 출석률
x3_data = [random.randint(4, 9)  for _ in range(n)]   # 수면시간
x4_data = [random.randint(5, 10) for _ in range(n)]   # 이전 성적

# 정답 식 + 약간의 노이즈로 y 생성
y_data = [
    2 * x1 + 1.5 * x2 + 1 * x3 + 1.5 * x4 + 5 + random.uniform(-0.5, 0.5)
    for x1, x2, x3, x4 in zip(x1_data, x2_data, x3_data, x4_data)
]

# ============================================================
# 2. 하이퍼파라미터
# ============================================================
learning_rate = 0.001
epochs = 30000

# ============================================================
# 3. 파라미터 초기화
#    feature 마다 가중치가 1개씩 필요 → w 가 4개로 늘어난다.
#    bias 는 단일 변수와 동일하게 1개.
# ============================================================
random.seed(42)
w1 = random.random()
w2 = random.random()
w3 = random.random()
w4 = random.random()
b  = random.random()

print("-" * 10)
print("Multiple Linear Regression (BGD)")
print("-" * 10)
print(f"데이터 수: {n}개,  feature 수: 4")
print(f"초기 w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f}, b={b:.4f}")
print()

# ============================================================
# 4. 학습 루프 (Batch Gradient Descent)
#    핵심: 매 epoch 마다 전체 데이터를 한 바퀴 돌면서
#         5개의 기울기(w1, w2, w3, w4, b) 를 모두 누적한 뒤
#         평균을 내서 한꺼번에 업데이트한다.
# ============================================================
for epoch in range(1, epochs + 1):
    # --- 5개의 gradient 누적 슬롯 (feature 마다 1개씩 + bias) ---
    w1_grad = 0.0
    w2_grad = 0.0
    w3_grad = 0.0
    w4_grad = 0.0
    b_grad  = 0.0
    loss = 0.0

    for x1, x2, x3, x4, y in zip(x1_data, x2_data, x3_data, x4_data, y_data):
        # --- 가설: 모든 feature 의 가중합 + bias ---
        predict = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + b

        # --- 오차: 모든 파라미터의 기울기에 공통으로 사용되는 값 ---
        error = predict - y

        # --- 각 가중치의 기울기 = (오차) × (자기 자신의 입력) ---
        #     같은 error 라도 곱해지는 입력이 다르므로 기울기가 다르다.
        w1_grad += 2 * x1 * error
        w2_grad += 2 * x2 * error
        w3_grad += 2 * x3 * error
        w4_grad += 2 * x4 * error
        # bias 의 기울기는 입력과 곱해지지 않으므로 오차 자체
        b_grad  += 2 * error

        loss += error ** 2

    # --- 평균 (전체 데이터의 평균 기울기) ---
    w1_grad /= n
    w2_grad /= n
    w3_grad /= n
    w4_grad /= n
    b_grad  /= n
    loss    /= n

    # --- 1번 업데이트: 5개 파라미터를 동시에 자기 기울기 방향으로 ---
    #     모든 파라미터는 "이번 epoch 의 (w, b)" 로 계산된 기울기를
    #     사용해야 한다. 한 파라미터를 먼저 갱신하고 그 값으로 다음
    #     기울기를 다시 계산하는 식으로 하면 안 됨 (= 동기 업데이트).
    w1 = w1 - learning_rate * w1_grad
    w2 = w2 - learning_rate * w2_grad
    w3 = w3 - learning_rate * w3_grad
    w4 = w4 - learning_rate * w4_grad
    b  = b  - learning_rate * b_grad

    if epoch % 3000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} "
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
    pred = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + b
    print(f"  (x1={x1}, x2={x2}, x3={x3}, x4={x4}) → 예측: {pred:.2f}, 정답: {y:.2f}")
