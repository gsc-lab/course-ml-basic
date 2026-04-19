"""
TITLE: Mini-batch GD(미니배치 경사하강법) 구현
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, optimization, batch
EVAL: stdio

DESCRIPTION:
Mini-batch Gradient Descent를 구현하세요.

Mini-batch GD는 데이터를 batch_size 크기로 나누어 배치 단위로 업데이트합니다.
BGD와 SGD의 중간 지점입니다:
  - 배치 내에서는 BGD처럼 기울기 "평균"을 구하고
  - 배치마다 업데이트하므로 SGD처럼 빈번하게 움직인다

  ┌───────────────┬──────────────────────────────────┐
  │  배치 내 처리  │  BGD와 동일: 기울기 누적 → 평균    │
  │  업데이트 빈도  │  SGD와 유사: 배치마다 즉시 업데이트  │
  └───────────────┴──────────────────────────────────┘

핵심 공식 (배치 내):
  w_grad += 2 * x * error   (누적)
  b_grad += 2 * error       (누적)
  w_grad /= batch_size      (평균)
  b_grad /= batch_size      (평균)
  w = w - learning_rate * w_grad
  b = b - learning_rate * b_grad

아래 TODO 부분을 채워 Mini-batch GD 학습을 완성하세요.

예시 출력:
  Epoch    1 | Loss: 25.7946 | w: 0.55, b: 0.04
  ...
  최종 파라미터: w = 0.55, b = 1.38

생각해보기:
  1. SGD(batch_size=1)와 비교하면 loss가 더 안정적인가?
  2. batch_size를 1, 10, 20으로 바꿔보자.
     batch_size=1이면? → SGD와 동일해진다
     batch_size=20이면? → BGD와 동일해진다
  3. 실무에서 batch_size = 32, 64, 128을 주로 사용하는 이유는?
     (안정성 vs 수렴 속도 vs GPU 활용률)
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: "----------\nMini-batch Gradient Descent 학습\n----------\n데이터 수: 20개, epochs: 500, batch_size: 4\nlearning_rate: 0.001\nepoch당 업데이트 횟수: 5회\n\nEpoch    1 | Loss: 25.7946 | w: 0.55, b: 0.04\nEpoch  100 | Loss: 0.5934 | w: 0.62, b: 0.45\nEpoch  200 | Loss: 0.3660 | w: 0.60, b: 0.77\nEpoch  300 | Loss: 0.2454 | w: 0.58, b: 1.02\nEpoch  400 | Loss: 0.1562 | w: 0.56, b: 1.22\nEpoch  500 | Loss: 0.1067 | w: 0.55, b: 1.38\n\n최종 파라미터: w = 0.55, b = 1.38\n최종 Loss: 0.1067\n정답과 비교: w(정답=0.5), b(정답=2.0)"

import random

# ============================================================
# 데이터셋 (20개)
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# Mini-batch GD 학습 함수
# ============================================================
def train_minibatch(x_data, y_data, learning_rate, epochs, batch_size,
                    w_init, b_init):
    """
    Mini-batch Gradient Descent로 학습한다.

    데이터를 batch_size 크기로 나누어, 배치 단위로 업데이트한다.
    → epoch당 업데이트 (n / batch_size)회

    BGD와 SGD의 중간 지점이다:
      BGD : 전체 n개 → 기울기 평균 → 1번 업데이트
      Mini-batch : batch_size개 → 기울기 평균 → 배치마다 업데이트
      SGD : 1개 → 기울기 그대로 → 매번 업데이트

    매 epoch마다 데이터 순서를 섞어야 한다.

    Args:
        x_data: 입력 데이터 리스트
        y_data: 정답 데이터 리스트
        learning_rate: 학습률
        epochs: 반복 횟수
        batch_size: 배치 크기
        w_init: 초기 기울기
        b_init: 초기 절편
    Returns:
        (w, b, loss_history) 튜플
    """
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        # TODO: 데이터 인덱스를 섞으세요.
        #   힌트: indices = list(range(n)) 를 만들고
        #         random.shuffle(indices) 로 섞는다.
        pass

        epoch_loss = 0.0

        # TODO: 배치 단위로 처리하세요.
        #   for start in range(0, n, batch_size):
        #
        #     1) 배치 인덱스 추출:
        #          batch_indices = indices[start:start + batch_size]
        #          bs = len(batch_indices)   ← 마지막 배치는 크기가 다를 수 있다
        #
        #     2) 배치 내 기울기 누적 (BGD와 동일한 방식):
        #          w_grad, b_grad, batch_loss = 0.0, 0.0, 0.0
        #          for idx in batch_indices:
        #              x, y = x_data[idx], y_data[idx]
        #              predict = w * x + b
        #              error = predict - y
        #              w_grad += 2 * x * error
        #              b_grad += 2 * error
        #              batch_loss += error ** 2
        #
        #     3) 배치 평균:
        #          w_grad /= bs
        #          b_grad /= bs
        #
        #     4) 배치마다 업데이트:
        #          w = w - learning_rate * w_grad
        #          b = b - learning_rate * b_grad
        #
        #     5) 손실 누적:
        #          epoch_loss += batch_loss
        pass

        loss = epoch_loss / n
        loss_history.append(loss)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: {w:.2f}, b: {b:.2f}")

    return w, b, loss_history


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    w_init, b_init = 0.0, 0.0
    learning_rate = 0.001
    epochs = 500
    batch_size = 4

    n = len(x_data)
    updates_per_epoch = (n + batch_size - 1) // batch_size

    print("-" * 10)
    print("Mini-batch Gradient Descent 학습")
    print("-" * 10)
    print(f"데이터 수: {n}개, epochs: {epochs}, batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"epoch당 업데이트 횟수: {updates_per_epoch}회")
    print()

    random.seed(42)
    w, b, loss_history = train_minibatch(
        x_data, y_data, learning_rate, epochs, batch_size, w_init, b_init
    )

    print()
    print(f"최종 파라미터: w = {w:.2f}, b = {b:.2f}")
    print(f"최종 Loss: {loss_history[-1]:.4f}")
    print(f"정답과 비교: w(정답=0.5), b(정답=2.0)")

