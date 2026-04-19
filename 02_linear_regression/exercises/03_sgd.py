"""
TITLE: SGD(확률적 경사하강법) 구현
DIFFICULTY: medium
TAGS: gradient-descent, sgd, optimization, shuffle
EVAL: stdio

DESCRIPTION:
SGD(Stochastic Gradient Descent)를 구현하세요.

SGD는 데이터를 한 개씩 보고 즉시 파라미터를 업데이트합니다.
  - 매 epoch마다 데이터 순서를 섞는다 (shuffle)
  - 샘플 1개의 기울기로 즉시 w, b를 업데이트한다
  - epoch당 업데이트 횟수 = 데이터 수(n)

핵심 공식:
  predict = w * x + b
  error = predict - y
  w_grad = 2 * x * error    (평균 없이 그대로)
  b_grad = 2 * error
  w = w - learning_rate * w_grad
  b = b - learning_rate * b_grad

아래 TODO 부분을 채워 SGD 학습을 완성하세요.

예시 출력:
  Epoch    1 | Loss: 38.5324 | w: 0.29, b: 0.02
  ...
  최종 파라미터: w = 0.60, b = 0.76

생각해보기:
  1. BGD에 비해 loss가 울퉁불퉁한 이유는 무엇인가?
  2. learning_rate를 0.001로 올리면 어떻게 되는가?
  3. epoch당 업데이트가 20회인데, BGD는 1회이다.
     같은 epoch 수라면 SGD가 더 많이 움직이는 셈이다.
     이것이 장점일까, 단점일까?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: "----------\nSGD (Stochastic Gradient Descent) 학습\n----------\n데이터 수: 20개, epochs: 500\nlearning_rate: 0.0001\nepoch당 업데이트 횟수: 20회 (= 데이터 수)\n\nEpoch    1 | Loss: 38.5324 | w: 0.29, b: 0.02\nEpoch  100 | Loss: 0.7482 | w: 0.64, b: 0.22\nEpoch  200 | Loss: 0.6241 | w: 0.63, b: 0.37\nEpoch  300 | Loss: 0.5220 | w: 0.61, b: 0.52\nEpoch  400 | Loss: 0.4370 | w: 0.61, b: 0.65\nEpoch  500 | Loss: 0.3659 | w: 0.60, b: 0.76\n\n최종 파라미터: w = 0.60, b = 0.76\n최종 Loss: 0.3659\n정답과 비교: w(정답=0.5), b(정답=2.0)"

import random

# ============================================================
# 데이터셋 (20개)
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# SGD 학습 함수
# ============================================================
def train_sgd(x_data, y_data, learning_rate, epochs, w_init, b_init):
    """
    Stochastic Gradient Descent로 학습한다.

    샘플 1개마다 즉시 파라미터를 업데이트한다.
    → epoch당 업데이트 n회 (데이터 수만큼)

    BGD와의 차이점:
      BGD : 전체 데이터의 기울기 "평균"을 구한 뒤 1번 업데이트
      SGD : 샘플 1개의 기울기로 "즉시" 업데이트 (평균 없음)

    매 epoch마다 데이터 순서를 섞어야(shuffle) 편향을 줄일 수 있다.

    Args:
        x_data: 입력 데이터 리스트
        y_data: 정답 데이터 리스트
        learning_rate: 학습률
        epochs: 반복 횟수
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

        # TODO: 섞인 순서대로 샘플을 하나씩 처리하세요.
        #   for idx in indices:
        #       1) x, y 가져오기: x = x_data[idx], y = y_data[idx]
        #       2) 예측: predict = w * x + b
        #       3) 오차: error = predict - y
        #       4) 기울기 계산 (평균 없이 그대로):
        #            w_grad = 2 * x * error
        #            b_grad = 2 * error
        #       5) 즉시 업데이트:
        #            w = w - learning_rate * w_grad
        #            b = b - learning_rate * b_grad
        #       6) 손실 누적: epoch_loss += error ** 2
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
    learning_rate = 0.0001
    epochs = 500

    print("-" * 10)
    print("SGD (Stochastic Gradient Descent) 학습")
    print("-" * 10)
    print(f"데이터 수: {len(x_data)}개, epochs: {epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"epoch당 업데이트 횟수: {len(x_data)}회 (= 데이터 수)")
    print()

    random.seed(42)
    w, b, loss_history = train_sgd(
        x_data, y_data, learning_rate, epochs, w_init, b_init
    )

    print()
    print(f"최종 파라미터: w = {w:.2f}, b = {b:.2f}")
    print(f"최종 Loss: {loss_history[-1]:.4f}")
    print(f"정답과 비교: w(정답=0.5), b(정답=2.0)")

