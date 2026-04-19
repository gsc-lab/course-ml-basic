"""
실습: 하이퍼파라미터(Epoch, Learning Rate)의 이해
==================================================

배경:
  경사하강법에서 학습의 질을 결정하는 것은 모델 구조뿐만 아니라
  **하이퍼파라미터(hyperparameter)** 설정이다.
  하이퍼파라미터란 학습 과정에서 사람이 직접 정해줘야 하는 값으로,
  모델이 스스로 학습하는 w, b와 구분된다.

  대표적인 하이퍼파라미터:
    - Epoch: 전체 데이터를 몇 번 반복 학습할 것인가
    - Learning Rate: 한 번의 업데이트에서 얼마나 크게 이동할 것인가

목표:
  - 수업 시연 코드(class/linear_regression_gradient_descent.py)를 기반으로
    학습 함수를 완성한다.
  - epoch과 learning_rate를 다양하게 바꿔가며 학습 결과를 비교한다.
  - 각 하이퍼파라미터가 학습에 미치는 영향을 직접 관찰한다.

지시사항:
  - TODO 로 표시된 부분을 채워 넣으세요.
  - 각 함수의 docstring을 참고하세요.

생각해보기:
  1. epoch=10일 때와 epoch=2000일 때 loss 차이가 얼마나 나는가?
  2. learning_rate=0.0001과 0.01의 결과를 비교해보자.
     같은 epoch인데 왜 결과가 다른가?
  3. learning_rate=1.0일 때 어떤 일이 벌어지는가?
     loss가 줄어드는가, 늘어나는가?
"""
import random

# ============================================================
# 데이터셋
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
x_data = [1, 2, 3, 4]
y_data = [2.5, 3.0, 3.5, 4.0]


# ============================================================
# 함수 정의
# ============================================================

def train(x_data, y_data, learning_rate, epochs):
    """
    경사하강법으로 선형 회귀 모델을 학습한다.

    수업 시연 코드의 학습 루프를 함수로 분리한 것이다.
    매 epoch마다 전체 데이터에 대해:
      1) 예측값 계산  →  H(x) = w * x + b
      2) 오차 계산    →  error = H(x) - y
      3) 기울기 누적  →  ∂Loss/∂w, ∂Loss/∂b
      4) 평균 후 파라미터 업데이트

    Args:
        x_data: 입력 데이터 리스트
        y_data: 실제값 리스트
        learning_rate: 학습률
        epochs: 반복 횟수
    Returns:
        (w, b, loss_history) 튜플
        loss_history는 매 epoch의 손실값 리스트
    """
    random.seed(42)
    w = random.random()
    b = random.random()
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        w_grad = 0.0
        b_grad = 0.0
        loss = 0.0

        for x, y in zip(x_data, y_data):
            # TODO: 예측값을 계산하세요. H(x) = w * x + b
            predict = None

            # TODO: 오차를 계산하세요. error = predict - y
            error = None

            # TODO: w의 기울기를 누적하세요. (2 * x * error)
            pass

            # TODO: b의 기울기를 누적하세요. (2 * error)
            pass

            # TODO: loss를 누적하세요. (error ** 2)
            pass

        # TODO: 기울기와 loss를 샘플 수(n)로 나누어 평균을 구하세요.
        pass

        # TODO: w와 b를 업데이트하세요.
        #   w = w - learning_rate * w_grad
        #   b = b - learning_rate * b_grad
        pass

        loss_history.append(loss)

        # 발산 감지: loss가 너무 커지면 학습 중단
        if loss > 1e10:
            print(f"  ⚠ epoch {epoch}에서 발산 감지! (loss = {loss:.2e}) 학습을 중단합니다.")
            break

    return w, b, loss_history


def print_result(label, w, b, loss_history):
    """학습 결과를 출력한다."""
    print(f"\n[ {label} ]")
    print(f"  최종 w: {w:.4f} (정답: 0.5)")
    print(f"  최종 b: {b:.4f} (정답: 2.0)")
    print(f"  최종 loss: {loss_history[-1]:.6f}")
    print(f"  loss 변화: {loss_history[0]:.4f} → {loss_history[-1]:.6f}")


# ============================================================
# 실험 1: Epoch의 영향
#   - learning_rate는 고정, epoch만 변경
#   - epoch이 너무 적으면? 충분하면?
# ============================================================
def experiment_epoch():
    """
    Epoch(반복 횟수)가 학습에 미치는 영향을 관찰한다.

    Epoch이란?
      전체 데이터를 한 번 학습하는 것을 1 epoch이라 한다.
      epoch이 적으면 파라미터가 정답에 도달하기 전에 학습이 끝나고 (underfitting),
      충분히 크면 정답에 수렴한다.

    TODO: epoch_list에 있는 각 epoch 값으로 학습을 실행하고 결과를 비교하세요.
          train() 함수를 호출하고, print_result()로 결과를 출력하면 됩니다.

    힌트: learning_rate는 0.01로 고정하세요.
    """
    print("-" * 10)
    print("실험 1: Epoch의 영향 (learning_rate = 0.01 고정)")
    print("-" * 10)

    epoch_list = [10, 100, 500, 2000]

    # TODO: epoch_list의 각 값에 대해 train()을 호출하고 print_result()로 출력하세요.
    pass


# ============================================================
# 실험 2: Learning Rate의 영향
#   - epoch은 고정, learning_rate만 변경
#   - 너무 작으면? 적당하면? 너무 크면?
# ============================================================
def experiment_learning_rate():
    """
    Learning Rate(학습률)가 학습에 미치는 영향을 관찰한다.

    Learning Rate란?
      기울기(gradient) 방향으로 한 번에 얼마나 이동할지 결정하는 값이다.
      - 너무 작으면: 수렴은 하지만 매우 느리다 (epoch을 많이 돌려야 한다)
      - 적당하면: 빠르고 안정적으로 수렴한다
      - 너무 크면: 최적점을 지나쳐 발산(diverge)할 수 있다

    TODO: lr_list에 있는 각 learning_rate 값으로 학습을 실행하고 결과를 비교하세요.
          train() 함수를 호출하고, print_result()로 결과를 출력하면 됩니다.

    힌트: epochs는 1000으로 고정하세요.
    """
    print("\n" + "-" * 10)
    print("실험 2: Learning Rate의 영향 (epochs = 1000 고정)")
    print("-" * 10)

    lr_list = [0.0001, 0.01, 0.1, 1.0]

    # TODO: lr_list의 각 값에 대해 train()을 호출하고 print_result()로 출력하세요.
    pass


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    experiment_epoch()
    experiment_learning_rate()

