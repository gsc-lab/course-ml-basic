"""
실습: BGD vs SGD vs Mini-batch GD 비교 실험
============================================

배경:
  같은 경사하강법이라도 한 번에 몇 개의 데이터를 보고 업데이트하느냐에 따라
  학습 과정이 크게 달라진다.

  ┌──────────────┬──────────────┬──────────────────────┐
  │    방식       │  batch_size  │  epoch당 업데이트 횟수  │
  ├──────────────┼──────────────┼──────────────────────┤
  │  BGD         │  n (전체)     │  1회                  │
  │  Mini-batch  │  k (예: 4)   │  n/k회                │
  │  SGD         │  1           │  n회                  │
  └──────────────┴──────────────┴──────────────────────┘

목표:
  - BGD, SGD, Mini-batch GD를 각각 함수로 구현한다.
  - 동일한 데이터·초기값·epoch에서 세 방식의 수렴 과정을 비교한다.
  - 어떤 상황에서 어떤 방식이 유리한지 생각해본다.

지시사항:
  - TODO 로 표시된 부분을 채워 넣으세요.
  - 각 함수의 docstring을 참고하세요.

생각해보기:
  1. 세 방식의 loss 감소 속도를 비교해보자. 어느 것이 가장 빠른가?
  2. SGD의 loss가 울퉁불퉁한 이유는 무엇인가?
     Mini-batch는 SGD보다 덜 울퉁불퉁한가?
  3. batch_size를 2, 10, 20으로 바꿔보자.
     batch_size=20이면 어떤 방식과 같아지는가?
  4. 실무에서 Mini-batch GD가 가장 많이 쓰이는 이유는 무엇일까?
"""
import random

# ============================================================
# 데이터셋 (20개)
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 함수 정의
# ============================================================

def train_bgd(x_data, y_data, learning_rate, epochs, w_init, b_init):
    """
    Batch Gradient Descent로 학습한다.

    전체 데이터의 기울기 평균을 구한 뒤 1번 업데이트한다.
    → epoch당 업데이트 1회

    Args:
        x_data, y_data: 학습 데이터
        learning_rate: 학습률
        epochs: 반복 횟수
        w_init, b_init: 초기 파라미터
    Returns:
        (w, b, loss_history) 튜플
    """
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        w_grad = 0.0
        b_grad = 0.0
        loss = 0.0

        # TODO: 전체 데이터를 순회하며 기울기와 loss를 "누적"하세요.
        #   for x, y in zip(x_data, y_data):
        #       1) predict = w * x + b
        #       2) error = predict - y
        #       3) w_grad += 2 * x * error
        #       4) b_grad += 2 * error
        #       5) loss += error ** 2
        pass

        # TODO: 누적된 값을 샘플 수(n)로 나누어 평균을 구하세요.
        pass

        # TODO: w, b를 업데이트하세요. (epoch당 1번)
        pass

        loss_history.append(loss)

    return w, b, loss_history


def train_sgd(x_data, y_data, learning_rate, epochs, w_init, b_init):
    """
    Stochastic Gradient Descent로 학습한다.

    샘플 1개마다 즉시 파라미터를 업데이트한다.
    → epoch당 업데이트 n회

    주의: 매 epoch마다 데이터 순서를 섞어야(shuffle) 편향을 줄일 수 있다.

    Args:
        x_data, y_data: 학습 데이터
        learning_rate: 학습률
        epochs: 반복 횟수
        w_init, b_init: 초기 파라미터
    Returns:
        (w, b, loss_history) 튜플
    """
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        # TODO: 데이터 인덱스를 섞으세요.
        #   indices = list(range(n))
        #   random.shuffle(indices)
        pass

        epoch_loss = 0.0

        # TODO: 섞인 순서대로 샘플을 하나씩 처리하세요.
        #   for idx in indices:
        #       1) x, y = x_data[idx], y_data[idx]
        #       2) predict = w * x + b
        #       3) error = predict - y
        #       4) w_grad = 2 * x * error  (평균 없이 그대로)
        #       5) b_grad = 2 * error
        #       6) 즉시 w, b 업데이트
        #       7) epoch_loss += error ** 2
        pass

        loss_history.append(epoch_loss / n)

    return w, b, loss_history


def train_minibatch(x_data, y_data, learning_rate, epochs, batch_size, w_init, b_init):
    """
    Mini-batch Gradient Descent로 학습한다.

    데이터를 batch_size 크기로 나누어, 배치 단위로 업데이트한다.
    → epoch당 업데이트 (n / batch_size)회

    BGD와 SGD의 중간 지점이다:
      - 배치 내에서는 BGD처럼 기울기 평균을 구하고
      - 배치마다 업데이트하므로 SGD처럼 빈번하게 움직인다

    주의: 매 epoch마다 데이터 순서를 섞어야 한다.

    Args:
        x_data, y_data: 학습 데이터
        learning_rate: 학습률
        epochs: 반복 횟수
        batch_size: 배치 크기
        w_init, b_init: 초기 파라미터
    Returns:
        (w, b, loss_history) 튜플
    """
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        # TODO: 데이터 인덱스를 섞으세요.
        #   indices = list(range(n))
        #   random.shuffle(indices)
        pass

        epoch_loss = 0.0

        # TODO: 배치 단위로 처리하세요.
        #   for start in range(0, n, batch_size):
        #       1) batch_indices = indices[start:start + batch_size]
        #       2) bs = len(batch_indices)
        #       3) 배치 내 기울기 누적 (BGD와 동일한 방식):
        #            w_grad, b_grad, batch_loss = 0.0, 0.0, 0.0
        #            for idx in batch_indices:
        #                predict, error 계산
        #                w_grad += 2 * x * error
        #                b_grad += 2 * error
        #                batch_loss += error ** 2
        #       4) 배치 평균: w_grad /= bs, b_grad /= bs
        #       5) 업데이트: w, b 갱신
        #       6) epoch_loss += batch_loss
        pass

        loss_history.append(epoch_loss / n)

    return w, b, loss_history


def print_comparison(results):
    """세 방식의 결과를 나란히 출력한다."""
    print(f"{'':18s}", end="")
    for name, _, _, _ in results:
        print(f" {name:>12s}", end="")
    print(f"   {'정답':>8s}")
    print("-" * 10)

    print(f"{'최종 w':18s}", end="")
    for _, w, _, _ in results:
        print(f" {w:12.4f}", end="")
    print(f"   {'0.5':>8s}")

    print(f"{'최종 b':18s}", end="")
    for _, _, b, _ in results:
        print(f" {b:12.4f}", end="")
    print(f"   {'2.0':>8s}")

    print(f"{'최종 loss':18s}", end="")
    for _, _, _, loss_h in results:
        print(f" {loss_h[-1]:12.6f}", end="")
    print()


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    # 동일한 초기값 사용
    random.seed(42)
    w_init = random.random()
    b_init = random.random()

    epochs = 300
    batch_size = 4

    # 각 방식에 맞는 learning_rate 설정
    # 업데이트 빈도가 높을수록 작은 lr이 필요하다
    bgd_lr = 0.001
    mini_lr = 0.001
    sgd_lr = 0.0001

    print("-" * 10)
    print("BGD vs Mini-batch GD vs SGD 비교 실험")
    print("-" * 10)
    print(f"데이터 수: {len(x_data)}개, epochs: {epochs}, batch_size: {batch_size}")
    print(f"초기 w: {w_init:.4f}, 초기 b: {b_init:.4f}")
    print(f"learning_rate → BGD: {bgd_lr}, Mini-batch: {mini_lr}, SGD: {sgd_lr}")
    print()

    # --- BGD 학습 ---
    print("[BGD 학습 과정] (epoch당 1번 업데이트)")
    w_bgd, b_bgd, loss_bgd = train_bgd(
        x_data, y_data, bgd_lr, epochs, w_init, b_init
    )
    for i in range(0, epochs, 50):
        print(f"  Epoch {i+1:4d} | Loss: {loss_bgd[i]:.6f}")
    print()

    # --- Mini-batch 학습 ---
    print(f"[Mini-batch GD 학습 과정] (epoch당 {len(x_data) // batch_size}번 업데이트)")
    random.seed(99)
    w_mini, b_mini, loss_mini = train_minibatch(
        x_data, y_data, mini_lr, epochs, batch_size, w_init, b_init
    )
    for i in range(0, epochs, 50):
        print(f"  Epoch {i+1:4d} | Loss: {loss_mini[i]:.6f}")
    print()

    # --- SGD 학습 ---
    print(f"[SGD 학습 과정] (epoch당 {len(x_data)}번 업데이트)")
    random.seed(99)
    w_sgd, b_sgd, loss_sgd = train_sgd(
        x_data, y_data, sgd_lr, epochs, w_init, b_init
    )
    for i in range(0, epochs, 50):
        print(f"  Epoch {i+1:4d} | Loss: {loss_sgd[i]:.6f}")
    print()

    # --- 비교 ---
    print("-" * 10)
    print("결과 비교")
    print("-" * 10)
    results = [
        ("BGD", w_bgd, b_bgd, loss_bgd),
        ("Mini-batch", w_mini, b_mini, loss_mini),
        ("SGD", w_sgd, b_sgd, loss_sgd),
    ]
    print_comparison(results)

    n = len(x_data)
    print()
    print(f"{'epoch당 업데이트':18s}", end="")
    print(f" {'1회':>12s} {str(n // batch_size) + '회':>12s} {str(n) + '회':>12s}")
    print(f"{'총 업데이트 횟수':18s}", end="")
    print(f" {epochs:12d} {epochs * (n // batch_size):12d} {epochs * n:12d}")

