"""
정답: BGD vs SGD vs Mini-batch GD 비교 실험
============================================
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
    """Batch Gradient Descent로 학습한다."""
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        w_grad = 0.0
        b_grad = 0.0
        loss = 0.0

        # 전체 데이터 순회 → 기울기 누적
        for x, y in zip(x_data, y_data):
            predict = w * x + b
            error = predict - y
            w_grad += 2 * x * error
            b_grad += 2 * error
            loss += error ** 2

        # 평균
        w_grad /= n
        b_grad /= n
        loss /= n

        # 업데이트 (epoch당 1번)
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        loss_history.append(loss)

    return w, b, loss_history


def train_sgd(x_data, y_data, learning_rate, epochs, w_init, b_init):
    """Stochastic Gradient Descent로 학습한다."""
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        # 데이터 순서를 섞는다 (shuffle)
        indices = list(range(n))
        random.shuffle(indices)

        epoch_loss = 0.0

        # 샘플 하나씩 처리 → 즉시 업데이트
        for idx in indices:
            x = x_data[idx]
            y = y_data[idx]

            predict = w * x + b
            error = predict - y

            w_grad = 2 * x * error
            b_grad = 2 * error

            # 즉시 업데이트 (샘플당 1번)
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad

            epoch_loss += error ** 2

        loss_history.append(epoch_loss / n)

    return w, b, loss_history


def train_minibatch(x_data, y_data, learning_rate, epochs, batch_size, w_init, b_init):
    """Mini-batch Gradient Descent로 학습한다."""
    w, b = w_init, b_init
    n = len(x_data)
    loss_history = []

    for epoch in range(1, epochs + 1):
        # 데이터 순서를 섞는다
        indices = list(range(n))
        random.shuffle(indices)

        epoch_loss = 0.0

        # 배치 단위로 처리
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            bs = len(batch_indices)

            # 배치 내 기울기 누적
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

            # 배치마다 업데이트
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad

            epoch_loss += batch_loss

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
    random.seed(42)
    w_init = random.random()
    b_init = random.random()

    epochs = 300
    batch_size = 4

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

