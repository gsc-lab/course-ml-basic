"""
정답: Mini-batch GD(미니배치 경사하강법) 구현
=============================================
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
# Mini-batch GD 학습 함수
# ============================================================
def train_minibatch(x_data, y_data, learning_rate, epochs, batch_size,
                    w_init, b_init):
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

