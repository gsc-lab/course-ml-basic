"""
정답: SGD(확률적 경사하강법) 구현
====================================
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
# SGD 학습 함수
# ============================================================
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

