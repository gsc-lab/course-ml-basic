"""
정답: SGD with Momentum 구현
============================
"""
import random

# ============================================================
# 데이터셋 (20개) - 변경 금지
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# Helper: Mini-batch MSE gradient (변경 금지)
#   convention: factor 2 포함, 배치 크기 m 으로 평균
#       dw = (2/m) Σ (w·x_i + b - y_i) · x_i
#       db = (2/m) Σ (w·x_i + b - y_i)
# ============================================================
def compute_mse_gradient(batch_x, batch_y, w, b):
    m = len(batch_x)
    dw = (2.0 / m) * sum((w * x + b - y) * x for x, y in zip(batch_x, batch_y))
    db = (2.0 / m) * sum(w * x + b - y for x, y in zip(batch_x, batch_y))
    return dw, db


# ============================================================
# Helper: Mini-batch MSE loss (변경 금지)
# ============================================================
def compute_mse_loss(batch_x, batch_y, w, b):
    m = len(batch_x)
    return (1.0 / m) * sum((w * x + b - y) ** 2 for x, y in zip(batch_x, batch_y))


# ============================================================
# Momentum 학습 함수
# ============================================================
def train_with_momentum(x_data, y_data, lr, epochs, batch_size,
                        beta, w_init, b_init):
    w, b = w_init, b_init
    vw, vb = 0.0, 0.0
    n = len(x_data)

    for epoch in range(1, epochs + 1):
        # shuffle - 출제자 영역
        indices = list(range(n))
        random.shuffle(indices)

        epoch_loss_sum = 0.0
        batch_count = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_x = [x_data[i] for i in batch_idx]
            batch_y = [y_data[i] for i in batch_idx]

            dw, db = compute_mse_gradient(batch_x, batch_y, w, b)
            batch_loss = compute_mse_loss(batch_x, batch_y, w, b)

            # 학생 TODO 영역 (정답)
            vw = beta * vw + dw
            vb = beta * vb + db
            w = w - lr * vw
            b = b - lr * vb

            epoch_loss_sum += batch_loss
            batch_count += 1

        avg_loss = epoch_loss_sum / batch_count

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | "
                  f"w: {w:.2f}, b: {b:.2f} | vw: {vw:.4f}, vb: {vb:.4f}")

    return w, b


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    w_init, b_init = 0.0, 0.0
    lr = 0.001
    epochs = 500
    batch_size = 4
    beta = 0.9

    n = len(x_data)
    print("-" * 10)
    print("SGD with Momentum 학습")
    print("-" * 10)
    print(f"데이터 수: {n}개, epochs: {epochs}, batch_size: {batch_size}")
    print(f"learning_rate: {lr}, beta(momentum): {beta}")
    print()

    random.seed(42)
    w, b = train_with_momentum(
        x_data, y_data, lr, epochs, batch_size, beta, w_init, b_init
    )

    print()
    print(f"최종 파라미터: w = {w:.2f}, b = {b:.2f}")
    print(f"정답과 비교: w(정답=0.5), b(정답=2.0)")
