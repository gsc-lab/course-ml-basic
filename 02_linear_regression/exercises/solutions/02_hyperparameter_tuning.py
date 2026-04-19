"""
정답: 하이퍼파라미터(Epoch, Learning Rate)의 이해
==================================================
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
    """경사하강법으로 선형 회귀 모델을 학습한다."""
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
            # 예측값 계산
            predict = w * x + b

            # 오차 계산
            error = predict - y

            # 기울기 누적
            w_grad += 2 * x * error
            b_grad += 2 * error

            # loss 누적
            loss += error ** 2

        # 평균
        w_grad /= n
        b_grad /= n
        loss /= n

        # 파라미터 업데이트
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

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
# ============================================================
def experiment_epoch():
    print("-" * 10)
    print("실험 1: Epoch의 영향 (learning_rate = 0.01 고정)")
    print("-" * 10)

    epoch_list = [10, 100, 500, 2000]

    for ep in epoch_list:
        w, b, loss_history = train(x_data, y_data, learning_rate=0.01, epochs=ep)
        print_result(f"epochs = {ep}", w, b, loss_history)


# ============================================================
# 실험 2: Learning Rate의 영향
# ============================================================
def experiment_learning_rate():
    print("\n" + "-" * 10)
    print("실험 2: Learning Rate의 영향 (epochs = 1000 고정)")
    print("-" * 10)

    lr_list = [0.0001, 0.01, 0.1, 1.0]

    for lr in lr_list:
        w, b, loss_history = train(x_data, y_data, learning_rate=lr, epochs=1000)
        print_result(f"learning_rate = {lr}", w, b, loss_history)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    experiment_epoch()
    experiment_learning_rate()

