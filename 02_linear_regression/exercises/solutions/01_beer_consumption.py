"""
실습 정답: 기온에 따른 맥주 소비량 예측 (Linear Regression)
============================================================

데이터셋 소개:
  브라질 상파울루 대학교(USP) 연구팀이 상파울루 시내 특정 지역에서
  약 1년간 수집한 실제 관측 데이터이다 (Kaggle 공개 데이터셋).
  연구팀은 매일의 평균 기온(°C)과 해당 지역의 맥주 소비량(리터)을
  기록하였으며, 기온이 높을수록 맥주 소비량이 선형적으로 증가하는
  패턴을 확인하였다.

  이 데이터는 실제로 맥주 회사의 생산량·재고 계획, 편의점·마트의
  발주량 예측 등에 활용될 수 있다. 예를 들어, 내일 기온이 35°C로
  예보된다면 맥주를 얼마나 준비해야 하는지 예측하는 것이다.

  본 실습에서는 원본 데이터(약 365건) 중 기온 구간별 대표값
  11건을 발췌하여 사용한다.
"""

# ============================================================
# 데이터셋: (평균 기온 °C, 맥주 소비량 리터)
# ============================================================
temperatures = [17.0, 19.0, 20.0, 22.0, 24.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0]
beer_consumption = [20.8, 21.5, 22.0, 23.5, 25.2, 26.0, 27.8, 29.0, 31.2, 32.5, 34.0]


# ============================================================
# 함수 정의
# ============================================================

def hypothesis(x_data, w, b):
    y_pred = []
    for x in x_data:
        y_pred.append(w * x + b)
    return y_pred


def compute_loss(y_pred, y_data):
    n = len(y_data)
    loss = 0.0
    for i in range(n):
        loss += (y_pred[i] - y_data[i]) ** 2
    loss = loss / n
    return loss


def compute_gradients(x_data, y_pred, y_data):
    n = len(y_data)
    grad_w = 0.0
    grad_b = 0.0
    for i in range(n):
        error = y_pred[i] - y_data[i]
        grad_w += error * x_data[i]
        grad_b += error
    grad_w = (2 / n) * grad_w
    grad_b = (2 / n) * grad_b
    return grad_w, grad_b


def train(x_data, y_data, learning_rate, epochs):
    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        y_pred = hypothesis(x_data, w, b)
        loss = compute_loss(y_pred, y_data)
        grad_w, grad_b = compute_gradients(x_data, y_pred, y_data)
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | w: {w:.4f}, b: {b:.4f}")

    return w, b


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    learning_rate = 0.0001
    epochs = 5000

    w, b = train(temperatures, beer_consumption, learning_rate, epochs)

    print("\n========== 학습 완료 ==========")
    print(f"학습된 w: {w:.4f}")
    print(f"학습된 b: {b:.4f}")
    print(f"의미: 기온이 1°C 오르면 맥주 소비량이 약 {w:.2f}리터 증가")
    print()

    print("예측 결과:")
    for x, y in zip(temperatures, beer_consumption):
        pred = w * x + b
        print(f"  기온 {x:5.1f}°C → 예측: {pred:.2f}L, 실제: {y:.1f}L")

    print()
    new_temps = [18.0, 26.0, 36.0]
    print("새로운 기온에 대한 예측:")
    for t in new_temps:
        pred = w * t + b
        print(f"  기온 {t:.1f}°C → 예측 소비량: {pred:.2f}L")
