"""
실습: 기온에 따른 맥주 소비량 예측 (Linear Regression)
======================================================

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

목표:
  - 가설, 손실함수, 기울기 계산, 학습 루프를 함수로 분리하여 구현한다.
  - 각 함수의 역할과 관계를 이해한다.

지시사항:
  - TODO 로 표시된 부분을 채워 넣으세요.
  - 각 함수의 docstring을 참고하세요.
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
    """
    가설 함수: H(x) = wx + b
    현재 파라미터(w, b)로 각 입력에 대한 예측값 리스트를 반환한다.

    Args:
        x_data: 입력 데이터 리스트
        w: 가중치
        b: 편향
    Returns:
        예측값 리스트
    """
    # TODO: for문을 이용해 각 x에 대해 w * x + b 를 계산하여 리스트로 반환하세요.
    pass


def compute_loss(y_pred, y_data):
    """
    손실 함수: Loss = (1/n) Σ (H(x) - y)²
    예측값과 실제값의 차이를 하나의 숫자로 요약한다.

    Args:
        y_pred: 예측값 리스트
        y_data: 실제값 리스트
    Returns:
        손실값 (float)
    """
    # TODO: MSE 손실을 계산하세요. (1/n) * Σ (y_pred[i] - y_data[i])²
    pass


def compute_gradients(x_data, y_pred, y_data):
    """
    기울기 계산:
        ∂Loss/∂w = (2/n) Σ (H(x) - y) · x
        ∂Loss/∂b = (2/n) Σ (H(x) - y)

    손실을 줄이기 위해 w, b를 어느 방향으로 얼마나 바꿔야 하는지 알려준다.

    Args:
        x_data: 입력 데이터 리스트
        y_pred: 예측값 리스트
        y_data: 실제값 리스트
    Returns:
        (grad_w, grad_b) 튜플
    """
    # TODO: grad_w, grad_b를 계산하여 반환하세요.
    pass


def train(x_data, y_data, learning_rate, epochs):
    """
    학습 루프:
        1. 가설로 예측값 계산
        2. 손실 계산
        3. 기울기 계산
        4. 파라미터 업데이트

    Args:
        x_data: 입력 데이터 리스트
        y_data: 실제값 리스트
        learning_rate: 학습률
        epochs: 반복 횟수
    Returns:
        학습된 (w, b) 튜플
    """
    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        # TODO: 위에서 만든 함수들을 순서대로 호출하여 학습을 진행하세요.
        # 1) hypothesis()로 예측값 구하기
        # 2) compute_loss()로 손실 구하기
        # 3) compute_gradients()로 기울기 구하기
        # 4) w, b 업데이트: w = w - learning_rate * grad_w
        pass

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | w: {w:.4f}, b: {b:.4f}")

    return w, b


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    # 하이퍼파라미터
    learning_rate = 0.0001
    epochs = 5000

    # 학습
    w, b = train(temperatures, beer_consumption, learning_rate, epochs)

    # 결과 출력
    print("\n========== 학습 완료 ==========")
    print(f"학습된 w: {w:.4f}")
    print(f"학습된 b: {b:.4f}")
    print(f"의미: 기온이 1°C 오르면 맥주 소비량이 약 {w:.2f}리터 증가")
    print()

    # 예측 테스트
    print("예측 결과:")
    for x, y in zip(temperatures, beer_consumption):
        pred = w * x + b
        print(f"  기온 {x:5.1f}°C → 예측: {pred:.2f}L, 실제: {y:.1f}L")

    # 새로운 기온으로 예측
    print()
    new_temps = [18.0, 26.0, 36.0]
    print("새로운 기온에 대한 예측:")
    for t in new_temps:
        pred = w * t + b
        print(f"  기온 {t:.1f}°C → 예측 소비량: {pred:.2f}L")
