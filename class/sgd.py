import random  # noqa: F401 - 수업 중 w, b 초기화 및 shuffle에 사용

# ============================================================
# 1. 데이터셋
#    정답: H(x) = 0.3x + 1  →  w=0.3, b=1
# ============================================================
x_data = [count for count in range(1, 51)]
y_data = [0.3 * x + 1 for x in x_data]
n = len(x_data)


# 2. 파라미터 초기화 (w, b)


# 3. 하이퍼파라미터 설정 (learning_rate, epochs)


# 4. 학습 루프 (epoch 반복)
#    4-1. 데이터 셔플 (SGD: 매 epoch마다 순서를 섞는다)
#
#    4-2. 샘플 하나씩 순회 (SGD: 샘플 1개마다 즉시 업데이트)
#         (a) 예측값 계산: H(x) = wx + b
#         (b) 오차 계산: error = 예측값 - 실제값
#         (c) 그래디언트 계산: grad_w = 2 * x * error
#                              grad_b = 2 * error
#         (d) 파라미터 즉시 업데이트: w = w - lr * grad_w
#
#    4-3. 손실(loss) 출력 (학습 경과 확인)


# 5. 최종 결과 출력
