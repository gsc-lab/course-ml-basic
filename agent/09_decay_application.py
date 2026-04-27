"""
TITLE: Decay 활용 - 진동 데이터에서 안정성 비교
DIFFICULTY: hard
TAGS: gradient-descent, minibatch, learning-rate, decay, application
EVAL: script

DESCRIPTION:
노이즈가 큰 데이터에서 Step Decay 의 효과를 정량적으로 비교하세요.

이 문제는 "lr 을 끝까지 크게 유지할 때 vs 점차 줄여 안착시킬 때 의 차이"를
측정해 보는 것이 목적입니다.

노이즈가 큰 데이터에서는 mini-batch 마다 gradient 방향이 들쭉날쭉하다.
이때 lr 을 끝까지 크게 유지하면 → 최적점 근처에서 계속 진동한다.
lr 을 점점 줄이면 → 보폭이 작아져서 진동이 작아지고 안정적으로 안착한다.

실험 설정:
  - 고정 lr   : base_lr 그대로
  - Step Decay: 100 epoch 마다 lr × 0.5

평가 지표 (마지막 50 epoch 기준):
  · 평균 loss / 최대 loss / 최소 loss
  · 스윙폭 = 최대 - 최소        (안정성)

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요. (노이즈가 큰 점에 주의)
  - 동일한 base_lr 로 (고정 lr / Step Decay) 두 케이스를 학습합니다.
  - 마지막 50 epoch 의 (평균, 최대, 최소, 스윙폭) loss 를 비교 출력하세요.

생각해보기:
  1. 스윙폭이 100배 줄어든다는 것은 어떤 의미인가?
  2. 노이즈가 없는 깨끗한 데이터라면 Decay 의 효과가 클까, 작을까?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 (노이즈 큰 20개) - 변경 금지
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-2.0, 2.0) for x in x_data]


# ============================================================
# 학생 구현 영역
#   - 고정 lr 과 Step Decay 두 케이스를 직접 비교 구현하세요.
#   - 마지막 50 epoch 의 (평균, 최대, 최소, 스윙폭) 을 출력하세요.
# ============================================================
