"""
TITLE: Decay 활용 - 진동 데이터에서 안정성 비교
DIFFICULTY: hard
TAGS: gradient-descent, minibatch, learning-rate, decay, application
EVAL: script

DESCRIPTION:
노이즈 큰 데이터에서 Step Decay 의 효과를 정량적으로 비교하세요.

배경:
  노이즈가 크면 mini-batch 마다 gradient 가 들쭉날쭉하다.
  lr 을 끝까지 크게 두면 → 최적점 근처에서 계속 진동한다.
  lr 을 점점 줄이면 → 보폭이 작아져 안정적으로 안착한다.

실험:
  같은 base_lr 로 (고정 lr / Step Decay 500-step ×0.5) 두 케이스를 학습한 뒤
  마지막 50 epoch 의 (평균, 최대, 최소, 스윙폭) loss 를 표로 비교.

수식 (per-step):
  times = (step - 1) // decay_steps
  current_lr = base_lr * (decay_rate ** times)

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요. (노이즈 큼)
  - global_step 카운터를 두고 매 batch 마다 lr 을 갱신하세요.
  - 두 케이스의 tail 메트릭을 비교 출력하세요.

사용 가능한 도구:
  · 권장: random.shuffle, random.seed, list slicing, sum/len/range, max/min
  · 금지: sklearn, torch, tensorflow, scipy.optimize.* (학습 알고리즘 우회)

생각해보기:
  - 스윙폭이 100배 줄어드는 것이 의미하는 바는?
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
# ============================================================
