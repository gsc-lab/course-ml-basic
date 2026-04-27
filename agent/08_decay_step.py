"""
TITLE: Step Decay 구현
DIFFICULTY: basic
TAGS: gradient-descent, minibatch, learning-rate, decay
EVAL: script

DESCRIPTION:
Mini-batch GD에 Step Decay를 적용하세요.

이 문제는 "lr 을 단계적으로 줄여가는 Step Decay 의 핵심 규칙"을 익히는 것이
목적입니다.

Step Decay 란?
  N epoch 마다 학습률(lr)에 일정 비율을 곱해 줄여나가는 기법.
  학습 초반엔 큰 보폭으로 빠르게 내려가고, 후반엔 작은 보폭으로
  세밀하게 최적점에 안착하기 위함이다.

Step Decay 수식:
  times      = (epoch - 1) // decay_step
  current_lr = base_lr · (decay_rate ^ times)

  ┌──────────────────────────────────────────────────┐
  │ epoch 1 ~ 100  : lr = base_lr × 1                │
  │ epoch 101~200  : lr = base_lr × 0.5              │
  │ epoch 201~300  : lr = base_lr × 0.5²  = 0.25배   │
  │ ...                                              │
  └──────────────────────────────────────────────────┘

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요.
  - Mini-batch Gradient Descent + Step Decay 학습을 직접 구현합니다.
  - 매 epoch 마다 학습률(lr)을 갱신해 사용하세요.
  - 학습 진행 상황(lr, loss, w, b)을 적절한 간격으로 출력하세요.

생각해보기:
  1. decay_rate 를 0.1 (10배 감소)로 바꾸면 학습이 더 안정적인가, 너무 빨리 멈추는가?
  2. decay_step 을 50 vs 200 으로 바꾸면 어떻게 달라지는가?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 (20개) - 변경 금지
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 학생 구현 영역
#   - Mini-batch GD + Step Decay 학습 코드를 직접 작성하세요.
#   - 학습 종료 후 최종 (w, b) 와 학습 결과를 출력하세요.
# ============================================================
