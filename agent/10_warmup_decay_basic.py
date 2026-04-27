"""
TITLE: Warmup → Constant → Step Decay (3단계 스케줄)
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, learning-rate, warmup, decay, scheduler
EVAL: script

DESCRIPTION:
Warmup 과 Step Decay 를 결합한 3단계 학습률 스케줄을 구현하세요.

이 문제는 "여러 lr 정책을 조합하는 스케줄러" 의 작성 패턴을 익히는 것이 목적입니다.

3단계 스케줄:

  ┌─────────────────────────────────────────────────────┐
  │ Phase 1: Warmup    (epoch ≤ warmup_epochs)          │
  │   lr = base_lr × (epoch / warmup_epochs)            │
  │                                                     │
  │ Phase 2: Constant  (warmup_epochs < epoch ≤ decay_start) │
  │   lr = base_lr                                      │
  │                                                     │
  │ Phase 3: Decay     (epoch > decay_start)            │
  │   times = (epoch - decay_start - 1) // decay_step + 1 │
  │   lr = base_lr × (decay_rate ^ times)               │
  └─────────────────────────────────────────────────────┘

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요.
  - 위 3단계 스케줄에 따라 매 epoch 마다 lr 을 계산하세요.
  - Mini-batch GD 학습을 직접 구현하고, 진행 상황을 출력하세요.

생각해보기:
  1. Phase 2 (constant) 가 없다면 어떻게 되는가? warmup 직후 바로 decay 가 시작되면 너무 이른가?
  2. decay_start 를 50, 400 으로 바꾸면 어떻게 달라지는가?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 (20개) - 변경 금지
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 학생 구현 영역
#   - Warmup → Constant → Step Decay 3단계 스케줄을 직접 작성하세요.
#   - Mini-batch GD 학습 코드를 직접 구현하세요.
# ============================================================
