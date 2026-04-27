"""
TITLE: Linear Warmup + Linear Decay (삼각형 스케줄)
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, learning-rate, warmup, decay, scheduler
EVAL: script

DESCRIPTION:
Linear Warmup 과 Linear Decay 가 결합된 삼각형 모양의 lr 스케줄을 구현하세요.

이 문제는 "양 끝이 0 이고 중간에 base_lr 이 되는 부드러운 스케줄" 의 작성을
연습하는 것이 목적입니다.

  ┌──────────────────────────────────────────────────┐
  │   lr ^                                           │
  │      │      ╱│╲                                  │
  │ base │     ╱ │ ╲                                 │
  │      │    ╱  │  ╲                                │
  │      │   ╱   │   ╲                               │
  │      │  ╱    │    ╲                              │
  │      │ ╱     │     ╲                             │
  │    0 │╱______│______╲___→ epoch                  │
  │       0   warmup   total                         │
  └──────────────────────────────────────────────────┘

수식:
  Phase 1 (epoch ≤ warmup_epochs):
      lr = base_lr · (epoch / warmup_epochs)
  Phase 2 (epoch > warmup_epochs):
      progress = (total_epochs - epoch) / (total_epochs - warmup_epochs)
      lr = base_lr · max(progress, 0.0)
      → epoch == total_epochs 에서 lr = 0 으로 수렴

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요.
  - 위 두 phase 분기를 그대로 구현하세요.
  - Mini-batch GD 학습 코드도 직접 작성합니다.

생각해보기:
  1. WD1 의 3단계 스케줄과 비교했을 때 비슷한 성능이 나오는 이유는?
  2. warmup_epochs 를 0 으로 두면 어떤 모양이 되는가? (단순 Linear Decay)
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
#   - Linear Warmup + Linear Decay (삼각형) 스케줄을 직접 작성하세요.
#   - Mini-batch GD 학습 코드를 직접 구현하세요.
# ============================================================
