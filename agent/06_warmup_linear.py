"""
TITLE: Linear Warmup 구현
DIFFICULTY: basic
TAGS: gradient-descent, minibatch, learning-rate, warmup
EVAL: script

DESCRIPTION:
Mini-batch GD에 Linear Warmup을 적용하세요.

이 문제는 "학습 초반에 lr 을 점진적으로 키우는 Warmup 의 핵심 규칙"을
익히는 것이 목적입니다.

Warmup 이란?
  학습 초반에 학습률(lr)을 0 에서 base_lr 까지 점진적으로 증가시키는 기법.
  초기 큰 lr 로 인한 학습 불안정 / 발산을 방지한다.

Linear Warmup 수식:
  current_lr = base_lr · (epoch / warmup_epochs)   if epoch ≤ warmup_epochs
  current_lr = base_lr                              otherwise

  ┌──────────────────────────────────────────────┐
  │ epoch=1   → lr = base_lr × 1/warmup_epochs   │
  │ epoch=N/2 → lr ≈ base_lr × 0.5               │
  │ epoch=N   → lr = base_lr                     │
  │ epoch>N   → lr = base_lr (고정)               │
  └──────────────────────────────────────────────┘

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요.
  - Mini-batch Gradient Descent + Linear Warmup 학습을 직접 구현합니다.
  - 매 epoch 마다 학습률(lr)을 갱신해 사용하세요.
  - 학습 진행 상황(loss, w, b)을 적절한 간격으로 출력하세요.

생각해보기:
  1. warmup_epochs = 0 이면 어떤 분기로 떨어져야 하는가?
  2. base_lr 을 키우면 Warmup 의 효과가 더 커지는가? (다음 문제에서 확인)
  3. epoch == warmup_epochs 일 때 lr 값은 정확히 base_lr 이 되는가?
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
#   - Mini-batch GD + Linear Warmup 학습 코드를 직접 작성하세요.
#   - 학습 종료 후 최종 (w, b) 와 학습 결과를 출력하세요.
# ============================================================
