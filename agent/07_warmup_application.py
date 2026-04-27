"""
TITLE: Warmup 활용 - 큰 학습률 안정화 비교
DIFFICULTY: hard
TAGS: gradient-descent, minibatch, learning-rate, warmup, application
EVAL: script

DESCRIPTION:
큰 base_lr 환경에서 warmup의 효과를 정량적으로 비교하세요.

이 문제는 "Warmup 이 학습 초기 안정성에 어떤 영향을 주는가" 를 직접 측정해서
보여주는 것이 목적입니다.

같은 base_lr 을 사용하되 warmup_epochs 를 0 / 5 / 50 으로 바꾸어 가며:
  · 학습 초기 30 epoch 동안의 최대 loss (peak loss)
  · 최종 loss
  · 최종 (w, b)
를 비교한다.

핵심 공식 (Linear Warmup):
  current_lr = base_lr · (epoch / warmup_epochs)   if epoch ≤ warmup_epochs
  current_lr = base_lr                              otherwise
  단, warmup_epochs == 0 이면 항상 base_lr 을 반환한다 (zero-division 방지).

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요.
  - 동일한 base_lr 로 warmup_epochs 0 / 5 / 50 세 케이스를 학습합니다.
  - 각 케이스에 대해 (초기 peak loss, 최종 loss, 최종 w, 최종 b) 를 출력하세요.
  - 비교가 한눈에 보이도록 표 형태로 정리하면 좋습니다.

생각해보기:
  1. 최종 loss 는 모든 케이스에서 동일하다. 그런데 왜 warmup 이 필요할까?
  2. peak loss 가 654 → 53 으로 줄어든다는 것이 실제 학습에서 어떤 의미인가?
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
#   - Mini-batch GD + Linear Warmup 학습 코드를 직접 작성하세요.
#   - warmup_epochs 0 / 5 / 50 세 케이스를 비교해 결과를 출력하세요.
# ============================================================
