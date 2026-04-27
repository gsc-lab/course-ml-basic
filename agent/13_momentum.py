"""
TITLE: SGD with Momentum 구현
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, optimization, momentum
EVAL: script

DESCRIPTION:
Mini-batch GD에 Momentum 을 적용하세요.

이 문제는 "Momentum 옵티마이저의 핵심 업데이트 규칙"을 익히는 것이 목적입니다.

Momentum 이란?
  매 step 의 gradient 방향이 들쭉날쭉할 때, 직전 step 들의 평균 방향(velocity)
  을 같이 사용해 더 부드럽고 빠르게 수렴하도록 만드는 기법.

이 문제에서 사용하는 convention (PyTorch SGD 와 동일):
  v_w ← β · v_w + dw
  v_b ← β · v_b + db
  w   ← w - lr · v_w
  b   ← b - lr · v_b

  · β  : momentum 계수 (보통 0.9)
  · v_w, v_b : 누적 velocity (초기값 0)
  · dw, db   : 현재 batch 의 평균 gradient

요구 사항:
  - 아래 데이터셋(x_data, y_data)은 변경하지 마세요.
  - Mini-batch GD + Momentum 학습 코드를 직접 작성하세요.
  - 학습 종료 후 최종 (w, b) 와 학습 결과를 출력하세요.

생각해보기:
  1. β 를 0(=Momentum 없음), 0.5, 0.9, 0.99 로 바꿔보면 수렴 양상이 어떻게 변하나?
  2. epoch 1 의 vw 값이 음수이고 절댓값이 큰 이유는? (초기 gradient 와 관계)
  3. 같은 lr 에서 Momentum 이 있으면 더 빠르게 수렴한다. 그러면 lr 을 더 작게
     써도 되는가, 아니면 더 크게 써도 되는가?
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
#   - Mini-batch GD + Momentum 학습 코드를 직접 작성하세요.
#   - velocity (vw, vb) 초기화부터 파라미터 업데이트까지 직접 구현합니다.
# ============================================================
