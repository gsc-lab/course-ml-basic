"""
TITLE: SGD with Momentum 구현
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, optimization, momentum
EVAL: script

DESCRIPTION:
Mini-batch GD에 Momentum 을 적용하세요.

배경:
  Momentum 은 직전 step 의 gradient 누적(velocity)을 함께 사용해
  진동을 줄이고 수렴을 빠르게 한다. PyTorch SGD 와 동일하게 매 step
  (= 매 batch) 마다 velocity 와 파라미터를 갱신한다.

수식 (PyTorch SGD convention, per-step):
  v_w = β * v_w + dw
  v_b = β * v_b + db
  w   = w - lr * v_w
  b   = b - lr * v_b

  · β  : momentum 계수 (보통 0.9)
  · v_w, v_b : 누적 velocity (초기값 0, 학습 전체에 걸쳐 유지)
  · dw, db   : 현재 batch 의 평균 gradient

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요.
  - velocity (vw, vb) 를 학습 시작 전에 0 으로 초기화하고, 매 batch 마다 갱신하세요.
  - 학습 진행 상황(epoch, loss, w, b, vw, vb) 을 출력하세요.

사용 가능한 도구:
  · 권장: random.shuffle, random.seed, list slicing, sum/len/range/zip
  · 금지: sklearn, torch, tensorflow, scipy.optimize.* (학습 알고리즘 우회)

생각해보기:
  - β 를 0(=Momentum 없음), 0.5, 0.9, 0.99 로 바꿔보면 수렴 양상이 어떻게 변하는가?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 - 변경 금지
#   정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 학생 구현 영역
# ============================================================
