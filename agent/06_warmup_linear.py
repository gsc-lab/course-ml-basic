"""
TITLE: Linear Warmup (per-step)
DIFFICULTY: basic
TAGS: gradient-descent, minibatch, learning-rate, warmup
EVAL: script

DESCRIPTION:
Mini-batch GD 에 Linear Warmup 을 적용하세요.

배경:
  학습 초반의 큰 lr 은 학습을 불안정하게 만든다. Warmup 은 lr 을 0 에서
  base_lr 까지 점진적으로 키워 이를 방지한다. PyTorch / HuggingFace 등
  실무 라이브러리는 epoch 이 아닌 step(= batch) 단위로 적용한다.

수식:
  lr = base_lr * min(1.0, global_step / warmup_steps)

  - min(1.0, ...) : warmup 구간이 끝난 뒤 lr 이 base_lr 을 초과하지 않도록
                    위에서 clip 하는 안전장치.
                    (global_step <= warmup_steps : 선형 증가,
                     global_step >  warmup_steps : 1.0 으로 고정 → lr = base_lr)
  - global_step   : 1 부터 시작, 매 batch 마다 +1
  - warmup_steps  : warmup 구간 길이 (step 수)

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요.
  - global_step 카운터를 두고 매 batch 마다 lr 을 갱신하세요.
  - 학습 결과(loss, w, b) 를 출력하세요.

사용 라이브러리(필요에 따라):
  - random.shuffle, random.seed, list slicing, sum/len/range/zip

생각해보기:
  - warmup_steps = 250 은 (epochs=500, batch_size=4, n=20) 에서 몇 epoch 분량인가?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 (변경 금지) -- 정답: H(x) = 0.5x + 2  (w=0.5, b=2)
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 학생 구현 영역
# ============================================================
