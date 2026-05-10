"""
TITLE: Linear Warmup + Linear Decay (삼각형 스케줄, per-step)
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, learning-rate, warmup, decay, scheduler
EVAL: script

DESCRIPTION:
양 끝이 0 이고 가운데가 base_lr 이 되는 삼각형 모양의 lr 스케줄을 적용하세요.

배경:
  Warmup 으로 천천히 lr 을 올린 뒤 곧바로 Decay 로 0 까지 줄이는 스케줄.
  Constant 구간 없이 한 봉우리로 학습이 진행되어 마지막 step 에서 lr=0 에
  안착한다.


수식:
  if global_step < warmup_steps:
      # Warmup: 0 → base_lr 선형 증가
      lr = base_lr * (global_step / warmup_steps)
  else:
      # Linear Decay: base_lr → 0 선형 감소
      # max(0, ...) : step 이 (warmup_steps + decay_steps) 를 초과할 경우
      #   lr 이 음수가 되는 것을 방지하는 안전장치.
      #   음수 학습률은 가중치를 잘못된 방향으로 업데이트해 모델을 발산시킬 수 있다.
      lr = base_lr * max(0, 1 - (global_step - warmup_steps) / decay_steps)

  - global_step  : 1 부터 시작, 매 batch 마다 +1
  - warmup_steps : warmup 구간 길이 (step 수)
  - decay_steps  : decay 구간 길이 (step 수)

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요.
  - global_step 카운터를 두고 매 batch 마다 lr 을 갱신하세요.
  - 학습 결과(loss, w, b) 를 출력하세요.

사용 라이브러리(필요에 따라):
  - random.shuffle, random.seed, list slicing, sum/len/range/zip, max

생각해보기:
  - warmup_steps = 0 으로 두면 어떤 모양이 되는가? (단순 Linear Decay)
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
