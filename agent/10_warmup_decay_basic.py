"""
TITLE: Warmup → Constant → Step Decay (3단계 스케줄, per-step)
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, learning-rate, warmup, decay, scheduler
EVAL: script

DESCRIPTION:
Warmup 과 Step Decay 를 결합한 3단계 학습률 스케줄을 per-step 으로 구현하세요.

배경:
  실무에서는 warmup → constant → decay 라는 3단계 스케줄을 자주 쓴다.
  학습 초반은 warmup 으로 안정화, 중간은 base_lr 로 빠르게 수렴, 후반은
  decay 로 미세 조정하여 최적점에 안착시키는 패턴이다.

수식 (per-step):
  Phase 1 — Warmup    (step ≤ warmup_steps):
      lr = base_lr * (step / warmup_steps)
  Phase 2 — Constant  (warmup_steps < step ≤ decay_start_step):
      lr = base_lr
  Phase 3 — Decay     (step > decay_start_step):
      times = (step - decay_start_step - 1) // decay_steps + 1
      lr = base_lr * (decay_rate ** times)

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요.
  - global_step 카운터를 두고 매 batch 마다 위 3단계 분기로 lr 을 갱신하세요.
  - 학습 진행 상황(epoch, step, lr, loss, w, b) 을 출력하세요.

사용 가능한 도구:
  · 권장: random.shuffle, random.seed, list slicing, sum/len/range/zip
  · 금지: sklearn, torch, tensorflow, scipy.optimize.* (학습 알고리즘 우회)

생각해보기:
  - Phase 2 (constant) 가 없다면 어떤 모양이 되며, 단점은 무엇일까?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 - 변경 금지
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]


# ============================================================
# 학생 구현 영역
# ============================================================
