"""
TITLE: 4가지 lr 스케줄 종합 벤치마크 (per-step)
DIFFICULTY: hard
TAGS: gradient-descent, minibatch, learning-rate, warmup, decay, scheduler, application
EVAL: script

DESCRIPTION:
4가지 학습률 스케줄을 같은 데이터/조건에서 per-step 으로 비교 평가하세요.

스케줄 4종:
  1. lr_constant       : 항상 base_lr
  2. lr_warmup_only    : 0 -> base_lr 까지 warmup, 이후 base_lr 유지
  3. lr_decay_only     : base_lr -> 0 선형 감소 (전체 학습에 걸쳐)
  4. lr_warmup_decay   : 삼각형 (warmup -> decay)

스케줄 시그니처(권장): (global_step, base_lr, warmup_steps, decay_steps) -> lr
  - 각 스케줄러는 필요한 인자만 사용 (예: lr_constant 는 둘 다 무시)
  - 학습 전체 길이 total_steps == warmup_steps + decay_steps 로 맞춰서 호출
  - max(0, ...) 안전장치를 사용해 lr 이 음수가 되지 않도록 한다

수식 참고:
  - lr_warmup_only:
        if step < warmup_steps:
            lr = base_lr * (step / warmup_steps)
        else:
            lr = base_lr
  - lr_decay_only (warmup 없음, decay 만):
        lr = base_lr * max(0, 1 - (step - 1) / decay_steps)
  - lr_warmup_decay (삼각형):
        if step < warmup_steps:
            lr = base_lr * (step / warmup_steps)
        else:
            lr = base_lr * max(0, 1 - (step - warmup_steps) / decay_steps)

평가 지표:
  - 수렴 속도   : loss <= threshold 에 처음 도달한 epoch
  - 정확도     : 마지막 50 epoch 평균 loss
  - 안정성     : 마지막 50 epoch 스윙폭 (max - min)
  - 최종 loss  : final epoch loss

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요. (적당한 노이즈 있음)
  - 4가지 스케줄러 함수와 학습 함수를 직접 작성하세요.
  - global_step 카운터를 두고 매 batch 마다 lr 을 갱신하세요.
  - 4가지 스케줄을 동일한 학습 루프에 주입하여 표로 비교 출력하세요.

사용 라이브러리(필요에 따라):
  - random.shuffle, random.seed, list slicing, sum/len/range/zip, max/min, enumerate

생각해보기:
  - 실무에서는 왜 warmup + decay 조합을 표준으로 쓸까?
"""
# META_TESTS:
# - stdin: ""
#   expected_stdout: ""

import random

# ============================================================
# 데이터셋 (적당한 노이즈, 변경 금지) -- 정답: H(x) = 0.5x + 2  (w=0.5, b=2)
# ============================================================
random.seed(0)
x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-1.0, 1.0) for x in x_data]


# ============================================================
# 학생 구현 영역
# ============================================================
