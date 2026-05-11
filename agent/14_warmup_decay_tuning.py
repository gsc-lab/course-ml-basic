"""
TITLE: Linear Warmup + Linear Decay 파라미터 튜닝 (per-step)
DIFFICULTY: medium
TAGS: gradient-descent, minibatch, learning-rate, warmup, decay, scheduler, tuning
EVAL: script

DESCRIPTION:
Linear Warmup + Linear Decay 스케줄을 함수 정의 없이 직접 구현하고, 서로
다른 warmup_steps 값을 비교하여 더 적절한 파라미터를 선택하시오.

수식:
  if global_step < warmup_steps:
      lr = base_lr * (global_step / warmup_steps)
  else:
      lr = base_lr * max(0, 1 - (global_step - warmup_steps) / decay_steps)

  - global_step  : 1 부터 시작, 매 batch 마다 +1
  - warmup_steps : warmup 구간 길이 (step 수)
  - decay_steps  : decay 구간 길이 (step 수)
  - 형상         : lr 이 0 → base_lr → 0 으로 변하는 삼각형 모양. 마지막
                   step 근처에서 lr = 0 에 안착한다.

요구 사항 (모두 함수 정의 없이 한 스크립트에 직접 작성):
  1. 주어진 데이터셋(x_data, y_data) 을 그대로 사용한다. (변경 금지)
  2. 공통 하이퍼파라미터로 두 조합 모두 학습한다.
       base_lr = 0.03, epochs = 75, batch_size = 4
       w_init = 0.0,  b_init = 0.0
       (전체 step 수 = epochs × ceil(n / batch_size) = 75 × 8 = 600)
  3. 비교 대상 두 조합 (warmup_steps + decay_steps = 600):
       조합 A : warmup_steps = 500, decay_steps = 100
       조합 B : warmup_steps = 100, decay_steps = 500
  4. 각 조합을 mini-batch GD 로 학습한다. global_step 카운터를 두고 매 batch
     마다 lr 을 갱신하며, 진행 상황을 일부 epoch 에서 stdout 으로 출력한다.
  5. 두 조합 학습이 끝나면 각 조합의 final_loss (전체 데이터 MSE) 와 (w, b)
     를 출력하여 비교한다.
  6. 본인이 더 적절하다고 판단한 조합과 그 근거 한 줄을 print() 로 stdout 에
     출력한다.

힌트:
  - 두 조합 비교의 재현성을 위해 각 조합 학습 직전 random.seed(42) 를 호출한다.
  - gradient 의 factor 2 포함/미포함, batch 평균/합산 convention 은 자유.

  사용 가능한 파이썬 표준 함수:
    · random.shuffle(seq)       — 리스트를 in-place 로 무작위 섞는다.
    · zip(seq_a, seq_b)         — 두 시퀀스를 같은 인덱스의 쌍으로 묶어 순회.
    · enumerate(seq, start=0)   — (인덱스, 원소) 쌍으로 순회.

생각해보기:
  1. warmup_steps 비율을 매우 크게 잡으면 lr 곡선과 학습 결과는 어떻게
     달라지는가?
  2. base_lr 이 동일할 때 warmup 의 본질적 역할은 무엇인가?
"""
# META_TESTS:
# - stdin: ""

import random

# ============================================================
# 데이터셋 (변경 금지)
#   x: 일일 평균 카페인 섭취량 (잔)
#   y: 야간 수면 부족 (시간) — 카페인이 많을수록 수면 시간 감소
# ============================================================
random.seed(0)
N = 32
x_data = [round(0.2 + 4.8 * i / (N - 1), 3) for i in range(N)]   # 0.2 ~ 5.0
y_data = [round(2.5 * x + 1.0 + random.uniform(-2.0, 2.0), 3) for x in x_data]


# ============================================================
# 학생 구현 영역
# ────────────────────────────────────────────────────────────
# 아래의 raise 는 미구현 상태를 알리는 placeholder 이다.
# 이 raise 줄을 삭제하고, 그 자리에 상단 docstring 의 "요구 사항" 을 함수
# 정의 없이 직접 작성하시오.
#
# 본인이 선택한 조합과 그 근거는 NotImplementedError 의 메시지가 아니라
# print() 로 stdout 에 출력해야 채점된다.
# ============================================================
raise NotImplementedError("학생 구현 영역 — 이 raise 줄을 삭제하고 작성하시오.")
