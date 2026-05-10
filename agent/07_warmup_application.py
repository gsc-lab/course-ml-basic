"""
TITLE: Warmup 활용 - 큰 학습률 안정화 비교
DIFFICULTY: hard
TAGS: gradient-descent, minibatch, learning-rate, warmup, application
EVAL: script

DESCRIPTION:
큰 base_lr 환경에서 warmup_steps 를 바꿔가며 효과를 측정하세요.

배경:
  Warmup 의 효과는 "초기 lr 이 너무 큰 상황" 에서 두드러진다.
  같은 base_lr 에서도 warmup_steps 를 0/적게/많이 주면 학습 초기의
  loss spike (peak loss) 가 크게 달라진다.

실험:
  base_lr 은 동일하게 두고 warmup_steps 만 0 / 25 / 250 으로 바꾸어
  학습한 뒤 다음을 비교한다.
    · 초기 30 epoch 동안의 최대 loss (peak loss)
    · 최종 loss
    · 최종 (w, b)

수식 (zero-division 처리 포함):
  if warmup_steps == 0 or step > warmup_steps:
      lr = base_lr
  else:
      lr = base_lr * step / warmup_steps

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요.
  - global_step 카운터를 두고 매 batch 마다 lr 을 갱신하세요.
  - 세 케이스를 같은 base_lr 로 학습하고 표 형태로 비교 출력하세요.

사용 가능한 도구:
  · 권장: random.shuffle, random.seed, list slicing, sum/len/range/zip, max/min
  · 금지: sklearn, torch, tensorflow, scipy.optimize.* (학습 알고리즘 우회)

생각해보기:
  - 최종 loss 는 세 케이스 모두 거의 같다. 그래도 warmup 이 필요한 이유는?
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
