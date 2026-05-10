"""
TITLE: Linear Decay (per-step)
DIFFICULTY: basic
TAGS: gradient-descent, minibatch, learning-rate, decay
EVAL: script

DESCRIPTION:
Mini-batch GD 에 Linear Decay 를 적용하세요.

배경:
  학습이 진행될수록 lr 을 줄여 최적점 근처에서 세밀하게 안착시킨다.
  Linear Decay 는 base_lr 에서 시작해 마지막 step 에서 0 까지 선형으로
  감소시킨다. (06번 Linear Warmup 의 거울 대칭)

수식:
  lr = base_lr * max(0.0, 1 - global_step / decay_steps)

  - max(0.0, ...) : global_step 이 decay_steps 를 초과하면 (1 - step/decay_steps)
                    가 음수가 된다. 음수 lr 은 가중치를 잘못된 방향으로 업데이트해
                    모델을 발산시킬 수 있으므로, 아래에서 0 으로 clip 하는 안전장치.
                    (06 Warmup 의 min(1.0, ...) 와 대칭 구조)
  - global_step   : 1 부터 시작, 매 batch 마다 +1
  - decay_steps   : 학습률이 0 에 도달할 때까지 걸리는 step 수

  예시 (decay_steps = 2500):
    - global_step = 1     : lr = base_lr 의 거의 100%
    - global_step = 1250  : lr = base_lr * 0.5
    - global_step = 2500  : lr = 0

요구 사항:
  - 데이터셋(x_data, y_data) 은 변경하지 마세요.
  - decay_steps 는 학습 끝 step 에서 lr=0 이 되도록 설정하세요
    (예: epochs * (n / batch_size)).
  - global_step 카운터를 두고 매 batch 마다 lr 을 갱신하세요.
  - 학습 결과(loss, w, b) 를 출력하세요.

사용 라이브러리(필요에 따라):
  - random.shuffle, random.seed, list slicing, sum/len/range/zip, max

생각해보기:
  - 06번 Linear Warmup 과 lr 변화 모양은 어떻게 다른가? (시간축 vs lr 그래프 머릿속에 그려보기)
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
