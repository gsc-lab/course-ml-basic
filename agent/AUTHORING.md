# ML 문제 출제 지침 (Agent YJU:Eval)

## 핵심 원칙 — ML 알고리즘 문제의 default 는 `EVAL: script`

ML 알고리즘 문제는 같은 결과로 가는 길이 여러 개다. gradient 의 factor 2
포함/미포함, batch 평균/합산, loss 의 1/2 factor, 정규화 위치 등 — 학생마다
다른 수식으로 풀어도 **수학적으로 모두 옳다** (lr 에 흡수). 그런데
`EVAL: stdio` 는 stdout 한 자리 차이로도 fail 처리한다.

→ **structure 가 옳은데 stdout 미세 차이로 fail** 은 학습 의도와 어긋남.

따라서 ML 알고리즘 문제는 `EVAL: script` 로 출제한다. AI 가 코드와 1회 실행
stdout 을 종합 평가:

- **구조**: mini-batch loop, gradient 부호·방향, warmup/decay 식, 업데이트 시점
- **수렴**: loss 의 단조 감소, target 파라미터로의 이동
- **convention 차이는 동등하게 인정**: `(1/m)·Σ` vs `(2/m)·Σ` 둘 다 옳음

---

## 모드 선택 가이드

| 문제 종류 | 모드 |
|---|---|
| 학습 곡선·수렴이 핵심 (GD, warmup, decay, scheduler, 정규화 등) | **script** ⭐ |
| 부동소수점 누적·난수 의존이 큰 numerical 문제 | **script** |
| 코드 스타일·리팩토링·구조 평가 | **script** |
| 출력 형식 자체가 학습 목표 (FizzBuzz, 패턴 출력, 별 찍기) | stdio |
| 단일 정답이 명확한 알고리즘 (정렬·검색·문자열 처리) | stdio 또는 testCases |

ML 기초 코스(`02_linear_regression`, 향후 `04_logistic_regression` 등) 의
모든 학습 알고리즘 실습은 `script` 가 기본.

---

## Hard Rules — `EVAL: script` 모드

### 1. description 에 "채점 기준 (AI 평가)" 섹션 명시

학생도 무엇으로 평가되는지 알아야 한다. AI 평가 항목을 1-3 줄로:

```python
"""
TITLE: Linear Warmup 구현
DIFFICULTY: basic
TAGS: gradient-descent, minibatch, warmup
EVAL: script

DESCRIPTION:
Mini-batch GD에 Linear Warmup을 적용하세요.

Linear Warmup 수식:
  current_lr = base_lr * (epoch / warmup_epochs)   if epoch <= warmup_epochs
  current_lr = base_lr                              otherwise

채점 기준 (AI 평가):
  1. 구조: get_warmup_lr() 가 위 수식대로 동작 — 선형 증가 후 base_lr 고정
  2. 학습: 매 epoch 시작 시 current_lr 갱신 + Mini-batch 업데이트 진행
  3. 수렴: 최종 loss 가 초기 loss 의 절반 이하로 감소, w/b 가 정답 방향으로 이동
  4. convention 자유: gradient 의 factor 2 / batch 평균/합산 등 어느 형태든 OK
                     (부호와 방향이 맞으면 정답)
"""
```

### 2. helper 로 잠그지 않는다 — 학생이 직접 구현

`stdio` 모드와 달리 `script` 는 convention 차이를 허용하므로 helper 로
gradient 식을 미리 채울 필요 없음. 학생이 직접 식을 쓰는 것이 학습 목표.

> 단 *알고리즘 구현* 이 학습 포인트가 아닌 보조 함수 (예: 데이터 로딩, 시각화) 는 미리 채워둬도 OK.

### 3. random.seed 고정 (재현성 보장)

AI 가 매번 같은 학생 코드에 같은 평가를 주려면 실행 결과가 deterministic 해야:

```python
random.seed(0)    # 데이터 생성 — 변경 금지
random.seed(42)   # 학습 직전 — 변경 금지
```

학생 영역의 추가 random 호출은 허용 (script 모드는 비교가 아니라 종합 검토라 무관).

### 4. META_TESTS — `expected_stdout` 사용 안 함

script 모드는 stdout 을 정확 비교하지 않는다. stdin 이 필요 없으면 빈 케이스 1개:

```python
# META_TESTS:
# - stdin: ""
```

`expected_stdout` 키는 빼거나 비워둠. 채점 시 무시됨.

### 5. canonical solution 이 60s 이내 정상 종료 + 의미 있는 stdout

- 무한 루프 / 발산 ❌
- AI 가 학습 진행을 볼 수 있게 epoch 별 loss / 파라미터 출력 포함
- target 값 근처로 수렴 (학생이 "내 코드가 잘못됐나" 의심하지 않게)

### 6. 학생 TODO ≤ 2개

script 모드도 TODO 가 너무 많으면 AI 평가 일관성 떨어짐. 핵심 학습 포인트 1~2개.

### 7. skeleton 그대로 실행 시 학생에게 명확한 신호

`pass` 만 두고 학생이 미구현 상태로 제출하면 AI 가 "구현 안 됨" 을 명확히
인식할 수 있도록:

```python
def get_warmup_lr(epoch, base_lr, warmup_epochs):
    # TODO: warmup 구간 안에서는 base_lr * (epoch / warmup_epochs) 를,
    #       그 이후에는 base_lr 을 그대로 반환하세요.
    raise NotImplementedError("get_warmup_lr 구현 필요")
```

`pass` + None 반환은 종종 silent error 를 만듦. `NotImplementedError` 또는
명시적 안내 print 권장.

---

## Hard Rules — `EVAL: stdio` 모드 (비-ML 문제 한정)

알고리즘이 아닌 **출력 자체가 학습 목표** 인 문제 (FizzBuzz, 패턴 출력 등)
에서만 사용. ML 문제는 위 script 모드 우선.

stdio 사용 시 적용할 추가 규칙 — 이전 회귀 사례 (Linear Warmup 의 factor 2
함정) 에서 도출:

- description 에 모호 가능한 모든 convention (loss factor, 정렬 순서 등) 을
  수식으로 명시
- expected_stdout 은 반드시 canonical solution 을 직접 실행한 결과
- random.seed 가 학생 영역 밖에 고정 (학생 추가 random 호출 금지 명시)
- 출력 format 은 skeleton 의 print 문에 못 박음 (학생은 변수만 채움)

---

## Workflow (반드시 이 순서)

1. **알고리즘·학습 목표 1줄 정의**
2. **모드 선택** — ML 학습 알고리즘이면 `script`, 출력이 답이면 `stdio`
3. **canonical solution 작성** → `*/exercises/solutions/<name>.py`
4. **채점 기준 정의** — description 의 "채점 기준 (AI 평가)" 섹션 (script 모드)
5. **skeleton 작성** → `*/exercises/<name>.py`
   - canonical 에서 학생이 채울 부분만 `# TODO` + `raise NotImplementedError`
6. **검증**:
   ```bash
   python agent/verify_problem.py <module>/exercises/<file>.py
   ```
7. **AI 채점 prompt 등록 (script 모드 단원의 경우)** — Agent YJU:Eval 의
   단원 채점 지침에 ML-specific prompt 추가 (아래 템플릿 참조)
8. **출제** — 단원 import 패널 또는 개별 등록

---

## AI 채점 prompt 권장 — `script` 모드 ML 단원

Agent YJU:Eval 의 단원 채점 지침에 추가:

```
이 단원은 ML 학습 알고리즘 실습. 학생 코드 평가 시:

평가 항목:
- 알고리즘 구조 (mini-batch loop, gradient 방향, warmup/decay 식, 업데이트
  시점) 의 정확성을 우선 평가
- loss 가 단조 감소하고 target 방향으로 수렴하는지 확인
- 학생이 직접 작성한 gradient·loss·update 식의 부호와 방향 검증

convention 차이는 정답으로 인정:
- gradient 의 factor 2 포함/미포함
- batch 평균(/m) vs 합산
- loss 의 (1/n) vs (1/2n)
- L2 reg 의 λ vs λ/2

감점 사유 (이런 경우 명확히 짚어주기):
- gradient 부호 반대 (학습이 발산)
- 업데이트 시점 오류 (배치 누적 누락 / 매 샘플 즉시 업데이트하면서 batch
  단위 변수 안 씀)
- warmup ramp 식이 선형이 아닌 step 함수
- learning rate 가 음수·0
- loss 출력이 NaN / inf 로 발산

출력 자릿수가 미세하게 다르거나, 학생이 print 형식을 살짝 바꿨다고 감점 ❌.
```

---

## Anti-pattern

- ❌ ML 학습 알고리즘 문제를 `EVAL: stdio` 로 → convention 차이로 false fail
- ❌ `script` 모드인데 helper 로 gradient 식 다 잠금 → 학습 의도 무력화
- ❌ description 에 채점 기준 안 적기 → 학생이 무엇으로 채점되는지 모름
- ❌ canonical 이 발산·무한루프 → AI 평가 자체가 불가
- ❌ random.seed 빠짐 → 매번 다른 stdout 으로 AI 평가 불일치
- ❌ skeleton 에 `pass` 만 두기 → 미구현 제출 시 silent error

---

## Self-check 체크리스트

- [ ] 학습 알고리즘 구조 평가가 목표면 `EVAL: script`
- [ ] description 에 "채점 기준 (AI 평가)" 섹션
- [ ] canonical solution 이 60s 이내 정상 종료
- [ ] canonical 의 stdout 에 학습 진행 흔적 (loss / w / b 출력)
- [ ] random.seed 가 출제자 영역에 고정
- [ ] skeleton 의 TODO 는 `raise NotImplementedError` 또는 명시적 안내
- [ ] 학생 TODO ≤ 2개
- [ ] `python agent/verify_problem.py <path>` PASS
- [ ] target 값에 합리적 수렴

---

## Claude Code 사용 prompt 예시

**새 ML 문제 출제**:
> `course-ml-basic/agent/AUTHORING.md` 따라 `EVAL: script` 모드로
> `02_linear_regression/exercises/<번호>_<이름>.py` 와 짝 solution 작성.
> description 에 "채점 기준 (AI 평가)" 섹션 포함, skeleton TODO 는
> `NotImplementedError` 로. 작성 후 `python agent/verify_problem.py <path>`
> 로 검증해 PASS 인지 확인.

**기존 stdio 문제 일괄 전환** (현재 그 작업 중이라면):
> `02_linear_regression/exercises/` 의 모든 .py 파일을 `EVAL: stdio` →
> `EVAL: script` 로 변환:
>   - 헤더 `EVAL: stdio` → `EVAL: script`
>   - description 끝에 "채점 기준 (AI 평가)" 섹션 추가 (구조·수렴·
>     convention 자유 항목)
>   - META_TESTS 의 `expected_stdout` 키 제거 (stdin 만 남김)
>   - skeleton 의 `pass` → `raise NotImplementedError(...)`
>
> solutions/ 는 그대로 둬도 됨. 변환 후 verify_problem.py --all 로 일괄 검증.
