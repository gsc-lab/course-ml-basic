# agent/ — ML 문제 출제 도우미

## 목적

ML 학습 알고리즘 문제는 학생마다 다른 convention 으로 풀어도 모두 옳다
(gradient factor 2 포함/미포함, batch 평균/합산 등 — lr 에 흡수). `EVAL: stdio`
는 stdout 한 자리 차이로 fail 처리하므로 ML 알고리즘 문제와 안 맞다.

이 폴더는 **`EVAL: script` (AI 가 코드 + 1회 실행 결과를 종합 평가) 를
ML 문제의 default 로 삼는 출제 워크플로** 와 그 도구들을 제공:

- **[AUTHORING.md](AUTHORING.md)** — 출제 지침. 모드 선택 가이드, Hard Rules,
  Workflow, AI 채점 prompt 템플릿, Self-check 체크리스트. Claude Code 에 prompt
  로 그대로 사용 가능.
- **[verify_problem.py](verify_problem.py)** — exercise + solution 짝 검증.
  script / stdio / testCases 모드별로 적절한 검사 수행.

---

## 새 문제 출제 시

1. [AUTHORING.md](AUTHORING.md) 의 모드 선택 가이드 — ML 알고리즘 문제는
   `EVAL: script` 가 default
2. AUTHORING.md 의 Workflow 따라 작성:
   - canonical solution 먼저 → exercise 의 META_TESTS 와 짝 맞춤
   - skeleton 은 canonical 에서 학생 TODO 부분만 비움
   - script 모드에서는 description 에 "채점 기준 (AI 평가)" 섹션 명시
3. 검증:
   ```bash
   python agent/verify_problem.py 02_linear_regression/exercises/06_warmup_linear.py
   ```
4. PASS 면 출제 완료. Agent YJU:Eval 단원 import 패널로 업로드.

---

## 일괄 검증

```bash
# 전체
python agent/verify_problem.py --all

# 특정 모듈만
python agent/verify_problem.py --module 02_linear_regression
```

---

## 검증 항목

| # | 항목 | 모드 |
|---|---|---|
| 1 | 헤더 (`TITLE` / `DIFFICULTY` / `EVAL`) | 모든 모드 |
| 2 | `# META_TESTS:` 블록 YAML 파싱 가능 | 모든 모드 |
| 3 | `solutions/<같은 파일명>.py` 존재 | 모든 모드 |
| 4 | solution 실행 stdout == META_TESTS expected_stdout (정확 일치) | **stdio** |
| 5 | solution 정상 종료 (60s 이내), stdout 비교는 안 함 | **script** |
| 6 | skeleton 그대로 실행 시 학생에게 명확한 신호 | 모든 모드 |

비교 정규화는 Agent YJU:Eval 서버와 동일:
- 줄 끝 공백 제거 / 마지막 빈 줄 제거 / `\r\n` → `\n`

---

## Claude Code 사용 prompt 예시

**새 문제 출제**:
> course-ml-basic/agent/AUTHORING.md 의 지침을 따라 새 문제를 작성해줘.
> ML 알고리즘 문제면 `EVAL: script` 로, description 에 "채점 기준 (AI 평가)"
> 섹션 포함. 작성 후 `python agent/verify_problem.py <path>` 로 PASS 확인.

**기존 stdio → script 일괄 변환**:
> 02_linear_regression/exercises/ 의 모든 .py 를 EVAL: stdio → EVAL: script
> 로 변환. AUTHORING.md 의 "기존 stdio 문제 일괄 전환" 항목 절차 따라줘.

---

## 의존성

- Python 3.10+
- PyYAML (`pip install pyyaml`)
