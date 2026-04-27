# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML 기초 교과목용 예제 및 실습 코드 저장소. 한국어로 작성되며, 순수 Python → NumPy → 프레임워크 순서로 개념을 쌓아가는 구조.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.10+. Dependencies: numpy, pandas, matplotlib, scikit-learn.

## Running Code

개별 스크립트를 직접 실행:
```bash
python 02_linear_regression/examples/01_simple_linear_regression.py
```

## Architecture

각 모듈은 동일한 디렉토리 패턴을 따른다:

```
XX_topic/
├── examples/           # 완성된 예제 코드 (학습용)
└── exercises/          # 학생 실습 문제 (TODO 빈칸)
    └── solutions/      # 실습 정답
```

- **examples**: 개념 설명이 포함된 완성 코드. 주석으로 가설/손실함수/옵티마이저 등 핵심 개념의 관계를 설명.
- **exercises**: TODO 주석으로 빈칸을 남겨두고 학생이 직접 구현. docstring에 힌트 제공.
- **solutions**: exercises와 동일한 파일명으로 정답 제공.

## Conventions

- 초반 모듈은 numpy 없이 순수 Python for문으로 구현하여 원리를 먼저 이해시킨다.
- 손실함수는 표준 MSE `(1/n) Σ (H(x)-y)²` 형태를 사용한다 (강의·예제 통일용).
- gradient 는 MSE 미분의 factor 2 를 포함하여 `(2/m) Σ` 형태로 examples / solutions
  통일. 단 **학생 솔루션 채점 시에는 `(1/m) Σ` 도 동등하게 정답으로 인정** —
  factor 는 lr 에 흡수되어 수학적으로 동일. (학생 자유, 통일은 강의용)
- 코드 주석과 출력 메시지는 한국어로 작성한다.
- exercises에서는 함수 분리를 도입하여 examples 대비 구조화 수준을 높인다.

## 새 연습 문제 출제 (필수 절차)

ML 학습 알고리즘 문제는 학생마다 다른 convention 으로 풀어도 수학적으로
모두 옳다 (gradient factor 2 포함/미포함, batch 평균/합산 등 — lr 에 흡수).
`EVAL: stdio` 는 stdout 한 자리 차이로 fail 처리하므로 ML 알고리즘과 안 맞다.

→ **ML 학습 알고리즘 문제의 default 채점 모드는 `EVAL: script`**. AI 가 코드
구조 + 1회 실행 stdout 을 종합 평가 (loss 단조 감소, gradient 부호·방향,
warmup/decay 식 등). `EVAL: stdio` 는 출력 형식 자체가 학습 목표인 비-ML
문제 (FizzBuzz, 패턴 출력 등) 에서만 사용.

출제 전 반드시:

1. [`agent/AUTHORING.md`](agent/AUTHORING.md) 의 모드 선택 가이드 + Hard Rules
   + Workflow 를 따른다.
2. 검증:
   ```bash
   python agent/verify_problem.py <module>/exercises/<file>.py
   ```
3. PASS 면 출제 가능 (단원 import 패널 업로드).
4. `script` 모드 단원이면 단원 채점 지침에 ML-specific AI 채점 prompt 추가
   (AUTHORING.md 의 템플릿 참조).

전체 일괄 검증:
```bash
python agent/verify_problem.py --all
```
