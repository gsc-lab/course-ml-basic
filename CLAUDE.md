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
- 손실함수는 표준 MSE `(1/n) Σ (H(x)-y)²` 형태를 사용한다.
- 코드 주석과 출력 메시지는 한국어로 작성한다.
- exercises에서는 함수 분리를 도입하여 examples 대비 구조화 수준을 높인다.
