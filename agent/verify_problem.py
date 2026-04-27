#!/usr/bin/env python3
"""
ML 연습 문제 검증 스크립트.

Agent YJU:Eval 의 단원 import 시 학생 솔루션이 expected_stdout 과
정확 일치해야 자동 채점 통과한다. 이 스크립트는 출제자가 작성한
exercise + solution 짝을 출제 전에 동일한 비교 로직으로 검증한다.

검증 항목:
  1. exercise 파일 헤더 (TITLE / DIFFICULTY / EVAL)
  2. # META_TESTS: 블록 형식 (YAML 파싱 가능)
  3. solutions/<같은 파일명>.py 존재
  4. solution 실행 stdout == META_TESTS expected_stdout (정규화 후 비교)
  5. (선택) skeleton 그대로 실행 시 학생에게 명확한 신호 (에러 또는 미구현)

사용법:
  python agent/verify_problem.py 02_linear_regression/exercises/06_warmup_linear.py
  python agent/verify_problem.py --all
  python agent/verify_problem.py --module 02_linear_regression
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.stderr.write('PyYAML 가 필요합니다. `pip install pyyaml`\n')
    sys.exit(2)

# Windows cp949 콘솔에서 ✓/✗/⚠ 등 unicode 깨짐 방지.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass


META_HEADER_RE = re.compile(r'^\s*#\s*META_TESTS\s*:\s*$', re.IGNORECASE)
META_LINE_RE = re.compile(r'^\s*#\s?(.*)$')
DOCSTRING_RE = re.compile(r"""^\s*("""r'"""'r"""|''')([\s\S]*?)\1""")
HEADER_FIELD_RE = re.compile(r'^\s*(TITLE|DIFFICULTY|TAGS|EVAL)\s*:\s*(.+?)\s*$')

ROOT = Path(__file__).resolve().parent.parent


# ───────────────────────────────────────────────────────────
# 파싱
# ───────────────────────────────────────────────────────────

def parse_header(content: str) -> dict[str, str] | None:
    """파일 상단 docstring 에서 헤더 필드 추출."""
    m = DOCSTRING_RE.match(content)
    if not m:
        return None
    doc = m.group(2)
    fields: dict[str, str] = {}
    for line in doc.splitlines():
        m2 = HEADER_FIELD_RE.match(line)
        if m2:
            fields[m2.group(1)] = m2.group(2)
    return fields


def parse_meta_tests(content: str) -> list[dict[str, Any]] | dict[str, str] | None:
    """# META_TESTS: 블록을 YAML list 로 반환. 에러는 dict {'__error__': ...}."""
    lines = content.splitlines()
    in_meta = False
    yaml_lines: list[str] = []
    for line in lines:
        if META_HEADER_RE.match(line):
            in_meta = True
            continue
        if not in_meta:
            continue
        m = META_LINE_RE.match(line)
        if not m:
            break
        yaml_lines.append(m.group(1))
    if not yaml_lines:
        return None
    try:
        parsed = yaml.safe_load('\n'.join(yaml_lines))
    except yaml.YAMLError as e:
        return {'__error__': str(e)}
    if not isinstance(parsed, list):
        return {'__error__': 'META_TESTS 가 list 형태가 아님'}
    return parsed


# ───────────────────────────────────────────────────────────
# 실행 / 비교
# ───────────────────────────────────────────────────────────

def normalize_stdout(s: str | None) -> str:
    """줄 끝 공백 제거 + 마지막 빈 줄 제거 (서버 비교 로직과 동일)."""
    if s is None:
        s = ''
    lines = [ln.rstrip() for ln in s.replace('\r\n', '\n').splitlines()]
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def run_python_file(path: Path, stdin: str = '', timeout: int = 60):
    """python 으로 파일 실행. UTF-8 강제."""
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    return subprocess.run(
        [sys.executable, str(path)],
        input=stdin,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout,
        env=env,
    )


def diff_lines(expected: str, actual: str, max_lines: int = 10) -> str:
    a = expected.splitlines()
    b = actual.splitlines()
    out: list[str] = []
    n = max(len(a), len(b))
    shown = 0
    for i in range(n):
        ax = a[i] if i < len(a) else '(EOF)'
        bx = b[i] if i < len(b) else '(EOF)'
        if ax != bx:
            out.append(f'  L{i+1}: expected = {ax!r}')
            out.append(f'  L{i+1}: actual   = {bx!r}')
            shown += 1
            if shown >= max_lines:
                out.append(f'  … (이후 차이 생략)')
                break
    return '\n'.join(out) if out else '  (차이 없음 — 정규화 후 일치)'


# ───────────────────────────────────────────────────────────
# 검증
# ───────────────────────────────────────────────────────────

def find_solution_path(exercise_path: Path) -> Path:
    return exercise_path.parent / 'solutions' / exercise_path.name


def verify_one(exercise_path: Path) -> tuple[bool, list[str], list[str]]:
    """단일 exercise 검증. (passed, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not exercise_path.exists():
        return False, [f'파일 없음: {exercise_path}'], []

    content = exercise_path.read_text(encoding='utf-8')

    # 1. 헤더
    header = parse_header(content)
    if not header:
        errors.append('상단 docstring 없음 (""" ... """ 필요)')
        return False, errors, warnings
    for key in ('TITLE', 'DIFFICULTY', 'EVAL'):
        if key not in header:
            errors.append(f'헤더 {key} 누락')

    # 2. META_TESTS
    meta = parse_meta_tests(content)
    if meta is None:
        errors.append('# META_TESTS: 블록 없음')
    elif isinstance(meta, dict) and '__error__' in meta:
        errors.append(f'META_TESTS 파싱 실패: {meta["__error__"]}')

    # 3. solution 존재
    solution_path = find_solution_path(exercise_path)
    if not solution_path.exists():
        errors.append(f'solution 없음: {solution_path.relative_to(ROOT)}')

    if errors:
        return False, errors, warnings

    eval_mode = header.get('EVAL', '').lower().strip()
    assert isinstance(meta, list)

    # 4. EVAL 모드별 검증
    if eval_mode == 'stdio':
        for i, tc in enumerate(meta):
            stdin = str(tc.get('stdin', '') or '')
            expected = str(tc.get('expected_stdout', '') or '')
            try:
                run = run_python_file(solution_path, stdin=stdin)
            except subprocess.TimeoutExpired:
                errors.append(f'TC{i}: solution 실행 시간 초과 (>60s)')
                continue
            if run.returncode != 0:
                errors.append(
                    f'TC{i}: solution 비정상 종료 (exit {run.returncode})\n'
                    f'  stderr (앞 500자):\n  {run.stderr[:500]}'
                )
                continue
            actual_norm = normalize_stdout(run.stdout)
            expected_norm = normalize_stdout(expected)
            if actual_norm != expected_norm:
                errors.append(
                    f'TC{i}: stdout 불일치\n'
                    f'{diff_lines(expected_norm, actual_norm)}'
                )

    elif eval_mode == 'testcases':
        warnings.append(
            'EVAL: testCases — 함수 기반 채점은 서버 런타임에서 검증됨 (이 스크립트는 stdio 만).'
        )
        # solution 실행 가능 여부만 체크
        run = run_python_file(solution_path)
        if run.returncode != 0:
            errors.append(f'solution 실행 실패: {run.stderr[:500]}')

    elif eval_mode == 'script':
        warnings.append('EVAL: script — AI 채점만, stdout 비교 안 함.')
        run = run_python_file(solution_path)
        if run.returncode != 0:
            errors.append(f'solution 실행 실패: {run.stderr[:500]}')

    else:
        errors.append(f'알 수 없는 EVAL 모드: {eval_mode!r}')

    # 5. skeleton 학생 신호 체크
    try:
        run_skel = run_python_file(exercise_path, timeout=15)
        if run_skel.returncode == 0 and not run_skel.stderr:
            warnings.append(
                '⚠ skeleton 그대로 실행해도 에러 없이 종료됨. 학생이 미구현 상태로 '
                '제출하면 빈 stdout 으로 fail 만 표시됨. TODO 부분에 '
                'NotImplementedError 또는 명시적 print 안내 권장.'
            )
    except subprocess.TimeoutExpired:
        warnings.append('skeleton 실행 시간 초과 — 학생 영역에 무한 루프 가능성?')

    return len(errors) == 0, errors, warnings


# ───────────────────────────────────────────────────────────
# 진입점
# ───────────────────────────────────────────────────────────

def collect_targets(args: argparse.Namespace) -> list[Path]:
    if args.all:
        return sorted(ROOT.glob('*/exercises/*.py'))
    if args.module:
        return sorted((ROOT / args.module / 'exercises').glob('*.py'))
    if not args.files:
        return []
    return [Path(f).resolve() for f in args.files]


def main() -> int:
    ap = argparse.ArgumentParser(description='ML 연습 문제 검증')
    ap.add_argument('files', nargs='*', help='exercise 파일 경로')
    ap.add_argument('--all', action='store_true', help='모든 exercises 검증')
    ap.add_argument('--module', help='특정 모듈 폴더만 (예: 02_linear_regression)')
    args = ap.parse_args()

    targets = collect_targets(args)
    if not targets:
        ap.print_help()
        return 1

    fail_count = 0
    warn_count = 0
    print(f'검증 대상: {len(targets)}개\n')
    for path in targets:
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path
        print(f'=== {rel} ===')
        passed, errors, warnings = verify_one(path)
        if passed:
            print('  ✓ PASS')
        else:
            fail_count += 1
            print('  ✗ FAIL')
            for e in errors:
                for ln in e.splitlines():
                    print(f'    {ln}')
        for w in warnings:
            warn_count += 1
            print(f'  ⚠ {w}')
        print()

    summary = f'총 {len(targets)}건 — 실패 {fail_count}, 경고 {warn_count}'
    print('=' * len(summary))
    print(summary)
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
