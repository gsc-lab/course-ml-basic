# ML 기초 (course-ml-basic)

머신러닝 기초 교과목 예제 및 실습 코드 저장소

## 커리큘럼

| # | 주제 | 내용 |
|---|------|------|
| 01 | NumPy | 배열 생성, 인덱싱, 연산, 브로드캐스팅 |
| 02 | Linear Regression | 가설, 손실함수, 경사하강법 |
| 03 | Logistic Regression | (예정) |

## 디렉토리 구조

```
XX_topic/
├── examples/           # 강의용 예제 코드 (완성본)
└── exercises/          # 학생 실습 문제 (TODO 빈칸)
    └── solutions/      # 실습 정답
```

## 설치 및 실행

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -e .

# 예제 실행
python 01_numpy/examples/01_array_basics.py
python 02_linear_regression/examples/01_simple_linear_regression.py
```

## 요구사항

- Python >= 3.10
- numpy, pandas, matplotlib, scikit-learn
