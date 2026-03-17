"""
NumPy 기초 - 배열 생성과 속성
=============================
Python 리스트와 NumPy 배열(ndarray)의 차이를 이해하고,
배열을 생성하는 다양한 방법을 익힌다.
"""

import numpy as np

# ============================================================
# 1. 리스트 vs ndarray
#    리스트는 범용 컨테이너, ndarray는 수치 연산에 최적화된 배열이다.
# ============================================================
py_list = [1, 2, 3, 4, 5]
np_array = np.array([1, 2, 3, 4, 5])

print("=== 리스트 vs ndarray ===")
print(f"리스트:  {py_list}, 타입: {type(py_list)}")
print(f"ndarray: {np_array}, 타입: {type(np_array)}")

# 리스트는 * 연산이 반복, ndarray는 요소별 곱셈
print(f"\n리스트 * 2  = {py_list * 2}")       # [1,2,3,4,5,1,2,3,4,5]
print(f"ndarray * 2 = {np_array * 2}")         # [2,4,6,8,10]

# ============================================================
# 2. 배열의 속성
#    shape: 배열의 형태 (행, 열)
#    dtype: 요소의 데이터 타입
#    ndim:  차원 수
#    size:  전체 요소 수
# ============================================================
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("\n=== 배열 속성 ===")
print(f"1D 배열: {arr_1d}")
print(f"  shape: {arr_1d.shape}")   # (3,)
print(f"  dtype: {arr_1d.dtype}")   # int64
print(f"  ndim:  {arr_1d.ndim}")    # 1
print(f"  size:  {arr_1d.size}")    # 3

print(f"\n2D 배열:\n{arr_2d}")
print(f"  shape: {arr_2d.shape}")   # (2, 3) → 2행 3열
print(f"  dtype: {arr_2d.dtype}")   # int64
print(f"  ndim:  {arr_2d.ndim}")    # 2
print(f"  size:  {arr_2d.size}")    # 6

# ============================================================
# 3. 배열 생성 함수
#    자주 쓰는 패턴: 0으로 채우기, 1로 채우기, 범위 생성, 균등 분할
# ============================================================
print("\n=== 배열 생성 함수 ===")

# zeros: 0으로 채운 배열
zeros = np.zeros((2, 3))
print(f"zeros(2,3):\n{zeros}")

# ones: 1로 채운 배열
ones = np.ones((2, 3))
print(f"\nones(2,3):\n{ones}")

# arange: 범위 지정 (start, stop, step) — Python의 range와 동일한 규칙
arange = np.arange(0, 10, 2)
print(f"\narange(0, 10, 2): {arange}")  # [0, 2, 4, 6, 8]

# linspace: 구간을 균등하게 나누기
linspace = np.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {linspace}")  # [0, 0.25, 0.5, 0.75, 1.0]

# ============================================================
# 4. dtype 지정
#    정수(int), 실수(float) 등 타입을 명시할 수 있다.
#    ML에서는 대부분 float로 다룬다.
# ============================================================
print("\n=== dtype 지정 ===")
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1, 2, 3], dtype=np.float64)
print(f"int32 배열:   {int_arr}, dtype: {int_arr.dtype}")
print(f"float64 배열: {float_arr}, dtype: {float_arr.dtype}")

# ============================================================
# 5. reshape: 형태 변환
#    데이터는 그대로, 모양만 바꾼다. 전체 요소 수는 동일해야 한다.
# ============================================================
print("\n=== reshape ===")
arr = np.arange(1, 7)          # [1, 2, 3, 4, 5, 6]
reshaped = arr.reshape(2, 3)   # 2행 3열로 변환
print(f"원본:    {arr}")
print(f"reshape(2,3):\n{reshaped}")
