"""
scikit-learn 손글씨 숫자(digits) 데이터셋 살펴보기
softmax 회귀(다중 분류) 실습에 들어가기 전에, 사용할 데이터를 먼저 이해한다.

03장까지는 정답이 두 가지(암/정상, 바위/기뢰)인 이진 분류였다.
이번 장의 정답은 0~9 — 열 가지 중 하나를 고르는 다중 분류(multi-class) 문제다.
sigmoid 하나로는 "맞다/아니다"만 말할 수 있으므로, 새 가설(softmax)이 필요하다.

손글씨 숫자 데이터셋 (Optical Recognition of Handwritten Digits, 1998)
  미국 표준기술연구소(NIST)가 수집한 손글씨 양식에서, 43명이 쓴 숫자를 잘라내
  32×32 흑백 비트맵으로 만든 것이 원본이다. 튀르키예 보아지치 대학의 Alpaydin 과
  Kaynak 은 이 32×32 그림을 4×4 블록 64개로 나누고, 블록마다 칠해진 픽셀 수(0~16)를
  세어 8×8 = 64개의 숫자로 줄였다. 작은 흔들림·굵기 차이에 덜 민감해지고,
  데이터도 16분의 1로 작아진다.
  scikit-learn 에 들어 있는 것은 그중 UCI 테스트 세트(1,797장)의 복사본이다.

  → 풀어야 할 문제: "64개의 픽셀 값만 보고, 이 사람이 쓴 숫자가 0~9 중 무엇인지 맞힐 수 있을까?"
    우편번호 자동 분류기가 하던 일을, 우리가 만든 softmax 한 줄이 흉내 내는 것이다.

왜 다중 분류 연습에 좋은가?
  1) 정답이 열 가지(0~9)        → 이진 분류로는 풀 수 없다. softmax 가 꼭 필요한 문제
  2) 입력이 "그림"이다           → 숫자 64개를 8×8 로 펼치면 눈으로 바로 확인된다.
                                    모델이 틀린 글씨를 보면 "나라도 헷갈리겠다"를 체감
  3) 1,797장 × 64특성, 전부 정수 0~16, 결측치 없음 → 전처리가 거의 필요 없다
  4) 클래스마다 약 180장씩 고르게 들어 있다 → 정확도(accuracy)가 공정한 지표
  5) 특성 64개의 단위와 범위가 전부 같다(픽셀 밝기 0~16) → 표준화 대신 16으로
     나누는 것만으로 충분. 03장과 달리 "같은 스케일"의 데이터를 다루어 본다
  6) 작아서 NumPy 경사하강법으로도 몇 초면 끝나고, softmax 회귀만으로 정확도 95% 이상
  7) 딥러닝의 "Hello World" 인 MNIST(28×28, 7만 장)의 축소판 →
     이 장에서 만든 코드가 그대로 신경망 입력층으로 이어진다

새로 등장하는 개념 (이 파일에서 미리 맛본다)
  - 다중 분류 정답을 벡터로 표현하는 원-핫 인코딩(one-hot encoding)   → 5번
  - 2차원 그림(8×8)과 1차원 특성 벡터(64,) 사이의 reshape                 → 1번
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 픽셀 밝기(0~16)를 다섯 단계 문자로 바꾸어 터미널에 그림을 그린다
SHADES = " ░▒▓█"


def to_text_art(img):
    """(8, 8) 배열을 8줄짜리 텍스트 그림(문자열 리스트)으로 바꾼다. 값 범위 0~16."""
    rows = []
    for row in img:
        # 0→' ', 1~4→'░', 5~8→'▒', 9~12→'▓', 13~16→'█'
        level = np.clip(np.ceil(row / 4), 0, 4).astype(int)
        rows.append("".join(SHADES[v] * 2 for v in level))   # 가로를 2배로 늘려 정사각형처럼
    return rows


def print_side_by_side(arts, titles, gap="   "):
    """여러 텍스트 그림을 가로로 나란히 출력한다."""
    print(gap.join(f"{t:16s}" for t in titles))
    for r in range(8):
        print(gap.join(art[r] for art in arts))


# ============================================================
# 1. 데이터 불러오기 — 같은 데이터가 두 가지 모양으로 들어 있다
#    data   : (1797, 64)    모델에 넣는 특성 벡터. 행 하나가 글씨 한 장
#    images : (1797, 8, 8)  사람이 보는 그림. 8×8 픽셀
#    둘은 reshape 관계일 뿐 값은 똑같다. 모델은 "그림"을 모른다 —
#    64개 숫자를 받을 뿐이고, 픽셀의 위치 정보는 가중치 w 가 알아서 학습한다.
# ============================================================
ds = datasets.load_digits()
X, y = ds.data, ds.target

print("Bunch 의 키:", list(ds.keys()))
print(f"X (data)  : {X.shape}  (N=글씨 수, D=픽셀 수 8×8)")
print(f"images    : {ds.images.shape}")
print(f"y (target): {y.shape}")
print("data[0] 을 (8, 8) 로 reshape 하면 images[0] 과 같은가?",
      np.array_equal(X[0].reshape(8, 8), ds.images[0]))
# print(ds.DESCR)   # 데이터셋 공식 설명문 전체 — 한 번쯤 읽어 볼 것

# ============================================================
# 2. 글씨 한 장을 들여다보기 — 숫자 64개가 곧 그림이다
#    각 칸의 값은 4×4 블록 안에서 칠해진 픽셀 수(0~16). 클수록 진하다.
# ============================================================
idx = 0
print(f"\n{idx}번 글씨의 픽셀 값 (8×8, 정답 y={y[idx]}):")
for row in ds.images[idx]:
    print("  " + " ".join(f"{int(v):2d}" for v in row))

print(f"\n같은 값을 밝기로 그려 보면:")
for line in to_text_art(ds.images[idx]):
    print("  " + line)

# 0~9 각 숫자의 첫 번째 글씨를 나란히
first_of_each = [np.where(y == d)[0][0] for d in range(10)]
print("\n0~9 각 숫자의 첫 번째 글씨:")
print_side_by_side([to_text_art(ds.images[i]) for i in first_of_each[:5]],
                   [f"y={y[i]} (#{i})" for i in first_of_each[:5]])
print()
print_side_by_side([to_text_art(ds.images[i]) for i in first_of_each[5:]],
                   [f"y={y[i]} (#{i})" for i in first_of_each[5:]])

# ============================================================
# 3. 정답(label) — 0~9, 열 가지 클래스
#    03장과 달리 정답이 "0 또는 1" 이 아니다. 클래스 수 K=10.
#    클래스마다 약 180장씩 고르게 있어서, 한 클래스만 찍는 모델은 정확도 10% 근처다.
# ============================================================
print("\n클래스 (target_names):", ds.target_names)
print("클래스별 글씨 수:")
for label in ds.target_names:
    count = np.sum(y == label)
    print(f"  y={label}  {count:3d}장 ({100 * count / len(y):4.1f}%)  {'■' * (count // 10)}")

# ============================================================
# 4. 특성(feature) 64개 — 전부 같은 단위, 같은 범위
#    feature_names 는 pixel_행_열. 값은 모두 0~16 이므로 03장의 유방암 데이터처럼
#    특성마다 표준화할 필요가 없고, 16으로 나누어 0~1 로 맞추기만 해도 된다.
#    다만 가장자리 픽셀은 거의 항상 0 → 그 픽셀의 가중치는 학습해도 쓸 일이 없다.
# ============================================================
print(f"\n특성 이름 (처음 4개): {list(ds.feature_names[:4])} ... {ds.feature_names[-1]}")
print(f"픽셀 값의 범위: {X.min():.0f} ~ {X.max():.0f}  (값은 정수지만 dtype 은 {X.dtype})")

pixel_max = X.max(axis=0)          # 픽셀마다 1797장 중 최댓값: (64,)
pixel_mean = X.mean(axis=0)        # 픽셀마다 평균 밝기: (64,)
always_zero = np.sum(pixel_max == 0)
print(f"1,797장 모두에서 값이 0인 픽셀: {always_zero}개 / 64개")

print("\n픽셀별 평균 밝기 (8×8) — 가운데가 밝고 가장자리는 거의 0:")
for row in pixel_mean.reshape(8, 8):
    print("  " + " ".join(f"{v:4.1f}" for v in row))

# 클래스별 평균 그림 — "0 의 평균은 동그라미, 1 의 평균은 세로선"
# 03장에서 클래스별 평균을 숫자로 비교했다면, 여기서는 그림으로 비교할 수 있다.
# 모델이 학습하는 w 는 결국 클래스마다 "이 자리가 밝으면 이 숫자일 가능성이 높다"는 지도다.
mean_imgs = [X[y == d].mean(axis=0).reshape(8, 8) for d in range(10)]
print("\n클래스별 평균 그림 (같은 숫자 약 180장을 겹쳐 놓은 모습):")
print_side_by_side([to_text_art(m) for m in mean_imgs[:5]], [f"평균 {d}" for d in range(5)])
print()
print_side_by_side([to_text_art(m) for m in mean_imgs[5:]], [f"평균 {d}" for d in range(5, 10)])

# ============================================================
# 5. 원-핫 인코딩(one-hot encoding) — 정답 3 을 왜 [0,0,0,1,0,0,0,0,0,0] 으로 바꾸나?
#    y=3 과 y=7 은 "숫자 7이 3보다 크다"는 뜻이 아니라 그냥 서로 다른 이름표다.
#    정수 그대로 쓰면 모델이 크기 순서가 있다고 오해한다.
#    → 길이 K=10 인 벡터에서 정답 자리만 1, 나머지는 0 으로 둔다.
#    softmax 의 출력도 길이 10 인 확률 벡터이므로, 둘을 나란히 놓고 비교(cross entropy)한다.
#    np.eye(10)[y] : 단위행렬의 y번째 행을 뽑으면 그것이 곧 원-핫 벡터다.
# ============================================================
K = len(ds.target_names)
Y_onehot = np.eye(K)[y]            # (1797, 10)
print(f"\n원-핫 인코딩: y {y.shape} → Y_onehot {Y_onehot.shape}")
for i in range(5):
    print(f"  y={y[i]} → {Y_onehot[i].astype(int)}")
print("  원-핫 벡터를 argmax 하면 원래 정수로 돌아온다:",
      np.array_equal(Y_onehot.argmax(axis=1), y))

# ============================================================
# 6. 학습용 / 테스트용 분할
#    처음 보는 사람의 글씨도 맞혀야 의미가 있다 → 일부를 떼어 시험용으로 남긴다.
#    stratify=y : 열 개 클래스의 비율을 두 묶음에서 똑같이 유지한다.
#    (이진 분류에서는 비율 하나만 맞추면 됐지만, 이제는 열 개를 동시에 맞춘다.)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)
print(f"\n학습 데이터: {len(X_train)}장, 테스트 데이터: {len(X_test)}장")
print("클래스별 장수 —  클래스:", " ".join(f"{d:4d}" for d in range(K)))
print("              학습  :", " ".join(f"{c:4d}" for c in np.bincount(y_train, minlength=K)))
print("              테스트:", " ".join(f"{c:4d}" for c in np.bincount(y_test, minlength=K)))

# 다음 단계 — 02번에서 한다
#   X 를 16으로 나누어 0~1 로 맞춘 뒤, 가설 softmax(XW + b) 로 열 개 확률을 한꺼번에 낸다.
#   W 의 모양은 (64, 10) — 03장의 w (30,) 가 열 개 클래스로 늘어난 것이다.
#   손실은 원-핫 정답과 확률 벡터를 비교하는 cross entropy,
#   예측은 확률이 가장 큰 자리(argmax) 로 정하고, 틀린 글씨는 2번처럼 그려서 직접 확인한다.
