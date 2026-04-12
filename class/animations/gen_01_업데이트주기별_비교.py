"""
업데이트 주기에 따른 경사하강법 특성 비교 애니메이션

세 가지 방식을 나란히 비교한다:
  - BGD (Batch)      : 전체 데이터를 보고 1번 업데이트  → 느리지만 안정적
  - Mini-batch GD    : 일부만 보고 업데이트            → 균형
  - SGD (Stochastic) : 데이터 1개마다 업데이트         → 빠르지만 흔들림

위쪽: 데이터에 직선을 맞추는 과정
아래쪽: 손실(MSE) 등고선 위 파라미터(w, b) 이동 궤적
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

# 한글 폰트 (macOS 기본)
rcParams["font.family"] = "AppleGothic"
rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 1. 데이터 생성: y = 2x + 1 + 노이즈
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)
N = 40
X = np.linspace(-3, 3, N)
true_w, true_b = 2.0, 1.0
y = true_w * X + true_b + rng.normal(0, 1.5, size=N)


def mse(w, b):
    return np.mean((w * X + b - y) ** 2)


def grad(w, b, xb, yb):
    pred = w * xb + b
    err = pred - yb
    dw = 2 * np.mean(err * xb)
    db = 2 * np.mean(err)
    return dw, db


# ----------------------------------------------------------------------
# 2. 세 가지 방식으로 학습 궤적 기록
#    "프레임 = 1번의 파라미터 업데이트" 로 통일
# ----------------------------------------------------------------------
def train(method, n_frames, lr):
    w, b = -3.0, -3.0  # 동일한 시작점
    traj = [(w, b)]
    idx = rng.permutation(N)
    cursor = 0

    for _ in range(n_frames):
        if method == "BGD":
            xb, yb = X, y
        elif method == "SGD":
            i = idx[cursor % N]
            xb, yb = np.array([X[i]]), np.array([y[i]])
            cursor += 1
            if cursor % N == 0:
                idx = rng.permutation(N)
        elif method == "MBGD":
            batch = idx[cursor % N : cursor % N + 8]
            if len(batch) < 8:
                idx = rng.permutation(N)
                batch = idx[:8]
                cursor = 0
            xb, yb = X[batch], y[batch]
            cursor += 8

        dw, db = grad(w, b, xb, yb)
        w -= lr * dw
        b -= lr * db
        traj.append((w, b))

    return np.array(traj)


N_FRAMES = 60
traj_bgd = train("BGD",  N_FRAMES, lr=0.05)
traj_mbg = train("MBGD", N_FRAMES, lr=0.05)
traj_sgd = train("SGD",  N_FRAMES, lr=0.05)

# ----------------------------------------------------------------------
# 3. 손실 등고선 준비
# ----------------------------------------------------------------------
ws = np.linspace(-4, 5, 120)
bs = np.linspace(-4, 5, 120)
WW, BB = np.meshgrid(ws, bs)
LL = np.zeros_like(WW)
for i in range(WW.shape[0]):
    for j in range(WW.shape[1]):
        LL[i, j] = mse(WW[i, j], BB[i, j])

# ----------------------------------------------------------------------
# 4. Figure 구성: 3열(방식) × 2행(직선 / 등고선)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
titles = [
    "BGD (전체 데이터)\n→ 느리지만 매끄럽게",
    "Mini-batch (일부)\n→ 균형 잡힌 업데이트",
    "SGD (1개씩)\n→ 빠르지만 들쭉날쭉",
]
trajs = [traj_bgd, traj_mbg, traj_sgd]
colors = ["#2a6fdb", "#2ca02c", "#d62728"]

line_artists, point_artists, path_artists, text_artists = [], [], [], []
x_line = np.linspace(-3.5, 3.5, 50)

for col in range(3):
    ax_top = axes[0, col]
    ax_top.scatter(X, y, s=18, color="gray", alpha=0.7)
    ax_top.set_xlim(-3.5, 3.5)
    ax_top.set_ylim(y.min() - 2, y.max() + 2)
    ax_top.set_title(titles[col], fontsize=11)
    ax_top.set_xlabel("x"); ax_top.set_ylabel("y")
    (ln,) = ax_top.plot([], [], color=colors[col], lw=2.5)
    line_artists.append(ln)
    txt = ax_top.text(
        0.03, 0.95, "", transform=ax_top.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )
    text_artists.append(txt)

    ax_bot = axes[1, col]
    ax_bot.contour(WW, BB, LL, levels=20, cmap="viridis", alpha=0.6)
    ax_bot.plot(true_w, true_b, "*", color="gold",
                markersize=15, markeredgecolor="black", label="정답")
    ax_bot.set_xlim(-4, 5); ax_bot.set_ylim(-4, 5)
    ax_bot.set_xlabel("기울기 w"); ax_bot.set_ylabel("절편 b")
    ax_bot.legend(loc="upper right", fontsize=8)
    (pt,) = ax_bot.plot([], [], "o", color=colors[col], markersize=8)
    (pa,) = ax_bot.plot([], [], "-", color=colors[col], lw=1.2, alpha=0.7)
    point_artists.append(pt)
    path_artists.append(pa)

fig.suptitle(
    "업데이트 주기가 학습 특성을 바꾼다  —  같은 업데이트 횟수, 다른 모습",
    fontsize=13, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])


# ----------------------------------------------------------------------
# 5. 애니메이션
# ----------------------------------------------------------------------
def update(frame):
    artists = []
    for col in range(3):
        traj = trajs[col]
        w, b = traj[frame]
        line_artists[col].set_data(x_line, w * x_line + b)
        loss = mse(w, b)
        text_artists[col].set_text(
            f"step {frame:02d}\nw={w:+.2f}, b={b:+.2f}\nMSE={loss:.2f}"
        )
        point_artists[col].set_data([w], [b])
        path_artists[col].set_data(traj[: frame + 1, 0], traj[: frame + 1, 1])
        artists += [line_artists[col], text_artists[col],
                    point_artists[col], path_artists[col]]
    return artists


anim = animation.FuncAnimation(
    fig, update, frames=N_FRAMES + 1, interval=200, blit=False
)

out_path = "class/animations/01_업데이트주기별_BGD_SGD_Minibatch_비교.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=6))
print(f"저장 완료: {out_path}")
