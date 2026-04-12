"""
SGD의 경로는 왜 흔들리는가?

왼쪽 (원인) : 매 step마다 뽑힌 "1개 샘플"과 그 샘플 한 개로 계산된 gradient
오른쪽 (결과) : 그 gradient들을 따라 실제로 이동한 파라미터 궤적

같은 step을 양쪽에서 동시에 보여주어
"sample이 달라지면 gradient가 달라지고 → 경로가 흔들린다" 를 시각화한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

rcParams["font.family"] = "AppleGothic"
rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 데이터 & 손실
# ----------------------------------------------------------------------
rng = np.random.default_rng(2)
N = 50
X = np.linspace(-3, 3, N)
true_w, true_b = 2.0, 1.0
y = true_w * X + true_b + rng.normal(0, 1.6, size=N)


def mse(w, b):
    return np.mean((w * X + b - y) ** 2)


def grad_one(w, b, i):
    err = (w * X[i] + b) - y[i]
    return 2 * err * X[i], 2 * err


# 등고선
ws = np.linspace(-1, 4.5, 90)
bs = np.linspace(-3, 4.5, 90)
WW, BB = np.meshgrid(ws, bs)
LL = np.array([[mse(w, b) for w in ws] for b in bs])

# ----------------------------------------------------------------------
# SGD 시뮬레이션 — 매 step마다 (선택된 sample, w, b)를 기록
# ----------------------------------------------------------------------
N_STEPS = 70
lr = 0.04
w, b = -0.5, -2.0
order = rng.permutation(N)
history = []  # (sample_idx, w_before, b_before, dw, db, w_after, b_after)
for step in range(N_STEPS):
    i = order[step % N]
    if step % N == 0 and step > 0:
        order = rng.permutation(N)
    dw, db = grad_one(w, b, i)
    w_new = w - lr * dw
    b_new = b - lr * db
    history.append((i, w, b, dw, db, w_new, b_new))
    w, b = w_new, b_new

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(
    3, 2, height_ratios=[0.55, 4, 0.4], hspace=0.35, wspace=0.25
)

# 상단 핵심 문구
ax_top = fig.add_subplot(gs[0, :]); ax_top.axis("off")
ax_top.text(
    0.5, 0.5,
    "SGD는 sample 1개만 보고 업데이트하기 때문에,\n"
    "각 sample의 특성이 gradient 방향에 크게 반영된다",
    ha="center", va="center", fontsize=14, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.6", fc="#fff7d6", ec="#d4a017", lw=1.5),
)

# 본문
ax_left  = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[1, 1])

# 하단 한 줄 정리
ax_bot = fig.add_subplot(gs[2, :]); ax_bot.axis("off")
ax_bot.text(
    0.5, 0.5,
    "SGD는 빠르게 이동할 수 있지만, 개별 sample의 영향이 커서 "
    "경로가 불안정해질 수 있다",
    ha="center", va="center", fontsize=12,
    color="#444",
    bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#888"),
)

# ── 왼쪽: 원인 ────────────────────────────────────────────────
ax_left.set_title("원인 ─ sample 1개만 본다", fontsize=12, fontweight="bold")
ax_left.scatter(X, y, s=22, color="lightgray", edgecolor="none")
hl_pt = ax_left.scatter([], [], s=130, color="#d62728",
                        edgecolor="black", linewidth=0.8, zorder=5)
# 현재 모델 직선
x_line = np.linspace(-3.5, 3.5, 50)
(line_model,) = ax_left.plot([], [], color="#1f77b4", lw=2)
# 오차를 잇는 점선
(err_line,) = ax_left.plot([], [], "--", color="#d62728", lw=1.2)
ax_left.set_xlim(-3.5, 3.5)
ax_left.set_ylim(y.min() - 2, y.max() + 2)
ax_left.set_xlabel("x"); ax_left.set_ylabel("y")
left_text = ax_left.text(
    0.02, 0.98,
    "• sample마다 값과 오차가 다르다\n"
    "• 따라서 sample마다 gradient도 다르다\n"
    "• SGD는 그 1개 기준으로 즉시 업데이트\n"
    "• → 매 step 방향이 조금씩 달라진다",
    transform=ax_left.transAxes, va="top", fontsize=9,
    bbox=dict(boxstyle="round", fc="white", ec="#d62728", alpha=0.9),
)
left_info = ax_left.text(
    0.98, 0.02, "", transform=ax_left.transAxes, va="bottom", ha="right",
    fontsize=9,
    bbox=dict(boxstyle="round", fc="#fff", ec="gray"),
)

# ── 오른쪽: 결과 ──────────────────────────────────────────────
ax_right.set_title("결과 ─ 경로가 흔들린다", fontsize=12, fontweight="bold")
ax_right.contour(WW, BB, LL, levels=20, cmap="viridis", alpha=0.55)
ax_right.plot(true_w, true_b, "*", color="gold",
              markersize=16, markeredgecolor="black", label="정답")
(path_line,) = ax_right.plot([], [], "-", color="#d62728", lw=1.4, alpha=0.85)
(path_pt,)   = ax_right.plot([], [], "o", color="#d62728", markersize=8,
                             markeredgecolor="black")
arrow_holder = [None]  # 매 프레임 새 화살표
ax_right.set_xlim(-1, 4.5); ax_right.set_ylim(-3, 4.5)
ax_right.set_xlabel("기울기 w"); ax_right.set_ylabel("절편 b")
ax_right.legend(loc="upper right", fontsize=8)
right_text = ax_right.text(
    0.02, 0.98,
    "• 전체적으로는 정답 쪽으로 이동\n"
    "• 하지만 매 step 방향이 들쭉날쭉\n"
    "• 최적점 근처에서도 진동이 생긴다\n"
    "• → 학습 경로가 매끄럽지 않다",
    transform=ax_right.transAxes, va="top", fontsize=9,
    bbox=dict(boxstyle="round", fc="white", ec="#d62728", alpha=0.9),
)

# ----------------------------------------------------------------------
# 애니메이션
# ----------------------------------------------------------------------
ws_path, bs_path = [], []


def update(frame):
    i, w0, b0, dw, db, w1, b1 = history[frame]

    # 왼쪽: 현재 모델 직선 + 선택된 샘플 + 오차 점선
    line_model.set_data(x_line, w0 * x_line + b0)
    hl_pt.set_offsets([[X[i], y[i]]])
    pred_y = w0 * X[i] + b0
    err_line.set_data([X[i], X[i]], [pred_y, y[i]])
    left_info.set_text(
        f"step {frame:02d}\n"
        f"sample #{i:2d} → x={X[i]:+.2f}, y={y[i]:+.2f}\n"
        f"이 샘플의 gradient = ({dw:+.2f}, {db:+.2f})"
    )

    # 오른쪽: 경로 누적 + 이번 step 화살표
    ws_path.append(w0); bs_path.append(b0)
    if frame == len(history) - 1:
        ws_path.append(w1); bs_path.append(b1)
    path_line.set_data(ws_path, bs_path)
    path_pt.set_data([w0], [b0])

    if arrow_holder[0] is not None:
        arrow_holder[0].remove()
    arrow_holder[0] = ax_right.annotate(
        "", xy=(w1, b1), xytext=(w0, b0),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
    )
    return []


anim = animation.FuncAnimation(
    fig, update, frames=len(history), interval=250, blit=False
)

out_path = "class/animations/03_SGD_경로가_흔들리는_이유.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=5), dpi=90)
print(f"저장 완료: {out_path}")
