"""
배치 크기에 따른 gradient의 안정성 비교 애니메이션

왼쪽 (큰 배치)                       오른쪽 (작은 배치)
- 여러 샘플의 오차가 평균됨           - 개별 샘플 영향이 큼
- gradient 방향이 안정적              - gradient 방향이 흔들림
- 계산량이 큼                         - 빠르게 업데이트 가능

각 프레임마다 새 배치를 뽑아
  (1) 어떤 샘플이 뽑혔는지 데이터 위에 표시하고
  (2) 그 배치가 만든 gradient 화살표를 손실 등고선 위에 그린다.
최근 화살표를 잔상으로 남겨 "방향이 얼마나 흩어지는가"를 보여준다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

rcParams["font.family"] = "AppleGothic"
rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 데이터: y = 2x + 1 + noise
# ----------------------------------------------------------------------
rng = np.random.default_rng(1)
N = 60
X = np.linspace(-3, 3, N)
true_w, true_b = 2.0, 1.0
y = true_w * X + true_b + rng.normal(0, 1.8, size=N)

# 평가 지점 (정답에서 약간 떨어진 곳에서 gradient를 계산해 화살표로 표시)
W0, B0 = 0.2, -1.0


def grad_at(idx):
    xb, yb = X[idx], y[idx]
    err = (W0 * xb + B0) - yb
    dw = 2 * np.mean(err * xb)
    db = 2 * np.mean(err)
    return dw, db


# 손실 등고선
def mse(w, b):
    return np.mean((w * X + b - y) ** 2)


ws = np.linspace(-1, 4, 80)
bs = np.linspace(-3, 4, 80)
WW, BB = np.meshgrid(ws, bs)
LL = np.array([[mse(w, b) for w in ws] for b in bs])

# "참" gradient (전체 데이터 평균) — 비교 기준
true_dw, true_db = grad_at(np.arange(N))

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                         gridspec_kw={"height_ratios": [1, 1.1]})

LEFT_K, RIGHT_K = 30, 2  # 큰 배치 vs 작은 배치
configs = [
    {"col": 0, "k": LEFT_K,  "color": "#2a6fdb",
     "title": f"큰 배치  (샘플 {LEFT_K}개)",
     "bullets": ["• 여러 샘플의 오차가 함께 반영",
                 "• 개별 샘플의 영향이 작음",
                 "• gradient 방향이 안정적",
                 "• 계산량이 큼"]},
    {"col": 1, "k": RIGHT_K, "color": "#d62728",
     "title": f"작은 배치  (샘플 {RIGHT_K}개)",
     "bullets": ["• 개별 샘플의 영향이 큼",
                 "• 빠르게 업데이트 가능",
                 "• gradient 방향이 흔들림",
                 "• 데이터 순서의 영향이 큼"]},
]

scatter_artists, highlight_artists = [], []
arrow_artists, ghost_arrow_lists = [], []
text_artists = []

for cfg in configs:
    col = cfg["col"]

    # 위: 데이터 + 강조된 배치
    ax_top = axes[0, col]
    ax_top.scatter(X, y, s=22, color="lightgray", edgecolor="none")
    hl = ax_top.scatter([], [], s=70, color=cfg["color"],
                        edgecolor="black", linewidth=0.6, zorder=5)
    highlight_artists.append(hl)
    ax_top.set_xlim(-3.5, 3.5)
    ax_top.set_ylim(y.min() - 2, y.max() + 2)
    ax_top.set_title(cfg["title"], fontsize=12, color=cfg["color"],
                     fontweight="bold")
    ax_top.set_xlabel("x"); ax_top.set_ylabel("y")
    ax_top.text(0.02, 0.97, "\n".join(cfg["bullets"]),
                transform=ax_top.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white",
                          ec=cfg["color"], alpha=0.9))

    # 아래: 손실 등고선 + gradient 화살표
    ax_bot = axes[1, col]
    ax_bot.contour(WW, BB, LL, levels=18, cmap="viridis", alpha=0.55)
    ax_bot.plot(true_w, true_b, "*", color="gold",
                markersize=15, markeredgecolor="black", label="정답")
    # 평가 지점
    ax_bot.plot(W0, B0, "o", color="black", markersize=6)
    # "참" gradient 방향(전체 데이터)을 옅게 표시
    ax_bot.annotate(
        "", xy=(W0 - 0.15 * true_dw, B0 - 0.15 * true_db),
        xytext=(W0, B0),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6),
    )
    ax_bot.set_xlim(-1, 4); ax_bot.set_ylim(-3, 4)
    ax_bot.set_xlabel("기울기 w"); ax_bot.set_ylabel("절편 b")
    ax_bot.legend(loc="upper right", fontsize=8)

    txt = ax_bot.text(
        0.02, 0.97, "", transform=ax_bot.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
    )
    text_artists.append(txt)
    arrow_artists.append(None)   # 현재 화살표 (매 프레임 새로 그림)
    ghost_arrow_lists.append([]) # 잔상

fig.suptitle("배치 크기가 gradient의 안정성을 바꾼다",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ----------------------------------------------------------------------
# 애니메이션
# ----------------------------------------------------------------------
N_FRAMES = 40
GHOST_MAX = 12
ARROW_SCALE = 0.18

# 프레임마다 사용할 배치 인덱스를 미리 뽑음
batches = {0: [], 1: []}
for f in range(N_FRAMES):
    batches[0].append(rng.choice(N, size=LEFT_K,  replace=False))
    batches[1].append(rng.choice(N, size=RIGHT_K, replace=False))


def update(frame):
    for col, cfg in enumerate(configs):
        idx = batches[col][frame]
        # 강조된 샘플
        highlight_artists[col].set_offsets(np.c_[X[idx], y[idx]])

        # gradient
        dw, db = grad_at(idx)
        # 참 방향과의 각도 차이 (0이면 일치)
        cos = (dw * true_dw + db * true_db) / (
            np.hypot(dw, db) * np.hypot(true_dw, true_db) + 1e-9
        )
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

        ax_bot = axes[1, col]

        # 이전 화살표 → 잔상으로
        if arrow_artists[col] is not None:
            arrow_artists[col].set_alpha(0.25)
            ghost_arrow_lists[col].append(arrow_artists[col])
            if len(ghost_arrow_lists[col]) > GHOST_MAX:
                old = ghost_arrow_lists[col].pop(0)
                old.remove()
            # 잔상 점점 흐리게
            for i, g in enumerate(ghost_arrow_lists[col]):
                g.set_alpha(0.08 + 0.15 * (i / GHOST_MAX))

        # 새 화살표
        end = (W0 - ARROW_SCALE * dw, B0 - ARROW_SCALE * db)
        arrow_artists[col] = ax_bot.annotate(
            "", xy=end, xytext=(W0, B0),
            arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=2.2),
        )

        text_artists[col].set_text(
            f"step {frame:02d}\n"
            f"gradient = ({dw:+.2f}, {db:+.2f})\n"
            f"참 방향과의 차이 ≈ {angle:4.1f}°"
        )
    return []


anim = animation.FuncAnimation(
    fig, update, frames=N_FRAMES, interval=350, blit=False
)

out_path = "class/animations/02_배치크기에_따른_gradient_안정성.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=4), dpi=90)
print(f"저장 완료: {out_path}")
