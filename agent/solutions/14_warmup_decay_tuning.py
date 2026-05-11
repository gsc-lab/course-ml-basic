"""
정답: Linear Warmup + Linear Decay 파라미터 튜닝 (모범 답안)
============================================================
함수 없이 모든 코드를 직접 작성한 형태. 학생 답안의 모범 예시.
"""
import random

# ============================================================
# 데이터셋 (변경 금지)
#   x: 일일 평균 카페인 섭취량 (잔)
#   y: 야간 수면 부족 (시간) — 카페인이 많을수록 수면 시간 감소
# ============================================================
random.seed(0)
N = 32
x_data = [round(0.2 + 4.8 * i / (N - 1), 3) for i in range(N)]   # 0.2 ~ 5.0
y_data = [round(2.5 * x + 1.0 + random.uniform(-2.0, 2.0), 3) for x in x_data]

# 공통 하이퍼파라미터
base_lr = 0.03
total_epochs = 80
batch_size = 4
w_init, b_init = 0.0, 0.0
n = len(x_data)

# ============================================================
# 조합 A : warmup 비율이 너무 큼 (총 200 중 160 epoch 을 ramp up 에 사용)
#         → 본 학습 구간이 짧아 수렴 부족할 가능성
# ============================================================
print("=" * 56)
print(f"[A] warmup_epochs = 70  (total {total_epochs} 의 87.5%)")
print("=" * 56)

w, b = w_init, b_init
warmup_epochs = 70
random.seed(42)

for epoch in range(1, total_epochs + 1):
    # lr 스케줄 (Phase 분기) — warmup 구간이 0 이면 곧장 decay 진입
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        progress = (total_epochs - epoch) / (total_epochs - warmup_epochs)
        lr = base_lr * max(progress, 0.0)

    # mini-batch 한 epoch
    indices = list(range(n))
    random.shuffle(indices)
    epoch_loss_sum = 0.0
    batch_count = 0
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        bx = [x_data[i] for i in idx]
        by = [y_data[i] for i in idx]
        m = len(bx)

        dw = (2.0 / m) * sum((w * x + b - y) * x for x, y in zip(bx, by))
        db = (2.0 / m) * sum((w * x + b - y) for x, y in zip(bx, by))
        loss = (1.0 / m) * sum((w * x + b - y) ** 2 for x, y in zip(bx, by))

        w = w - lr * dw
        b = b - lr * db

        epoch_loss_sum += loss
        batch_count += 1

    if epoch == 1 or epoch % 20 == 0 or epoch == total_epochs:
        print(f"  epoch {epoch:3d} | lr {lr:.5f} | "
              f"avg-batch-loss {epoch_loss_sum / batch_count:7.4f} | "
              f"w {w:6.3f} b {b:6.3f}")

final_loss_A = (1.0 / n) * sum((w * x + b - y) ** 2 for x, y in zip(x_data, y_data))
w_A, b_A = w, b

# ============================================================
# 조합 B : warmup 30 epoch + linear decay (삼각형 스케줄)
# ============================================================
print()
print("=" * 56)
print(f"[B] warmup_epochs = 15  (total {total_epochs} 의 18.75%)")
print("=" * 56)

w, b = w_init, b_init
warmup_epochs = 15
random.seed(42)

for epoch in range(1, total_epochs + 1):
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        progress = (total_epochs - epoch) / (total_epochs - warmup_epochs)
        lr = base_lr * max(progress, 0.0)

    indices = list(range(n))
    random.shuffle(indices)
    epoch_loss_sum = 0.0
    batch_count = 0
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        bx = [x_data[i] for i in idx]
        by = [y_data[i] for i in idx]
        m = len(bx)

        dw = (2.0 / m) * sum((w * x + b - y) * x for x, y in zip(bx, by))
        db = (2.0 / m) * sum((w * x + b - y) for x, y in zip(bx, by))
        loss = (1.0 / m) * sum((w * x + b - y) ** 2 for x, y in zip(bx, by))

        w = w - lr * dw
        b = b - lr * db

        epoch_loss_sum += loss
        batch_count += 1

    if epoch == 1 or epoch == warmup_epochs or epoch % 20 == 0 or epoch == total_epochs:
        print(f"  epoch {epoch:3d} | lr {lr:.5f} | "
              f"avg-batch-loss {epoch_loss_sum / batch_count:7.4f} | "
              f"w {w:6.3f} b {b:6.3f}")

final_loss_B = (1.0 / n) * sum((w * x + b - y) ** 2 for x, y in zip(x_data, y_data))
w_B, b_B = w, b

# ============================================================
# 비교 + 선택 + 근거
# ============================================================
print()
print("=" * 56)
print("최종 비교")
print("=" * 56)
print(f"  A (warmup 70) : final_loss = {final_loss_A:.4f} | w = {w_A:.3f}, b = {b_A:.3f}")
print(f"  B (warmup 15) : final_loss = {final_loss_B:.4f} | w = {w_B:.3f}, b = {b_B:.3f}")
print()
print("[선택]   B (warmup_epochs = 15)")
print("[근거]   A 는 warmup 비율이 87.5% 라 base_lr 도달 후 곧장 decay 로 0 에 수렴, 본")
print("        학습 구간이 너무 짧아 수렴 부족. B 는 ~19% warmup 으로 안정적 ramp-up 후")
print("        충분한 본학습 + decay 까지 확보 — warmup 비율은 보통 total 의 5~20% 권장.")
