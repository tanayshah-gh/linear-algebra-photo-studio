"""
benchmark.py  (Step 6)
Compare our hand-written matrix_ops against NumPy.

  1. Correctness : same answers as NumPy on random matrices?
  2. Speed       : how does run time grow with matrix size?
  3. Images      : per-pixel loop with our matmul vs one vectorised product
  4. Hard case   : Hilbert matrices, where accuracy depends on cond(A)
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import matrix_ops as mo
from utils import apply_color_matrix, make_sample_image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(0)


def timeit(fn, repeats=3):
    """Best of `repeats` runs, in seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ------------------------------------------------------------ 1. correctness
print("1. CORRECTNESS vs NumPy (random matrices)")
print(f"{'n':>4s} {'matmul max diff':>17s} {'det rel diff':>14s} {'inverse max diff':>18s} {'|A·A⁻¹ − I| max':>17s}")
for n in (3, 10, 30, 60):
    A = rng.random((n, n)) + n * np.eye(n) * 0.2     # keep it comfortably invertible
    B = rng.random((n, n))
    d_mm = np.abs(mo.matmul(A, B) - A @ B).max()
    det_ours, det_np = mo.determinant(A), np.linalg.det(A)
    d_det = abs(det_ours - det_np) / abs(det_np)
    Ainv = mo.inverse(A)
    d_inv = np.abs(Ainv - np.linalg.inv(A)).max()
    resid = np.abs(A @ Ainv - np.eye(n)).max()
    print(f"{n:4d} {d_mm:17.2e} {d_det:14.2e} {d_inv:18.2e} {resid:17.2e}")

# ------------------------------------------------------------------ 2. speed
print("\n2. SPEED (seconds, best of a few runs)")
sizes_mm = [10, 20, 40, 80]
sizes_inv = [10, 25, 50, 100, 200]
t_mm_ours, t_mm_np = [], []
print(f"{'matmul n':>9s} {'ours':>10s} {'NumPy':>10s} {'slowdown':>10s}")
for n in sizes_mm:
    A, B = rng.random((n, n)), rng.random((n, n))
    a = timeit(lambda: mo.matmul(A, B), repeats=2)
    b = timeit(lambda: A @ B, repeats=20)
    t_mm_ours.append(a); t_mm_np.append(b)
    print(f"{n:9d} {a:10.5f} {b:10.7f} {a / b:9.0f}x")

t_inv_ours, t_inv_np = [], []
print(f"\n{'inverse n':>9s} {'ours':>10s} {'NumPy':>10s} {'slowdown':>10s}")
for n in sizes_inv:
    A = rng.random((n, n)) + n * np.eye(n) * 0.2
    a = timeit(lambda: mo.inverse(A), repeats=2)
    b = timeit(lambda: np.linalg.inv(A), repeats=20)
    t_inv_ours.append(a); t_inv_np.append(b)
    print(f"{n:9d} {a:10.5f} {b:10.7f} {a / b:9.0f}x")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, sizes, ours, npy, title in [
        (axes[0], sizes_mm, t_mm_ours, t_mm_np, "Matrix multiplication"),
        (axes[1], sizes_inv, t_inv_ours, t_inv_np, "Matrix inverse")]:
    ax.loglog(sizes, ours, "o-", label="our implementation")
    ax.loglog(sizes, npy, "s-", label="NumPy")
    ax.set_xlabel("matrix size n  (n x n)"); ax.set_ylabel("seconds")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step6_speed.png"), dpi=110)
plt.close()

# ------------------------------------------------------------------ 3. images
print("\n3. APPLYING A 3x3 COLOUR MATRIX TO A 128x128 IMAGE (16,384 pixels)")
img = make_sample_image(128)
M = np.array([[0.636, 0.461, 0.113], [0.209, 0.812, 0.101], [0.163, 0.320, 0.479]])


def per_pixel_loop():
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = mo.matmul(M, img[i, j].reshape(3, 1)).ravel()   # our matmul, one pixel at a time
    return out


t_loop = timeit(per_pixel_loop, repeats=1)
t_vec = timeit(lambda: apply_color_matrix(img, M), repeats=20)
same = np.allclose(per_pixel_loop(), apply_color_matrix(img, M))
print(f"  per-pixel loop with our matmul : {t_loop:8.3f} s")
print(f"  one vectorised product         : {t_vec:8.5f} s   ({t_loop / t_vec:,.0f}x faster)")
print(f"  results identical              : {same}")
print(f"  (a 12-megapixel photo would take roughly {t_loop / img.shape[0] / img.shape[1] * 12e6:.0f} seconds with the loop)")

# ------------------------------------------------------------------ 4. Hilbert
print("\n4. HARD CASE: Hilbert matrices H[i][j] = 1/(i+j+1)  (famously ill-conditioned)")
print(f"{'n':>3s} {'cond(H)':>11s} {'ours |H·H⁻¹−I|':>20s} {'NumPy |H·H⁻¹−I|':>17s}")
for n in (4, 6, 8, 10, 12):
    H = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1)
    theirs = np.abs(H @ np.linalg.inv(H) - np.eye(n)).max()
    try:
        ours = f"{np.abs(H @ mo.inverse(H) - np.eye(n)).max():.2e}"
        cond = f"{mo.condition_number(H):.2e}"
    except ValueError:
        ours, cond = "refuses (singular)", f"{np.linalg.cond(H):.2e}*"
    print(f"{n:3d} {cond:>11s} {ours:>20s} {theirs:17.2e}")
print("  * = cond from NumPy, shown only if our inverse refuses a matrix.")
print("  Reading the table:")
print("   - For BOTH methods the error grows with cond(H), so most of the lost accuracy comes from the matrix.")
print("   - Ours is similar to NumPy for small, mild matrices, but the gap opens as cond(H) grows (tens of times")
print("     worse by n=10). NumPy calls LAPACK, an LU-based routine, which is generally more accurate than a")
print("     plain Gauss-Jordan inverse. Exact ratios vary by machine.")
print("   - At n=12, cond(H) ~ 1/machine-eps (1e16): both answers are wrong (residual ~1 and ~0.1 instead of ~1e-16).")
print("     A huge condition number means 'do not trust this inverse', whatever the library.")
print("\nSaved outputs/step6_speed.png")