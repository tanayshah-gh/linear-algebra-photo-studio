"""Step 5 demo: undoing filters with matrix inverses."""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import color_filters as cf
import convolution_filters as cv
import inversion_recovery as ir
import matrix_ops as mo
from utils import apply_color_matrix, load_image, make_sample_image, save_image, to_uint8

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "images", "sample.png")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(IMG_PATH):
    os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
    save_image(make_sample_image(), IMG_PATH)
img = load_image(IMG_PATH)
rng = np.random.default_rng(42)

# =====================================================================
# PART A. Colour filters: exact recovery, then recovery after noise
# =====================================================================
print("PART A - colour filters: apply M, then M^-1")
print(f"{'filter':14s} {'cond(M)':>10s} {'no-noise RMSE':>15s} {'noisy RMSE':>12s} {'amplification':>14s}")
NOISE = 0.5   # about the size of 8-bit rounding error (grey levels)
rows = []
for name in ["Swap R<->B", "Brighten", "Warm tone", "Saturate", "Soft sepia", "Sepia (full)"]:
    M = cf.FILTERS[name]()
    cond = mo.condition_number(M)
    _, back = ir.roundtrip_color(img, M)
    exact_err = ir.rmse(back, img)
    noisy, back_n = ir.noisy_roundtrip_color(img, M, NOISE, rng)
    noisy_err = ir.rmse(back_n, img)
    rows.append((name, cond, exact_err, noisy_err))
    print(f"{name:14s} {cond:10.2f} {exact_err:15.2e} {noisy_err:12.3f} {noisy_err / NOISE:13.1f}x")
print(f"(noise added to the filtered image: std = {NOISE} grey levels)")

# figure: soft sepia (safe) vs full sepia (fragile) after the SAME small noise
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for r, name in enumerate(["Soft sepia", "Sepia (full)"]):
    M = cf.FILTERS[name]()
    noisy, back_n = ir.noisy_roundtrip_color(img, M, NOISE, np.random.default_rng(7))
    axes[r, 0].imshow(to_uint8(img)); axes[r, 0].set_title("Original")
    axes[r, 1].imshow(to_uint8(apply_color_matrix(img, M))); axes[r, 1].set_title(f"{name}\ncond = {mo.condition_number(M):.1f}")
    axes[r, 2].imshow(to_uint8(noisy)); axes[r, 2].set_title(f"+ noise (std {NOISE})\n(looks identical)")
    axes[r, 3].imshow(to_uint8(back_n)); axes[r, 3].set_title(f"Recovered via M⁻¹\nRMSE = {ir.rmse(back_n, img):.2f}")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step5_color_recovery.png"), dpi=110)
plt.close()

# storing as an 8-bit image CLIPS values >255, which no inverse can restore
M = cf.soft_sepia_matrix()
stored = to_uint8(apply_color_matrix(img, M)).astype(float)       # clip + round to 0..255
back8 = apply_color_matrix(stored, mo.inverse(M))
clipped = (apply_color_matrix(img, M) > 255).any(axis=2).mean() * 100
print(f"\nSoft sepia saved as 8-bit then inverted: RMSE = {ir.rmse(back8, img):.2f}  "
      f"({clipped:.1f}% of pixels had a channel clipped at 255 - that information is gone)")

# =====================================================================
# PART B. Singular case: grayscale
# =====================================================================
print("\nPART B - grayscale (singular)")
G = cf.grayscale_matrix()
try:
    mo.inverse(G)
except ValueError as e:
    print("  mo.inverse(G) ->", e)
c1, c2 = ir.null_direction_example()
print(f"  colour 1 = {np.round(c1, 1)}  -> grey {np.round(G @ c1, 3)}")
print(f"  colour 2 = {np.round(c2, 1)}  -> grey {np.round(G @ c2, 3)}")
print("  Different colours, identical output: the original cannot be determined from the grey pixel.")

# =====================================================================
# PART C. Deblurring: T^-1 on a blurred 16x16 image
# =====================================================================
print("\nPART C - deblurring a 16x16 grayscale image (T is 256x256)")
n = 16
x_img = ir.downsample_gray(img, n)
x = x_img.ravel()
kernel = cv.gaussian_kernel(5, 1.0)
T = cv.convolution_matrix(kernel, n, n)
cond_T = mo.condition_number(T)
print(f"  Gaussian blur matrix: cond(T) = {cond_T:.3g}")

y = T @ x
t0 = time.time()
x_clean = ir.naive_deblur(T, y)
print(f"  Inverse of clean blurred image: RMSE = {ir.rmse(x_clean, x):.2e}   ({time.time() - t0:.1f}s for our 256x256 inverse)")

y_noisy = y + rng.normal(0, 0.5, y.shape)         # tiny noise, about 8-bit rounding
x_naive = ir.naive_deblur(T, y_noisy)
print(f"  Inverse after noise std 0.5:    RMSE = {ir.rmse(x_naive, x):.2e}   (blurred image itself is off by only {ir.rmse(y_noisy, y):.2f})")

print("  Ridge deblur (x = (TᵀT + λI)⁻¹ Tᵀ y):")
best = None
for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    xr = ir.ridge_deblur(T, y_noisy, lam)
    err = ir.rmse(xr, x)
    print(f"    lambda = {lam:<8g} RMSE = {err:8.2f}")
    if best is None or err < best[0]:
        best = (err, lam, xr)
print(f"  Best lambda in this list = {best[1]:g} (picked by comparing with the true image, so this is a best case; real use needs a rule for choosing lambda)")
print(f"  Error of the blurred image vs original, for reference: {ir.rmse(y, x):.2f}")

fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
panels = [(x, "Original 16x16"), (y, "Blurred\n(Gaussian 5x5)"),
          (x_clean, f"T⁻¹ y, no noise\nRMSE {ir.rmse(x_clean, x):.1e}"),
          (x_naive, f"T⁻¹ y, noise 0.5\nRMSE {ir.rmse(x_naive, x):.1e}"),
          (best[2], f"Ridge λ={best[1]:g}\nRMSE {best[0]:.1f}")]
for ax, (arr, title) in zip(axes, panels):
    ax.imshow(arr.reshape(n, n), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step5_deblur.png"), dpi=110)
plt.close()
print("\nSaved outputs/step5_color_recovery.png and step5_deblur.png")