"""Step 4 demo: convolution filters, edge detection, and the matrix views of convolution."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import convolution_filters as cv
import matrix_ops as mo
from utils import load_image, make_sample_image, save_image, to_uint8

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "images", "sample.png")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(IMG_PATH):
    os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
    save_image(make_sample_image(), IMG_PATH)
img = load_image(IMG_PATH)

# ------------------------------------------------ 1. filter grid
names = list(cv.FILTERS)
mag, gx, gy = cv.sobel_edges(img)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
axes[0].imshow(to_uint8(img)); axes[0].set_title("Original")
for ax, name in zip(axes[1:], names):
    out, kernel = cv.apply_filter(img, name)
    ax.imshow(to_uint8(out)); ax.set_title(f"{name}\nkernel sum = {kernel.sum():.2f}")
axes[6].imshow(mag, cmap="gray"); axes[6].set_title("Sobel edges\n sqrt(Gx² + Gy²)")
axes[7].imshow(to_uint8(cv.convolve(img, cv.identity_kernel()))); axes[7].set_title("Identity kernel\n(unchanged)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step4_filters.png"), dpi=110)
plt.close()
print("Saved outputs/step4_filters.png")

# ------------------------------------------------ 2. verify against SciPy
gray = cv.to_gray(img)
mine = cv.convolve(gray, cv.gaussian_kernel(5, 1.2))
try:
    from scipy.signal import convolve2d
    ref = convolve2d(gray, cv.gaussian_kernel(5, 1.2), mode="same", boundary="fill")
    print(f"\nim2col vs SciPy convolve2d (Gaussian): max difference = {np.abs(mine - ref).max():.2e}")
    mine_s = cv.convolve(gray, cv.sobel_x_kernel())
    ref_s = convolve2d(gray, cv.sobel_x_kernel(), mode="same", boundary="fill")
    print(f"im2col vs SciPy convolve2d (Sobel x):  max difference = {np.abs(mine_s - ref_s).max():.2e}")
except ImportError:
    print("\nSciPy not installed - skipping SciPy comparison")

# ------------------------------------------------ 3. convolution as ONE big matrix
def report(name, kern, h, w, patch):
    T = cv.convolution_matrix(kern, h, w)
    via_T = (T @ patch.ravel()).reshape(h, w)
    err = np.abs(via_T - cv.convolve(patch, kern)).max()
    cond = mo.condition_number(T)
    print(f"{name:22s} T is {T.shape[0]}x{T.shape[1]} | T@vec(img) vs im2col diff = {err:.1e} "
          f"| invertible = {mo.is_invertible(T)} | cond = {cond:.3g}")

print("\nConvolution as one big matrix T (7x7 patch -> T is 49x49):")
p7 = gray[100:107, 100:107]
report("Box blur 3x3", cv.box_blur_kernel(3), 7, 7, p7)
report("Gaussian blur 5x5", cv.gaussian_kernel(5, 1.2), 7, 7, p7)
report("Sharpen", cv.sharpen_kernel(), 7, 7, p7)

print("\nSame box blur on an 8x8 patch (T is 64x64):")
p8 = gray[100:108, 100:108]
report("Box blur 3x3 (8x8)", cv.box_blur_kernel(3), 8, 8, p8)
print("  -> exactly singular: a 3-tap box blur of width n has a zero eigenvalue when (n+1) is divisible by 3.")
print("\nTakeaway for Step 5: blur matrices are ill-conditioned (Gaussian worst), so deblurring")
print("by inversion amplifies noise; sharpen is well-conditioned and easy to undo.")