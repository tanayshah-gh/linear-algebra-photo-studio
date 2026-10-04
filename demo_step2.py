"""Step 2 demo: apply each colour filter, print matrix facts, save a comparison grid."""

import os

import matplotlib
matplotlib.use("Agg")  # no display needed; remove this line to pop up a window
import matplotlib.pyplot as plt
import numpy as np

import color_filters as cf
from utils import load_image, make_sample_image, save_image, to_uint8

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "images", "sample.png")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# use your own photo if images/sample.png exists, else generate one
if not os.path.exists(IMG_PATH):
    os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
    save_image(make_sample_image(), IMG_PATH)
img = load_image(IMG_PATH)

names = list(cf.FILTERS)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
axes[0].imshow(to_uint8(img)); axes[0].set_title("Original")

for ax, name in zip(axes[1:], names):
    out, M = cf.apply_filter(img, name)
    info = cf.describe_matrix(M)
    ax.imshow(to_uint8(out))
    ax.set_title(f"{name}\ndet={info['determinant']:.2e}")
    print(f"\n{name}\n{np.round(M, 3)}")
    print(f"  det = {info['determinant']:.3e} | invertible = {info['invertible']} "
          f"| cond = {info['condition_number']:.2f}")
    save_image(out, os.path.join(OUT_DIR, f"{name.split()[0].lower()}_{names.index(name)}.png"))

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step2_grid.png"), dpi=110)
print("\nSaved outputs/step2_grid.png")