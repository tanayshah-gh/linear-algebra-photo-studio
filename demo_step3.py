"""Step 3 demo: geometric transforms, composition order, and inverse round-trip."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import geometric_transforms as gt
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
h, w = img.shape[:2]

C = lambda M: gt.about_center(M, w, h)   # act around the centre

transforms = {
    "Rotate 30°": C(gt.rotation(30)),
    "Scale 0.6": C(gt.scaling(0.6)),
    "Shear x=0.4": C(gt.shear(shx=0.4)),
    "Flip horizontal": gt.flip_horizontal(w),
    "Translate (40,20)": gt.translation(40, 20),
}

# ---- figure 1: each transform with its matrix determinant
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
axes[0].imshow(to_uint8(img)); axes[0].set_title("Original")
for ax, (name, M) in zip(axes[1:], transforms.items()):
    ax.imshow(to_uint8(gt.warp(img, M)))
    ax.set_title(f"{name}\ndet = {mo.determinant(M):.3f}")
    print(f"{name:18s} det = {mo.determinant(M):8.4f}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3_transforms.png"), dpi=110)
plt.close()

# ---- figure 2: order of multiplication matters
R, S = C(gt.rotation(40)), C(gt.shear(shx=0.5))
rot_then_shear = gt.compose(R, S)      # S @ R
shear_then_rot = gt.compose(S, R)      # R @ S
print("\nS@R equals R@S ?", np.allclose(rot_then_shear, shear_then_rot))

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
axes[0].imshow(to_uint8(img)); axes[0].set_title("Original")
axes[1].imshow(to_uint8(gt.warp(img, rot_then_shear))); axes[1].set_title("Rotate, then shear")
axes[2].imshow(to_uint8(gt.warp(img, shear_then_rot))); axes[2].set_title("Shear, then rotate")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3_order.png"), dpi=110)
plt.close()

# ---- recovery: apply T, then T^-1, compare on pixels that were never cut off
print("\nRound trip  img -> T -> T^-1  (error only where the pixel survived):")
for name, M in transforms.items():
    Minv = mo.inverse(M)
    back = gt.warp(gt.warp(img, M), Minv)
    # mask of pixels that stay inside the frame through both warps
    ones = np.ones((h, w, 1))
    keep = gt.warp(gt.warp(ones, M), Minv)[..., 0] > 0.99
    if keep.sum() == 0:
        continue
    mae = np.abs(back[keep] - img[keep]).mean()
    print(f"  {name:18s} kept {keep.mean()*100:5.1f}% of pixels | mean abs error = {mae:.3f}")
    if name == "Rotate 30°":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(to_uint8(img)); axes[0].set_title("Original")
        axes[1].imshow(to_uint8(gt.warp(img, M))); axes[1].set_title("After T (rotate 30°)")
        axes[2].imshow(to_uint8(back)); axes[2].set_title("After T⁻¹ (recovered)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "step3_recovery.png"), dpi=110)
        plt.close()

print("\nSaved outputs/step3_transforms.png, step3_order.png, step3_recovery.png")