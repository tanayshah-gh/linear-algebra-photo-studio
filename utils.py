"""
utils.py
Image I/O and helpers shared by all filter modules.

Convention used everywhere in the project:
    an RGB image is a NumPy array of shape (H, W, 3), dtype float, values 0-255.
"""

import numpy as np
from PIL import Image


def load_image(path):
    """Load an image file as a float RGB array of shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=float)


def save_image(img, path):
    """Clip to 0-255, convert to uint8 and save."""
    Image.fromarray(to_uint8(img)).save(path)


def to_uint8(img):
    """Clip values to the valid pixel range and cast to uint8."""
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def apply_color_matrix(img, M):
    """Apply a 3x3 matrix M to every pixel's RGB vector.

    For one pixel p = [R, G, B]^T the new pixel is p' = M @ p.
    Doing this for all H*W pixels at once: reshape the image into a
    (H*W, 3) table of row vectors P, then P' = P @ M^T.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError("Colour matrix must be 3x3")
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)          # (H*W, 3)
    out = flat @ M.T                   # same as M @ pixel for every pixel
    return out.reshape(h, w, 3)


def make_sample_image(size=256):
    """Build a colourful test image (gradients, shapes) so the project runs
    without needing an external photo."""
    y, x = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3))
    img[..., 0] = x / (size - 1) * 255                 # red grows left -> right
    img[..., 1] = y / (size - 1) * 255                 # green grows top -> bottom
    img[..., 2] = (1 - x / (size - 1)) * 200 + 30      # blue grows right -> left

    # a white disc and a dark square, so edges exist for later filters
    cy, cx, r = size // 3, size // 3, size // 6
    disc = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
    img[disc] = [255, 255, 255]
    s = size // 5
    img[2 * size // 3: 2 * size // 3 + s, 2 * size // 3: 2 * size // 3 + s] = [20, 20, 20]
    return img