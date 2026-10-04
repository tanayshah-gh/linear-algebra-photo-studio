"""
geometric_transforms.py
Rotate / scale / shear / flip / translate an image with 3x3 homogeneous matrices.

A 2-D point (x, y) is written as the vector [x, y, 1]^T. In this form
translation also becomes a matrix product, so ALL transforms can be combined
by multiplying matrices:

    p' = M @ p          M = M_n @ ... @ M_2 @ M_1   (M_1 is applied first)

Coordinates: x = column index, y = row index (y points down in images).

Warping uses INVERSE MAPPING: for every pixel of the output image we compute
the source location  p_src = M^-1 @ p_out  and sample the input there.
This leaves no holes (forward mapping would).
"""

import numpy as np

import matrix_ops as mo


# ---------------------------------------------------------- basic 3x3 matrices
def translation(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)


def rotation(degrees):
    """Counter-clockwise on screen. Rotation matrices satisfy R^-1 = R^T, det = 1."""
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c,  s, 0],
                     [-s, c, 0],
                     [0,  0, 1]], dtype=float)


def scaling(sx, sy=None):
    sy = sx if sy is None else sy
    return np.diag([sx, sy, 1.0])


def shear(shx=0.0, shy=0.0):
    """shx shifts x in proportion to y; shy shifts y in proportion to x."""
    return np.array([[1,   shx, 0],
                     [shy, 1,   0],
                     [0,   0,   1]], dtype=float)


def flip_horizontal(width):
    """Mirror left <-> right: x -> (width-1) - x."""
    return np.array([[-1, 0, width - 1],
                     [0,  1, 0],
                     [0,  0, 1]], dtype=float)


def flip_vertical(height):
    return np.array([[1, 0,  0],
                     [0, -1, height - 1],
                     [0, 0,  1]], dtype=float)


# ------------------------------------------------------------- combining
def compose(*mats):
    """compose(A, B, C) applies A first, then B, then C  ->  C @ B @ A.
    Uses our own matmul. Order matters: matrix multiplication is not commutative."""
    M = np.eye(3)
    for A in mats:
        M = mo.matmul(A, M)
    return M


def about_center(M, width, height):
    """Make M act around the image centre instead of the top-left corner:
    move centre to origin, apply M, move back."""
    cx, cy = (width - 1) / 2, (height - 1) / 2
    return compose(translation(-cx, -cy), M, translation(cx, cy))


# ---------------------------------------------------------------- warping
def warp(img, M, out_shape=None, bilinear=True):
    """Apply the homogeneous transform M to an (H, W, 3) image.

    Pixels that map outside the source image are filled with black.
    """
    h, w = img.shape[:2]
    oh, ow = out_shape if out_shape is not None else (h, w)

    Minv = mo.inverse(M)  # <- Gauss-Jordan inverse from matrix_ops.py

    # homogeneous coordinates of every output pixel, shape (3, oh*ow)
    ys, xs = np.mgrid[0:oh, 0:ow]
    pts = np.vstack([xs.ravel(), ys.ravel(), np.ones(oh * ow)])

    src = Minv @ pts                       # where each output pixel came from
    sx, sy = src[0], src[1]

    if bilinear:
        out = _sample_bilinear(img, sx, sy)
    else:
        out = _sample_nearest(img, sx, sy)
    return out.reshape(oh, ow, -1)


def _sample_nearest(img, sx, sy):
    h, w = img.shape[:2]
    xi, yi = np.rint(sx).astype(int), np.rint(sy).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros((sx.size, img.shape[2]))
    out[valid] = img[yi[valid], xi[valid]]
    return out


def _sample_bilinear(img, sx, sy):
    """Weighted average of the 4 surrounding pixels (smooth result)."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    fx = (sx - x0)[valid][:, None]
    fy = (sy - y0)[valid][:, None]
    x0v, x1v, y0v, y1v = x0[valid], x1[valid], y0[valid], y1[valid]

    top = img[y0v, x0v] * (1 - fx) + img[y0v, x1v] * fx
    bot = img[y1v, x0v] * (1 - fx) + img[y1v, x1v] * fx

    out = np.zeros((sx.size, img.shape[2]))
    out[valid] = top * (1 - fy) + bot * fy
    return out