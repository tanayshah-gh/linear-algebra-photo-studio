"""
color_filters.py
Every colour filter is a 3x3 matrix M acting on a pixel vector [R, G, B]^T.

    new_pixel = M @ pixel

Whether the filter can be undone depends on M:
    det(M) != 0  ->  M^-1 exists  ->  original colours can be recovered
    det(M) == 0  ->  information is destroyed (e.g. grayscale)
"""

import numpy as np

import matrix_ops as mo
from utils import apply_color_matrix

# ------------------------------------------------------------------ the matrices
LUMA = np.array([0.299, 0.587, 0.114])  # perceived brightness weights


def grayscale_matrix():
    """Every output channel = weighted brightness. All 3 rows equal -> rank 1,
    det = 0, NOT invertible."""
    return np.tile(LUMA, (3, 1))


def sepia_matrix():
    """Classic sepia tone matrix (nearly singular: rows are almost dependent)."""
    return np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])


def soft_sepia_matrix(t=0.6):
    """Blend between identity (t=0) and full sepia (t=1): M = (1-t)I + t*Sepia.
    For moderate t this is well-conditioned and invertible, so it can be undone."""
    return (1 - t) * np.eye(3) + t * sepia_matrix()


def channel_swap_matrix(order=(2, 1, 0)):
    """Permutation matrix. Default (2,1,0) swaps R and B. det = +/-1, and the
    inverse is simply the transpose."""
    return np.eye(3)[list(order)]


def brightness_matrix(k=1.2):
    """Scale all channels by k (diagonal matrix). Inverse scales by 1/k."""
    return k * np.eye(3)


def saturation_matrix(s=1.5):
    """s=0 -> grayscale, s=1 -> unchanged, s>1 -> more vivid.
    M = s*I + (1-s)*G where G is the grayscale matrix."""
    return s * np.eye(3) + (1 - s) * grayscale_matrix()


def warm_tone_matrix():
    """Boost red slightly, cut blue slightly (diagonal)."""
    return np.diag([1.15, 1.0, 0.85])


# --------------------------------------------------------------------- registry
FILTERS = {
    "Grayscale": grayscale_matrix,
    "Sepia (full)": sepia_matrix,
    "Soft sepia": soft_sepia_matrix,
    "Swap R<->B": channel_swap_matrix,
    "Brighten": brightness_matrix,
    "Saturate": saturation_matrix,
    "Warm tone": warm_tone_matrix,
}


def apply_filter(img, name, **kwargs):
    """Look up a filter by name, build its matrix and apply it to the image.
    Returns (filtered_image, matrix)."""
    M = FILTERS[name](**kwargs)
    return apply_color_matrix(img, M), M


def describe_matrix(M):
    """Facts we show in the demo/viva: determinant, invertibility, conditioning."""
    return {
        "determinant": mo.determinant(M),
        "invertible": mo.is_invertible(M),
        "condition_number": mo.condition_number(M),
    }