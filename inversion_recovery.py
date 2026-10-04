"""
inversion_recovery.py
Undoing filters with matrix inverses, and why it sometimes fails.

Core facts used here
    filtered = M @ original      =>   original = M^-1 @ filtered   (if M^-1 exists)

    * If det(M) = 0 (singular) no inverse exists: information was destroyed.
    * If M is invertible but ill-conditioned (large cond(M)) the inverse exists
      but tiny errors in `filtered` (rounding, noise) are amplified by up to
      cond(M) in the recovered image.
    * Ridge (Tikhonov) regularisation trades a little bias for much less noise
      amplification:  x = (T^T T + lam*I)^-1 T^T y
"""

import numpy as np

import matrix_ops as mo
from utils import apply_color_matrix


# ------------------------------------------------------------- colour filters
def roundtrip_color(img, M):
    """Apply M, then M^-1 (our Gauss-Jordan). Returns (filtered, recovered)."""
    filtered = apply_color_matrix(img, M)
    recovered = apply_color_matrix(filtered, mo.inverse(M))
    return filtered, recovered


def noisy_roundtrip_color(img, M, noise_std, rng):
    """Filter, add Gaussian noise (simulates rounding / sensor error), then invert."""
    filtered = apply_color_matrix(img, M)
    noisy = filtered + rng.normal(0.0, noise_std, filtered.shape)
    recovered = apply_color_matrix(noisy, mo.inverse(M))
    return noisy, recovered


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------- deblurring
def downsample_gray(img, n):
    """Shrink an RGB image to an (n, n) grayscale image by block averaging."""
    gray = img @ np.array([0.299, 0.587, 0.114])
    h, w = gray.shape
    bh, bw = h // n, w // n
    return gray[:bh * n, :bw * n].reshape(n, bh, n, bw).mean(axis=(1, 3))


def naive_deblur(T, y):
    """x = T^-1 y using our own Gauss-Jordan inverse."""
    return mo.inverse(T) @ y


def ridge_deblur(T, y, lam):
    """x = (T^T T + lam I)^-1 T^T y. Adding lam*I lifts the tiny eigenvalues
    that make plain inversion explode, so noise is no longer amplified."""
    n = T.shape[1]
    A = T.T @ T + lam * np.eye(n)
    return mo.inverse(A) @ (T.T @ y)


# ------------------------------------------------------- the singular case
def null_direction_example():
    """Two DIFFERENT colours that grayscale maps to the SAME output, proving the
    grayscale matrix cannot be inverted.

    The grayscale matrix G has every row = [0.299, 0.587, 0.114]. Any vector d
    with  0.299*d0 + 0.587*d1 + 0.114*d2 = 0  satisfies G @ d = 0 (null space).
    Take d = [0.587, -0.299, 0]:  0.299*0.587 - 0.587*0.299 = 0.
    """
    d = np.array([0.587, -0.299, 0.0])
    c1 = np.array([100.0, 100.0, 100.0])
    c2 = c1 + 100.0 * d
    return c1, c2