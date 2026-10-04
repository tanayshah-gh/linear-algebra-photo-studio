"""
convolution_filters.py
Blur, sharpen, edge detection and emboss written as MATRIX operations.

Two equivalent matrix views of convolution (both implemented here):

1) im2col  - every pixel's k x k neighbourhood is flattened into one row of a
   matrix P with shape (H*W, k*k). Then  output = P @ kernel_vector.

2) convolution matrix T - for an image flattened into a vector x (length H*W),
   filtering is one big matrix-vector product  y = T @ x.
   T has shape (H*W, H*W) and every row holds the kernel weights for one pixel.
   Because it is a matrix, we can ask whether it is invertible (deblurring!).

Borders use zero padding so both views match exactly.
Convolution flips the kernel (rotates it 180 deg) before sliding it. For
symmetric kernels (blur, sharpen) this changes nothing; for Sobel it only
flips the sign, which disappears in the edge magnitude.
"""

import numpy as np


# ----------------------------------------------------------------- the kernels
def identity_kernel():
    k = np.zeros((3, 3)); k[1, 1] = 1
    return k


def box_blur_kernel(size=3):
    """Every neighbour weighted equally; weights sum to 1 (keeps brightness)."""
    return np.ones((size, size)) / (size * size)


def gaussian_kernel(size=5, sigma=1.0):
    """Weights fall off with distance from the centre: exp(-(x^2+y^2)/(2 sigma^2))."""
    r = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(r, r)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def sharpen_kernel():
    """Identity + (identity - blur)-like boost of the centre. Sums to 1."""
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]], dtype=float)


def sobel_x_kernel():
    """Responds to horizontal intensity change -> vertical edges."""
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=float)


def sobel_y_kernel():
    return sobel_x_kernel().T


def laplacian_kernel():
    """Second-derivative edge detector; weights sum to 0 (flat areas -> 0)."""
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]], dtype=float)


def emboss_kernel():
    return np.array([[-2, -1, 0],
                     [-1, 1, 1],
                     [0, 1, 2]], dtype=float)


# --------------------------------------------------- view 1: im2col + matmul
def im2col(channel, k):
    """Turn an (H, W) array into a matrix of shape (H*W, k*k).

    Row r holds the k x k neighbourhood around pixel r (zero padded),
    flattened left-to-right, top-to-bottom.
    """
    h, w = channel.shape
    pad = k // 2
    padded = np.pad(channel, pad, mode="constant")
    cols = np.empty((h * w, k * k))
    idx = 0
    for dy in range(k):
        for dx in range(k):
            cols[:, idx] = padded[dy:dy + h, dx:dx + w].ravel()
            idx += 1
    return cols


def convolve_channel(channel, kernel):
    """Convolve one (H, W) channel: im2col matrix times the flipped kernel vector."""
    k = kernel.shape[0]
    P = im2col(channel, k)
    kvec = np.flip(kernel).ravel()          # flip = true convolution
    return (P @ kvec).reshape(channel.shape)


def convolve(img, kernel, bias=0.0):
    """Apply a kernel to a grayscale (H, W) or colour (H, W, 3) image."""
    if img.ndim == 2:
        return convolve_channel(img, kernel) + bias
    out = np.stack([convolve_channel(img[..., c], kernel) for c in range(img.shape[2])], axis=-1)
    return out + bias


# ------------------------------------------- view 2: one big convolution matrix
def convolution_matrix(kernel, h, w):
    """Build T of shape (h*w, h*w) so that  T @ image.ravel() == convolve(image).ravel().

    Row for output pixel (i, j) holds, in the columns of the input pixels it
    reads, the matching (flipped) kernel weights. Intended for SMALL images
    (T has (h*w)^2 entries).
    """
    k = kernel.shape[0]
    pad = k // 2
    kf = np.flip(kernel)
    T = np.zeros((h * w, h * w))
    for i in range(h):
        for j in range(w):
            row = i * w + j
            for dy in range(k):
                for dx in range(k):
                    ii, jj = i + dy - pad, j + dx - pad
                    if 0 <= ii < h and 0 <= jj < w:
                        T[row, ii * w + jj] = kf[dy, dx]
    return T


# ------------------------------------------------------------ edge detection
def to_gray(img):
    return img @ np.array([0.299, 0.587, 0.114])


def sobel_edges(img):
    """Gradient magnitude sqrt(Gx^2 + Gy^2) scaled to 0-255. Returns (mag, gx, gy)."""
    gray = to_gray(img) if img.ndim == 3 else img
    gx = convolve(gray, sobel_x_kernel())
    gy = convolve(gray, sobel_y_kernel())
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag = mag / mag.max() * 255
    return mag, gx, gy


# ------------------------------------------------------------------- registry
# name -> (kernel, bias added after convolution)
FILTERS = {
    "Box blur 5x5": (box_blur_kernel(5), 0.0),
    "Gaussian blur": (gaussian_kernel(5, 1.2), 0.0),
    "Sharpen": (sharpen_kernel(), 0.0),
    "Laplacian": (laplacian_kernel(), 128.0),
    "Emboss": (emboss_kernel(), 128.0),
}


def apply_filter(img, name):
    kernel, bias = FILTERS[name]
    return convolve(img, kernel, bias), kernel