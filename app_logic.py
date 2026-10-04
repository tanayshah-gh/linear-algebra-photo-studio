"""
app_logic.py
All computation behind the Streamlit app, with no Streamlit code in it, so it
can be tested on its own (see tests/test_app_logic.py).
"""

import numpy as np
from PIL import Image, ImageOps

import color_filters as cf
import convolution_filters as cv
import geometric_transforms as gt
import inversion_recovery as ir
import matrix_ops as mo
from utils import apply_color_matrix, make_sample_image, to_uint8

MAX_SIDE = 384          # uploaded photos are shrunk so the app stays responsive
DEBLUR_N = 16           # deblurring lab works on an N x N image (T is N^2 x N^2)

COLOR_FILTER_NAMES = list(cf.FILTERS) + ["Custom matrix"]
CONV_NAMES = ["Identity", "Box blur", "Gaussian blur", "Sharpen", "Laplacian", "Emboss", "Sobel edges"]


# ------------------------------------------------------------------- image I/O
def load_image(source=None, max_side=MAX_SIDE):
    """Return an (H, W, 3) float RGB array. `source` is a path or file-like object;
    None gives the built-in sample image."""
    if source is None:
        return make_sample_image(256)
    pil = ImageOps.exif_transpose(Image.open(source)).convert("RGB")   # phone photos store rotation in EXIF
    pil.thumbnail((max_side, max_side))          # keeps aspect ratio
    return np.asarray(pil, dtype=float)


# ----------------------------------------------------------------- colour mode
def color_matrix(name, t=0.6, k=1.2, s=1.5, custom=None):
    """Build the 3x3 matrix for a named colour filter."""
    if name == "Custom matrix":
        M = np.array(custom, dtype=float)
        if M.shape != (3, 3) or not np.all(np.isfinite(M)):
            raise ValueError("Custom matrix must be 3x3 with every cell filled in")
        return M
    if name == "Soft sepia":
        return cf.soft_sepia_matrix(t)
    if name == "Brighten":
        return cf.brightness_matrix(k)
    if name == "Saturate":
        return cf.saturation_matrix(s)
    return cf.FILTERS[name]()


def matrix_facts(M):
    return {
        "det": mo.determinant(M),
        "cond": mo.condition_number(M),
        "invertible": mo.is_invertible(M),
    }


def color_experiment(img, M, noise_std=0.0, store_8bit=False, seed=0):
    """Filter with M, optionally store as 8-bit and/or add noise, then undo with M^-1.

    Returns a dict. `recovered` is None when M is singular.
    """
    filtered = apply_color_matrix(img, M)
    stored = to_uint8(filtered).astype(float) if store_8bit else filtered
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        observed = stored + rng.normal(0.0, noise_std, stored.shape)
    else:
        observed = stored

    if mo.is_invertible(M):
        recovered = apply_color_matrix(observed, mo.inverse(M))
        err = ir.rmse(recovered, img)
    else:
        recovered, err = None, None

    amp = err / noise_std if (err is not None and noise_std > 0) else None
    return {"filtered": filtered, "observed": observed, "recovered": recovered,
            "rmse": err, "amplification": amp}


# --------------------------------------------------------------- geometry mode
def geometry_matrix(w, h, rotate=0.0, scale=1.0, shear_x=0.0, flip_h=False,
                    flip_v=False, tx=0.0, ty=0.0, shear_first=False):
    """Compose one 3x3 homogeneous matrix. Order of application:
    scale -> (rotate, shear in the chosen order) -> flips -> translate."""
    C = lambda M: gt.about_center(M, w, h)
    R, Sh, Sc = C(gt.rotation(rotate)), C(gt.shear(shx=shear_x)), C(gt.scaling(scale))
    steps = [Sc] + ([Sh, R] if shear_first else [R, Sh])
    if flip_h:
        steps.append(gt.flip_horizontal(w))
    if flip_v:
        steps.append(gt.flip_vertical(h))
    steps.append(gt.translation(tx, ty))
    return gt.compose(*steps)


def geometry_experiment(img, T):
    """Warp with T, then warp back with T^-1; report error on surviving pixels."""
    h, w = img.shape[:2]
    warped = gt.warp(img, T)
    back = gt.warp(warped, mo.inverse(T))
    keep = gt.warp(gt.warp(np.ones((h, w, 1)), T), mo.inverse(T))[..., 0] > 0.99
    mae = float(np.abs(back[keep] - img[keep]).mean()) if keep.any() else None
    return {"warped": warped, "recovered": back, "det": mo.determinant(T),
            "kept_pct": float(keep.mean() * 100), "mae": mae}


# ------------------------------------------------------------ convolution mode
def conv_experiment(img, name, size=3, sigma=1.0):
    """Apply a named convolution filter. Returns output image, kernels to display,
    and the shape of the im2col matrix that was multiplied."""
    h, w = img.shape[:2]
    if name == "Sobel edges":
        mag, _, _ = cv.sobel_edges(img)
        return {"output": mag, "bias": 0.0, "im2col_shape": (h * w, 9),
                "kernels": [("Gx", cv.sobel_x_kernel()), ("Gy", cv.sobel_y_kernel())]}

    bias = 0.0
    if name == "Identity":
        kernel = cv.identity_kernel()
    elif name == "Box blur":
        kernel = cv.box_blur_kernel(size)
    elif name == "Gaussian blur":
        kernel = cv.gaussian_kernel(size, sigma)
    elif name in cv.FILTERS:                       # Sharpen, Laplacian, Emboss
        kernel, bias = cv.FILTERS[name]
    else:
        raise ValueError(f"Unknown filter {name}")

    k = kernel.shape[0]
    return {"output": cv.convolve(img, kernel, bias), "bias": bias,
            "im2col_shape": (h * w, k * k), "kernels": [("Kernel", kernel)]}


# ------------------------------------------------------------------ deblur lab
def deblur_experiment(img, sigma=1.0, noise_std=0.0, lam=1e-3, seed=0, n=DEBLUR_N):
    """Blur an n x n grayscale version with matrix T, add noise, then try to undo it
    (a) with the plain inverse T^-1 and (b) with ridge regularisation."""
    if min(img.shape[:2]) < n:
        raise ValueError(f"Image must be at least {n}x{n} pixels")
    x_img = ir.downsample_gray(img, n)
    x = x_img.ravel()
    T = cv.convolution_matrix(cv.gaussian_kernel(5, sigma), n, n)
    y = T @ x
    rng = np.random.default_rng(seed)
    y_obs = y + rng.normal(0.0, noise_std, y.shape) if noise_std > 0 else y.copy()

    try:
        Tinv = mo.inverse(T)
    except ValueError:
        return {"singular": True}
    naive = Tinv @ y_obs
    ridge = ir.ridge_deblur(T, y_obs, lam)
    sh = (n, n)
    return {
        "singular": False,
        "cond": mo.norm_1(T) * mo.norm_1(Tinv),
        "original": x_img, "blurred": y.reshape(sh), "observed": y_obs.reshape(sh),
        "naive": naive.reshape(sh), "ridge": ridge.reshape(sh),
        "rmse_blurred": ir.rmse(y_obs, x),
        "rmse_naive": ir.rmse(naive, x),
        "rmse_ridge": ir.rmse(ridge, x),
    }