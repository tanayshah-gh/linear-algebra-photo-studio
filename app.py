"""
app.py  -  Linear Algebra Photo Studio (Streamlit UI)

Run with:   streamlit run app.py

All maths lives in the other modules; this file only draws the interface.
"""

import numpy as np
import pandas as pd
import streamlit as st

import app_logic as al
from utils import to_uint8

st.set_page_config(page_title="Linear Algebra Photo Studio", layout="wide")


# ----------------------------------------------------------------- helpers
def show(arr, caption):
    """Display a float image/array (clipped to 0-255)."""
    st.image(to_uint8(arr), caption=caption)


def show_matrix(M, decimals=4):
    st.dataframe(pd.DataFrame(np.round(np.asarray(M, dtype=float), decimals)))


def show_facts(facts):
    c1, c2, c3 = st.columns(3)
    c1.metric("determinant", f"{facts['det']:.4g}")
    c2.metric("invertible?", "yes" if facts["invertible"] else "NO (singular)")
    c3.metric("condition number", "infinite" if not np.isfinite(facts["cond"]) else f"{facts['cond']:.4g}")


def pixelated(arr, scale=12):
    """Enlarge a tiny array without smoothing so individual pixels stay visible."""
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


# Heavy computations are cached, so moving one slider does not redo everything.
@st.cache_data(show_spinner=False)
def run_color(img, M, noise, store8, seed):
    return al.color_experiment(img, M, noise_std=noise, store_8bit=store8, seed=seed)


@st.cache_data(show_spinner=False)
def run_geometry(img, T):
    return al.geometry_experiment(img, T)


@st.cache_data(show_spinner=False)
def run_conv(img, name, size, sigma):
    return al.conv_experiment(img, name, size=size, sigma=sigma)


@st.cache_data(show_spinner=False)
def run_deblur(img, sigma, noise, lam, seed):
    return al.deblur_experiment(img, sigma=sigma, noise_std=noise, lam=lam, seed=seed)


# ----------------------------------------------------------------- sidebar
st.title("Linear Algebra Photo Studio")
st.caption("An image is a matrix. Photo filters are matrix operations. "
           "Inverting the matrix undoes the filter, if the matrix allows it.")

with st.sidebar:
    st.header("1. Choose an image")
    upload = st.file_uploader("Upload a photo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    try:
        img = al.load_image(upload)
    except Exception as exc:                      # unreadable file
        st.error(f"Could not read that image: {exc}")
        img = al.load_image(None)
    h, w = img.shape[:2]
    st.image(to_uint8(img), caption=f"{h} x {w} pixels, 3 channels (R, G, B)")
    st.caption(f"As data: an array of shape ({h}, {w}, 3), one {h}x{w} matrix per colour channel.")

    st.header("2. Choose a mode")
    mode = st.radio("Mode", ["Colour filters", "Geometric transforms", "Convolution filters",
                             "Deblurring lab", "Key ideas"])


# ----------------------------------------------------------- colour filters
if mode == "Colour filters":
    st.header("Colour filters: a 3x3 matrix applied to every pixel")
    st.write("Each pixel is a vector [R, G, B]. The filter computes **new pixel = M x pixel**. "
             "To undo it, multiply by **M⁻¹**.")

    name = st.selectbox("Filter", al.COLOR_FILTER_NAMES, index=al.COLOR_FILTER_NAMES.index("Soft sepia"))
    t, k, s, custom = 0.6, 1.2, 1.5, None
    if name == "Soft sepia":
        t = st.slider("Sepia strength t  (0 = original, 1 = full sepia)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    elif name == "Brighten":
        k = st.slider("Brightness factor k", min_value=0.5, max_value=2.0, value=1.2, step=0.05)
    elif name == "Saturate":
        s = st.slider("Saturation s  (0 = grey, 1 = original)", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    elif name == "Custom matrix":
        st.write("Edit the nine numbers. Rows are the new R, G, B; columns are the old R, G, B.")
        custom_df = st.data_editor(pd.DataFrame(np.eye(3), columns=["old R", "old G", "old B"],
                                                index=["new R", "new G", "new B"]))
        custom = custom_df.to_numpy(dtype=float)

    try:
        M = al.color_matrix(name, t=t, k=k, s=s, custom=custom)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    facts = al.matrix_facts(M)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("The matrix M")
        show_matrix(M)
    with right:
        st.subheader("What the matrix tells us")
        show_facts(facts)

    st.subheader("Apply the filter, then undo it")
    c1, c2 = st.columns(2)
    noise = c1.slider("Noise added to the filtered image (grey levels)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    store8 = c2.checkbox("Save filtered result as 8-bit first (clips values above 255)", value=False)
    st.caption("Noise stands in for rounding and sensor error. Try full sepia with a little noise.")

    res = run_color(img, M, noise, store8, 0)
    a, b, c, d = st.columns(4)
    with a:
        show(img, "Original")
    with b:
        show(res["filtered"], "Filtered  (M x pixel)")
    with c:
        show(res["observed"], "What we have to invert")
    with d:
        if res["recovered"] is None:
            st.error("Singular matrix: M⁻¹ does not exist, so the original cannot be recovered.")
        else:
            show(res["recovered"], f"Recovered  (M⁻¹ x pixel)\nRMSE = {res['rmse']:.3g}")

    if res["recovered"] is not None and res["amplification"] is not None:
        st.write(f"The noise was amplified by about **{res['amplification']:.1f}x** "
                 f"(the condition number, {facts['cond']:.4g}, is an upper-bound guide).")
    if not facts["invertible"]:
        st.info("Different colours can map to the same output here (for grayscale, every row is identical), "
                "so no inverse can tell them apart.")


# ------------------------------------------------------ geometric transforms
elif mode == "Geometric transforms":
    st.header("Geometric transforms: 3x3 homogeneous matrices")
    st.write("A point (x, y) becomes [x, y, 1], so rotation, scaling, shear, flips and translation are all "
             "3x3 matrices that combine by **matrix multiplication**. Warping uses **inverse mapping**: "
             "for each output pixel, M⁻¹ finds where it came from.")

    c1, c2, c3 = st.columns(3)
    rotate = c1.slider("Rotate (degrees)", min_value=-180, max_value=180, value=30, step=5)
    scale = c1.slider("Scale", min_value=0.2, max_value=2.0, value=1.0, step=0.05)
    shear_x = c2.slider("Shear (x direction)", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    tx = c2.slider("Translate x (pixels)", min_value=-100, max_value=100, value=0, step=5)
    ty = c3.slider("Translate y (pixels)", min_value=-100, max_value=100, value=0, step=5)
    flip_h = c3.checkbox("Flip horizontally", value=False)
    flip_v = c3.checkbox("Flip vertically", value=False)
    shear_first = st.checkbox("Apply shear BEFORE rotation (the order of multiplication matters)", value=False)

    T = al.geometry_matrix(w, h, rotate=rotate, scale=scale, shear_x=shear_x, flip_h=flip_h,
                           flip_v=flip_v, tx=tx, ty=ty, shear_first=shear_first)
    res = run_geometry(img, T)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Combined matrix T")
        show_matrix(T)
    with right:
        st.metric("determinant", f"{res['det']:.4g}")
        st.caption("det = 1: area preserved.  |det| = scale²: area scaled.  det < 0: mirror image.")
        st.write(f"Warping back with T⁻¹ keeps **{res['kept_pct']:.1f}%** of pixels inside the frame"
                 + (f"; on those pixels the mean absolute error is **{res['mae']:.3f}** (out of 255)."
                    if res["mae"] is not None else "."))

    a, b, c = st.columns(3)
    with a:
        show(img, "Original")
    with b:
        show(res["warped"], "After T")
    with c:
        show(res["recovered"], "After T⁻¹ (recovered)")
    st.caption("Corners that left the frame are gone for good, so recovery is exact only for pixels that stayed inside.")


# ------------------------------------------------------ convolution filters
elif mode == "Convolution filters":
    st.header("Convolution filters: a kernel slid across the image")
    st.write("Each output pixel is a weighted sum of its neighbourhood. In matrix form: every pixel's neighbourhood "
             "becomes one row of a big matrix **P**, and **output = P x kernel**.")

    name = st.selectbox("Filter", al.CONV_NAMES, index=al.CONV_NAMES.index("Gaussian blur"))
    size, sigma = 3, 1.0
    if name in ("Box blur", "Gaussian blur"):
        size = st.select_slider("Kernel size", options=[3, 5, 7], value=5)
    if name == "Gaussian blur":
        sigma = st.slider("Sigma (spread of the blur)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

    res = run_conv(img, name, size, sigma)
    a, b = st.columns(2)
    with a:
        show(img, "Original")
    with b:
        show(res["output"], f"{name}")

    st.subheader("The kernel(s)")
    kcols = st.columns(len(res["kernels"]))
    for col, (label, kern) in zip(kcols, res["kernels"]):
        with col:
            st.write(f"**{label}**  (sum = {kern.sum():.3g})")
            show_matrix(kern, decimals=3)
    st.write(f"The matrix **P** that was multiplied has shape **{res['im2col_shape'][0]:,} x {res['im2col_shape'][1]}** "
             f"(one row per pixel, one column per kernel cell).")
    st.caption("Kernel sum 1 keeps overall brightness. Sum 0 (Laplacian, Sobel) shows only changes, i.e. edges.")


# ------------------------------------------------------------ deblurring lab
elif mode == "Deblurring lab":
    st.header("Deblurring lab: why inverting a blur is dangerous")
    st.write(f"To keep the matrix small, the image is shrunk to {al.DEBLUR_N} x {al.DEBLUR_N} grayscale, so the blur "
             f"matrix **T** is {al.DEBLUR_N**2} x {al.DEBLUR_N**2}. We blur with T, add noise, then try to undo it.")

    c1, c2, c3 = st.columns(3)
    sigma = c1.slider("Blur sigma", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
    noise = c2.slider("Noise std (grey levels)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
    lam = c3.select_slider("Ridge strength λ", options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], value=1e-3)

    try:
        res = run_deblur(img, sigma, noise, lam, 0)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if res["singular"]:
        st.error("This blur matrix is singular at the chosen settings, so T⁻¹ does not exist.")
    else:
        st.metric("condition number of T", f"{res['cond']:.4g}")
        cols = st.columns(4)
        panels = [("Original", res["original"], None),
                  ("Blurred + noise", res["observed"], res["rmse_blurred"]),
                  ("Plain inverse T⁻¹ y", res["naive"], res["rmse_naive"]),
                  (f"Ridge (λ = {lam:g})", res["ridge"], res["rmse_ridge"])]
        for col, (title, arr, err) in zip(cols, panels):
            with col:
                st.image(to_uint8(pixelated(arr)), caption=title + ("" if err is None else f"\nRMSE = {err:.3g}"))
        st.write("RMSE is the error against the original (0 to 255 scale). With **no noise** the plain inverse is "
                 "almost exact. Add even a little noise and it gets worse than the blurred image itself, because "
                 "small singular values of T amplify the noise. Ridge regularisation, "
                 "x = (TᵀT + λI)⁻¹ Tᵀ y, keeps the noise under control.")
        st.caption("Choosing λ matters: too small stays noisy, too large over-smooths.")


# ----------------------------------------------------------------- key ideas
else:
    st.header("Key ideas")
    st.markdown("""
**Image = matrix.** A grayscale image is an H x W matrix; a colour image is three of them (R, G, B).

**Matrix multiplication = applying a filter.** A 3x3 colour matrix times each pixel's [R, G, B] vector gives the new
colour. Geometric transforms use 3x3 homogeneous matrices, and combining transforms means multiplying matrices.
The order matters because matrix multiplication is **not commutative**.

**Inverse = undoing a filter.** If original → M x original, then original = M⁻¹ x filtered. We compute M⁻¹ with
**Gauss-Jordan elimination** on [M | I].

**Determinant** tells us *whether* an inverse exists (det = 0 means singular). |det| also gives the area scaling of a
geometric transform; a negative sign means a flip.

**Condition number** tells us *how safely*. A large value means tiny errors (rounding, noise) are amplified in the
recovered image, up to about that factor. Full sepia is invertible but extremely ill-conditioned.

**Singular matrices lose information.** Grayscale maps different colours to the same grey, so it cannot be undone.
Clipping at 255 and cropping at the frame edge lose information in the same way.

**Convolution is matrix multiplication too.** Via im2col (**output = P x kernel**) or one big matrix T acting on the
image vector. Blur matriaces are ill-conditioned, so deblurring by plain inversion fails with noise; **ridge
regularisation** fixes most of it.
""")