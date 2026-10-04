import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_logic as al
import matrix_ops as mo

IMG = al.load_image(None)          # built-in 256x256 sample


def test_load_sample_shape():
    assert IMG.shape == (256, 256, 3)


def test_color_recovery_exact_without_noise():
    for name in ("Swap R<->B", "Soft sepia", "Saturate", "Brighten"):
        M = al.color_matrix(name)
        res = al.color_experiment(IMG, M)
        assert res["recovered"] is not None
        assert res["rmse"] < 1e-8, name


def test_singular_filter_has_no_recovery():
    res = al.color_experiment(IMG, al.color_matrix("Grayscale"))
    assert res["recovered"] is None and res["rmse"] is None


def test_noise_amplification_follows_conditioning():
    good = al.color_experiment(IMG, al.color_matrix("Soft sepia"), noise_std=0.5)
    bad = al.color_experiment(IMG, al.color_matrix("Sepia (full)"), noise_std=0.5)
    assert good["amplification"] < 10
    assert bad["amplification"] > 1000


def test_custom_matrix_validation():
    for bad in (np.ones((2, 2)), np.full((3, 3), np.nan)):
        try:
            al.color_matrix("Custom matrix", custom=bad)
            assert False, "should have raised"
        except ValueError:
            pass
    assert np.allclose(al.color_matrix("Custom matrix", custom=np.eye(3)), np.eye(3))


def test_geometry_identity_and_roundtrip():
    h, w = IMG.shape[:2]
    T = al.geometry_matrix(w, h)                       # all defaults = identity
    assert np.allclose(T, np.eye(3))
    assert np.allclose(al.geometry_experiment(IMG, T)["warped"], IMG)

    T = al.geometry_matrix(w, h, rotate=25, scale=0.8, shear_x=0.2, tx=10, ty=-5)
    res = al.geometry_experiment(IMG, T)
    assert res["mae"] is not None and res["mae"] < 3
    assert 0 < res["kept_pct"] <= 100


def test_geometry_order_matters():
    h, w = IMG.shape[:2]
    a = al.geometry_matrix(w, h, rotate=40, shear_x=0.5, shear_first=False)
    b = al.geometry_matrix(w, h, rotate=40, shear_x=0.5, shear_first=True)
    assert not np.allclose(a, b)


def test_conv_identity_and_shapes():
    res = al.conv_experiment(IMG, "Identity")
    assert np.allclose(res["output"], IMG)
    assert res["im2col_shape"] == (256 * 256, 9)
    assert al.conv_experiment(IMG, "Box blur", size=5)["im2col_shape"] == (256 * 256, 25)


def test_every_conv_filter_runs():
    for name in al.CONV_NAMES:
        out = al.conv_experiment(IMG, name)["output"]
        assert out.shape[:2] == IMG.shape[:2] and np.all(np.isfinite(out))


def test_deblur_exact_without_noise_and_unstable_with_noise():
    clean = al.deblur_experiment(IMG, sigma=1.0, noise_std=0.0)
    assert not clean["singular"] and clean["rmse_naive"] < 1e-6
    noisy = al.deblur_experiment(IMG, sigma=1.0, noise_std=0.5)
    # with noise the plain inverse is WORSE than the blurred image it started from...
    assert noisy["rmse_naive"] > 2 * noisy["rmse_blurred"]
    # ...while ridge regularisation is better than both
    assert noisy["rmse_ridge"] < noisy["rmse_blurred"] < noisy["rmse_naive"]


def test_deblur_rejects_tiny_image():
    try:
        al.deblur_experiment(np.zeros((8, 8, 3)))
        assert False, "should have raised"
    except ValueError:
        pass


def _jpeg_bytes(size, orientation=None, mode="RGB", fmt="JPEG"):
    import io
    from PIL import Image
    im = Image.new(mode, size, (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128))
    kwargs = {}
    if orientation:
        exif = im.getexif(); exif[0x0112] = orientation
        kwargs["exif"] = exif
    buf = io.BytesIO(); im.save(buf, format=fmt, **kwargs); buf.seek(0)
    return buf


def test_upload_respects_exif_orientation():
    # 40 wide x 20 tall, tagged "rotate 90": a phone photo held upright
    assert al.load_image(_jpeg_bytes((40, 20), orientation=6)).shape == (40, 20, 3)
    assert al.load_image(_jpeg_bytes((40, 20))).shape == (20, 40, 3)      # no tag: unchanged


def test_upload_is_shrunk_and_flattened_to_rgb():
    big = al.load_image(_jpeg_bytes((1600, 1200)))
    assert max(big.shape[:2]) == al.MAX_SIDE and big.shape[2] == 3
    assert abs(big.shape[1] / big.shape[0] - 1600 / 1200) < 0.02          # aspect ratio kept
    rgba = al.load_image(_jpeg_bytes((50, 50), mode="RGBA", fmt="PNG"))   # transparency -> plain RGB
    assert rgba.shape == (50, 50, 3)


def test_unreadable_upload_raises():
    import io
    try:
        al.load_image(io.BytesIO(b"not an image"))
        assert False, "should have raised"
    except Exception:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)