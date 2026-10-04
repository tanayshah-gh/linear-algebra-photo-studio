import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matrix_ops as mo


def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    A, B = rng.random((4, 5)), rng.random((5, 3))
    assert np.allclose(mo.matmul(A, B), A @ B)


def test_matmul_shape_mismatch():
    try:
        mo.matmul(np.ones((2, 3)), np.ones((2, 3)))
        assert False, "should have raised"
    except ValueError:
        pass


def test_determinant_matches_numpy():
    rng = np.random.default_rng(1)
    for n in (2, 3, 5):
        A = rng.random((n, n))
        assert np.isclose(mo.determinant(A), np.linalg.det(A))


def test_inverse_matches_numpy():
    rng = np.random.default_rng(2)
    A = rng.random((4, 4)) + 4 * np.eye(4)
    assert np.allclose(mo.inverse(A), np.linalg.inv(A))
    assert np.allclose(mo.matmul(A, mo.inverse(A)), np.eye(4))


def test_singular_matrix():
    S = [[1, 2], [2, 4]]  # row 2 = 2 * row 1
    assert mo.determinant(S) == 0.0
    assert not mo.is_invertible(S)
    try:
        mo.inverse(S)
        assert False, "should have raised"
    except ValueError:
        pass
    assert mo.condition_number(S) == float("inf")


def test_condition_number_identity():
    assert np.isclose(mo.condition_number(np.eye(3)), 1.0)


def test_scale_independence():
    # well-conditioned matrices must stay invertible whatever the size of the entries
    for scale in (1e-13, 1e-6, 1.0, 1e8):
        A = scale * np.eye(3)
        assert mo.is_invertible(A)
        assert np.allclose(mo.inverse(A) * scale, np.eye(3))
        assert np.isclose(mo.determinant(A), scale ** 3)
        assert np.isclose(mo.condition_number(A), 1.0)


def test_zero_matrix_is_singular():
    assert not mo.is_invertible(np.zeros((3, 3)))
    assert mo.determinant(np.zeros((3, 3))) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)