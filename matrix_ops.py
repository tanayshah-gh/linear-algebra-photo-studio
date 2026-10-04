"""
matrix_ops.py
Hand-written linear algebra core for the Photo Filter project.

Everything here is implemented from first principles (no np.linalg.* inside the
algorithms) so each step can be explained in the viva. NumPy is used only as a
container for the numbers and for verification in tests.
"""

import numpy as np

MACHINE_EPS = np.finfo(float).eps   # ~2.2e-16, smallest relative step of a float


def _pivot_tolerance(M):
    """Pivots smaller than this count as zero.

    The threshold is RELATIVE to the size of the matrix entries (n * eps * max|a_ij|),
    not a fixed number. A fixed threshold wrongly calls 1e-13 * I singular and
    wrongly accepts a huge matrix whose pivots are tiny relative to its entries."""
    n = M.shape[0]
    return n * MACHINE_EPS * float(np.max(np.abs(M)))


# ---------------------------------------------------------------- multiplication
def matmul(A, B):
    """Matrix product C = A @ B using the definition C[i][j] = sum_k A[i][k] * B[k][j].

    Shapes: A is (m x n), B is (n x p)  ->  C is (m x p).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul expects two 2-D matrices")
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError(f"Shape mismatch: {A.shape} cannot multiply {B.shape}")

    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


# ------------------------------------------------------------------ determinant
def determinant(A):
    """Determinant via Gaussian elimination with partial pivoting.

    Reduce A to upper-triangular form U. det(A) = (+/-1) * product of diagonal
    of U, where the sign flips once for every row swap.
    """
    M = np.array(A, dtype=float)
    n = _check_square(M)
    sign = 1.0
    tol = _pivot_tolerance(M)

    for col in range(n):
        pivot = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[pivot, col]) <= tol:
            return 0.0
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
            sign = -sign
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]

    return sign * float(np.prod(np.diag(M)))


# ---------------------------------------------------------------------- inverse
def inverse(A):
    """Inverse via Gauss-Jordan elimination on the augmented matrix [A | I].

    Row operations turn [A | I] into [I | A^-1]. Raises ValueError if A is
    singular (no pivot available), because then A^-1 does not exist.
    """
    M = np.array(A, dtype=float)
    n = _check_square(M)
    aug = np.hstack([M, np.eye(n)])
    tol = _pivot_tolerance(M)

    for col in range(n):
        # 1. partial pivoting: pick the largest entry in this column
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) <= tol:
            raise ValueError("Matrix is singular: inverse does not exist")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]

        # 2. scale pivot row so the pivot becomes 1
        aug[col] /= aug[col, col]

        # 3. eliminate this column from every other row
        for row in range(n):
            if row != col:
                aug[row] -= aug[row, col] * aug[col]

    return aug[:, n:]


# ------------------------------------------------------------ conditioning / norm
def norm_1(A):
    """Matrix 1-norm: maximum absolute column sum."""
    A = np.asarray(A, dtype=float)
    return float(np.max(np.sum(np.abs(A), axis=0)))


def condition_number(A):
    """cond(A) = ||A||_1 * ||A^-1||_1. Large value => small errors get amplified
    when inverting, so recovered images will be noisier. Returns inf if singular."""
    try:
        return norm_1(A) * norm_1(inverse(A))
    except ValueError:
        return float("inf")


def is_invertible(A):
    """A square matrix is invertible exactly when det(A) != 0, i.e. when
    Gauss-Jordan finds a pivot in every column.

    We test it by attempting the inversion rather than checking |det| against a threshold:
    the size of a determinant depends on the matrix scale (a 49x49 blur matrix
    has det ~ 1e-47 yet is perfectly invertible), so a fixed threshold on det
    gives wrong answers. cond(A) tells you HOW invertible it is."""
    try:
        inverse(A)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------- helpers
def _check_square(M):
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")
    return M.shape[0]


if __name__ == "__main__":
    A = [[2, 1, 1], [1, 3, 2], [1, 0, 0]]
    print("det(A) =", determinant(A))
    Ainv = inverse(A)
    print("A^-1 =\n", Ainv)
    print("A @ A^-1 =\n", np.round(matmul(A, Ainv), 10))
    print("cond(A) =", condition_number(A))