import numpy as np

from numpy.linalg import solve as linsolve
from scipy.linalg import ordqz


def solve(A, B, C, method="qz", options={}):

    if method == "ti":

        sol, evs = solve_ti(A, B, C, **options)

    else:

        sol, evs = solve_qz(A, B, C, **options)

    return sol, evs


def solve_ti(A, B, C, T=10000, tol=1e-10, verbose=False):

    n = A.shape[0]

    X0 = np.random.randn(n, n)

    for t in range(T):

        X1 = linsolve(A @ X0 + B, -C)
        e = abs(X0 - X1).max()

        if np.isnan(e):
            raise Exception("Invalid value")

        X0 = X1
        if e < tol:
            return X0, None

    raise Exception("No convergence")


def solve_qz(A, B, C, tol=1e-15):
    """Solves AX² + BX + C = 0 for X using a QZ decomposition."""
    n = A.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Generalised eigenvalue problem
    F = np.block([[Z, I], [-C, -B]])
    G = np.block([[I, Z], [Z, A]])
    T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol)) <= 1)
    λ_all = vgenev(α, β, tol=tol)
    λ = λ_all[np.abs(λ_all) <= 1]

    Λ = np.diag(λ)
    Z11, Z12, Z21, Z22 = decompose_blocks(Z)
    X = Z21 @ np.linalg.inv(Z11)

    return X, sorted(λ_all)


def decompose_blocks(Z):
    n = Z.shape[0] // 2
    Z11 = Z[:n, :n]
    Z12 = Z[:n, n:]
    Z21 = Z[n:, :n]
    Z22 = Z[n:, n:]
    return Z11, Z12, Z21, Z22


def genev(α, β, tol=1e-9):
    """Computes the eigenvalues λ = α/β."""
    if not np.isclose(β, 0, atol=tol):
        return α / β
    else:
        if np.isclose(α, 0, atol=tol):
            return np.nan
        else:
            return np.inf


vgenev = np.vectorize(genev, excluded=["tol"])


def moments(X, Y, Σ):
    """
    Compute conditional and unconditional moments of process $y_t = X y_{t-1} + Y e_t$
    """

    Σ0 = Y @ Σ @ Y.T
    n = X.shape[0]

    # Compute the unconditional variance
    Σ = (np.linalg.inv(np.eye(n**2) - np.kron(X, X)) @ Σ0.flatten()).reshape(n, n)

    return Σ0, Σ
