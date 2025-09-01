from dyno.modfile import Modfile

model = Modfile("examples/modfiles/model_KR2000_STAT.mod")

r, A, B, C, D = model.compute(diff=True)

A
B
C

import dyno.solver

dyno.solver.solve_ti(A, B, C, verbose=True)


import numpy as np

n = A.shape[0]
I = np.eye(n)
Z = np.zeros((n, n))


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

from scipy.linalg import ordqz

tol = 1e-10


def decompose_blocks(Z):
    n = Z.shape[0] // 2
    Z11 = Z[:n, :n]
    Z12 = Z[:n, n:]
    Z21 = Z[n:, :n]
    Z22 = Z[n:, n:]
    return Z11, Z12, Z21, Z22


# Generalised eigenvalue problem
F = np.block([[Z, I], [-C, -B]])
G = np.block([[I, Z], [Z, A]])

T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol)) <= 1)
λ_all = vgenev(α, β, tol=tol)
λ = λ_all[np.abs(λ_all) <= 1]


Λ = np.diag(λ)
Z11, Z12, Z21, Z22 = decompose_blocks(Z)
X = Z21 @ np.linalg.inv(Z11)
