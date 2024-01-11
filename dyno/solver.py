import numpy as np
import pandas as pd
from numpy.linalg import solve as linsolve
from scipy.linalg import ordqz
import xarray
from colorama import Fore, Style

def simulate(dr,T=40):

    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ

    n = X.shape[1]
    v0 = np.zeros(n)
    m0 = np.zeros(Y.shape[1])
    ss = [v0]

    for t in range(T):
        e = np.random.multivariate_normal(m0, Σ)
        ss.append(X@ss[-1] + Y@e)

    res = np.concatenate([e[None,:] for e in ss], axis=0)
    # # rr = pd.DataFrame(res, columns=X.coords['y_t'])
    # return res
    dim_1 = [*range(T+1)]
    dim_2 = [*dr.X.coords['y_t'].data]

    return xarray.DataArray(res, coords=(('T', dim_1 ), ('V', dim_2)))


def solve(A,B,C, T=10000, tol=1e-10):
            
    n = A.shape[0]

    X0 = np.random.randn(n,n)

    for t in range(T):

        X1 = - linsolve(A@X0 + B, -C)
        e = abs(X0-X1).max()

        if np.isnan(e):
            raise Exception("Invalid value")
        
        X0 = X1
        if e<tol:
            return X0


    #     X1 = - linsolve(A@X0 + B, C)
    #     e = abs(X0-X1).max()

    #     X0 = X1
    #     if e<tol:
    #         return X0
        
    # raise Exception("No convergence")

def solve_qz(A, B, C, tol=1e-15, verbose=False):
    """Solves AX² + BX + C = 0 for X using a QZ decomposition."""
    n  = A.shape[0]
    I  = np.eye(n)
    Z  = np.zeros((n, n))
    
    # Generalised eigenvalue problem
    F  = block_matrix(Z, I, -C, -B)
    G  = block_matrix(I, Z, Z, A)
    T, S, α, β, Q, Z = ordqz(F, G, sort=select_eigenvalues_bool)
    λ, ind, λ_all = select_eigenvalues(α, β, tol=tol)
    
    if verbose:
        print_colored_tab(λ_all, select_eigenvalues_bool(α, β, tol=tol))
        
    Λ  = np.diag(λ)
    Z11, Z12, Z21, Z22 = decompose_blocks(Z)
    X  = Z21 @ np.linalg.inv(Z11)
    
    return X


def block_matrix(A, B, C, D):
    """Builds the block matrix [[A, B], [C, D]]"""
    tl  = np.block([[A]])
    tr  = np.block([[B]]) 
    bl  = np.block([[C]])
    br  = np.block([[D]])
    return np.block([[tl, tr], [bl, br]])


def decompose_blocks(Z):
    n = Z.shape[0] // 2
    Z11 = Z[:n, :n]
    Z12 = Z[:n, n:]
    Z21 = Z[n:, :n]
    Z22 = Z[n:, n:]
    return Z11, Z12, Z21, Z22


def select_eigenvalues(α, β, tol=1e-9):
    """Computes the eigenvalues λ = α/β and select those with modulus less or equal to 1."""
    λ_all = []  # For all the eigenvalues
    λ     = []  # Selected ones
    ind   = []  # Indexes of the the selected ones
    
    for i in range(len(α)):
        if not np.isclose(β[i], 0, atol=tol):
            val = α[i]/β[i]
            λ_all.append(val)
            if np.linalg.norm(val) <= 1:
                λ.append(val)
                ind.append(i)
        else:
            if np.isclose(α[i], 0, atol=tol):
                λ_all.append(np.nan)
            else:
                λ_all.append(np.inf)
                
    return λ, ind, λ_all


def select_eigenvalues_bool(α, β, tol=1e-9):
    """Computes the eigenvalues λ = α/β and select those with modulus less or equal to 1."""
    decision = []
    
    for i in range(len(α)):
        if not np.isclose(β[i], 0, atol=tol):
            val = α[i]/β[i]
            if np.linalg.norm(val) <= 1:
                decision.append(True)
            else:
                decision.append(False)
        else:
            decision.append(False)
                
    return np.array(decision)


def print_colored_tab(tab, tab_bool):
    for i, (value, is_true) in enumerate(zip(tab, tab_bool)):
        if is_true:
            print(Fore.RED + str(value) + Style.RESET_ALL, end='')
        else:
            print(value, end='')
        if i < len(tab) - 1:
            print(', ', end='')
    print()
