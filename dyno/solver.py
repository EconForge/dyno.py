import numpy as np
from numpy.linalg import solve as linsolve
from scipy.linalg import ordqz

def solve(A, B, C, method="qz", options={}):
    """Solves AX² + BX + C = 0 for X using the chosen method

    Parameters
    ----------
    A : (N,N) ndarray
        
    B : (N,N) ndarray
        
    C : (N,N) ndarray
        
    method : str, optional
        chosen solver: either "ti" for fixed-point iteration or "qz" for generalized Schur decomposition, by default "qz"
    
    options : dict, optional
        dictionary of optional parameters to pass to the chosen solver, by default {}

    Returns
    -------
    tuple containing:
    - X : (N,N) ndarray
        solution of the equation
    
    - evs : List[float]
        sorted list of associated generalized eigenvalues if the chosen method is "qz", None otherwise
    """

    if method == "ti":

        sol, evs = solve_ti(A, B, C, **options)

    else:

        sol, evs = solve_qz(A, B, C, **options)

    return sol, evs

class NoConvergence(Exception):
    """An exception raised when the convergence threshold is not reached within the maximal allowed number of iterations"""
    pass

def solve_ti(A, B, C, T=10000, tol=1e-10):
    """Solves AX² + BX + C = 0 for X using fixed-point iteration.

    Parameters
    ----------
    A : (N,N) ndarray
        
    B : (N,N) ndarray
        
    C : (N,N) ndarray
        
    T : int, optional
        Maximum number of iterations. If more are needed, `NoConvergence` is raised, by default 10000
    
    tol : float, optional
        convergence threshold, by default 1e-10

    Returns
    -------
    tuple containing:

    - X : (N,N) ndarray
        solution of the equation
    
    - evs : None
        not applicable, returned for the sake of having a uniform interface over solvers
    
    Raises
    ------
    NoConvergence :
        when the convergence threshold is not reached within the maximal allowed number of iterations
    LinAlgError :
        when a non-singular matrix is obtained while iterating
    ValueError :
        when a matrix containing a NaN is obtained while iterating
    """
    n = A.shape[0]

    X0 = np.random.randn(n, n)

    for t in range(T):

        X1 = linsolve(A @ X0 + B, -C)
        e = abs(X0 - X1).max()

        if np.isnan(e):
            # impossible situation?
            raise ValueError("Invalid value")

        X0 = X1
        if e < tol:
            return X0, None

    raise NoConvergence("The maximal number of iterations was exceeded.")


def solve_qz(A, B, C, tol=1e-15):
    """Solves AX² + BX + C = 0 for X using QZ decomposition.

    Parameters
    ----------
    A : (N,N) ndarray
        
    B : (N,N) ndarray
       
    C : (N,N) ndarray
        
    tol : float, optional
        error tolerance, by default 1e-15

    Returns
    -------
    tuple containing:
    - X : (N,N) ndarray
        solution of the equation
    
    - evs : List[float]
        sorted list of associated generalized eigenvalues
    """
    n = A.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Generalised eigenvalue problem
    F = np.block([[Z, I], [-C, -B]])
    G = np.block([[I, Z], [Z, A]])
    T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol)) <= 1)
    λ_all = vgenev(α, β, tol=tol)
    #λ = λ_all[np.abs(λ_all) <= 1] # unused?

    #Λ = np.diag(λ) # unused?
    Z11, Z12, Z21, Z22 = decompose_blocks(Z)
    X = Z21 @ np.linalg.inv(Z11)

    return X, sorted(λ_all)


def decompose_blocks(Z):
    """Decomposes square matrix Z into four square blocks Z11, Z12, Z21, Z22 such that Z can be written as:
    ```
    [Z11, Z12]
    [Z21, Z22]
    ```

    Parameters
    ----------
    Z : (N,N) ndarray
    
    Returns
    -------
    Z11 : (N//2, N//2) ndarray

    Z12 : (N//2, N-N//2) ndarray

    Z21 : (N-N//2, N//2) ndarray

    Z22 : (N-N//2, N-N//2) ndarray
    """
    n = Z.shape[0] // 2
    Z11 = Z[:n, :n]
    Z12 = Z[:n, n:]
    Z21 = Z[n:, :n]
    Z22 = Z[n:, n:]
    return Z11, Z12, Z21, Z22


def genev(α, β, tol=1e-9):
    """Computes the generalized eigenvalues λ = α/β.
    
    Parameters
    ----------

    α, β : (N,) ndarrays
        output of scipy.linalg.ordqz
    
    Returns
    -------
    λ : (N,) ndarray
        vector of generalized eigenvalues computed as λ = α/β
    """
    if not np.isclose(β, 0, atol=tol):
        return α / β
    else:
        if np.isclose(α, 0, atol=tol):
            return np.nan
        else:
            return np.inf


vgenev = np.vectorize(genev, excluded=["tol"])
"""vectorized version of `genev`"""

def moments(X, Y, Σ):
    """
    Computes conditional and unconditional moments of process $y_t = X y_{t-1} + Y e_t$
    """

    Σ0 = Y @ Σ @ Y.T
    n = X.shape[0]

    # Compute the unconditional variance
    Σ = (np.linalg.inv(np.eye(n**2) - np.kron(X, X)) @ Σ0.flatten()).reshape(n, n)

    return Σ0, Σ
