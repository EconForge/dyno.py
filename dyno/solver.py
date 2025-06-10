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
    (X, evs) : Tuple[(N,N) ndarray, (2*N,) ndarray|None]
        solution of the equation as well as sorted list of associated generalized eigenvalues if the chosen method is "qz" and None otherwise
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
    (X, evs) : Tuple[(N,N) ndarray, None]
        solution of the equation and None (necessary to have a common solver interface)
    
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
    (X, evs) : Tuple[(N,N) ndarray, (2*N, ) ndarray]
        solution of the equation as well as sorted list of associated generalized eigenvalues
    """
    n = A.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Generalised eigenvalue problem
    F = np.block([[Z, I], [-C, -B]])
    G = np.block([[I, Z], [Z, A]])
    T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol)) <= 1)
    λ_all = vgenev(α, β, tol=tol)
    λ = λ_all[np.abs(λ_all) <= 1] # unused? should be used to ensure that Blanchard-Kahn conditions are verified

    Λ = np.diag(λ) # unused?
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
    Z : (2*N,2*N) ndarray
    
    Returns
    -------
    Z11, Z12, Z21, Z22 : (N,N) ndarrays
    """
    n = Z.shape[0] // 2
    Z11 = Z[:n, :n]
    Z12 = Z[:n, n:]
    Z21 = Z[n:, :n]
    Z22 = Z[n:, n:]
    return Z11, Z12, Z21, Z22


def genev(α, β, tol=1e-9):
    """
    Computes the generalized eigenvalue λ = α/β
    
    Parameters
    ----------

    α, β : floats
    
    Returns
    -------
    λ : float
        Generalized eigenvalue computed as λ = α/β with the conventions x/0 = ∞ for x > 0 and 0/0 = NaN
    """
    if not np.isclose(β, 0, atol=tol):
        return α / β
    else:
        if np.isclose(α, 0, atol=tol):
            return np.nan
        else:
            return np.inf

def vgenev(α, β, tol=1e-9):
    """
    Computes the generalized eigenvalues λ = α/β, vectorized version of `genev`

    Parameters
    ----------

    α, β : (2*N,) ndarrays
        output of scipy.linalg.ordqz
    
    Returns
    -------
    λ : (2*N,) ndarray
        vector of generalized eigenvalues computed as λ = α/β
    """
    return np.array([genev(a,b) for a,b in zip(α, β)])


def moments(X, Y, Σ):
    """
    Computes conditional and unconditional moments of stationary process $y_t = X y_{t-1} + Y e_t$

    Parameters
    ----------
    X, Y : (N,N) ndarrays
        matrices defining the stochastic process
    
    Σ : (N,N) ndarray
        covariance matrix of the independant idententically distributed error terms e_t
    
    Returns
    -------
    Γ₀, Γ : (N,N) ndarrays
        conditional and unconditional covariance matrices of the stationary process y_t respectively
    
    Notes
    -----
    The unconditional covariance matrix Γ is computed in the following way:

    Applying the linear covariance operator to both sides of the equation $y_t = X y_{t-1} + Y e_t$ yields
    $$
    \mathrm{Cov}(y_t) = X ⋅ \mathrm{Cov}(y_{t-1}) ⋅ X^* + Y ⋅ \mathrm{Cov}(e_t) ⋅ Y^*
    $$
    By stationarity of $y_t$, $\mathrm{Cov}(y_t) = \mathrm{Cov}(y_{t-1}) := \Gamma$, so
    $$
    Γ = X Γ X^* + Γ₀
    $$
    By applying the [Vec-operator](https://en.wikipedia.org/wiki/Vectorization_(mathematics)#Compatibility_with_Kronecker_products), we get the following equation:
    $$
    \mathrm{Vec}(Γ) = (X ⊗ X) \mathrm{Vec}(Γ)  + \mathrm{Vec}(Γ₀)
    $$
    Which gives the following solution
    $$
    \mathrm{Vec}(Γ) = (I_{N^2} - X ⊗ X)^{-1} \mathrm{Vec}(Γ₀)
    $$
    """

    Γ0 = Y @ Σ @ Y.T
    n = X.shape[0]

    # Compute the unconditional variance
    Γ = (np.linalg.inv(np.eye(n**2) - np.kron(X, X)) @ Γ0.flatten()).reshape(n, n)

    return Γ0, Γ
