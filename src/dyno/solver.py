import numpy as np
from numpy.linalg import solve as linsolve
from scipy.linalg import ordqz
from .typedefs import TVector, TMatrix, Solver


def solve(
    A: TMatrix, B: TMatrix, C: TMatrix, method: Solver = "qz", options={}
) -> tuple[TMatrix, TVector | None]:
    """Solves AX² + BX + C = 0 for X using the chosen method

    Parameters
    ----------
    A, B, C : (N,N) Matrix

    method : str, optional
        chosen solver: either "ti" for fixed-point iteration or "qz" for generalized Schur decomposition, by default "qz"

    options : dict, optional
        dictionary of optional parameters to pass to the chosen solver, by default {}

    Returns
    -------
    (X, evs) : tuple[(N,N) Matrix, 2N Vector|None]
        solution of the equation as well as sorted list of associated generalized eigenvalues if the chosen method is "qz" and None otherwise

    Raises
    ------
    NoConvergence :
        when the convergence threshold is not reached within the maximal allowed number of iterations (only in `solve_ti`)
    LinAlgError :
        when a singular matrix is obtained during iterations in `solve_ti` or when Blachard-Kahn conditions are not verified in `solve_qz`
    ValueError :
        when a matrix containing a NaN is obtained
    """

    if method == "ti":

        sol, evs = solve_ti(A, B, C, **options)

    else:

        sol, evs = solve_qz(A, B, C, **options)

    return sol, evs


class NoConvergence(Exception):
    """An exception raised when the convergence threshold is not reached within the maximal allowed number of iterations"""

    pass


def solve_ti(
    A: TMatrix, B: TMatrix, C: TMatrix, T: int = 10000, tol: float = 1e-10
) -> tuple[TMatrix, None]:
    """Solves AX² + BX + C = 0 for X using fixed-point iteration.

    Parameters
    ----------
    A, B, C : (N,N) Matrix

    T : int, optional
        Maximum number of iterations. If more are needed, `NoConvergence` is raised, by default 10000

    tol : float, optional
        convergence threshold, by default 1e-10

    Returns
    -------
    (X, evs) : tuple[(N,N) Matrix, None]
        solution of the equation and None (necessary to have a common solver interface)

    Raises
    ------
    NoConvergence :
        when the convergence threshold is not reached within the maximal allowed number of iterations
    LinAlgError :
        when a singular matrix is obtained while iterating
    ValueError :
        when a matrix containing a NaN is obtained while iterating
    """
    n = A.shape[0]

    # Reshape necessary for static type checking
    X0 = np.random.randn(n, n).reshape((n, n))

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


def solve_qz(
    A: TMatrix, B: TMatrix, C: TMatrix, tol: float = 1e-15
) -> tuple[TMatrix, TVector]:
    """Solves AX² + BX + C = 0 for X using QZ decomposition.

    Parameters
    ----------
    A, B, C : (N,N) Matrix

    tol : float, optional
        error tolerance, by default 1e-15

    Returns
    -------
    (X, evs) : tuple[(N,N) Matrix, 2N Vector]
        solution of the equation as well as sorted list of associated generalized eigenvalues

    Raises
    ------
    LinAlgError :
        when Blanchard-Kahn conditions are not verified (less than N generalized eigenvalues inside the unit ball)
    """
    n = A.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Generalised eigenvalue problem
    F = np.block([[Z, I], [-C, -B]])
    G = np.block([[I, Z], [Z, A]])
    T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol)) <= 1)  # type: ignore
    λ_all = vgenev(α, β, tol=tol)
    Z11, Z12, Z21, Z22 = decompose_blocks(Z)
    # TODO: verify whether Blanchard-Kahn conditions are valid

    X = (Z21 @ np.linalg.inv(Z11)).reshape(
        (n, n)
    )  # Reshape necessary for static type checking

    return X, np.sort(λ_all).reshape(2 * n)


def decompose_blocks(Z: TMatrix) -> tuple[TMatrix, TMatrix, TMatrix, TMatrix]:
    """Decomposes square matrix Z into four square blocks Z11, Z12, Z21, Z22 such that Z can be written as:
    ```
    [Z11, Z12]
    [Z21, Z22]
    ```

    Parameters
    ----------
    Z : (2N,2N) Matrix

    Returns
    -------
    Z11, Z12, Z21, Z22 : (N,N) Matrix
    """
    n = Z.shape[0] // 2
    # Reshapes necessary for static type checking
    Z11 = Z[:n, :n].reshape((n, n))
    Z12 = Z[:n, n:].reshape((n, n))
    Z21 = Z[n:, :n].reshape((n, n))
    Z22 = Z[n:, n:].reshape((n, n))
    return Z11, Z12, Z21, Z22


def genev(α: float, β: float, tol: float = 1e-9) -> float:
    """
    Computes the generalized eigenvalue λ = α/β

    Parameters
    ----------

    α, β : float

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


def vgenev(α: TVector, β: TVector, tol: float = 1e-9) -> TVector:
    """
    Computes the generalized eigenvalues λ = α/β, vectorized version of `genev`

    Parameters
    ----------

    α, β : 2N Vector
        output of scipy.linalg.ordqz

    Returns
    -------
    λ : 2N Vector
        vector of generalized eigenvalues computed as λ = α/β
    """
    return (np.array([genev(a, b) for a, b in zip(α, β)])).reshape(len(α))


def moments(X: TMatrix, Y: TMatrix, Σ: TMatrix) -> tuple[TMatrix, TMatrix]:
    """
    Computes conditional and unconditional moments of stationary process $y_t = X y_{t-1} + Y e_t$

    Parameters
    ----------
    X, Y : (N,N) Matrix
        matrices defining the stochastic process

    Σ : (N,N) Matrix
        covariance matrix of the independant idententically distributed error terms e_t

    Returns
    -------
    Γ₀, Γ : (N,N) Matrix
        conditional and unconditional covariance matrices of the stationary process y_t respectively

    Notes
    -----
    The unconditional covariance matrix Γ is computed in the following way:

    Applying the linear covariance operator to both sides of the equation $y_t = X y_{t-1} + Y e_t$ yields
    $$
    \\mathrm{Cov}(y_t) = X ⋅ \\mathrm{Cov}(y_{t-1}) ⋅ X^* + Y ⋅ \\mathrm{Cov}(e_t) ⋅ Y^*
    $$
    By stationarity of $y_t$, $\\mathrm{Cov}(y_t) = \\mathrm{Cov}(y_{t-1}) := Γ$, so
    $$
    Γ = X Γ X^* + Γ₀
    $$
    By applying the [Vec-operator](https://en.wikipedia.org/wiki/Vectorization_(mathematics)#Compatibility_with_Kronecker_products), we get the following equation:
    $$
    \\mathrm{Vec}(Γ) = (X ⊗ X) \\mathrm{Vec}(Γ)  + \\mathrm{Vec}(Γ₀)
    $$
    Which gives the following solution
    $$
    \\mathrm{Vec}(Γ) = (I_{N^2} - X ⊗ X)^{-1} \\mathrm{Vec}(Γ₀)
    $$
    """

    Γ0 = Y @ Σ @ Y.T
    n = X.shape[0]

    # Compute the unconditional variance
    Γ = (np.linalg.inv(np.eye(n**2) - np.kron(X, X)) @ Γ0.flatten()).reshape(n, n)

    return Γ0, Γ


def deterministic_solve(model, x0=None, T=None, method='hybr'):

    import scipy.optimize
    import pandas

    if x0 is None:
        v0 = model.deterministic_guess(T=T)
    else:
        v0 = np.array(x0)

    T = v0.shape[0]-1
    u0 = np.array(v0).ravel(),

    res = scipy.optimize.root(
        lambda u: model.deterministic_residuals(u, jac=True),
        u0,
        method=method,
        jac=True
    )
    
    w0 = res.x.reshape(v0.shape)

    df = pandas.DataFrame({e: w0[:,i] for i,e in enumerate(model.variables)})
    df.index=range(T+1)
    df.index.name='t'

    return df
