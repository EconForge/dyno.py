from __future__ import annotations

import logging
import numpy as np
from numpy.linalg import solve as linsolve
from numpy.typing import NDArray
from .typedefs import TVector, TMatrix, Solver
from .errors import BlanchardKahnError, ConvergenceWarning

_log = logging.getLogger(__name__)

from typing import Any, TYPE_CHECKING

from typing_extensions import Self
from .typedefs import IRFType, SimulateMode, UnitsType

if TYPE_CHECKING:
    from .model import AbstractModel
    from .simul import IRFSimulation, SimulationResult

__all__ = [
    "RecursiveDecisionRule",
    "PerturbationSolution",
    "NoConvergence",
    "BlanchardKahnError",
    "ConvergenceWarning",
    "solve",
    "solve_ti",
    "solve_qz",
    "moments",
    "deterministic_solve",
]


class RecursiveDecisionRule:
    """VAR(1) representing a linearized model

    Attributes
    ----------
    X, Y, Σ: (N,N) Matrix
        parameters of the stationary VAR process $y_t = Xy_{t-1} + Yε_t$, where Σ is the covariance matrix of $ε_t$

    symbols: dict[str, list[str]]
        dictionary containing the symbols used in the model, the only allowed keys are "endogenous", "exogenous" and "parameters"

    x0: N Vector | None
        the state around which the linearization is done, generally the steady state, by default None

    """

    def __init__(
        self: Self,
        X: TMatrix,
        Y: TMatrix,
        Σ: TMatrix,
        symbols: dict[str, list[str]],
        x0: TVector | None = None,
        model: "AbstractModel | None" = None,
    ) -> None:

        self.x0 = x0
        self.X = X
        self.Y = Y
        self.Σ = Σ

        self.symbols = symbols
        self._model = model

    def moments(self):

        return moments(self.X, self.Y, self.Σ)

    def coefficients_as_df(self):
        import pandas as pd

        assert self.x0 is not None

        ss = pd.DataFrame(
            [self.x0], columns=["{}".format(e) for e in self.symbols["endogenous"]]
        )
        hh_y = self.X
        hh_e = self.Y
        df = pd.DataFrame(
            np.concatenate([hh_y, hh_e], axis=1),
            columns=["{}[t-1]".format(e) for e in self.symbols["endogenous"]]
            + ["{}[t]".format(e) for e in (self.symbols["exogenous"])],
        )
        df.index = pd.Index(["{}[t]".format(e) for e in self.symbols["endogenous"]])
        return ss, df

    def _repr_html_(self):

        Σ0, Σ = moments(self.X, self.Y, self.Σ)

        ss, df = self.coefficients_as_df()

        html = f"""
        <h3>Decision Rule</h3>
        <h4>Steady-state</h4>
        {ss.to_html(index=False)}
        <h4>Jacobian</h4>
        {df.to_html()}
        """
        return html

    def irfs(self, type: IRFType = "log-deviation", T: int = 40) -> "IRFSimulation":

        from .simul import irfs

        assert self._model is not None

        sim = irfs(self._model, self, type=type, T=T)
        return sim

    def simulate(
        self,
        T: int = 40,
        mode: SimulateMode = "auto",
        N: int = 1,
        units: UnitsType = "deviation",
        shocks: dict[str, dict[int, float]] | np.ndarray | None = None,
        initial_states: dict[str, float] | np.ndarray | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "SimulationResult":
        """Simulate the solved model.

        Parameters
        ----------
        T : int, default 40
            Simulation horizon.
        mode : {'auto', 'irf', 'random', 'transition', 'deterministic'}, default 'auto'
            When ``'auto'``, defaults to ``'irf'`` unless ``N > 1``, ``shocks``, or
            ``initial_states`` are explicitly supplied.
        N : int, default 1
            Number of Monte Carlo trajectories when ``mode='random'``.
        units : {'deviation', 'level', 'percent', 'log-deviation'}, default 'deviation'
            Default output units.
        rng : numpy.random.Generator | int | None, optional
            Generator or seed for the random shocks when ``mode='random'``.
        """
        from .simul import simulate

        resolved_mode: SimulateMode
        if mode == "auto":
            if shocks is not None or initial_states is not None or N > 1:
                resolved_mode = "random"
            else:
                resolved_mode = "irf"
        else:
            resolved_mode = mode

        if resolved_mode == "irf":
            return self.irfs(type=units, T=T)

        return simulate(
            self,
            T=T,
            shocks=shocks,
            initial_states=initial_states,
            N=N,
            mode="random",
            units=units,
            rng=rng,
        )

    def plot(
        self,
        type: IRFType = "log-deviation",
        units: UnitsType | None = None,
        variables: list[str] | None = None,
        T: int = 40,
        **kwargs: Any,
    ):
        target_units: UnitsType = units if units is not None else type
        sim = self.irfs(type=target_units, T=T)
        return sim.plot(variables=variables, T=T, units=target_units, **kwargs)


class PerturbationSolution:
    """Container for perturbation outputs.

    Attributes
    ----------
    decision_rule: RecursiveDecisionRule
        First-order recursive decision rule.
    """

    def __init__(
        self: Self,
        decision_rule: RecursiveDecisionRule,
        evs: TVector | None = None,
    ) -> None:
        self.decision_rule = decision_rule
        self.evs = evs

    def __getattr__(self, name: str) -> Any:
        # Delegate any other method/property access to the underlying decision_rule
        return getattr(self.decision_rule, name)


def solve(
    A: TMatrix,
    B: TMatrix,
    C: TMatrix,
    method: Solver = "qz",
    options: dict | None = None,
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

    options = options or {}

    if method == "ti":
        sol, evs = solve_ti(A, B, C, **options)
    elif method == "qz":
        sol, evs = solve_qz(A, B, C, **options)
    else:
        raise ValueError(
            f"Unknown solver method: {method!r}. Valid choices are 'ti' and 'qz'."
        )

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

    # Deterministic initial guess: the solution of the backward-looking part
    # (A = 0). Falls back to zeros when B is singular.
    X0: TMatrix
    try:
        X0 = np.asarray(linsolve(B, -C), dtype=float).reshape((n, n))
    except np.linalg.LinAlgError:
        X0 = np.zeros((n, n))

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
    from scipy.linalg import ordqz

    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0)), np.array([], dtype=float)
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Generalised eigenvalue problem
    F = np.block([[Z, I], [-C, -B]])
    G = np.block([[I, Z], [Z, A]])
    tol_sort = 1e-6
    T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol_sort)) <= 1 + tol_sort)  # type: ignore
    λ_all = vgenev(α, β, tol=tol_sort)
    Z11, Z12, Z21, Z22 = decompose_blocks(Z)

    λ_all = np.abs(λ_all)

    # TODO: verify whether Blanchard-Kahn conditions are valid
    evs = np.sort(λ_all).reshape(2 * n)
    n = len(evs) // 2
    l1 = evs[n - 1]
    l2 = evs[n]
    if l1 <= l2 < 1:
        raise BlanchardKahnError(
            f"Eigenvalue condition not satisfied: l_(n)={l1}, l_(n+1)={l2}. Too many stable solutions.",
            evs=evs,
        )
    if 1 < l1 <= l2:
        raise BlanchardKahnError(
            f"Eigenvalue condition not satisfied: l_(n)={l1}, l_(n+1)={l2}. No stable solution.",
            evs=evs,
        )

    X = (Z21 @ np.linalg.inv(Z11)).reshape(
        (n, n)
    )  # Reshape necessary for static type checking

    return X, evs


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


def genev(α: complex, β: complex, tol: float = 1e-9) -> complex:
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


def vgenev(α: NDArray[Any], β: NDArray[Any], tol: float = 1e-9) -> NDArray[Any]:
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
    return (np.array([genev(a, b, tol=tol) for a, b in zip(α, β)])).reshape(len(α))


def moments(X: TMatrix, Y: TMatrix, Σ: TMatrix) -> tuple[TMatrix, TMatrix]:
    """
    Computes conditional and unconditional moments of stationary process $y_t = X y_{t-1} + Y e_t$

    Parameters
    ----------
    X : (N,N) Matrix
        Transition matrix defining the stochastic process.
    Y : (N,N) Matrix
        Shock impact matrix defining the stochastic process.
    Sigma : (N,N) Matrix
        Covariance matrix of the independent identically distributed error terms e_t.

    Returns
    -------
    Gamma_0, Gamma : (N,N) Matrix
        Conditional and unconditional covariance matrices of the stationary process y_t respectively.

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


def serial_solve(a, b):
    return np.linalg.solve(a, b)


class NewtonResult(list):
    """Result container subclassing list for [x, it] unpacking while exposing metadata."""

    def __init__(self, x: np.ndarray, it: int, converged: bool, error: float):
        super().__init__([x, it])
        self.x = x
        self.it = it
        self.converged = converged
        self.error = error


def newton(f, x, verbose=False, tol=1e-6, maxit=5, jactype="serial"):
    """Solve nonlinear system using safeguarded Newton iterations


    Parameters
    ----------

    Return
    ------
    """
    it = 0
    error = 10
    converged = False
    maxbacksteps = 30

    x0 = x

    if jactype == "sparse":
        from scipy.sparse.linalg import spsolve as solve
    elif jactype == "full":
        from numpy.linalg import solve
    else:
        solve = serial_solve

    error_0 = float(error)

    while it < maxit and not converged:

        [v, dv] = f(x)

        # TODO: rewrite starting here

        #        print("Time to evaluate {}".format(ss-tt)0)

        error_0 = float(abs(v).max())

        if error_0 < tol:

            if verbose:
                _log.debug(
                    "> System was solved after iteration %d. Residual=%s",
                    it,
                    error_0,
                )
            converged = True

        else:

            it += 1

            dx = solve(dv, v)

            # norm_dx = abs(dx).max()

            xx = x
            err = error_0
            bck = 0
            for bck in range(maxbacksteps):
                xx = x - dx * (2 ** (-bck))
                vm = f(xx)[0]
                err = float(abs(vm).max())
                if err < error_0:
                    break

            x = xx

            if verbose:
                _log.debug("\t> %d | %s | %s", it, err, bck)

    final_error = error_0
    if not converged:
        final_v = f(x)[0]
        final_error = float(abs(final_v).max())
        if final_error < tol:
            converged = True

    if not converged:
        import warnings

        warnings.warn(
            f"Deterministic / perfect foresight solver did not converge after {it} iterations "
            f"(maximum residual: {final_error:.2e}). The computed solution is incorrect.",
            ConvergenceWarning,
            stacklevel=2,
        )

    return NewtonResult(x, it, converged, final_error)


def deterministic_solve(
    model,
    x0=None,
    T=None,
    method="hybr",
    verbose=False,
    continuation="stationary",
    growth_rate=None,
    growth_type="geometric",
    return_iterations=False,
    units: UnitsType = "level",
    **args,
):
    from .simul import TransitionSimulation

    continuation = args.pop("terminal_condition", continuation)
    growth_rate = args.pop("growth_rate", growth_rate)
    growth_type = args.pop("growth_type", growth_type)

    if x0 is None:
        v0 = model.deterministic_guess(T=T)
    else:
        v0 = np.array(x0)

    T = v0.shape[0] - 1

    u0 = np.array(v0).ravel()

    newton_res = newton(
        lambda u: model.deterministic_residuals_with_jacobian(
            u,
            sparsify=True,
            continuation=continuation,
            growth_rate=growth_rate,
            growth_type=growth_type,
            **args,
        ),
        u0,
        jactype="sparse",
        verbose=verbose,
        maxit=args.get("maxit", 10),
        tol=args.get("tol", 1e-8),
    )

    u_sol, nit = newton_res[0], newton_res[1]
    converged = getattr(newton_res, "converged", True)
    final_error = getattr(newton_res, "error", None)

    w0 = u_sol.reshape(v0.shape)

    variables = list(model.symbols["variables"])
    c = getattr(model, "context", {})
    ss_map = c.get("steady_states", {})
    val_map = c.get("values", {})
    ss_vec = np.array(
        [ss_map.get(name, val_map.get(name, {}).get(0, np.nan)) for name in variables],
        dtype=float,
    )

    attrs: dict[str, Any] = {
        "iterations": nit,
        "converged": converged,
    }
    if final_error is not None:
        attrs["residual"] = final_error

    sim = TransitionSimulation(
        data=w0,
        variables=variables,
        steady_state=ss_vec,
        units=units,
        canonical_units="level",
        attrs=attrs,
    )

    if return_iterations:
        return sim, nit

    return sim
