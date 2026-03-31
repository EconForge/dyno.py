from __future__ import annotations

import numpy as np
from numpy.linalg import solve as linsolve
from scipy.linalg import ordqz
from .typedefs import TVector, TMatrix, Solver

from typing import Any, TYPE_CHECKING

from typing_extensions import Self
from .typedefs import IRFType

if TYPE_CHECKING:
    from .model import AbstractModel


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

        # df_cmoments = pd.DataFrame(
        #     Σ0,
        #     columns=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
        #     index=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
        # )

        # df_umoments = pd.DataFrame(
        #     Σ,
        #     columns=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
        #     index=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
        # )
        ss, df = self.coefficients_as_df()

        html = f"""
        <h3>Decision Rule</h3>
        <h4>Steady-state</h4>
        {ss.to_html(index=False)}
        <h4>Jacobian</h4>
        {df.to_html()}
        """
        return html

    def irfs(self, type: IRFType = "log-deviation", T=40):

        from .simul import irfs

        assert self._model is not None

        sim = irfs(self._model, self, type=type, T=T)
        return sim

    def plot(self, type: IRFType = "log-deviation"):

        from .simul import sim_to_nsim

        import plotly.express as px

        sim = self.irfs(type=type)
        plots = sim_to_nsim(sim)

        fig = px.line(
            plots,
            x="t",
            y="value",
            color="shock",
            facet_col="variable",
            facet_col_wrap=2,
        )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(title_text="", matches=None)
        fig.update_xaxes(title_text="")

        return fig


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

    # Backward-compatible passthroughs
    @property
    def X(self):
        return self.decision_rule.X

    @property
    def Y(self):
        return self.decision_rule.Y

    @property
    def Σ(self):
        return self.decision_rule.Σ

    @property
    def x0(self):
        return self.decision_rule.x0

    @property
    def symbols(self):
        return self.decision_rule.symbols

    def moments(self):
        return self.decision_rule.moments()

    def coefficients_as_df(self):
        return self.decision_rule.coefficients_as_df()

    def irfs(self, type: IRFType = "log-deviation", T=40):
        return self.decision_rule.irfs(type=type, T=T)

    def plot(self, type: IRFType = "log-deviation"):
        return self.decision_rule.plot(type=type)

    def _repr_html_(self):
        return self.decision_rule._repr_html_()


# Backward-compatible alias
RecursiveSolution = RecursiveDecisionRule

        # html = f"""
        # <h3>Eigenvalues</h3>
        # {evv.to_html()}
        # <h3>Decision Rule</h3>
        # <h4>Steady-state</h4>
        # {ss.to_html(index=False)}
        # <h4>Jacobian</h4>
        # {df.to_html()}
        # <h3>Moments</h3>
        # <h4>Unconditional moments</h4>
        # {df_umoments.to_html()}
        # <h4>Conditional moments</h4>
        # {df_cmoments.to_html()}
        # <h3>IRFs</h3>
        # {fig.to_html(full_html=False, include_plotlyjs=False)}
        # """
        # return html


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
    tol = 1e-6
    T, S, α, β, Q, Z = ordqz(F, G, sort=lambda a, b: np.abs(vgenev(a, b, tol=tol)) <= 1 + tol)  # type: ignore
    λ_all = vgenev(α, β, tol=tol)
    Z11, Z12, Z21, Z22 = decompose_blocks(Z)

    # TODO: verify whether Blanchard-Kahn conditions are valid
    evs = np.sort(λ_all).reshape(2 * n)
    n = len(evs)//2
    l1 = evs[n-1]
    l2 = evs[n]
    if l1<=l2<1:
        raise Exception(f"Eigenvalue condition not satisfied: l1={l1}, l2={l2}. Too many stable solutions.")
    if 1<l1<=l2:
        raise Exception(f"Eigenvalue condition not satisfied: l1={l1}, l2={l2}. No stable solution.")


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


import time

old_print = print


def serial_solve(a, b):
    return np.linalg.solve(a, b)


def newton(f, x, verbose=False, tol=1e-6, maxit=5, jactype="serial"):
    """Solve nonlinear system using safeguarded Newton iterations


    Parameters
    ----------

    Return
    ------
    """
    if verbose:
        print = lambda txt: old_print(txt)
    else:
        print = lambda txt: None

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

    while it < maxit and not converged:

        [v, dv] = f(x)

        # TODO: rewrite starting here

        #        print("Time to evaluate {}".format(ss-tt)0)

        error_0 = abs(v).max()

        if error_0 < tol:

            if verbose:
                print(
                    "> System was solved after iteration {}. Residual={}".format(
                        it, error_0
                    )
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
                err = abs(vm).max()
                if err < error_0:
                    break

            x = xx

            if verbose:
                print("\t> {} | {} | {}".format(it, err, bck))

    if not converged:
        import warnings

        warnings.warn("Did not converge")
    return [x, it]


def deterministic_solve(model, x0=None, T=None, method="hybr", verbose=False, **args):

    import pandas

    if x0 is None:
        v0 = model.deterministic_guess(T=T)
    else:
        v0 = np.array(x0)

    T = v0.shape[0] - 1
    u0 = (np.array(v0).ravel(),)

    # res = scipy.optimize.root(
    #     lambda u: model.deterministic_residuals(u, jac=True),
    #     u0,
    #     method=method,
    #     jac=True,
    # )
    # w0 = res.x.reshape(v0.shape)

    u0 = np.array(v0).ravel()

    res, nit = newton(
        lambda u: model.deterministic_residuals_with_jacobian(u, sparsify=True),
        u0,
        jactype="sparse",
        verbose=verbose,
        maxit=args.get("maxit", 10),
        tol=args.get("tol", 1e-8),
    )

    w0 = res.reshape(v0.shape)

    df = pandas.DataFrame(
        {e: w0[:, i] for i, e in enumerate(model.symbols["variables"])}
    )
    df.index = pandas.RangeIndex(T + 1, name="t")
    df.reset_index(inplace=True)

    return df
