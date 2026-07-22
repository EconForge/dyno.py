from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from dyno import DynoModel


ObjectiveCallback = Callable[[int, dict[str, np.ndarray], DynoModel], float]
TerminalCallback = Callable[[np.ndarray, int, DynoModel, int], tuple[float, np.ndarray]]


@dataclass
class RamseyOptions:
    tol: float = 1e-8
    maxit: int = 50
    max_backsteps: int = 10
    fd_eps: float = 1e-6
    verbose: bool = False
    terminal_mode: str | TerminalCallback = "lambda_T_zero"


@dataclass
class RamseySolveResult:
    path: pd.DataFrame
    iterations: int
    converged: bool
    max_residual: float
    terminal_mode: str


def _sparse_newton(
    f: Callable[[np.ndarray], tuple[np.ndarray, sp.csr_matrix]],
    x: np.ndarray,
    *,
    tol: float,
    maxit: int,
    maxbacksteps: int,
    verbose: bool,
) -> tuple[np.ndarray, int]:
    it = 0
    converged = False

    while it < maxit and not converged:
        residuals, jacobian = f(x)
        error_0 = float(np.max(np.abs(residuals)))

        if verbose:
            print(f"iter={it} residual={error_0:.3e}")

        if error_0 < tol:
            converged = True
            break

        it += 1
        dx = spsolve(jacobian, residuals)

        trial = x
        trial_error = error_0
        accepted = False
        for bck in range(maxbacksteps):
            trial = x - dx * (2.0 ** (-bck))
            trial_residuals = f(trial)[0]
            if not np.all(np.isfinite(trial_residuals)):
                continue
            trial_error = float(np.max(np.abs(trial_residuals)))
            if trial_error < error_0:
                accepted = True
                break

        if accepted:
            x = trial

        if verbose:
            if accepted:
                print(f"  accepted_residual={trial_error:.3e}")
            else:
                print("  no improving finite step found")

    return x, it


def _as_path_dict(v: np.ndarray, var_names: list[str]) -> dict[str, np.ndarray]:
    return {name: v[:, i] for i, name in enumerate(var_names)}


def _constant(model: DynoModel, name: str) -> float:
    return float(model.symbolic.context["constants"][name])


def _neoclassical_log_utility_residual(
    t: int,
    v: np.ndarray,
    var_names: list[str],
    model: DynoModel,
) -> float:
    beta = _constant(model, "β")
    alpha = _constant(model, "α")
    delta = _constant(model, "δ")

    k_i = var_names.index("k")
    y_i = var_names.index("y")
    c_i = var_names.index("c")

    c_t = v[t, c_i]
    k_t = v[t, k_i]
    y_tp1 = v[t + 1, y_i]
    c_tp1 = v[t + 1, c_i]

    return -1.0 / c_t + beta * (alpha * y_tp1 / k_t + 1.0 - delta) / c_tp1


def _closure_row_for_t(
    objective_callback: ObjectiveCallback,
    t: int,
    v: np.ndarray,
    var_names: list[str],
    model: DynoModel,
    instrument_i: int,
    eps: float,
) -> tuple[float, np.ndarray]:
    del objective_callback, instrument_i

    p = v.shape[1]
    n_rows = v.shape[0]
    row = np.zeros((n_rows, p))

    g0 = _neoclassical_log_utility_residual(t, v, var_names, model)

    dependency_points = [
        (t, var_names.index("c")),
        (t, var_names.index("k")),
        (t + 1, var_names.index("y")),
        (t + 1, var_names.index("c")),
    ]

    for s, j in dependency_points:
        vp = v.copy()
        vm = v.copy()
        vp[s, j] += eps
        vm[s, j] -= eps
        gp = _neoclassical_log_utility_residual(t, vp, var_names, model)
        gm = _neoclassical_log_utility_residual(t, vm, var_names, model)
        row[s, j] = (gp - gm) / (2.0 * eps)

    return g0, row


def _terminal_condition(
    terminal_mode: str | TerminalCallback,
    v: np.ndarray,
    model: DynoModel,
    instrument_i: int,
) -> tuple[float, np.ndarray, str]:
    p = v.shape[1]
    T = v.shape[0] - 1
    grad = np.zeros(p)

    if callable(terminal_mode):
        residual, grad = terminal_mode(v, T, model, instrument_i)
        return float(residual), np.asarray(grad, dtype=float), "custom"

    if terminal_mode == "lambda_T_zero":
        grad[instrument_i] = 1.0
        return float(v[T, instrument_i]), grad, terminal_mode

    if terminal_mode == "steady_state_terminal":
        y, e = model.__steady_state_vectors__
        ss = np.concatenate([y, e])
        grad[instrument_i] = 1.0
        return float(v[T, instrument_i] - ss[instrument_i]), grad, terminal_mode

    raise ValueError(f"Unsupported terminal_mode={terminal_mode!r}")


def _ramsey_residuals_with_jacobian(
    u: np.ndarray,
    model: DynoModel,
    objective_callback: ObjectiveCallback,
    instrument_name: str,
    options: RamseyOptions,
) -> tuple[np.ndarray, sp.csr_matrix]:
    var_names = list(model.symbols["variables"])
    p = len(var_names)
    q = len(model.equations)
    T = int(np.prod(u.shape) / p - 1)

    v = u.reshape((T + 1, p))
    instrument_i = var_names.index(instrument_name)

    res_flat, J_base = model.deterministic_residuals_with_jacobian(u, sparsify=True)
    res = res_flat.reshape((T + 1, p))
    J = J_base.tolil()

    closure_row = q

    for t in range(1, T):
        residual, grad = _closure_row_for_t(
            objective_callback,
            t,
            v,
            var_names,
            model,
            instrument_i,
            options.fd_eps,
        )
        res[t, closure_row] = residual
        row_i = t * p + closure_row
        J.rows[row_i] = []
        J.data[row_i] = []
        for j in range(p):
            val = grad[j]
            if val != 0.0:
                J[row_i, t * p + j] = val

    terminal_residual, terminal_grad, _ = _terminal_condition(
        options.terminal_mode,
                if instrument_name != "k":
                    raise NotImplementedError(
                        "Current prototype derives the Ramsey closure for the capital path, so instrument_name must be 'k'"
                    )
        v,
        model,
        instrument_i,
    )
    res[T, closure_row] = terminal_residual
    row_i = T * p + closure_row
    J.rows[row_i] = []
    J.data[row_i] = []
    for j in range(p):
        val = terminal_grad[j]
        if val != 0.0:
            J[row_i, T * p + j] = val

    return res.ravel(), J.tocsr()


def run_ramsey_prototype(
                    for s in range(T + 1):
                        for j in range(p):
                            val = grad[s, j]
                            if val != 0.0:
                                J[row_i, s * p + j] = val
    options: RamseyOptions | None = None,
) -> RamseySolveResult:
    options = options or RamseyOptions()
    model_path = Path(model_path)

    if model_path.suffix != ".dyno":
        raise ValueError("This prototype only supports .dyno files")

    model = DynoModel(str(model_path))
    p = len(model.symbols["variables"])
    q = len(model.equations)

    if p != q + 1:
        raise ValueError(
            "Model must satisfy one-extra-variable condition: len(variables) == len(equations) + 1"
        )

    if instrument_name not in model.symbols["variables"]:
        raise ValueError(f"Unknown instrument {instrument_name!r}")

    u0 = model.deterministic_guess(T=T).ravel()

    f = lambda uu: _ramsey_residuals_with_jacobian(  # noqa: E731
        uu,
        model,
        objective_callback,
        instrument_name,
        options,
    )

    sol, nit = _sparse_newton(
        f,
        u0,
        tol=options.tol,
        maxit=options.maxit,
        maxbacksteps=options.max_backsteps,
        verbose=options.verbose,
    )

    w = sol.reshape((T + 1, p))
    res_final, _ = f(sol)
    max_residual = float(np.max(np.abs(res_final)))
    converged = bool(max_residual <= options.tol)

    df = pd.DataFrame({name: w[:, i] for i, name in enumerate(model.symbols["variables"])})
    df.index = pd.RangeIndex(T + 1, name="t")
    df.reset_index(inplace=True)

    terminal_label = "custom" if callable(options.terminal_mode) else str(options.terminal_mode)
    return RamseySolveResult(
        path=df,
        iterations=nit,
        converged=converged,
        max_residual=max_residual,
        terminal_mode=terminal_label,
    )


def _demo_objective(t: int, path: dict[str, np.ndarray], _model: DynoModel) -> float:
    c_t = max(path["c"][t], 1e-12)
    return np.log(c_t)


def _run_demo() -> None:
    model_file = Path(__file__).resolve().parents[1] / "examples" / "neoclassical_ramsey.dyno"
    options = RamseyOptions(terminal_mode="steady_state_terminal", verbose=True)
    result = run_ramsey_prototype(
        model_file,
        objective_callback=_demo_objective,
        instrument_name="k",
        T=30,
        options=options,
    )
    print("Converged:", result.converged)
    print("Iterations:", result.iterations)
    print("Max residual:", result.max_residual)
    print(result.path.head())


if __name__ == "__main__":
    _run_demo()
