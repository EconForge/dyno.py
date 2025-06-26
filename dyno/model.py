from numpy.linalg import solve as linsolve

import numpy as np
import yaml

from .solver import solve
from .misc import jacobian

from abc import ABC, abstractmethod

from typing import Callable, overload, Literal, Any
from typing_extensions import Self
from .typedefs import TVector, TMatrix, IRFType, Solver, SymbolType, DynamicFunction
from pandas import DataFrame
from .language import Normal


class RecursiveSolution:
    """VAR(1) representing a linearized model

    Attributes
    ----------
    X, Y, Σ: (N,N) Matrix
        parameters of the stationary VAR process $y_t = Xy_{t-1} + Yε_t$, where Σ is the covariance matrix of $ε_t$

    symbols: dict[SymbolType, list[str]]
        dictionary containing the symbols used in the model, the only allowed keys are "endogenous", "exogenous" and "parameters"

    x0: N Vector | None
        the state around which the linearization is done, generally the steady state, by default None

    evs: 2N Vector | None
        eigenvalues containing information about the stability of the model,
        only available if the qz solver was used for linearization, by default None
    """

    def __init__(
        self: Self,
        X: TMatrix,
        Y: TMatrix,
        Σ: TMatrix,
        symbols: dict[SymbolType, list[str]],
        x0: TVector | None = None,
        evs: TVector | None = None,
    ) -> None:

        self.x0 = x0
        self.X = X
        self.Y = Y
        self.Σ = Σ

        self.evs = evs

        self.symbols = symbols


class Model(ABC):
    """Abstract class representing an economic model"""

    name: str | None
    """Name of the model if described in input"""

    symbols: dict[SymbolType, list[str]]
    """Symbols dictionary, allowed keys are 'endogenous', 'exogenous' and 'parameters'"""

    equations: list[tuple[str, str]]
    """List of the equations of the model: the equation LHS = RHS is written as (LHS, RHS)"""

    calibration: dict[str, float]
    """Dictionary of parameter values and initial values of endogenous and exogenous variables"""

    exogenous: Normal
    """Description of shocks on exogenous variables, only stochastic shocks are supported for now"""

    _dynamic: DynamicFunction
    """Temporary storage for dynamic method"""

    data: Any
    """Format-dependant internal representation of the data"""

    def __init__(
        self: Self, filename: str | None = None, txt: str | None = None
    ) -> None:
        match filename, txt:
            case (None, None):
                raise ValueError(
                    "Neither the file name nor content were passed to constructor. One of the two should be passed."
                )
            case (filename, None):
                assert filename is not None  # to reassure Mypy
                self.import_file(filename)
            case (None, txt):
                assert txt is not None
                self.import_model(txt)
            case _:
                raise ValueError(
                    "File name and content were both passed to constructor. Only one of the two should be passed."
                )
        self._set_name()
        self._set_symbols()
        self._set_equations()
        self._set_calibration()
        self._set_exogenous()
        self._set_dynamic()

    def _set_name(self: Self) -> None:
        # should be overridden for file types with name information
        self.name = None

    @abstractmethod
    def _set_symbols(self: Self) -> None:
        pass

    @abstractmethod
    def _set_equations(self: Self) -> None:
        pass

    @abstractmethod
    def _set_calibration(self: Self) -> None:
        pass

    @abstractmethod
    def _set_exogenous(self: Self) -> None:
        pass

    @abstractmethod
    def import_model(self: Self, txt: str) -> None:
        """sets data attribute from model text description"""
        pass

    def import_file(self: Self, filename: str) -> None:
        """sets data attribute from file"""
        txt = open(filename, "rt", encoding="utf-8").read()
        assert txt is not None
        return self.import_model(txt)

    def get_calibration(self, **kwargs):
        c = self.calibration.copy()
        c.update(**kwargs)
        return c

    def _set_dynamic(self: Self) -> None:
        """generates dynamic method from the equations of the model using Dolang"""
        dict_eq = {
            f"out{i+1}": f"({rhs}) - ({lhs})"
            for i, (lhs, rhs) in enumerate(self.equations)
        }
        symbols = self.symbols
        from dolang.symbolic import stringify_symbol
        from dolang.function_compiler import FlatFunctionFactory as FFF
        from dolang.function_compiler import make_method_from_factory

        spec = dict(
            y_f=[stringify_symbol((e, 1)) for e in symbols["endogenous"]],
            y_0=[stringify_symbol((e, 0)) for e in symbols["endogenous"]],
            y_p=[stringify_symbol((e, -1)) for e in symbols["endogenous"]],
            e=[stringify_symbol((e, 0)) for e in symbols["exogenous"]],
            p=[stringify_symbol(e) for e in symbols["parameters"]],
        )
        fff = FFF(dict(), dict_eq, spec, "f_dynamic")
        fun = make_method_from_factory(fff, compile=False, debug=False)
        self._dynamic = fun

    def describe(self: Self) -> str:
        """Returns a string representation of the model's symbols"""
        return f"""
symbols: {self.symbols}
        """

    @overload
    def dynamic(
        self: Self, y0: TVector, y1: TVector, y2: TVector, e: TVector, p: TVector
    ) -> TVector:
        pass

    @overload
    def dynamic(
        self: Self,
        y0: TVector,
        y1: TVector,
        y2: TVector,
        e: TVector,
        p: TVector,
        diff: bool,
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        pass

    def dynamic(
        self: Self,
        y0: TVector,
        y1: TVector,
        y2: TVector,
        e: TVector,
        p: TVector,
        diff: bool = False,
    ) -> TVector | tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        """function f describing the behavior of the dynamic system $f(y_{t+1}, y_t, y_{t-1}, ε_t, p) = 0$

        Parameters
        ----------
        y0,y1,y2 : Vector
            the system's endogenous variable values at times t+1, t and t-1 respectively
        e : Vector
            exogenous variable values
        p : Vector
            parameter values
        diff : bool, optional
            if set to True returns the function's partial derivatives with regards to y0, y1, y2 and e as well, by default False

        Returns
        -------
        Vector|tuple[Vector, Matrix, Matrix, Matrix, Matrix]
            value of f(y0, y1, y2, e, p), as well as partial derivatives w.r.t. y0, y1, y2 and e if diff is set to True
        """
        r = np.zeros(len(y0))
        self._dynamic(y0, y1, y2, e, p, r)
        d = np.zeros(len(self.symbols["exogenous"]))

        if diff:
            f = lambda a, b, c, d, e: self.dynamic(a, b, c, d, e)
            r1 = jacobian(lambda u: f(u, y1, y2, e, p), y0)
            r2 = jacobian(lambda u: f(y0, u, y2, e, p), y1)
            r3 = jacobian(lambda u: f(y0, y1, u, e, p), y2)
            r4 = jacobian(lambda u: f(y0, y1, y2, u, p), d)
            return r, r1, r2, r3, r4

        return r

    @overload
    def compute(self: Self, calibration: dict[str, float] = {}) -> TVector:
        pass

    @overload
    def compute(
        self: Self, calibration: dict[str, float] = {}, diff: bool = False
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        pass

    def compute(
        self: Self, calibration: dict[str, float] = {}, diff: bool = False
    ) -> TVector | tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        """Computes the dynamic function's value based on calibration state and parameters

        Parameters
        ----------
        calibration : dict[str, float], optional
            dictionary containing the value of each parameter and variable of the model, indexed by their symbols, by default {}
        diff : bool, optional
            if set to True returns the dynamic function's partial derivatives as well, by default False

        Returns
        -------
        TVector|tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]
            value of the dynamic function at the state described by calibration, as well as its partial derivatives if diff is set to True
        """
        c = self.get_calibration(**calibration)
        v = self.symbols["endogenous"]
        p = self.symbols["parameters"]

        endogenous_values = [c[e] for e in v]
        parameter_values = [c[e] for e in p]
        # Reshapes necessary for static type checking
        y0 = np.reshape(endogenous_values, len(endogenous_values))
        p0 = np.reshape(parameter_values, len(parameter_values))

        e = np.zeros(len(self.symbols["exogenous"]))
        return self.dynamic(y0, y0, y0, e, p0, diff=diff)

    def solve(
        self: Self, calibration: dict[str, float] = {}, method: Solver = "qz"
    ) -> RecursiveSolution:
        """linearizes the model

        Parameters
        ----------
        calibration : dict[str, float], optional
            dictionary containing the value of each parameter and variable of the model, indexed by their symbols, by default {}
        method : Solver, optional
            chosen solver: either "ti" for fixed-point iteration or "qz" for generalized Schur decomposition, by default "qz"

        Returns
        -------
        RecursiveSolution
            linearized model
        """
        from .solver import solve as solveit

        r, A, B, C, D = self.compute(diff=True, calibration=calibration)

        X, evs = solveit(A, B, C, method=method)
        Y = linsolve(A @ X + B, -D)

        v = self.symbols["endogenous"]
        e = self.symbols["exogenous"]
        p = self.symbols["parameters"]

        Σ = self.exogenous.Σ

        c = self.get_calibration(**calibration)

        # a bit stupid
        endogenous_values = [c[e] for e in v]
        parameter_values = [c[e] for e in p]
        # Reshapes necessary for static type checking
        y0 = np.reshape(endogenous_values, len(endogenous_values))
        p0 = np.reshape(parameter_values, len(parameter_values))

        return RecursiveSolution(
            X, Y, Σ, {"endogenous": v, "exogenous": e}, evs=evs, x0=y0
        )


def irfs(
    model: Model, dr: RecursiveSolution, type: IRFType = "log-deviation"
) -> dict[str, DataFrame]:
    """Impulse response function simulation in response to shocks on each exogenous variable

    Parameters
    ----------
    dr : RecursiveSolution
        linearized model, contains all variables and parameters
    T : int, optional
        time horizon over which the simulation is done, by default 40
    type : IRFType, optional
        can be "level", "log-deviation" or "deviation", by default "level"

    Returns
    -------
    pd.DataFrame
        impulse response function of all endogenous variables to shocks on each exogenous variable
    """
    from .simul import irf

    res = {}
    for i, e in enumerate(model.symbols["exogenous"]):
        res[e] = irf(dr, i, type=type)

    return res


def evaluate(expression: str, calibration: dict[str, float] = {}) -> float:
    """safely evaluates mathematical expression based on calibration

    Two safety checks are in place to avoid arbitrary code execution :
        1. The abstract syntax tree of the expression is computed and each of its nodes is checked against a whitelist of allowed AST nodes.
        2. A dictionary containing only allowed functions is passed to `eval` eliminating the possibility of using arbitrary function calls.

    Parameters
    ----------
    expression : str
        mathematical expression that only makes use of supported functions (see below) and variables defined in calibration

    calibration: dict[str,float], optional
        dictionary of previously defined variables, by default {}

    Returns
    -------
    float
        result of evaluation

    Note
    ----
    - `^` is assumed to be the exponentiation operator and not exclusive or (as opposed to python syntax)
    - List of supported functions : exp, log, ln, log10, sqrt, cbrt,
                    sign, abs, max, min, sin, cos, tan, asin, acos,
                    atan, sinh, cosh, tanh, asinh, acosh, atanh

    Examples
    --------
    >>> evaluate("exp(a)", {'a': 1})
    2.718281828459045
    >>> evaluate("a^b", {'a': 2, 'b': 4})
    16.0
    >>> evaluate("cbrt(8)")
    2.0
    >>> evaluate("(x > 0) + (x < 0.5)", {'x': 0.25})
    2.0
    """
    import ast

    expression = expression.replace("^", "**")
    tree = ast.parse(expression, mode="eval")

    whitelist = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.operator,
        ast.unaryop,
        ast.cmpop,
        ast.Num,
        ast.Compare,
    )

    valid = all(isinstance(node, whitelist) for node in ast.walk(tree))

    if valid:
        from math import exp, log, log10, sqrt
        from numpy import cbrt, sign
        from math import sin, cos, tan
        from math import asin, acos, atan
        from math import sinh, cosh, tanh
        from math import asinh, acosh, atanh

        # abs, max and min are builtins
        safe_list = [
            exp,
            log,
            log10,
            sqrt,
            cbrt,
            sign,
            abs,
            max,
            min,
            sin,
            cos,
            tan,
            asin,
            acos,
            atan,
            sinh,
            cosh,
            tanh,
            asinh,
            acosh,
            atanh,
        ]
        safe_dict = {f.__name__: f for f in safe_list}  # type: ignore
        safe_dict["ln"] = log  # Add alias for compatibility with Dynare
        safe_dict.update(calibration)
        return float(
            eval(
                compile(tree, filename="", mode="eval"),
                {"__builtins__": None},
                safe_dict,
            )
        )
    else:
        raise ValueError("Invalid Mathematical expression")
