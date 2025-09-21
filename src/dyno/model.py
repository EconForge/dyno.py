from numpy.linalg import solve as linsolve

import numpy as np
import yaml

from .solver import solve, moments
from .misc import jacobian

from abc import ABC, abstractmethod

from typing import Callable, overload, Literal, Any
from typing_extensions import Self
from .typedefs import TVector, TMatrix, IRFType, Solver, DynamicFunction
import pandas as pd
from .language import Exogenous, Normal, Deterministic, ProductNormal

from .errors import SteadyStateError

from dyno.solver import RecursiveSolution

class Model(ABC):
    """Abstract class representing an economic model"""

    name: str | None
    """Name of the model if described in input"""

    symbols: dict[str, list[str]]
    """Symbols dictionary, allowed keys are 'endogenous', 'exogenous' and 'parameters'"""

    equations: list[str]
    """List of the equations of the model written in the form LHS = RHS"""

    calibration: dict[str, float]
    """Dictionary of parameter values and initial values of endogenous and exogenous variables"""

    processes: ProductNormal | None

    values: dict[str, dict[int, float]] | None
    
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
        # self._set_dynamic()

    def _set_name(self: Self) -> None:
        # should be overridden for file types with name information
        self.name = None

    @property
    def checks(self):
        checks = {}
        
        checks['deterministic'] = False if self.processes is not None else True
        
        return checks

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

    @property
    def variables(self):
        return self.symbols["endogenous"] + self.symbols["exogenous"]

    @property
    def parameters(self):
        return self.symbols["parameters"]

    # def _set_dynamic(self: Self) -> None:
    #     """generates dynamic method from the equations of the model using Dolang"""
    #     from dolang import stringify

    #     str_equations = [stringify(eq) for eq in self.equations]

    #     equations = []
    #     for streq in str_equations:
    #         lst = streq.split("=")

    #         match len(lst):
    #             case 1:
    #                 eq = streq.strip()
    #             case 2:
    #                 eq = f"({lst[0].strip()}) - ({lst[1].strip()})"
    #             case _:
    #                 raise ValueError("More than one equation on the same line")

    #         equations.append(eq)

    #     dict_eq = {f"out{i+1}": eq for i, eq in enumerate(equations)}
    #     symbols = self.symbols
    #     from dolang.symbolic import stringify_symbol
    #     from dolang.function_compiler import FlatFunctionFactory as FFF
    #     from dolang.function_compiler import make_method_from_factory

    #     spec = dict(
    #         y_f=[stringify_symbol((e, 1)) for e in symbols["endogenous"]],
    #         y_0=[stringify_symbol((e, 0)) for e in symbols["endogenous"]],
    #         y_p=[stringify_symbol((e, -1)) for e in symbols["endogenous"]],
    #         e=[stringify_symbol((e, 0)) for e in symbols["exogenous"]],
    #         p=[stringify_symbol(e) for e in symbols["parameters"]],
    #     )
    #     fff = FFF(dict(), dict_eq, spec, "f_dynamic")
    #     fun = make_method_from_factory(fff, compile=False, debug=False)
    #     self._dynamic = fun

    def describe(self: Self) -> str:
        """Returns a string representation of the model's symbols"""
        return f"""<h3>Model</h3>
<ul>
<li>name: {self.name if self.name is not None else "Unnamed"}</li>
<li>symbols:
    <ul>
        <li>variables:
            <ul>
                <li>endogenous: {str.join(",",self.symbols["endogenous"])}</li>
                <li>exogenous: {str.join(",",self.symbols["exogenous"])}</li>
            </ul>
        </li>
        <li> parameters: {str.join(",",self.symbols["parameters"])}</li>
    </ul>
</li>
</ul>
"""

    def _repr_html_(self):
        # from IPython.display import display, Markdown, HTML

        txt = self.describe()
        return txt

    def residuals(self):

        v = self.symbols["endogenous"]
        p = self.symbols["parameters"]

        p = [c[e] for e in p]
        y,e = self.steady_state
        return self.dynamic(y,y,y,e,p)
    
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

        # TODO add support for Deterministic
        assert isinstance(self.processes, Normal) or isinstance(
            self.processes, ProductNormal
        )

        Σ = self.processes.Σ

        c = self.get_calibration(**calibration)

        # a bit stupid

        endogenous_values = [c[e] for e in v]

        # Reshapes necessary for static type checking
        y0 = np.reshape(endogenous_values, len(endogenous_values))

        return RecursiveSolution(
            X, Y, Σ, {"endogenous": v, "exogenous": e}, evs=evs, x0=y0, model=self
        )

