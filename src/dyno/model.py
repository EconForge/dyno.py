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

import plotly.express as px


class RecursiveSolution:
    """VAR(1) representing a linearized model

    Attributes
    ----------
    X, Y, Σ: (N,N) Matrix
        parameters of the stationary VAR process $y_t = Xy_{t-1} + Yε_t$, where Σ is the covariance matrix of $ε_t$

    symbols: dict[str, list[str]]
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
        symbols: dict[str, list[str]],
        x0: TVector | None = None,
        evs: TVector | None = None,
        model=None,
    ) -> None:

        self.x0 = x0
        self.X = X
        self.Y = Y
        self.Σ = Σ

        self.evs = evs

        self.symbols = symbols
        self._model = model

    def _repr_html_(self):
        evv = pd.DataFrame(
            [np.abs(self.evs)],
            columns=[i + 1 for i in range(len(self.evs))],
            index=["λ"],
        )
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
        df.index = ["{}[t]".format(e) for e in self.symbols["endogenous"]]

        Σ0, Σ = moments(self.X, self.Y, self.Σ)

        df_cmoments = pd.DataFrame(
            Σ0,
            columns=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
            index=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
        )

        df_umoments = pd.DataFrame(
            Σ,
            columns=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
            index=["{}[t]".format(e) for e in (self.symbols["endogenous"])],
        )

        sim = irfs(self._model, self, type="log-deviation")
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

        html = f"""
        <h3>Eigenvalues</h3>
        {evv.to_html()}
        <h3>Decision Rule</h3>
        <h4>Steady-state</h4>
        {ss.to_html(index=False)}
        <h4>Jacobian</h4>
        {df.to_html()}
        <h3>Moments</h3>
        <h4>Unconditional moments</h4>
        {df_umoments.to_html()}
        <h4>Conditional moments</h4>
        {df_cmoments.to_html()}
        <h3>IRFs</h3>
        {fig.to_html(full_html=False, include_plotlyjs=False)}
        """
        return html


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

    exogenous: Exogenous | None
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

    @property
    def variables(self):
        return self.symbols["endogenous"] + self.symbols["exogenous"]

    @property
    def parameters(self):
        return self.symbols["parameters"]

    def _set_dynamic(self: Self) -> None:
        """generates dynamic method from the equations of the model using Dolang"""
        from dolang import stringify

        str_equations = [stringify(eq) for eq in self.equations]

        equations = []
        for streq in str_equations:
            lst = streq.split("=")

            match len(lst):
                case 1:
                    eq = streq.strip()
                case 2:
                    eq = f"({lst[0].strip()}) - ({lst[1].strip()})"
                case _:
                    raise ValueError("More than one equation on the same line")

            equations.append(eq)

        dict_eq = {f"out{i+1}": eq for i, eq in enumerate(equations)}
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


def irfs(
    model: Model, dr: RecursiveSolution, type: IRFType = "log-deviation"
) -> dict[str, pd.DataFrame]:
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


def sim_to_nsim(irfs):

    pdf = pd.concat(irfs).reset_index()
    ppdf = pdf.rename(columns={"level_0": "shock", "level_1": "t"})

    ppdf = ppdf.melt(id_vars=["shock", "t"])

    return ppdf
