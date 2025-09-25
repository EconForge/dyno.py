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

class DynoModel(ABC):
    """Abstract class representing an economic model"""

    name: str | None
    """Name of the model if described in input"""

    symbols: dict[str, list[str]]
    """Symbols dictionary, allowed keys are 'endogenous', 'exogenous' and 'parameters'"""

    equations: list[str]
    """List of the equations of the model written in the form LHS = RHS"""

    processes: ProductNormal | None

    paths: dict[str, dict[int, float]] | None
    
    data: Any
    """Format-dependant internal representation of the data"""

    def __init__(
        self: Self, filename: str = None, txt: str | None = None
    , **kwargs) -> None:
        match filename, txt:
            case (None, None):
                raise ValueError(
                    "Neither the file name nor content were passed to constructor. One of the two should be passed."
                )
            case (None, txt): # This can't happen because of the default value
                assert txt is not None
                filename = "<anonymous>.dyno"
                self.filename = filename
                self.import_model(txt,**kwargs)
            case (filename, None):
                assert filename is not None  # to reassure Mypy
                self.filename = filename
                self.import_file(filename,**kwargs)
            case _:
                self.filename = filename
                self.import_model(txt,**kwargs)
                # raise ValueError(
                #     "File name and content were both passed to constructor. Only one of the two should be passed."
                # )

        self.__steady_state__ = None
        self._set_name()
        self._set_context()
        self._set_symbols()
        self._set_exogenous()
        # self._set_dynamic()

    @property
    def checks(self):
        return {
            "deterministic": self.processes is None
        }

    @abstractmethod
    def import_model(self: Self, txt: str) -> None:
        """sets data attribute from model text description"""
        pass

    def import_file(self: Self, filename: str, **kwargs) -> None:
        """sets data attribute from file"""
        txt = open(filename, "rt", encoding="utf-8").read()
        assert txt is not None
        return self.import_model(txt, **kwargs)


    def _set_name(self: Self) -> None:
        # should be overridden for file types with name information
        import os.path
        filename = self.filename
        self.name = os.path.basename(filename).split(".")[0]

    def _set_symbols(self: Self) -> None:
        
        c = self.context

        exo = set(sum(c['processes'].keys(), ()))
        variables = [*c['variables'].keys()]

        # this is needed because in dyno files
        # some exogenous variables not appearing in equations
        # are not declared as variables
        variables = variables + [e for e in exo if e not in variables]
                
        exogenous = [v for v in variables if v in exo]
        endogenous = [v for v in variables if v not in exogenous]

        try:
            # in mofdiles parameters and constants can differ
            parameters = [*self.data.parameters]
        except:
            parameters = [*c['constants'].keys()]

        self.symbols = {
            "variables": variables,
            "endogenous": endogenous,
            "exogenous": exogenous,
            "parameters": parameters,
        }

    def _set_exogenous(self):

        from dyno.language import ProductNormal, Normal

        pps = self.context['processes'].values()
        if len(pps)==0:
            self.processes = None
        else:
            self.processes = ProductNormal(
                *[e for e in pps]
            )


    @property
    def steady_state(self):
        if self.__steady_state__ is None:
            self.__steady_state__ = self.context['steady_states']
        return self.__steady_state__
        
    
    @property
    def residuals(self):
        y,e = self.__steady_state_vectors__
        return self.compute_residuals(y,y,y,e)

    @property
    def jacobians(self):
        
        y,e = self.__steady_state_vectors__
        return self.compute_jacobians(y,y,y,e)


#     # def derivatives(order=1):
#     #     """Computes the derivatives of the model up to the specified order at the steady-state. Caches the result."""
#     #     raise Exception("Not implemented yet.")

    @property
    def __steady_state_vectors__(self):

        # it would be nice to do steady_state("groups") to get groups of variables
        # or steady_state.groups("all") to get all variables
        c = self.steady_state
        from math import nan
        y = [c.get(name,nan) for name in self.symbols["endogenous"]]
        e = [c.get(name,nan) for name in self.symbols["exogenous"]]
        return y,e
        # try:
        # except KeyError as e:
        #     raise Exception("Steady state not fully specified: missing value for `{str(e)}`")


    def _repr_html_(self):
        # from IPython.display import display, Markdown, HTML
        """Returns a string representation of the model's symbols"""
        print(self.symbols)
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
        <li> constants: {str.join(",",self.symbols["parameters"])}</li>
    </ul>
</li>
</ul>
"""


    def solve(
        self: Self, method: Solver = "qz"
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

        r, A, B, C, D = self.jacobians

        X, evs = solveit(A, B, C, method=method)
        Y = linsolve(A @ X + B, -D)

        v = self.symbols["endogenous"]
        e = self.symbols["exogenous"]

        Σ = self.processes.Σ

        y,e = self.__steady_state_vectors__

        # Reshapes necessary for static type checking
        y0 = np.reshape(y, len(y))

        return RecursiveSolution(
            X, Y, Σ, {"endogenous": v, "exogenous": e}, evs=evs, x0=y0, model=self
        )



    def deterministic_guess(model, T=None):

        if T is None:
            T = model.calibration.get('T', 50)

        y,e = model.steady_state

        # initial guess
        v0 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)


        # works if the is one and exactly one exogenous variable?
        # does it?
        for key,value in model.data.evaluator.values.items():
            i = model.symbols['variables'].index(key)
            for a,b in value.items():
                v0[a,i] = b

        return v0