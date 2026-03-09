from __future__ import annotations

from abc import ABC, abstractmethod
from math import nan
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.linalg import solve as linsolve
from typing_extensions import Self

from .errors import SteadyStateError
from .language import ProductNormal
from .model_render import (
    model_repr_data,
    render_model_html,
    render_model_markdown,
    render_model_text,
)

if TYPE_CHECKING:
    from .solver import PerturbationSolution
from .typedefs import IRFType, Solver, TVector, TMatrix


class AbstractModel(ABC):
    """Abstract class representing an economic model."""

    name: str | None
    filename: str
    context: dict[str, Any]
    symbols: dict[str, list[str]]
    processes: ProductNormal | None
    paths: dict[str, dict[int, float]] | None
    symbolic: Any
    __steady_state__: dict[str, float] | None

    def __init__(
        self: Self,
        filename: str | None = None,
        txt: str | None = None,
        **kwargs: Any,
    ) -> None:
        match filename, txt:
            case (None, None):
                raise ValueError(
                    "Neither the file name nor content were passed to constructor. One of the two should be passed."
                )
            case (None, txt):
                assert txt is not None
                self.filename = "*anonymous*.dyno"
                self.import_model(txt, **kwargs)
            case (filename, None):
                assert filename is not None
                self.filename = filename
                self.import_file(filename, **kwargs)
            case _:
                assert filename is not None
                assert txt is not None
                self.filename = filename
                self.import_model(txt, **kwargs)

        self.__steady_state__ = None
        self._set_name()
        self._set_context()
        self._set_symbols()
        self._set_exogenous()

    def copy(self: Self):
        import copy

        return copy.deepcopy(self)

    def _repr_data(self: Self) -> dict[str, Any]:
        return model_repr_data(self)

    def _render_repr_text(self: Self, data: dict[str, Any]) -> str:
        return render_model_text(data)

    def _render_repr_html(self: Self, data: dict[str, Any]) -> str:
        return render_model_html(data)

    def __repr__(self: Self) -> str:
        return self._render_repr_text(self._repr_data())

    @property
    def data(self):
        """Backward-compatible alias for `symbolic`."""
        return self.symbolic

    @data.setter
    def data(self, value):
        self.symbolic = value

    @property
    def checks(self) -> dict[str, bool]:
        return {"deterministic": self.processes is None}

    @property
    def metadata(self) -> dict[str, Any]:
        """Convenience accessor for model metadata parsed from source files."""
        return self.context.get("metadata", {})

    @property
    def is_deterministic(self) -> bool:
        return self.processes is None

    @abstractmethod
    def import_model(self: Self, txt: str, **kwargs: Any) -> None:
        """Set `symbolic` attribute from model text description."""

    @abstractmethod
    def _set_context(self: Self) -> None: ...

    @abstractmethod
    def compute_residuals(self: Self, y2, y1, y0, e) -> TVector: ...

    @abstractmethod
    def compute_jacobians(
        self: Self, y2, y1, y0, e
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix]: ...

    def import_file(self: Self, filename: str, **kwargs: Any) -> None:
        txt = open(filename, "rt", encoding="utf-8").read()
        self.import_model(txt, **kwargs)

    def _set_name(self: Self) -> None:
        import os.path

        self.name = os.path.basename(self.filename).split(".")[0]

    def _set_symbols(self: Self) -> None:
        c = self.context

        # exogenous are either defined as processes or by specifying values
        # Preserve declaration/insertion order (avoid set() which scrambles order)
        exo_order: list[str] = []

        for exo_tuple in c.get("processes", {}).keys():
            for name in exo_tuple:
                if name not in exo_order:
                    exo_order.append(name)

        for name in c.get("values", {}).keys():
            if name not in exo_order:
                exo_order.append(name)

        exo_set = set(exo_order)

        declared_vars = list(c.get("variables", {}).keys())
        variables: list[str] = []
        for v in declared_vars:
            if v not in exo_set:
                variables.append(v)
        for v in declared_vars:
            if v in exo_set and v not in variables:
                variables.append(v)
        for v in exo_order:
            if v not in variables:
                variables.append(v)

        exogenous = [v for v in variables if v in exo_set]
        endogenous = [v for v in variables if v not in exogenous]

        # try:
        #     parameters = [*self.symbolic.parameters]
        # except Exception:
        parameters = [*c["constants"].keys()]

        self.symbols = {
            "variables": variables,
            "endogenous": endogenous,
            "exogenous": exogenous,
            "parameters": parameters,
        }

    def _set_exogenous(self: Self) -> None:
        pps = self.context["processes"].values()
        if len(pps) == 0:
            self.processes = None
        else:
            self.processes = ProductNormal(*[e for e in pps])

    @property
    def steady_state(self) -> dict[str, float]:
        if self.__steady_state__ is None:
            self.__steady_state__ = self.context["steady_states"]
        return self.__steady_state__

    @property
    def residuals(self):
        y, e = self.__steady_state_vectors__
        return self.compute_residuals(y, y, y, e)

    @property
    def jacobians(self):
        y, e = self.__steady_state_vectors__
        return self.compute_jacobians(y, y, y, e)

    @property
    def __steady_state_vectors__(self) -> tuple[list[float], list[float]]:
        c = self.context
        y = [c["steady_states"].get(name, nan) for name in self.symbols["endogenous"]]

        e: list[float] = []
        for name in self.symbols["exogenous"]:
            if name in c["steady_states"]:
                e.append(c["steady_states"][name])
            elif name in c["values"]:
                e.append(c["values"][name].get(0, 0.0))
            else:
                e.append(nan)
        return y, e

    def describe(self: Self) -> None:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print(Markdown(self._markdown_()))

    def _markdown_(self: Self) -> str:
        return render_model_markdown(self._repr_data(), self.filename)

    def _repr_html_(self: Self) -> str:
        return self._render_repr_html(self._repr_data())

    def solve(self: Self, **args: Any) -> "PerturbationSolution | pd.DataFrame":
        if self.is_deterministic:
            from .solver import deterministic_solve

            return deterministic_solve(self, **args)
        return self.perturb(**args)

    def perturb(self: Self, method: Solver = "qz") -> "PerturbationSolution":
        from .solver import PerturbationSolution, RecursiveDecisionRule
        from .solver import solve as solve_quadratic_matrix

        r, A, B, C, D = self.jacobians

        X, evs = solve_quadratic_matrix(A, B, C, method=method)
        Y = linsolve(A @ X + B, -D)

        v = self.symbols["endogenous"]
        e = self.symbols["exogenous"]

        assert self.processes is not None
        Σ = self.processes.Σ

        y, _ = self.__steady_state_vectors__
        y0 = np.reshape(y, len(y))

        dr = RecursiveDecisionRule(
            X,
            Y,
            Σ,
            {"endogenous": v, "exogenous": e},
            x0=y0,
            model=self,
        )
        return PerturbationSolution(dr, evs=evs)

    def deterministic_guess(self: Self, T: int | None = None):
        if T is None:
            T = int(self.context.get("constants", {}).get("T", 50))

        y, e = self.__steady_state_vectors__
        v0 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)

        for key, value in self.context.get("values", {}).items():
            i = self.symbols["variables"].index(key)
            v0[:, i] = 0.0
            for a, b in value.items():
                v0[a, i] = b

        return v0
