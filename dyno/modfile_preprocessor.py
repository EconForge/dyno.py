import dynare_preprocessor
import json
from dyno.model import Model, evaluate
from dyno.language import pad_list, Normal, Deterministic
from math import sqrt
import numpy as np

from typing_extensions import Self
from typing import Any

from dyno.util_json import json_safe_eval, UnsupportedDynareFeature, get_allowed_functions


class Modfile(Model):

    def import_model(self: Self, txt: str) -> None:
        """imports model written in `.mod` format into data attribute using Dynare's preprocessor

        Parameters
        ----------
        txt : str
            the model being imported in `.mod` form
        """
        prefixstr = "//-- BEGIN JSON --//"
        suffixstr = "//-- END JSON --// \nJSON written after Parsing step.\n"
        jsontxt = (
            dynare_preprocessor.preprocess(txt)
            .removeprefix(prefixstr)
            .removesuffix(suffixstr)
        )
        self.data = json.loads(jsontxt)

    def _set_symbols(self: Self) -> None:
        """sets symbols attribute of Model from json data"""
        self.symbols = {}
        for t in ["endogenous", "exogenous", "parameters"]:
            self.symbols[t] = [s["name"] for s in self.data["modfile"][t]]
        self.symbols["model_local_variables"] = [
            loc["variable"] for loc in self.data["modfile"]["model_local_variables"]
        ]

    def _set_equations(self: Self) -> None:
        """sets equations attribute of Model from json data"""

        self.equations = [
            f"{eq['lhs']} = {eq['rhs']}" for eq in self.data["modfile"]["model"]
        ]

    def _set_calibration(self: Self) -> None:
        """retrieves calibration values and exogenous variable definitions from json data"""

        self.calibration = {}
        calibration = self.calibration
        statements = self.data["modfile"]["statements"]

        def calibrate(name, value):
            calibration[name.strip()] = evaluate(value, calibration)

        for s in statements:
            match s["statementName"]:
                case "param_init":
                    calibrate(s["name"], s["value"])
                case "initval":
                    for v in s["vals"]:
                        calibrate(v["name"], v["value"])
                case "native":
                    try:
                        assignment = s["string"][:-1]  # remove semicolon
                        name, value = assignment.split("=")
                        calibrate(name.strip(), value.strip())
                    except:
                        pass  # ignore native statement if it is not a parameter definition
                case _:
                    pass

        self.locals = {}
        for loc in self.data["modfile"]["model_local_variables"]:
            var = loc["variable"]
            try:
                val = evaluate(loc["value"], calibration)
            except:
                continue  # TODO: Real evaluation and error checking
            calibration[var] = val
            self.locals[var] = val

        if "steady_state_model" in self.data.keys():
            for eq in self.data["steady_state_model"]["steady_state_model"]:
                calibrate(eq["lhs"], eq["rhs"])

        for k in self.variables:
            if k not in calibration.keys():
                calibration[k] = 0.0

        for k in self.parameters + list(self.locals.keys()):
            if k not in calibration.keys():
                calibration[k] = np.nan

    def _set_exogenous(self: Self) -> None:
        self.exogenous = None
        statements = self.data["modfile"]["statements"]
        c = self.get_calibration()

        deterministic_model = None
        varexo = self.symbols["exogenous"]
        n = len(varexo)

        # Deterministic case
        horizon = 0
        """time horizon over which perfect foresight simulation is done"""
        det_vals: dict[str, list[float]] = {v: [] for v in varexo}
        """det_vals[v][t] is the value of deterministic variable v in period t+1 (where periods start at 1)"""

        # Stochastic case
        var_index = {v: i for i, v in enumerate(varexo)}
        covar = np.zeros((n, n))
        """covar[i,j] is the covariance of ith and jth exogenous variables"""

        for s in statements:
            match s["statementName"]:
                case "shocks":
                    if s["overwrite"]:
                        deterministic_model = None
                        det_vals = {v: [] for v in varexo}
                        covar = np.zeros((n, n))
                    deterministic_shock = "deterministic_shocks" in s.keys()
                    if deterministic_model is None:
                        deterministic_model = deterministic_shock
                    if deterministic_model and deterministic_shock:
                        # perfect foresight model
                        for shock in s["deterministic_shocks"]:
                            v = shock["var"]
                            # preprocessor ensures no overlap between seasons for the same variable
                            for season in shock["values"]:
                                val = evaluate(season["value"], c)
                                p1 = int(season["period1"])
                                p2 = int(season["period2"])
                                pad_list(det_vals[v], p2)
                                det_vals[v][p1 - 1 : p2] = [val] * (p2 - p1 + 1)
                    elif (not deterministic_model) and (not deterministic_shock):
                        # dgse model
                        # preprocessor ensures no overlap between cases below
                        for v in s["variance"]:
                            i = var_index[v["name"]]
                            covar[i, i] = evaluate(v["variance"], c)
                        for v in s["stderr"]:
                            i = var_index[v["name"]]
                            covar[i, i] = evaluate(v["stderr"], c) ** 2
                        for couple in s["covariance"]:
                            i = var_index[couple["name"]]
                            j = var_index[couple["name2"]]
                            covar[i, j] = evaluate(couple["covariance"], c)
                            covar[j, i] = covar[i, j]
                        for couple in s["correlation"]:
                            i = var_index[couple["name"]]
                            j = var_index[couple["name2"]]
                            std_i = sqrt(covar[i, i])
                            std_j = sqrt(covar[j, j])
                            covar[i, j] = (
                                evaluate(couple["correlation"], c) * std_i * std_j
                            )
                            covar[j, i] = covar[i, j]
                    else:
                        raise UnsupportedDynareFeature(
                            "Mixing deterministic and stochastic exogenous variables is not supported (yet)."
                        )
                case "perfect_foresight_setup":
                    try:
                        horizon = s["options"]["periods"]
                    except:
                        pass
        if deterministic_model is None:
            return  # No temporary shocks were defined, perhaps permanent ones were in initval

        if deterministic_model:
            self.exogenous = Deterministic(horizon, det_vals)
        else:
            self.exogenous = Normal(Σ=covar)

    def _set_dynamic(self: Self) -> None:
        pass

    def _dynamic(self, y0, y1, y2, e, p, r) -> None:
        endo = self.symbols["endogenous"]
        exo = self.symbols["exogenous"]
        params = self.symbols["parameters"]
        local_vars = self.symbols["model_local_variables"]
        context = {
            "endogenous_present": {v: y1[i] for i, v in enumerate(endo)},
            "endogenous_future": {v: y0[i] for i, v in enumerate(endo)},
            "endogenous_past": {v: y2[i] for i, v in enumerate(endo)},
            "exogenous": {v: e[i] for i, v in enumerate(exo)},
            "parameters": {v: p[i] for i, v in enumerate(params)},
            "local_variables": self.locals,
        }
        ast = self.data["modfile"]["abstract_syntax_tree"]
        assert r.shape[0] == len(ast)
        for i, eq in enumerate(ast):
            r[i] = json_safe_eval(eq["AST"], context)
