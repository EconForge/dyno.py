import dynare_preprocessor
import json
from dyno.model import Model

from typing_extensions import Self


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

    def _set_equations(self: Self) -> None:
        """sets equations attribute of Model from json data"""
        self.equations = [
            (eq["lhs"], eq["rhs"]) for eq in self.data["modfile"]["model"]
        ]  # Maybe add line numbers too

    def _set_calibration(self: Self) -> None:
        """retrieves calibration values and exogenous variable definitions from json data"""
        self.calibration = {}
        calibration = self.calibration
        statements = self.data["modfile"]["statements"]
        ss_model = self.data["steady_state_model"]

        def calibrate(name, value):
            from dyno.model import evaluate

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
                        calibrate(name, value)
                    except:
                        pass  # ignore native statement if it is not a parameter definition
                case _:
                    pass

    def _set_exogenous(self: Self) -> None:
        pass


import time

t1 = time.time()
model = Modfile(filename="examples/example3.mod")
t2 = time.time()

print("Elapsed : ", t2 - t1)
