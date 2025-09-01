import dolang

from dyno.model import Model
import yaml
from dolang.symbolic import parse_string, str_expression, stringify_symbol
from dolang.language import eval_data

from typing_extensions import Self


class YAMLFile(Model):

    def import_model(self: Self, txt: str) -> None:
        self.data = yaml.compose(txt)

    def _set_name(self: Self):
        node = self.data.value[0]
        nodetype = node[0].value
        if nodetype == "name":
            self.name = node[1].value

    def _set_calibration(self: Self) -> None:

        from dolang.symbolic import remove_timing

        from .model import evaluate

        symbols = self.symbols
        calibration = dict()
        for k, v in self.data.get("calibration", {}).items():
            if v.tag == "tag:yaml.org,2002:str":

                expr = parse_string(v)
                expr = remove_timing(expr)
                expr = str_expression(expr)
            else:
                expr = float(v.value)
            kk = remove_timing(parse_string(k))
            kk = str_expression(kk)

            calibration[kk] = expr

        initial_values = {
            "exogenous": float("nan"),
            "endogenous": float("nan"),
            "parameters": float("nan"),
        }

        for symbol_group in symbols:
            if symbol_group in initial_values:
                default = initial_values[symbol_group]
            else:
                default = float("nan")
            for s in symbols[symbol_group]:
                if s not in calibration:
                    calibration[s] = default

        self.calibration = calibration

    def get_calibration(self, **kwargs):

        from dolang.triangular_solver import solve_triangular_system

        calibration = super().get_calibration(**kwargs)

        return solve_triangular_system(calibration)

    #     self.calibration =  solve_triangular_system(calibration)

    # return self.calibration

    def _set_exogenous(self):

        from .language import ProductNormal

        if "exogenous" not in self.data:
            self.exogenous = None

        exo = self.data["exogenous"]
        calibration = self.get_calibration()
        from dolang.language import eval_data

        exogenous = eval_data(exo, calibration)

        # new style
        syms = self.symbols["exogenous"]
        # first we check that shocks are defined in the right order
        ssyms = []
        for k in exo.keys():
            vars = [v.strip() for v in k.split(",")]
            ssyms.append(vars)
        ssyms = tuple(sum(ssyms, []))

        self.exogenous = ProductNormal(*exogenous.values())

    def _set_symbols(self: Self) -> None:
        data = self.data

        self._tree = dolang.parse_string(data["equations"], start="equation_block")

        tree = self._tree

        stree = dolang.grammar.sanitize(tree)

        symlist = dolang.list_symbols(stree)

        vars = list(set(e[0] for e in symlist.variables))
        pars = symlist.parameters

        # check exogenous variables
        try:
            l = [
                [h.strip() for h in k.split(",")] for k in self.data["exogenous"].keys()
            ]
            exovars = sum(l, [])
            #
            #  exovars = self.data['exogenous'].keys()
        except:
            exovars = []

        symbols = {
            "endogenous": [e for e in vars if e not in exovars],
            "parameters": pars,
            "exogenous": exovars,
        }

        self.symbols = symbols

    def _set_equations(self: Self):
        # equations = [f"({stringify(eq.children[1])})-({stringify(eq.children[0])})"  for eq in tree.children]
        self.equations = [str_expression(eq) for eq in self._tree.children]
