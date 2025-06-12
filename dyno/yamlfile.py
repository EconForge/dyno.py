import dolang
from dolang import stringify
from dolang.function_compiler import FlatFunctionFactory as FFF
from dolang.symbolic import str_expression, stringify_symbol
from dolang.function_compiler import make_method_from_factory

from dyno.model import Model
import yaml
from dolang.symbolic import sanitize, parse_string, str_expression
from dolang.language import eval_data
from dolang.symbolic import str_expression


class YAMLFile(Model):

    def __init__(self, data):

        self.data = data
        self.__update_equations__()
        self.__update_calibration__()
        self.exogenous = self.__get_exogenous__()

    def __update_calibration__(self):

        from dolang.symbolic import remove_timing

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

        self.__calibration__ = calibration

    def get_calibration(self, **kwargs):

        import copy
        from dolang.triangular_solver import solve_triangular_system

        calibration = copy.copy(self.__calibration__)
        calibration.update(**kwargs)

        return solve_triangular_system(calibration)

    #     self.__calibration__ =  solve_triangular_system(calibration)

    # return self.__calibration__

    def __get_exogenous__(self):

        from .language import ProductNormal

        if "exogenous" not in self.data:
            return {}

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

        return ProductNormal(*exogenous.values())

    def __update_equations__(self):

        data = self.data

        tree = dolang.parse_string(data["equations"], start="equation_block")

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

        n = len(tree.children)

        # equations = [f"({stringify(eq.children[1])})-({stringify(eq.children[0])})"  for eq in tree.children]
        equations = [stringify(str_expression(eq)) for eq in tree.children]

        self.equations = equations

        equations = [
            ("({1})-({0})".format(*eq.split("=")) if "=" in eq else eq)
            for eq in equations
        ]

        self.equations = equations

        dict_eq = dict([(f"out{i+1}", equations[i]) for i in range(n)])
        spec = dict(
            y_f=[stringify_symbol((e, 1)) for e in symbols["endogenous"]],
            y_0=[stringify_symbol((e, 0)) for e in symbols["endogenous"]],
            y_p=[stringify_symbol((e, -1)) for e in symbols["endogenous"]],
            e=[stringify_symbol((e, 0)) for e in symbols["exogenous"]],
            p=[stringify_symbol(e) for e in symbols["parameters"]],
        )

        fff = FFF(dict(), dict_eq, spec, "f_dynamic")

        fun = make_method_from_factory(fff, compile=False, debug=False)

        self.__functions__ = {"dynamic": fun}


cache: list[tuple[int, int, YAMLFile]] = []


def import_file(filename) -> YAMLFile:

    txt = open(filename, "rt", encoding="utf-8").read()
    return import_model(txt)


def import_model(txt) -> YAMLFile:

    data = yaml.compose(txt)

    v = hash(txt)
    v_eq = hash(data["equations"].value)

    existing = [m[0] for m in cache]

    if v in existing:
        i = existing.index(v)
        model = cache[i][2]
        return model

    else:
        model = YAMLFile(data)
        cache.append((v, v_eq, model))
        return model
