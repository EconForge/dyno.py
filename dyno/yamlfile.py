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

class UnsupportedDynareFeature(Exception):

    pass


def get_allowed_functions():
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
    return safe_dict


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
        safe_dict = get_allowed_functions()
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
