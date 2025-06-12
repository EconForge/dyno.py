import os
import dolang
from lark import Token, Lark
import numpy as np


def ast_to_yaml(node, indent=""):
    indent += "  "
    if isinstance(node, Token):
        yield f"{indent}- type: {node.type}"
        yield f"{indent}  value: {repr(node.value)}"
    else:
        yield f"{indent}- type: {node.data}"
        yield f"{indent}  children:"
        for child in node.children:
            yield from ast_to_yaml(child, indent)


dir_path = os.path.dirname(os.path.realpath(__file__))

modfile_grammar = open(f"{dir_path}/modfile_grammar.lark").read()
modfile_parser = Lark(modfile_grammar, propagate_positions=True)


class UnsupportedDynareFeature(Exception):

    pass


from dyno.model import Model
from lark import Visitor


class CheckFunCalls(Visitor):

    def call(self, tree):

        funname = tree.children[0].value
        if funname == "steady_state":
            raise UnsupportedDynareFeature(
                "Calling 'steady_state' value within model is not supported (yet)."
            )
        else:
            accepted = ["sin", "cos", "exp", "log"]
            if funname not in accepted:
                raise UnsupportedDynareFeature(
                    f"Calling external function `{funname}` is not allowed."
                )


class Modfile(Model):

    def __init__(self, filename=None, txt=None):

        if txt is not None:
            if filename is None:
                filename = "<string>.mod"
        self.filename = filename

        if txt is None:
            txt = open(filename).read()

        try:
            self.data = modfile_parser.parse(txt)
        except Exception as e:
            raise e

        yml_ = ast_to_yaml(self.data)
        with open("temp2.yml", "w") as f:
            for g in yml_:
                f.write(g)
                f.write("\n")

        self.__check_supported__()

        self.symbols = self.__find_symbols__()
        self.calibration = self.__get_calibration__()
        self.exogenous = self.__find_sigma__()

        self.__update_equations__()

    def __check_supported__(self):

        # for ch in self.data.children:

        #     if ch.data.value == "steady_block":
        #         raise UnsupportedDynareFeature("'steady_state_model' block is not supported yet.")

        CheckFunCalls().visit(self.data)

    @property
    def variables(self):
        return self.symbols["endogenous"] + self.symbols["exogenous"]

    @property
    def parameters(self):
        return self.symbols["parameters"]

    def __find_sigma__(self):

        import numpy as np
        from dyno.language import Normal

        ne = len(self.symbols["exogenous"])

        Sigma = np.zeros((ne, ne))

        for l in self.data.children:

            if l.data.value == "shocks_block":

                for ch in l.children:

                    if (
                        ch.data.value == "setstdvar_stmt"
                        or ch.data.value == "setvar_stmt"
                    ):

                        k = ch.children[0].children[0].value
                        ve = ch.children[1]  # .value

                        if isinstance(ve, str):
                            v = ve
                        else:
                            v = dolang.str_expression(ve)

                        from math import exp

                        context = {"exp": exp}
                        cc = self.calibration.copy()
                        vv = eval(v.replace("^", "**"), context, cc)
                        i = self.symbols["exogenous"].index(k)
                        if ch.data.value == "setstdvar_stmt":
                            Sigma[i, i] = vv**2
                        else:
                            Sigma[i, i] = vv

                    elif ch.data.value == "setcovar_stmt":

                        k = ch.children[0].children[0].value
                        l = ch.children[1].children[0].value
                        ve = ch.children[2]

                        if isinstance(ve, str):
                            v = ve
                        else:
                            v = dolang.str_expression(ve)

                        context = {"exp": exp}
                        cc = self.calibration.copy()

                        vv = eval(v.replace("^", "**"), context, cc)

                        i = self.symbols["exogenous"].index(k)
                        j = self.symbols["exogenous"].index(l)

                        Sigma[i, j] = vv
                        Sigma[j, i] = vv

        return Normal(Σ=Sigma)

    def __get_calibration__(self):

        calibration = {}
        for l in self.data.children:

            if l.data.value == "parassignment":

                k = l.children[0].children[0].value
                ve = l.children[1]

                v = dolang.str_expression(ve)
                try:
                    vv = eval(v.replace("^", "**"))
                except:
                    vv = v

                calibration[k] = vv

            elif l.data.value == "initval_block":
                for ll in l.children:
                    k = ll.children[0].children[0].value
                    ve = ll.children[1]

                    v = dolang.str_expression(ve)
                    try:
                        vv = eval(v.replace("^", "**")) # code injection risk?
                    except:
                        vv = v
                    calibration[k] = vv

            elif l.data.value == "steady_block":
                for ll in l.children:
                    k = ll.children[0].children[0].value
                    ve = ll.children[1]

                    v = dolang.str_expression(ve)
                    try:
                        vv = eval(v.replace("^", "**"))
                    except:
                        vv = v
                    calibration[k] = vv

        for k in self.symbols["endogenous"] + self.symbols["exogenous"]:
            if k not in calibration.keys():
                calibration[k] = 0.0

        for k in self.symbols["parameters"]:
            if k not in calibration.keys():
                calibration[k] = np.nan

        return calibration

    def get_calibration(self, **kwargs):

        import copy
        from dolang.triangular_solver import solve_triangular_system

        calibration = self.__get_calibration__()
        calibration.update(**kwargs)

        return solve_triangular_system(calibration)
        # return self.__get_calibration__()

    def __find_symbols__(self):

        # so far we discard latex and names
        get_name = lambda x: x.children[0].children[0].value

        dfs = []
        for l in self.data.children:
            if l.data.value == "var_statement":
                for tp in l.children:
                    name = get_name(tp)
                    dfs.append((name, "endogenous"))
            elif l.data.value == "varexo_statement":
                for tp in l.children:
                    name = get_name(tp)
                    dfs.append((name, "exogenous"))
            elif l.data.value == "par_statement":
                for tp in l.children:
                    name = get_name(tp)
                    dfs.append((name, "parameters"))

        symbols = {
            "endogenous": tuple(e[0] for e in dfs if e[1] == "endogenous"),
            "exogenous": tuple(e[0] for e in dfs if e[1] == "exogenous"),
            "parameters": tuple(e[0] for e in dfs if e[1] == "parameters"),
        }

        return symbols

    def __update_equations__(self):

        mod = self
        variables = mod.variables

        mm = [e for e in mod.data.children if e.data == "model_block"]
        assert len(mm) == 1

        from dolang.grammar import sanitize, str_expression, stringify, stringify_symbol
        from dolang.function_compiler import FlatFunctionFactory as FFF
        from dolang.function_compiler import make_method_from_factory

        symbols = self.symbols
        variables = self.variables

        equations = []
        for ll in mm[0].children:
            eq_tree = ll.children[-1]
            eq = sanitize(eq_tree, variables=variables)
            eq = stringify(eq)
            eq = str_expression(eq)

            if "=" in eq:
                lhs, rhs = eq.split("=")
                eq = f"({rhs}) - ({lhs})"

            equations.append(eq)

        n = len(equations)

        dict_eq = {f"out{i+1}": equations[i] for i in range(n)}

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
