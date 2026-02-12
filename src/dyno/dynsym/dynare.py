import os
from lark import Token, Lark
import numpy as np
from typing_extensions import Self
from ..typedefs import TVector, TMatrix, IRFType, Solver, DynamicFunction
import math
from dyno.language import Normal

dir_path = os.path.dirname(os.path.realpath(__file__))

modfile_grammar = open(f"{dir_path}/grammars/modfile_grammar.lark").read()

from lark import Tree, Token
from lark.visitors import Transformer


# from dyno.util_json import UnsupportedDynareFeature

# from dyno.model import Model

from lark import Tree, Lark
from lark.visitors import Transformer, v_args

from dyno.dynsym.analyze import (
    FormulaEvaluator,
    AssignmentEvaluator,
    EquationsEvaluator,
)


@v_args(tree=True)
class ModFileTransformer(Transformer):

    def __init__(self):

        self.variables = []
        self.parameters = []
        self.variables_exo = []
        self.variables_pred = []
        self.equations = []

        super().__init__()

    def var_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            if name not in self.variables:
                self.variables.append(name)
        return tree

    def varexo_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            if name not in self.variables_exo:
                self.variables_exo.append(name)
            if name not in self.variables:
                self.variables.append(name)
        return tree

    def par_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            if name not in self.parameters:
                self.parameters.append(name)
        return tree

    def pred_statement(self, tree):
        for child in tree.children:
            name = str(child.children[0].children[0])
            if name not in self.variables_pred:
                self.variables_pred.append(name)
        return tree

    def constant(self, tree):
        name = str(tree.children[0].children[0])

        if name in self.variables + self.variables_exo + self.variables_pred:
            return Tree(
                "variable",
                [Tree("name", [name]), Tree("index", ["t"]), Tree("shift", ["0"])],
            )
        return tree

    def variable(self, tree):

        return Tree(
            "variable", [tree.children[0], Tree("index", ["t"]), tree.children[1]]
        )

    def call(self, tree):

        return Tree("call", [Tree("name", [str(tree.children[0])]), *tree.children[1:]])


class InterpretModfile(AssignmentEvaluator, EquationsEvaluator):

    def __init__(self, **calibration):

        super().__init__()

        self.steady_state = True  # variables are evaluatated at their steady state
        self.steady_states = {}
        self.constants = {}
        self.covariances = {}
        self.current_block = None
        self.unknown_as_nan = True

        self.calibration = calibration

    def var_statement(self, tree):
        return tree

    def varexo_statement(self, tree):
        return tree

    def par_statement(self, tree):
        return tree

    def pred_statement(self, tree):
        return tree

    def initval_block(self, tree):

        self.current_block = "initval"
        res = [self.visit(ch) for ch in tree.children]
        self.current_block = None

    def steady_block(self, tree):

        self.current_block = "steady_block"
        res = [self.visit(ch) for ch in tree.children]
        self.current_block = None

    def parassignment(self, tree):
        name = str(tree.children[0].children[0].children[0])
        formula = tree.children[1]
        value = self.visit(formula)
        if self.current_block is None:
            if name in self.calibration:
                # override value in the model with calibraiton value
                print("overriding parameter value:", name, "=", self.calibration[name])
                value = self.calibration[name]
            self.constants[name] = value
        elif self.current_block == "initval":
            self.steady_states[name] = value
        elif self.current_block == "steady_block":
            self.steady_states[name] = value

    def setvar_stmt(self, tree):
        name = str(tree.children[0].children[0].children[0])
        formula = tree.children[1]
        value = self.visit(formula)
        self.covariances[(name, name)] = value
        return tree

    def setstdvar_stmt(self, tree):
        name = str(tree.children[0].children[0].children[0])
        formula = tree.children[1]
        value = self.visit(formula)
        self.covariances[(name, name)] = value**2
        return tree

    # Function calls
    def call(self, tree):
        """Handle function calls: func_name(arg)"""
        func_node = tree.children[0]
        # Depending on the transformer, the function name can arrive as a Tree('name', [...])
        try:
            if getattr(func_node, "data", None) == "name" and getattr(
                func_node, "children", None
            ):
                func_name = str(func_node.children[0])
            else:
                func_name = str(func_node)
        except Exception:
            func_name = str(func_node)

        if func_name in self.function_table:
            args = [self.visit(c) for c in tree.children[1:]]
            return self.function_table[func_name](*args)
        elif func_name == "steady_state":
            assert tree.children[1].data == "variable"
            name = str(tree.children[1].children[0].children[0])
            return self.steady_states.get(name, math.nan)
        else:
            print(func_name)
            print(self.function_table)
            raise ValueError(f"Undefined function: {func_name}")

    def lequation(self, tree):
        self.equations.append(tree.children[1])
        # val =  self.visit(tree.children[1])
        # return val

    def equality(self, tree):

        # maybe not exactly conform
        a = self.visit(tree.children[0])
        b = self.visit(tree.children[1])
        return b - a


# from .errors import LARKParserError, ParserError
# from lark.exceptions import UnexpectedInput


# class DynareModel(Model):

#     def import_model(self, txt):

#         trans = ModFileTransformer()
#         parser = Lark(
#             modfile_grammar,
#             propagate_positions=True,
#             parser="lalr",
#             transformer=trans,
#             cache=True,
#             # strict=True,
#         )
#         try:
#             tree = parser.parse(txt)
#         except UnexpectedInput as e:
#             raise LARKParserError(e, txt) from e

#         variables = trans.variables
#         variables_exo = trans.variables_exo
#         parameters = trans.parameters
#         # variables_pred = trans.variables_pred

#         self.symbols = {
#             "variables": variables + variables_exo,
#             "endogenous": variables,
#             "exogenous": variables_exo,
#             "parameters": parameters,
#         }

#         fe = InterpretModfile(steady_state=True)
#         fe.visit(tree)

#         # # count variable in equations and compute residuals
#         # fe.steady_state = True
#         # self.residuals = [fe.visit(eq) for eq in fe.equations]
#         # fe.steady_state = False
#         # self.evaluator = fe

#     # def _check_supported(self):
#     #     CheckFunCalls().visit(self.data)

#     def _set_exogenous(self):

#         n_e = len(self.symbols["exogenous"])
#         mat = np.zeros((n_e, n_e))
#         for ind, k in self.evaluator.covariances.items():
#             i = self.symbols["exogenous"].index(ind[0])
#             j = self.symbols["exogenous"].index(ind[1])
#             mat[i, j] = k
#             mat[j, i] = k
#         # self.covariances = mat
#         self.processes = Normal(Σ=mat)

#     def _set_calibration(self):

#         self.constants = self.evaluator.constants.copy()
#         self.steady_states = self.evaluator.steady_states.copy()
#         for e in self.symbols["exogenous"]:
#             if e not in self.steady_states:
#                 self.steady_states[e] = 0.0
#         self.calibration = self.constants | self.steady_states

#     def _set_symbols(self):

#         pass

#     def _set_equations(self):

#         pass

#     def latex_equations(self):

#         from dyno.dynsym.latex import latex

#         eqs_str = [latex(eq) for eq in self.evaluator.equations]
#         latex_str = str.join("\n", ["$${}$$".format(eq) for eq in eqs_str])
#         return latex_str

#     @property
#     def steady_state(self):

#         endogenous = self.symbols["endogenous"]
#         exogenous = self.symbols["exogenous"]
#         fe = self.evaluator

#         y = [self.steady_states[name] for name in (endogenous)]
#         e = [self.steady_states[name] for name in (exogenous)]
#         return y, e

#     def compute_derivatives(self, y2, y1, y0, e):

#         import numpy as np
#         from dyno.dynsym.autodiff import DNumber as DN

#         fe = self.evaluator
#         endogenous = self.symbols["endogenous"]
#         exogenous = self.symbols["exogenous"]

#         for i, name in enumerate(endogenous):
#             fe.variables[name] = {
#                 -1: DN(y0[i], {(name, -1): 1}),
#                 0: DN(y1[i], {(name, 0): 1}),
#                 1: DN(y2[i], {(name, 1): 1}),
#             }
#         for i, name in enumerate(exogenous):
#             fe.variables[name] = {0: DN(e[i], {(name, 0): 1})}

#         results = [fe.visit(eq) for eq in fe.equations]

#         neq = len(results)
#         nv = len(endogenous)
#         ne = len(exogenous)

#         r = np.array([el.value for el in results])
#         A = np.zeros((neq, nv))
#         B = np.zeros((neq, nv))
#         C = np.zeros((neq, nv))
#         J = [A, B, C]
#         D = np.zeros((neq, ne))

#         for n, eq in enumerate(results):
#             for (name, shift), v in eq.derivatives.items():
#                 if name in endogenous:
#                     i = endogenous.index(name)
#                     J[1 - shift][n, i] = v
#                 elif name in exogenous:
#                     i = exogenous.index(name)
#                     D[n, i] = v

#         return r, A, B, C, D

#     def compute(
#         self: Self, calibration: dict[str, float] = {}, diff: bool = False
#     ) -> TVector | tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
#         """Computes the dynamic function's value based on calibration state and parameters

#         Parameters
#         ----------
#         calibration : dict[str, float], optional
#             dictionary containing the value of each parameter and variable of the model, indexed by their symbols, by default {}
#         diff : bool, optional
#             if set to True returns the dynamic function's partial derivatives as well, by default False

#         Returns
#         -------
#         TVector|tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]
#             value of the dynamic function at the state described by calibration, as well as its partial derivatives if diff is set to True
#         """

#         if not diff:
#             return np.array(self.residuals)

#         from dyno.dynsym.analyze import DN

#         assert len(calibration) == 0, "calibration not supported yet"

#         ys, es = self.steady_state

#         return self.compute_derivatives(ys, ys, ys, es)

#         import copy

#     def deterministic_guess(model, T=None):

#         if T is None:
#             T = model.calibration.get("T", 50)

#         y, e = model.steady_state

#         # initial guess
#         v0 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)

#         y, e = model.steady_state

#         # works if the is one and exactly one exogenous variable?
#         # does it?
#         for key, value in model.evaluator.values.items():
#             i = model.variables.index(key)
#             for a, b in value.items():
#                 v0[a, i] = b

#         return v0

#     def deterministic_residuals(model, v, jac=False, **kwargs):

#         if jac:
#             return model.deterministic_residuals_with_jacobian(v, **kwargs)

#         flat = v.ndim == 1

#         p = len(model.variables)
#         T = int(np.prod(v.shape) / p - 1)

#         v = v.reshape((T + 1, p))

#         v_f = np.concatenate([v[1:, :], v[-1, :][None, :]], axis=0)
#         v_b = np.concatenate([v[0, :][None, :], v[:-1, :]], axis=0)

#         context = {}
#         for i, name in enumerate(model.variables):
#             context[name] = {-1: v_b[:, i], 0: v[:, i], 1: v_f[:, i]}

#         E = model.evaluator
#         E.variables.update(context)

#         results = [E.visit(eq) for eq in E.equations]

#         # number of variables not pinned down by dynamic equations
#         n_exo = len(model.symbols["variables"]) - len(results)

#         # the following works if there is one and exactly one exogenous variable
#         assert n_exo == 1

#         y, e = model.steady_state

#         v1 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)
#         for key, value in model.evaluator.values.items():
#             i = model.variables.index(key)
#             for a, b in value.items():
#                 v1[a, i] = b

#         exo = v1[:, -1].copy()

#         res = np.column_stack(results + [v[:, -1] - exo])

#         res[0, :] = v[0, :] - y  # slightly inconsistent

#         if flat:
#             return res.ravel()
#         else:
#             return res

#     def deterministic_residuals_with_jacobian(model, v, sparsify=False):

#         from dyno.dynsym.autodiff import DNumber

#         flat = v.ndim == 1

#         p = len(model.variables)
#         T = int(np.prod(v.shape) / p - 1)

#         v = v.reshape((T + 1, p))

#         v_f = np.concatenate([v[1:, :], v[-1, :][None, :]], axis=0)
#         v_b = np.concatenate([v[0, :][None, :], v[:-1, :]], axis=0)

#         context = {}
#         for i, name in enumerate(model.variables):
#             context[name] = {
#                 -1: DNumber(v_b[:, i], {(name, -1): 1.0}),
#                 0: DNumber(v[:, i], {(name, 0): 1.0}),
#                 1: DNumber(v_f[:, i], {(name, 1): 1.0}),
#             }

#         E = model.evaluator
#         E.variables.update(context)

#         results = [E.visit(eq) for eq in E.equations]

#         y, e = model.steady_state

#         # get exo values
#         # works if the is one and exactly one exogenous variable?
#         # does it?
#         v1 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)
#         for key, value in model.evaluator.values.items():
#             i = model.variables.index(key)
#             for a, b in value.items():
#                 v1[a, i] = b

#         exo = v1[:, -1].copy()

#         res = np.column_stack([e.value for e in results] + [v[:, -1] - exo])

#         res[0, :] = v[0, :] - y  # slightly inconsistent

#         N = v.shape[0]

#         p = len(model.variables)
#         q = len(model.equations)

#         D = np.zeros((N, q, p, 3))  # would be easier with 4d struct

#         for i_q in range(q):

#             for k, deriv in results[i_q].derivatives.items():
#                 s, t = k  # symbol, time
#                 i_var = model.variables.index(s)
#                 D[:, i_q, i_var, t + 1] = deriv

#         # add exogenous equations
#         DD = np.zeros((N, p, p, 3))
#         DD[:, :q, :, :] = D
#         DD[:, 2, 2, 1] = 1.0

#         if not flat:
#             return res, DD
#         else:
#             J = np.zeros((N * p, N * p))
#             for n in range(N):
#                 if n == 0:
#                     # J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,0] + DD[n,:,:,1]
#                     # J[p*n:p*(n+1),p*(n+1):p*(n+2)] = DD[n,:,:,2]
#                     J[p * n : p * (n + 1), p * n : p * (n + 1)] = np.eye(p, p)
#                 elif n == N - 1:
#                     J[p * n : p * (n + 1), p * (n - 1) : p * (n)] = DD[n, :, :, 0]
#                     J[p * n : p * (n + 1), p * n : p * (n + 1)] = (
#                         DD[n, :, :, 1] + DD[n, :, :, 2]
#                     )
#                 else:
#                     J[p * n : p * (n + 1), p * (n - 1) : p * (n)] = DD[n, :, :, 0]
#                     J[p * n : p * (n + 1), p * n : p * (n + 1)] = DD[n, :, :, 1]
#                     J[p * n : p * (n + 1), p * (n + 1) : p * (n + 2)] = DD[n, :, :, 2]

#             if sparsify:
#                 import scipy

#                 J = scipy.sparse.csr_matrix(J)

#             return res.ravel(), J
