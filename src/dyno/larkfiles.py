import math
from lark import Tree
from dyno.dynsym.grammar import parser, str_expression
from dyno.dynsym.analyze import FormulaEvaluator, AssignmentEvaluator, EquationsEvaluator
from typing_extensions import Self
import numpy as np
from typing import List, Dict, Any
from dyno.errors import LARKParserError, ParserError
from lark.exceptions import UnexpectedInput


class SymbolicFile:
    filename: str
    content: str
    tree: Tree
    context: Dict
    equations: List[Tree]
    # declarations: List[Tree]

    def __init__(self, content: str, filename="<none>.dyno") -> None:
        # store provided filename and content on the instance
        self.filename = filename
        self.content = content

    # @property
    # def context(self):
    #     return self.__context__

    # def get_context(self: Self) -> Dict[str, Any]:

    #     ev = self.evaluator
    #     context = {
    #         "constants": ev.constants,
    #         "variables": ev.variables,
    #         "values": ev.values,
    #         "processes": self.processes,
    #         "steady_states": ev.steady_states,
    #     }
    #     return context

    def latex_equations(self):

        from dyno.dynsym.latex import latex

        eqs_str = [latex(eq) for eq in self.equations]
        latex_str = str.join("\n", ["$${}$$".format(eq) for eq in eqs_str])
        return latex_str


    def eval_residuals(self, context = {}) -> None:

        if len(context)==0:
            context = self.context

        fe = EquationsEvaluator(context)
        fe.steady_state = True
        residuals = [fe.visit(eq) for eq in self.equations]
        return residuals



class DynoFile(SymbolicFile):
    """Class for .dyno files"""

    def __init__(self, content: str, filename="<none>.dyno") -> None:
        super().__init__(content, filename)
        self.content = content
        self.parse()

    def parse(self: Self) -> None:

        txt = self.content

        try:
            tree = parser.parse(txt, 
                start="free_block",
            )
        except UnexpectedInput as e:
            raise LARKParserError(e, txt) from e

        self.tree = tree
        
        self.process_assignments()


        # Separate assignments processing from parsing.

    def process_assignments(self, **calib) -> Tree:

        fe = AssignmentEvaluator(calibration=calib)
        fe.visit(self.tree)
        context = {
            'constants': fe.constants,
            'variables': fe.variables,
            'values': fe.values,
            'processes': fe.processes,
            'steady_states': fe.steady_states,
        }
        self.context = context
        self.equations = fe.equations
        self.processes = fe.processes
        self.residuals = self.eval_residuals()

from dyno.dynsym.dynare import (
    ModFileTransformer,
    modfile_grammar,
    InterpretModfile,
)

class LModFile(SymbolicFile):
    """Class for LARK .mod files"""

    def __init__(self, content: str, filename="<none>.dyno") -> None:
        super().__init__(content, filename)
        self.content = content
        self.parse()
        self.residuals = self.eval_residuals()

    def parse(self: Self) -> None:

        from lark import Lark


        content = self.content

        trans = ModFileTransformer()
        parser = Lark(
            modfile_grammar,
            propagate_positions=True,
            parser="lalr",
            transformer=trans,
            cache=True,
            # strict=True,
        )
        try:
            tree = parser.parse(content)
            self.tree = tree
        except UnexpectedInput as e:
            raise LARKParserError(e, content) from e

        variables = trans.variables
        variables_exo = trans.variables_exo
        parameters = trans.parameters

        self.symbols = {
            "variables": variables,
            "endogenous": variables,
            "exogenous": variables_exo,
            "parameters": parameters,
        }

        self.process_assignments()

    def process_assignments(self, **calib) -> None:
        
        ## calib ignored so far
        import math
        function_table={'exp':math.exp, 'log':math.log, 'sqrt':math.sqrt, 'abs':math.fabs}
        
        fe = InterpretModfile(**calib, function_table=function_table)
        fe.visit(self.tree)

        self.equations = fe.equations

        

        # Set processes
        from dyno.language import Normal
        covs = fe.covariances
        exo = tuple(self.symbols["exogenous"])
        n_e = len(exo)
        mat = np.zeros((n_e, n_e))
        for ind, k in covs.items():
            i = exo.index(ind[0])
            j = exo.index(ind[1])
            mat[i, j] = k
            mat[j, i] = k

        processes = {exo: Normal(mat)}

        context = {
            'constants': fe.constants,
            'variables': fe.variables,
            'values': fe.values,
            'processes': processes,
            'steady_states': fe.steady_states | {e: 0.0 for e in exo},        # set exogenous steady states to zero

        }


        self.context = context