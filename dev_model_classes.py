from lark import Tree
from dyno.dynsym.grammar import parser, str_expression
from dyno.dynsym.analyze import FormulaEvaluator
from typing_extensions import Self
import numpy as np
from typing import List, Dict, Any
from dyno.errors import LARKParserError, ParserError
from lark.exceptions import UnexpectedInput

from rich import inspect, print

class SymbolicFile:

    content: str
    tree: Tree
    equations: List[Tree]
    # declarations: List[Tree]

    def __init__(self, content: str, filename="<none>.dyno") -> None:

        self.filename = filename
        self.content = None
    
    @property
    def context(self):
        return self.__context__
    
    def get_context(self: Self) -> Dict[str, Any]:
        ev = self.evaluator        
        context = {
            'constants': ev.constants,
            'variables': ev.variables,
            'values': ev.values,
            'processes': self.processes,
            'steady_states': ev.steady_states,

        }
        return context

    def latex_equations(self):

        from dyno.dynsym.latex import latex
        eqs_str = [latex(eq) for eq in self.equations]
        latex_str = str.join("\n",["$${}$$".format(eq) for eq in eqs_str])
        return latex_str

class DynoFile(SymbolicFile):

    """Class for .dyno files"""

    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.content = content
        self.parse()
        self.__context__ = self.get_context()
    

    @property
    def context(self):
        return self.__context__

    def parse(self: Self) -> None:
        txt = self.content
        
        try:
            tree = parser.parse(txt, start='free_block')
        except UnexpectedInput as e:
            raise LARKParserError(e, txt) from e
        
        self.tree = tree

        fe = FormulaEvaluator()
        self.evaluator = fe

        # process assignments
        fe.visit(tree)

        # count variable in equations and compute residuals
        fe.steady_state = True
        self.residuals = [fe.visit(eq) for eq in fe.equations]
        fe.steady_state = False

        self.equations = fe.equations
        self.processes = fe.processes


class LModFile(SymbolicFile):

    """Class for LARK .mod files"""

    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.content = content
        self.parse()
        self._set_processes()
        self.__context__ = self.get_context()
    
    def parse(self: Self) -> None:

        from lark import Lark
        from dyno.modfile_lark import ModFileTransformer, modfile_grammar, InterpretModfile

        
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
        except UnexpectedInput as e:
            raise LARKParserError(e, content) from e

        variables = trans.variables
        variables_exo = trans.variables_exo
        parameters = trans.parameters
        # variables_pred = trans.variables_pred

        self.symbols = {
            "variables": variables + variables_exo,
            "endogenous": variables,
            "exogenous": variables_exo,
            "parameters": parameters,
        }


        fe = InterpretModfile(steady_state=True)
        fe.visit(tree)

        # count variable in equations and compute residuals
        fe.steady_state = True
        self.residuals = [fe.visit(eq) for eq in fe.equations]
        fe.steady_state = False

        self.evaluator = fe
        self.equations = fe.equations
        

    def _set_processes(self):

        from dyno.language import Normal
        fe = self.evaluator

        covs = self.evaluator.covariances.copy()

        exo = tuple(self.symbols['exogenous'])
        n_e = len(exo)
        mat = np.zeros((n_e,n_e))
        for ind,k in covs.items():
            i = exo.index(ind[0])
            j = exo.index(ind[1])
            mat[i,j] = k
            mat[j,i] = k

        self.evaluator.processes = {exo: Normal(mat)}
        self.processes = self.evaluator.processes
        for e in exo:
            self.evaluator.steady_states[e] = 0.0        
        


# f = DynoFile(open("examples/RBC.dyno").read())

# f = LModFile(open("examples/modfiles/RBC.mod").read())

fname_mod = "examples/modfiles/RBC.mod"
fname_dyno = "examples/RBC.dyno"
# f = LModFile(open("examples/RBC.dyno").read())
f = DynoFile(open(fname_dyno).read())
print(f.context)
txt1 = (f.latex_equations())
f = LModFile(open(fname_mod).read())

print(f.context)
txt2 = (f.latex_equations())

# the two models have exactly the same equations
assert(txt1 == txt2)