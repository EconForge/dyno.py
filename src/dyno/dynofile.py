from dyno.model import Model
import yaml
from .typedefs import TVector, TMatrix, IRFType, Solver, DynamicFunction

from dyno.dynsym.grammar import parser, str_expression
from dyno.dynsym.analyze import FormulaEvaluator
from typing_extensions import Self

import numpy as np

class DynoModel(Model):

    def import_model(self: Self, txt: str) -> None:

        self.data = parser.parse(txt, start='free_block')

        tree = self.data

        fe = FormulaEvaluator()
        self.evaluator = fe

        # process assignments
        from dyno.language import Normal
        fe.visit(tree)

        # count variable in equations and compute residuals
        fe.steady_state = True
        self.residuals = [fe.visit(eq) for eq in fe.equations]
        fe.steady_state = False



    def _set_name(self: Self):

        self.name = "anonymous"

    def _set_calibration(self: Self) -> None:

        pass

    def get_calibration(self, **kwargs):

        return self.evaluator.constants | self.evaluator.steady_states

    def _set_exogenous(self):

        from dyno.language import ProductNormal, Normal

        self.processes = ProductNormal(
            *[Normal([[e.sigma**2]], [e.mu]) for e in self.evaluator.processes.values()]
        )

        self.paths = self.evaluator.values

    def _set_symbols(self: Self) -> None:
        
        fe = self.evaluator

        variables = (fe.variables).keys()

        exogenous = [v for v in variables if (v in fe.processes)]
        endogenous = [v for v in variables if v not in exogenous]

        self.symbols = {
            "endogenous": endogenous,
            "parameters": list(fe.constants.keys()),
            "exogenous": exogenous,
        }

    def _set_dynamic(self):
        
        pass

    def _set_equations(self: Self):

        self.equations = [str_expression(eq) for eq in self.evaluator.equations]

    def compute_residuals():
            pass

    def steady_state(self):

        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]
        fe = self.evaluator

        y = [fe.steady_states[name] for name in  (endogenous)]
        e = [fe.steady_states[name] for name in  (exogenous)]
        return y,e


    def compute_derivatives(self,y2,y1,y0,e):


        import numpy as np
        from dyno.dynsym.autodiff import DNumber as DN

        fe = self.evaluator
        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]

        for i,name in enumerate(endogenous):
            fe.variables[name] = { 
                -1: DN(y0[i], {(name,-1): 1}),
                0: DN(y1[i], {(name,0): 1}),
                1: DN(y2[i], {(name,1): 1}) }
        for i,name in enumerate(exogenous):
            fe.variables[name] = { 0: DN(e[i], {(name,0): 1}) }

        results = [fe.visit(eq) for eq in fe.equations]

        neq = len(results)
        nv = len(endogenous)
        ne = len(exogenous)

        r = np.array([el.value for el in results])
        A = np.zeros((neq,nv))
        B = np.zeros((neq,nv))
        C = np.zeros((neq,nv))
        J = [A,B,C]
        D = np.zeros((neq,ne))


        for n,eq in enumerate(results):
            for ((name, shift),v) in eq.derivatives.items():
                if name in endogenous:
                    i = endogenous.index(name)
                    J[1-shift][n,i] = v
                elif name in exogenous:
                    i = exogenous.index(name)
                    D[n,i] = v

        return r, A,B,C,D




    def compute(
        self: Self, calibration: dict[str, float] = {}, diff: bool = False
    ) -> TVector | tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        """Computes the dynamic function's value based on calibration state and parameters

        Parameters
        ----------
        calibration : dict[str, float], optional
            dictionary containing the value of each parameter and variable of the model, indexed by their symbols, by default {}
        diff : bool, optional
            if set to True returns the dynamic function's partial derivatives as well, by default False

        Returns
        -------
        TVector|tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]
            value of the dynamic function at the state described by calibration, as well as its partial derivatives if diff is set to True
        """

        if not diff:
            return np.array(self.residuals)
        
        from dyno.dynsym.analyze import DN

        assert len(calibration)==0, "calibration not supported yet"

        ys, es = self.steady_state()
        
        return self.compute_derivatives(ys,ys,ys,es)

        import copy
       