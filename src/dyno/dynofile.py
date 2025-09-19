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

        self.calibration = self.get_calibration()


    def get_calibration(self, **kwargs):

        return self.evaluator.constants | self.evaluator.steady_states

    def _set_exogenous(self):

        from dyno.language import ProductNormal, Normal

        if len(self.evaluator.processes.values())==0:
            self.processes = None
        else:
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

    def latex_equations(self):

        from dyno.dynsym.latex import latex
        eqs_str = [latex(eq) for eq in self.evaluator.equations]
        latex_str = str.join("\n",["$${}$$".format(eq) for eq in eqs_str])
        return latex_str

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
    

    def deterministic_guess(model, T=None):

        if T is None:
            T = model.calibration.get('T', 50)

        y,e = model.steady_state()

        # initial guess
        v0 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)

        y,e = model.steady_state()

        # works if the is one and exactly one exogenous variable?
        # does it?
        for key,value in model.evaluator.values.items():
            i = model.variables.index(key)
            for a,b in value.items():
                v0[a,i] = b

        return v0
        
    def deterministic_residuals(model, v, jac=False, **kwargs):

        if jac:
            return model.deterministic_residuals_with_jacobian(v, **kwargs)

        flat = v.ndim == 1

        p = len(model.variables)
        T = int(np.prod(v.shape)/p-1)

        v = v.reshape((T+1,p))

        v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
        v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

        context = {}
        for i,name in enumerate(model.variables):
            context[name] = { -1: v_b[:,i], 0: v[:,i], 1: v_f[:,i] }

        E = (model.evaluator)
        E.variables.update(context)

        results = [E.visit(eq) for eq in E.equations]

        # number of variables not pinned down by dynamic equations
        n_exo = len(model.symbols['variables']) - len(results)

        # the following works if there is one and exactly one exogenous variable
        assert n_exo ==1
        
        y,e = model.steady_state()
        
        v1 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)
        for key,value in model.evaluator.values.items():
            i = model.variables.index(key)
            for a,b in value.items():
                v1[a,i] = b

        exo = v1[:,-1].copy()

        res = np.column_stack(
            results + [v[:,-1] - exo]
        )

        res[0,:] = v[0,:] - y # slightly inconsistent

        if flat:
            return res.ravel()
        else:
            return res

    def deterministic_residuals_with_jacobian(model, v, sparsify=False):

        from dyno.dynsym.autodiff import DNumber

        flat = v.ndim == 1

        p = len(model.variables)
        T = int(np.prod(v.shape)/p-1)

        v = v.reshape((T+1,p))

        v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
        v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

        context = {}
        for i,name in enumerate(model.variables):
            context[name] = {
                -1: DNumber(v_b[:,i], {(name,-1): 1.0}),
                0: DNumber(v[:,i], {(name,0): 1.0}),
                1: DNumber(v_f[:,i],  {(name,1): 1.0})
            }

        E = (model.evaluator)
        E.variables.update(context)

        results = [E.visit(eq) for eq in E.equations]
        
        y,e = model.steady_state()

        # get exo values
        # works if the is one and exactly one exogenous variable?
        # does it?
        v1 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)
        for key,value in model.evaluator.values.items():
            i = model.variables.index(key)
            for a,b in value.items():
                v1[a,i] = b

        exo = v1[:,-1].copy()

        res = np.column_stack(
            [e.value for e in results] + [v[:,-1] - exo]
        )
        
        res[0,:] = v[0,:] - y # slightly inconsistent

        N = v.shape[0]

        p = len(model.variables)
        q = len(model.equations)
        
        D = np.zeros( (N, q, p, 3 ))  # would be easier with 4d struct

        for i_q in range(q):
            
            for k,deriv in results[i_q].derivatives.items():
                s,t = k # symbol, time
                i_var = model.variables.index(s)
                D[:, i_q, i_var, t+1] = deriv

        # add exogenous equations
        DD = np.zeros( (N, p, p, 3))
        DD[:,:q,:,:] = D
        DD[:,2,2,1] = 1.0

        if not flat:
            return res, DD
        else:
            J = np.zeros((N*p, N*p))
            for n in range(N):
                if n==0:
                    # J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,0] + DD[n,:,:,1]
                    # J[p*n:p*(n+1),p*(n+1):p*(n+2)] = DD[n,:,:,2]
                    J[p*n:p*(n+1),p*n:p*(n+1)] = np.eye(p,p)
                elif n==N-1:
                    J[p*n:p*(n+1),p*(n-1):p*(n)] = DD[n,:,:,0]
                    J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,1] + DD[n,:,:,2]
                else:
                    J[p*n:p*(n+1),p*(n-1):p*(n)] = DD[n,:,:,0]
                    J[p*n:p*(n+1),p*n:p*(n+1)] = DD[n,:,:,1]
                    J[p*n:p*(n+1),p*(n+1):p*(n+2)] = DD[n,:,:,2]

            if sparsify:
                import scipy
                J = scipy.sparse.csr_matrix(J)

            return res.ravel(), J
