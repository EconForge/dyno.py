from dyno.model import DynoModel
import yaml
from .typedefs import TVector, TMatrix, IRFType, Solver, DynamicFunction

from dyno.dynsym.grammar import parser, str_expression
from dyno.dynsym.analyze import FormulaEvaluator
from typing_extensions import Self

import numpy as np

from .errors import LARKParserError, ParserError
from lark.exceptions import UnexpectedInput
from dyno.larkfiles import DynoFile, LModFile
from dyno.language import ProductNormal, Normal

class LDynoModel(DynoModel):

    def import_model(self: Self, txt: str) -> None:

        try:
            if self.filename.endswith('.mod'):
                self.data = LModFile(content=txt, filename=self.filename)
            elif self.filename.endswith('.dyno'):
                self.data = DynoFile(content=txt, filename=self.filename)
        except UnexpectedInput as e:
            raise LARKParserError(e, txt) from e

    def _set_context(self: Self) -> None:
        context = self.data.context.copy()
        self.context = context


    @property
    def equations(self):

        return [str_expression(eq) for eq in self.data.equations]

    def latex_equations(self):

        from dyno.dynsym.latex import latex
        eqs_str = [latex(eq) for eq in self.evaluator.equations]
        latex_str = str.join("\n",["$${}$$".format(eq) for eq in eqs_str])
        return latex_str


    def compute_residuals(self, y2, y1, y0, e):

        fe = self.data.evaluator
        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]

        for i,name in enumerate(endogenous):
            fe.variables[name] = { 
                -1: y0[i],
                0: y1[i],
                1: y2[i] }
        for i,name in enumerate(exogenous):
            fe.variables[name] = { 0: e[i] }

        results = [fe.visit(eq) for eq in fe.equations]

        r = np.array([float(el) for el in results])

        return r


    def compute_jacobians(self,y2,y1,y0,e):

        import numpy as np
        from dyno.dynsym.autodiff import DNumber as DN

        fe = self.data.evaluator
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









    def deterministic_guess(model, T=None):

        if T is None:
            T = model.context['constants'].get('T', 50)

        y,e = model.__steady_state_vectors__

        # initial guess
        v0 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)


        # works if the is one and exactly one exogenous variable?
        # does it?
        for key,value in model.data.evaluator.values.items():
            i = model.symbols['variables'].index(key)
            for a,b in value.items():
                v0[a,i] = b

        return v0
        
    def deterministic_residuals(model, v, jac=False, **kwargs):

        if jac:
            return model.deterministic_residuals_with_jacobian(v, **kwargs)

        flat = v.ndim == 1

        p = len(model.symbols['variables'])
        T = int(np.prod(v.shape)/p-1)

        v = v.reshape((T+1,p))

        v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
        v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

        context = {}
        for i,name in enumerate(model.symbols['variables']):
            context[name] = { -1: v_b[:,i], 0: v[:,i], 1: v_f[:,i] }

        E = (model.data.evaluator)
        E.variables.update(context)

        results = [E.visit(eq) for eq in E.equations]

        # number of variables not pinned down by dynamic equations
        n_exo = len(model.symbols['variables']) - len(results)

        # the following works if there is one and exactly one exogenous variable
        assert n_exo ==1
        
        y,e = model.steady_state
        
        v1 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)
        for key,value in model.data.evaluator.values.items():
            i = model.symbols['variables'].index(key)
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

        p = len(model.symbols['variables'])
        T = int(np.prod(v.shape)/p-1)

        v = v.reshape((T+1,p))

        v_f = np.concatenate([v[1:,:], v[-1,:][None,:]], axis=0)
        v_b = np.concatenate([v[0,:][None,:], v[:-1,:]], axis=0)

        context = {}
        for i,name in enumerate(model.symbols['variables']):
            context[name] = {
                -1: DNumber(v_b[:,i], {(name,-1): 1.0}),
                0: DNumber(v[:,i], {(name,0): 1.0}),
                1: DNumber(v_f[:,i],  {(name,1): 1.0})
            }

        E = (model.data.evaluator)
        E.variables.update(context)

        results = [E.visit(eq) for eq in E.equations]
        
        y,e = model.__steady_state_vectors__

        # get exo values
        # works if the is one and exactly one exogenous variable?
        # does it?
        v1 = np.concatenate([y,e])[None,:].repeat(T+1,axis=0)
        for key,value in model.data.evaluator.values.items():
            i = model.symbols['variables'].index(key)
            for a,b in value.items():
                v1[a,i] = b

        exo = v1[:,-1].copy()

        res = np.column_stack(
            [e.value for e in results] + [v[:,-1] - exo]
        )
        
        res[0,:] = v[0,:] - y # slightly inconsistent

        N = v.shape[0]

        p = len(model.symbols['variables'])
        q = len(model.equations)
        
        D = np.zeros( (N, q, p, 3 ))  # would be easier with 4d struct

        for i_q in range(q):
            
            for k,deriv in results[i_q].derivatives.items():
                s,t = k # symbol, time
                i_var = model.symbols['variables'].index(s)
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
