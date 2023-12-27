import dolang
from dolang import stringify
from dolang.function_compiler import FlatFunctionFactory as FFF
from dolang.symbolic import str_expression, stringify_symbol
from dolang.function_compiler import make_method_from_factory

from numpy.linalg import solve as linsolve

import numpy as np
import yaml
import xarray

from .solver import solve

cache = []

def jacobian(func,initial,delta=1e-3):
  f = func
  nrow = len(f(initial))
  ncol = len(initial)
  output = np.zeros(nrow*ncol)
  output = output.reshape(nrow,ncol)
  for i in range(nrow):
    for j in range(ncol):
      ej = np.zeros(ncol)
      ej[j] = 1
      dij = (f(initial+ delta * ej)[i] - f(initial- delta * ej)[i])/(2*delta)
      output[i,j] = dij
  return output

class RecursiveSolution:

    def __init__(self, X, Y, Σ):

        self.X = X
        self.Y = Y
        self.Σ = Σ

class Normal:

    def __init__(self, Σ, vars):

        self.Σ = Σ
        self.variables = tuple(*vars)

class Model:

    def __init__(self, data):

        self.data = data
        self.__update_equations__()
        self.__update_calibration__()
        self.exogenous = self.__get_exogenous__()


    def __get_exogenous__(self):

        # TODO: write something here
        s = self.data['exogenous']['e_z']
        s = np.array([[0.001**2]])
        return Normal(s, ['e_z'])

    def describe(self):

        return f"""
symbols: {self.symbols}
        """

    def __update_equations__(self):

        data = self.data

        tree = dolang.parse_string(data['equations'], start="equation_block")

        stree = dolang.grammar.sanitize(tree)

        symlist = dolang.list_symbols(stree)

        vars = list( set(e[0] for e in symlist.variables) )
        pars = symlist.parameters

        # check exogenous variables
        try:
            exovars = self.data['exogenous'].keys()
        except:
            exovars = []

        symbols = {
            'variables': [e for e in vars if e not in exovars],
            'parameters': pars,
            'exogenous': exovars
        }

        self.symbols = symbols

        n = len(tree.children)

        # equations = [f"({stringify(eq.children[1])})-({stringify(eq.children[0])})"  for eq in tree.children]
        equations = [stringify(str_expression(eq))  for eq in tree.children]
        
        self.equations = equations

        equations = [
                ("({1})-({0})".format(*eq.split("=")) if "=" in eq else eq) for eq in equations
        ]

        self.equations = equations

        dict_eq = dict([(f"out{i+1}", equations[i]) for i in range(n)])
        spec = dict(
            y_f=[stringify_symbol((e,1)) for e in symbols['variables']],
            y_0=[stringify_symbol((e,0)) for e in symbols['variables']],
            y_p=[stringify_symbol((e,-1)) for e in symbols['variables']],
            e=[stringify_symbol((e,0)) for e in symbols['exogenous']],
            p=[stringify_symbol(e) for e in symbols['parameters']]
        )

        fff = FFF(
            dict(),
            dict_eq,
            spec,
            "f_dynamic"
        )

        fun = make_method_from_factory(fff, compile=False, debug=False)

        self.__functions__ = {'dynamic': fun}

    def dynamic(self, y0, y1, y2, e, p, diff=False):
        

        r = np.zeros(len(y0))
        self.__functions__['dynamic'](y0, y1, y2, e, p, r)
        d = np.zeros(len(self.symbols['exogenous']))

        if diff:
            f = lambda a,b,c,d,e: self.dynamic(a,b,c,d, e)
            r1 = jacobian(lambda u: f(u,y1,y2,e,p), y0)
            r2 = jacobian(lambda u: f(y0,u,y2,e,p), y1)
            r3 = jacobian(lambda u: f(y0,y1,u,e,p), y2)
            r4 = jacobian(lambda u: f(y0,y1,y2,u,p), d)
            return r,r1,r2,r3,r4
        
        return r

    def compute(self, diff=False):

        c = self.calibration
        v = self.symbols['variables']
        p = self.symbols['parameters']

        y0 = np.array([c[e] for e in v])
        p0 = np.array([c[e] for e in p])
        e = np.zeros(len(self.symbols['exogenous']))
        return self.dynamic(y0,y0,y0,e,p0,diff=diff)
    
    def __update_calibration__(self):

        syms = self.symbols['variables'] + self.symbols['parameters']

        data = self.data
        nan = float("nan")

        calibration = {k: nan for k in syms}

        for k,vv in data['calibration'].items():
            v = vv.value
            calibration[k] = eval(v)

        self.calibration = calibration

    def solve(self, **args)->RecursiveSolution:

        r,A,B,C,D = self.compute(diff=True)
        X = solve(A,B,C, **args)
        Y = linsolve(A@X + B, -D)

        v = self.symbols['variables']
        e = self.symbols['exogenous']

        Σ = self.exogenous.Σ

        return RecursiveSolution(
            xarray.DataArray(X, coords=(("y_t",v), ("y_{t-1}",v))),
            xarray.DataArray(Y, coords=(("y_t",v), ("e_t",e))),
            Σ
        )


def import_file(filename)->Model:
    
    txt = open(filename, "rt", encoding="utf-8").read()
    return import_model(txt)

def import_model(txt)->Model:

    data = yaml.compose(txt)

    v = hash(txt)
    v_eq = hash(data['equations'].value)

    existing = [m[0] for m in cache]

    if v in existing:
        i = existing.index(v)
        model = cache[i][2]
        return model

    else:
        model = Model(data)
        cache.append((v,v_eq,model))
        return model