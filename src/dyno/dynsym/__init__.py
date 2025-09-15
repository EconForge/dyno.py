from .grammar import parser, str_expression
from .analyze import FormulaEvaluator as Analyzer

import numpy as np

def read_model(filename, diff=True):

    with open(filename, "rt", encoding="utf-8") as f:
        txt = f.read()
    tree = parser.parse(txt, start="free_block")

    if diff is False:
        an = Analyzer(steady_state=True, diff=False)
        res = an.evaluate(tree)
        return res

    an = Analyzer(steady_state=False, diff=True)
    res = an.evaluate(tree)

    variables = an.symbols['variables']
    exogenous = an.symbols['exogenous']
    endogenous = [v for v in variables if v not in exogenous]
    # constants = an.symbols['constants']
    # values = an.symbols['values']
        
    A = np.zeros((len(res), len(endogenous)))
    B = np.zeros((len(res), len(endogenous)))
    C = np.zeros((len(res), len(endogenous)))
    D = np.zeros((len(res), len(exogenous)))

    J = [A,B,C]

    for i,v in enumerate(endogenous):
        for k,_s in enumerate(["[t+1]", "[t]", "[t-1]"]):
            vfull = v+_s
            for n,r in enumerate(res):
                if vfull not in r.derivatives:
                    continue
                J[k][n,i] = r.derivatives[vfull]
    for i,v in enumerate(exogenous):
        vfull = v+"[t]"
        for n,r in enumerate(res):
            if vfull not in r.derivatives:
                continue
            D[n,i] = r.derivatives[vfull]

    return r, A, B, C, D
