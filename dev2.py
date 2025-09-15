from dynsym.analyze import FormulaEvaluator
from dynsym.grammar import parser
# from rich import print, inspect
from dynsym.grammar import str_expression

fe = FormulaEvaluator()


def import_model(filename):

    txt = open(filename, mode="r", encoding="utf-8").read()
    tree = parser.parse(txt, start="free_block")

    fe = FormulaEvaluator()
    fe.visit(tree)

    for eq in fe.equations:
        print(str_expression(eq))

    fe.steady_state = True
    residuals = [fe.visit(eq) for eq in fe.equations]
    fe.steady_state = False

    variables = (fe.variables).keys()

    exogenous = [v for v in variables if (v in fe.processes)]
    endogenous = [v for v in variables if v not in exogenous]


    import copy
    def compute_residuals():
        pass

    def steady_state():
        y = [fe.steady_states[name] for name in  (endogenous)]
        e = [fe.steady_states[name] for name in  (exogenous)]
        return y,e


    from dynsym.analyze import DN
    def compute_derivatives(y2,y1,y0,e):

        for i,name in enumerate(endogenous):
            fe.variables[name] = { 
                -1: DN(y0[i], {(name,-1): 1}),
                 0: DN(y1[i], {(name,0): 1}),
                 1: DN(y2[i], {(name,1): 1}) }
        for i,name in enumerate(exogenous):
            fe.variables[name] = { 0: DN(e[i], {(name,0): 1}) }

        results = [fe.visit(eq) for eq in fe.equations]

        import numpy as np
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
        
    ys, es = steady_state()
    
    r, A, B, C, D = compute_derivatives(ys, ys, ys, es)


   

    return fe, ys, r, A, B, C, D


import time
# warm up
res = import_model("tests/rbc.dyno")


# import timeit
# K = 100
# tt = timeit.timeit('import_model("tests/rbc.dyno")', globals=globals(), number=K)
# print(f"Timeit: {tt/K}")



# from rich import print, inspect

fe = res[0]
print(f"[bold]Constants[/bold]: {fe.constants}")
print(f"[bold]Values[/bold]: {fe.values}")
print(f"[bold]Steady-states[/bold]: {fe.steady_states}")
print(f"[bold]Processes[/bold]: {fe.processes}")





print(f"* Steady-state: {res[1]}")
print(f"* Residuals: {res[2]}")
print(f"* A\n: {res[3]}")
print(f"* B\n: {res[4]}")
print(f"* C\n: {res[5]}")
print(f"* D\n: {res[6]}")

# print(f"Timeit: {tt/K}")
