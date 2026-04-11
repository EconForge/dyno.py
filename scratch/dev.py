# implement steady-state solver

from dyno import DynoModel

txt = """
@name: TestModel

a <- 0.1

x[~] <- 0.0

x[t] = 0.1 + a * x[t-1]
"""

model = DynoModel(txt=txt)

model.residuals

# compute steady state  

def compute_steady_state(model, tol=1e-6, max_iter=1000):
    
    y0,e0 = model.__steady_state_vectors__
    def fun(y, diff=False):

        if not diff:
            return model.compute_residuals(y,y,y,e0)
        else:
            r,A,B,C,D = model.compute_jacobians(y,y,y,e0)
            return A+B+C
        
    print(fun([e+0.1 for e in y0], diff=False))
    print(fun(y0,diff=True))
        
    import scipy.optimize
    res = scipy.optimize.newton(fun, y0, fprime=lambda u: fun(u,diff=True), full_output=True)
    print(res)
    if not res.success:
        raise ValueError("Steady state solver did not converge: " + res.message)
    return res.x

compute_steady_state(model)