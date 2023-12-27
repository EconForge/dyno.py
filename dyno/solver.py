import numpy as np
import pandas as pd
from numpy.linalg import solve as linsolve
import xarray

def simulate(dr,T=40):

    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ

    n = X.shape[1]
    v0 = np.zeros(n)
    m0 = np.zeros(Y.shape[1])
    ss = [v0]

    for t in range(T):
        e = np.random.multivariate_normal(m0, Σ)
        ss.append(X@ss[-1] + Y@e)

    res = np.concatenate([e[None,:] for e in ss], axis=0)
    # # rr = pd.DataFrame(res, columns=X.coords['y_t'])
    # return res
    dim_1 = [*range(T+1)]
    dim_2 = [*dr.X.coords['y_t'].data]

    return xarray.DataArray(res, coords=(('T', dim_1 ), ('V', dim_2)))


def solve(A,B,C, T=10000, tol=1e-10):
            
    n = A.shape[0]

    X0 = np.random.randn(n,n)

    for t in range(T):

        X1 = - linsolve(A@X0 + B, -C)
        e = abs(X0-X1).max()

        if np.isnan(e):
            raise Exception("Invalid value")
        
        X0 = X1
        if e<tol:
            return X0


    #     X1 = - linsolve(A@X0 + B, C)
    #     e = abs(X0-X1).max()

    #     X0 = X1
    #     if e<tol:
    #         return X0
        
    # raise Exception("No convergence")
