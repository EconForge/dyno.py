import numpy as np
import pandas as pd


def irf(dr, i, T=40):

    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ

    n = X.shape[1]
    v0 = np.zeros(n)
    m0 = np.zeros(Y.shape[1])
    ss = [v0]

    m0[i] = np.sqrt(Σ[i, i])

    ss.append(X @ ss[-1] + Y @ m0)

    for t in range(T - 1):

        ss.append(X @ ss[-1])

    res = np.concatenate([e[None, :] for e in ss], axis=0)

    # dim_1 = [*range(T+1)]
    # dim_2 = [*dr.variables["endogenous"]]

    return pd.DataFrame(res, columns=dr.symbols["endogenous"])


def simulate(dr, T=40):

    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ

    n = X.shape[1]
    v0 = np.zeros(n)
    m0 = np.zeros(Y.shape[1])
    ss = [v0]

    for t in range(T):
        e = np.random.multivariate_normal(m0, Σ)
        ss.append(X @ ss[-1] + Y @ e)

    res = np.concatenate([e[None, :] for e in ss], axis=0)

    return pd.DataFrame(res, columns=dr.symbols["endogenous"])
