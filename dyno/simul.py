import numpy as np
import pandas as pd
from .model import RecursiveSolution

from .types import IRFType

def irf(dr : RecursiveSolution, i : int, T: int=40, type: IRFType="level") -> pd.DataFrame:
    """Impulse response function."""
    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ

    assert(X.shape is not None) # Necessary for static typechecking
    # v0 = np.zeros((X.shape[1],1))
    v0 = np.zeros(X.shape[1])

    assert(Y.shape is not None) # Necessary for static typechecking
    m0 = np.zeros(Y.shape[1])
    
    ss = [v0]

    m0[i] = np.sqrt(Σ[i, i])

    ss.append(X @ ss[-1] + Y @ m0) # type: ignore

    for t in range(T - 1):

        ss.append(X @ ss[-1]) # type: ignore

    res = np.concatenate([e[None, :] for e in ss], axis=0)

    assert(dr.x0 is not None) # Necessary for static typechecking

    if type == "level":
        res = res + dr.x0[None, :]
    elif type == "log-deviation":
        res = (res / dr.x0[None, :]) * 100
    elif type == "deviation":
        pass

    return pd.DataFrame(res, columns=dr.symbols["endogenous"])


def simulate(dr : RecursiveSolution, T: int=40) -> pd.DataFrame:

    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ
    assert(X.shape is not None) # Necessary for static typechecking

    n = X.shape[1]
    v0 = np.zeros(n)
    
    assert(Y.shape is not None) # Necessary for static typechecking
    m0 = np.zeros(Y.shape[1])
    ss = [v0]

    for t in range(T):
        e = np.random.multivariate_normal(m0, Σ)
        ss.append(X @ ss[-1] + Y @ e) # type: ignore

    res = np.concatenate([e[None, :] for e in ss], axis=0)

    return pd.DataFrame(res, columns=dr.symbols["endogenous"])
