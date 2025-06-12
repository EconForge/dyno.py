import numpy as np
import pandas as pd
from .model import RecursiveSolution

from .types import IRFType

def irf(dr : RecursiveSolution, i : int, T: int=40, type: IRFType="level") -> pd.DataFrame:
    """Impulse response function simulation in response to a shock on a specific exogenous variable

    Parameters
    ----------
    dr : RecursiveSolution
        linearized model, contains all variables and parameters
    i : int
        index representing the exogenous variable of the shock
    T : int, optional
        time horizon over which the simulation is done, by default 40
    type : IRFType, optional
        can be "level", "log-deviation" or "deviation", by default "level"

    Returns
    -------
    pd.DataFrame
        impulse response function of all endogenous variables to a shock on the ith exogenous variable
    """
    X = dr.X.data
    Y = dr.Y.data
    Σ = dr.Σ

    assert(X.shape is not None)
    # v0 = np.zeros((X.shape[1],1))
    v0 = np.zeros(X.shape[1])

    assert(Y.shape is not None)
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
    """Simulates the evolution of the endogenous variables

    Parameters
    ----------
    dr : RecursiveSolution
        linearized model, contains all variables and parameters
    T : int, optional
        time horizon over which the simulation is done, by default 40

    Returns
    -------
    pd.DataFrame
        evolution of the endogenous variables over time
    """
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
