import numpy as np
from .types import TVector, TMatrix
from typing import Callable


def jacobian(
    func: Callable[[TVector], TVector], initial: TVector, delta: float = 1e-3
) -> TMatrix:
    """Jacobian calculation with finite differences

    Parameters
    ----------
    func : p Vector → N Vector
        the function for which we seek the Jacobian
    initial : p Vector
        the value at which the Jacobian is calculated
    delta : float, optional
        step size for Jacobian calculation, by default 1e-3

    Returns
    -------
    output : (N,p) Matrix
        Jacobian matrix of func at point initial
    """
    f = func
    f0 = f(initial)
    nrow = len(f0)
    ncol = len(initial)
    output = np.zeros((nrow, ncol))
    for j in range(ncol):
        ej = np.zeros(ncol)
        ej[j] = 1
        x = (initial + delta * ej).reshape(ncol)
        #   dj = (f(initial+ delta * ej) - f(initial- delta * ej))/(2*delta)
        dj = (f(x) - f0) / (delta)
        output[:, j] = dj

    return output
