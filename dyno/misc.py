import numpy as np
from .types import Vector, Matrix
from typing import Callable

def jacobian(func: Callable[[Vector], Vector], initial: Vector, delta: float = 1e-3) -> Matrix:
    """Jacobian calculation with finite differences

    Parameters
    ----------
    func : Callable[(p,) ndarray, (N,) ndarray]
        the function for which we seek the Jacobian
    initial : (p,) ndarray
        the value at which the Jacobian is calculated
    delta : float, optional
        step size for Jacobian calculation, by default 1e-3

    Returns
    -------
    output : (N,p)
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
