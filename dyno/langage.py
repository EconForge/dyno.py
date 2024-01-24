# defines the dolang language elements recognized in the yaml file.

# copied from dolo

from typing import List, TypeVar, Generic, Union, Any, Callable  # type: ignore
from typing import Iterator, Tuple  # type: ignore

from dolang.language import greek_tolerance, language_element  # type: ignore

Vector = List[float]
# Matrix = List[Vector]

import numpy as np

@language_element
def Matrix(*lines):
    mat = np.array(lines, np.float64)
    assert mat.ndim == 2
    return mat


@language_element
def Vector(*elements):
    mat = np.array(elements, np.float64)
    assert mat.ndim == 1
    return mat

@language_element
class Normal:

    Μ: Vector  # this is capital case μ, not M... 😭
    Σ: Matrix

    signature = {"Σ": "Matrix", "Μ": "Optional[Vector]"}

    @greek_tolerance
    def __init__(self, Σ=None, Μ=None):

        Sigma = Σ
        mu = Μ

        self.Σ = np.atleast_2d(np.array(Sigma, dtype=float))
        self.d = len(self.Σ)
        if mu is None:
            self.Μ = np.array([0.0] * self.d)
        else:
            self.Μ = np.array(mu, dtype=float)
        
        assert self.Σ.shape[0] == self.d
        assert self.Σ.shape[0] == self.d

        # this class wraps functionality from scipy
        import scipy.stats

        self._dist_ = scipy.stats.multivariate_normal(mean=self.Μ, cov=self.Σ)


@language_element
class ProductNormal:

    ### This class represents the product of processes

    def __init__(self, *l):
        self.processes = l
        self.d = sum([e.d for e in self.processes])

    @property
    def Σ(self):
        assert len(self.processes) == 1
        return self.processes[0].Σ