# defines the dolang language elements recognized in the yaml file.

# copied from dolo

from .types import Vector, Matrix

from dolang.language import greek_tolerance, language_element  # type: ignore

import numpy as np

class NotPositiveSemidefinite(np.linalg.LinAlgError):
    """An exception raised when a normal process is created with a covariance matrix that is not positive semidefinite"""
    pass

@language_element
class Normal:
    """Multivariate normal process class, can be used to model white noise.

    Parameters
    ----------
    Μ: d Vector | None
        Mean vector for the normal process, taken to be equal to zero if not passed

    Σ : (d,d) Matrix
        Covariance matrix for the normal process, needs to be positive semidefinite

    Attributes
    ----------
    
    Μ: d Vector
        Mean vector

    Σ : (d,d) Matrix
        Covariance matrix

    d : int
        dimension
    """
    Μ: Vector  # this is capital case μ, not M... 😭
    Σ: Matrix

    @greek_tolerance
    def __init__(self, Σ, Μ=None):

        Sigma = Σ
        mu = Μ

        self.Σ = np.atleast_2d(np.array(Sigma, dtype=float))
        try:
            assert(np.array_equal(Sigma, Sigma.T))
            np.linalg.cholesky(Sigma)
        except AssertionError | np.linalg.LinAlgError:
            raise(NotPositiveSemidefinite, "Σ can't be used as a covariance matrix as it is not positive semidefinite")
        
        self.d = len(self.Σ)
        if mu is None:
            self.Μ = np.array([0.0] * self.d)
        else:
            self.Μ = np.array(mu, dtype=float)

        assert self.Σ.shape[0] == self.d
        assert self.Σ.shape[0] == self.d

        # this class wraps functionality from scipy
        import scipy.stats

        self._dist_ = scipy.stats.multivariate_normal(
            mean=self.Μ, cov=self.Σ, allow_singular=True
        )


@language_element
class ProductNormal:
    """Product of multivariate normal processes

    Parameters
    ----------
    N1, …, Nk : Normal
        Multivariate normal processes

    Attributes
    ----------
    processes : list[Normal]
        list of processes N1, …, Nk
    d : int
        sum of dimensions for all processes involved
    """

    def __init__(self, *l):
        self.processes = l
        self.d = sum([e.d for e in self.processes])

    @property
    def Σ(self):
        """Covariance matrix if only composed of a single process

        Returns
        -------
        Matrix
            Covariance matrix of the process
        """
        assert len(self.processes) == 1
        return self.processes[0].Σ
