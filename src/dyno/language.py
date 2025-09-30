# defines the dolang language elements recognized in the yaml file.

# copied from dolo

from .typedefs import TVector, TMatrix

from dolang.language import greek_tolerance, language_element  # type: ignore

import numpy as np

from scipy.stats import multivariate_normal


class NotPositiveSemidefinite(np.linalg.LinAlgError):
    """An exception raised when a normal process is created with a covariance matrix that is not positive semidefinite"""

    pass


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


class Exogenous:
    pass


class Deterministic(Exogenous):
    """Deterministic exogenous variables clas

    Parameters
    ----------
    horizon: int
        time horizon over which the perfect foresight solver will simulate the model

    values_dict: dict[str, list[float]]
        dictionary containing the values that each exogenous variable takes on at each time period before the horizon,
        empty fields are assumed to be zero
    """

    values: dict[str, list[float]]
    """values taken on by each exogenous variable"""

    def __init__(self, values_dict: dict[str, list[float]]):
        horizon = max([len(periods) for periods in values_dict.values()])
        for var, values in values_dict.items():
            if len(values) < horizon:
                values_dict[var] = pad_list(values, horizon)
        self.horizon = horizon
        self.values = values_dict


@language_element
class Normal(Exogenous):
    """Multivariate normal process class, can be used to model white noise.

    Parameters
    ----------
    Μ: d Vector | None
        Mean vector for the normal process, taken to be equal to zero if not passed

    Σ: (d,d) Matrix
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

    Μ: TVector  # this is capital case μ, not M... 😭
    Σ: TMatrix

    signature = {"Σ": "Matrix", "Μ": "Optional[Vector]"}

    @greek_tolerance
    def __init__(self, Σ, Μ=None):
        Sigma = np.array(Σ)
        mu = Μ
        self.Σ = np.atleast_2d(np.array(Sigma, dtype=float))
        try:
            assert np.array_equal(Sigma, Sigma.T)
            assert np.all(np.linalg.eigvalsh(Sigma) > -1e-12)
        except AssertionError:
            raise NotPositiveSemidefinite(
                "Σ can't be used as a covariance matrix as it is not positive semidefinite",
            )
        self.d = len(self.Σ)
        if mu is None:
            self.Μ = np.array([0.0] * self.d)
        else:
            self.Μ = np.array(mu, dtype=float)

        assert self.Σ.shape[0] == self.d
        assert self.Σ.shape[1] == self.d
        # this class wraps functionality from scipy
        self._dist_ = multivariate_normal(mean=self.Μ, cov=self.Σ, allow_singular=True)


@language_element
class ProductNormal(Exogenous):
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
        TMatrix
            Covariance matrix of the process
        """
        # assert len(self.processes) == 1
        from scipy.linalg import block_diag

        return block_diag(*(e.Σ for e in self.processes))


def pad_list(l: list, n: int) -> list:
    return l + [0] * (n - len(l))
