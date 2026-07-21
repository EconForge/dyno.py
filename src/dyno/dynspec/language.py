from functools import wraps
from inspect import signature

import numpy as np
from scipy.stats import multivariate_normal


_ASCII_TO_GREEK = {
    "Sigma": "Σ",
    "Mu": "Μ",
    "sigma": "σ",
    "mu": "μ",
}


def greek_tolerance(func):
    params = set(signature(func).parameters.keys())
    alias_to_greek = {
        alias: greek for alias, greek in _ASCII_TO_GREEK.items() if greek in params
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        remapped: dict[str, object] = {}
        for key, value in kwargs.items():
            target = alias_to_greek.get(key, key)
            if target in remapped:
                raise TypeError(
                    f"{func.__name__}() received duplicate values for '{target}'"
                )
            remapped[target] = value
        return func(*args, **remapped)

    return wrapper


class NotPositiveSemidefinite(np.linalg.LinAlgError):
    pass


class Normal:
    @greek_tolerance
    def __init__(self, Σ, Μ=None):
        Sigma = np.array(Σ)
        mu = Μ
        self.Σ = np.atleast_2d(np.array(Sigma, dtype=float))
        try:
            assert np.array_equal(Sigma, Sigma.T)
            assert np.all(np.linalg.eigvalsh(Sigma) > -1e-12)
        except AssertionError as exc:
            raise NotPositiveSemidefinite(
                "Σ can't be used as a covariance matrix as it is not positive semidefinite",
            ) from exc
        self.d = len(self.Σ)
        if mu is None:
            self.Μ = np.array([0.0] * self.d)
        else:
            self.Μ = np.array(mu, dtype=float)

        assert self.Σ.shape[0] == self.d
        assert self.Σ.shape[1] == self.d
        self._dist_ = multivariate_normal(mean=self.Μ, cov=self.Σ, allow_singular=True)


__all__ = ["Normal", "NotPositiveSemidefinite", "greek_tolerance"]