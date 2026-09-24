"""Dynare in Python subpackage.

This package implements the new version of Dynare in Python, incubated within
Dyno before being extracted into an independent, standalone package.
"""

from .model import DynareModel

__all__ = ["DynareModel"]
