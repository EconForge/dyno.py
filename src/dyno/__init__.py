"""Dyno package."""

from pathlib import Path

# from .model import *
from .solver import *
from .simul import *

from .dyno_model import DynoModel
from .dynare_model import DynareModel
from .report import DynoRunResults, DynareRunResults, Report, RunResults


def examples_path(*parts: str) -> Path:
    """Return the path to the repository examples directory.

    Optional path parts are joined to the examples directory.
    """
    root = Path(__file__).resolve().parents[2]
    return root.joinpath("examples", *parts)
