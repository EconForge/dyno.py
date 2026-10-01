"""Dyno package."""

from pathlib import Path

# from .model import *
from .solver import *
from .simul import *

from .dyno_model import DynoModel
from . import dynare
from .dynare import DynareModel
from .report import RunResults
from .variants import (
    ModelVariants,
    RunResultsVariants,
    SimulationVariants,
    SolutionVariants,
    VariantCollection,
)


def examples_path(*parts: str) -> Path:
    """Return the path to the bundled examples directory.

    Installed packages ship the examples inside ``dyno/examples``; a source
    checkout uses the repository ``examples`` directory. Optional path parts
    are joined to the examples directory.
    """
    here = Path(__file__).resolve().parent
    root = here / "examples"
    if not root.is_dir():
        root = here.parents[1] / "examples"
    return root.joinpath(*parts)
