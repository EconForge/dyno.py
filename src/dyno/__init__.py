"""Dyno package.

The names listed in ``__all__`` form the public API. Lower-level building
blocks (matrix solvers, simulation helpers, ...) remain available from their
submodules, e.g. ``dyno.solver.solve_qz`` or ``dyno.simul.simulate``.
"""

from pathlib import Path

from .dyno_model import DynoModel
from .dynare import DynareModel
from .errors import (
    BlanchardKahnError,
    ConvergenceWarning,
    DynareParserError,
    DynoError,
    LARKParserError,
    ParserError,
    RedefinitionWarning,
    SteadyStateError,
    SystemStructureError,
    UndefinedSymbolError,
    UndefinedSymbolWarning,
    UnsupportedFeatureError,
)
from .report import RunResults
from .simul import (
    IRFSimulation,
    RandomSimulation,
    SimulationResult,
    TransitionSimulation,
)
from .solver import PerturbationSolution, RecursiveDecisionRule
from .variants import (
    ModelVariants,
    RunResultsVariants,
    SimulationVariants,
    SolutionVariants,
    VariantCollection,
)

__all__ = [
    # Models
    "DynoModel",
    "DynareModel",
    # Results
    "RunResults",
    "PerturbationSolution",
    "RecursiveDecisionRule",
    "SimulationResult",
    "IRFSimulation",
    "RandomSimulation",
    "TransitionSimulation",
    # Variants
    "VariantCollection",
    "ModelVariants",
    "SolutionVariants",
    "SimulationVariants",
    "RunResultsVariants",
    # Errors and warnings
    "DynoError",
    "ParserError",
    "LARKParserError",
    "DynareParserError",
    "UnsupportedFeatureError",
    "UndefinedSymbolError",
    "SystemStructureError",
    "SteadyStateError",
    "BlanchardKahnError",
    "UndefinedSymbolWarning",
    "RedefinitionWarning",
    "ConvergenceWarning",
    # Utilities
    "examples_path",
]


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
