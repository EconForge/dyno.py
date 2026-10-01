"""The top-level ``dyno`` namespace exposes exactly the public API (issue #18)."""

import importlib

import dyno

EXPECTED_ALL = {
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
}


def test_all_contents():
    assert len(dyno.__all__) == len(set(dyno.__all__))
    assert set(dyno.__all__) == EXPECTED_ALL


def test_all_names_importable():
    namespace: dict[str, object] = {}
    exec("from dyno import *", namespace)
    for name in dyno.__all__:
        assert getattr(dyno, name) is namespace[name]


def test_low_level_names_not_at_top_level():
    for name in (
        "solve",
        "solve_qz",
        "solve_ti",
        "moments",
        "irf",
        "irfs",
        "simulate",
        "sim_to_nsim",
        "deterministic_solve",
        "NoConvergence",
    ):
        assert not hasattr(dyno, name), name


def test_low_level_names_reachable_from_submodules():
    solver = importlib.import_module("dyno.solver")
    simul = importlib.import_module("dyno.simul")
    for name in solver.__all__:
        assert hasattr(solver, name)
    for name in simul.__all__:
        assert hasattr(simul, name)
