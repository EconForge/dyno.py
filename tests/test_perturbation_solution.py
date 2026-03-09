from dyno import DynoModel
from dyno.solver import PerturbationSolution, RecursiveDecisionRule


def test_perturb_returns_perturbation_solution_with_decision_rule():
    txt = """
rho := 0.9
x[~] := 0

e[t] := N(0, 1)

x[t] = rho*x[t-1] + e[t]
"""

    model = DynoModel(txt=txt)
    sol = model.perturb()

    assert isinstance(sol, PerturbationSolution)
    assert isinstance(sol.decision_rule, RecursiveDecisionRule)
    assert hasattr(sol, "evs")
    assert not hasattr(sol.decision_rule, "evs")

    # Backward-compatible accessors
    assert sol.X.shape == sol.decision_rule.X.shape
    assert sol.Y.shape == sol.decision_rule.Y.shape
    assert sol.symbols == sol.decision_rule.symbols
