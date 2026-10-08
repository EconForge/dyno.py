from dyno import DynoModel
from dyno import examples_path
from jax import numpy as jnp

model = DynoModel(examples_path("consumption_savings_iid.dyno"))


dsym = model.symbolic
from typing import Dict, Tuple


def CartesianGrid(
    domain: Dict[str, Tuple[float, float]], num_points: int
) -> Dict[str, jnp.ndarray]:
    """
    Create a Cartesian grid for the given domain and number of points.

    Args:
        domain: A dictionary where keys are variable names and values are tuples of (min, max) for the variable.
        num_points: The number of points to generate for each variable.
    Returns:
        A dictionary where keys are variable names and values are 1D arrays of grid points for that variable.
    """
    grid = {}
    for var, (min_val, max_val) in domain.items():
        grid[var] = jnp.linspace(min_val, max_val, num_points)
    return grid


def initial_guess(s):
    return {"c": s["w"] * 0.8}


domain = model.metadata["domain"]
s = CartesianGrid(domain, num_points=10)
N = len(s["w"])
x = initial_guess(s)
e = {"y": jnp.zeros(N)}

import numpy as np

mvn = dsym.context["processes"][("y",)]
Σ = mvn.Σ
Μ = mvn.Μ
# discretize the normal shock using gauss hermite nodes
from numpy.polynomial.hermite import hermgauss

nodes, weights = hermgauss(5)
nodes *= np.sqrt(2) * np.sqrt(Σ[0, 0])


g_eq_t = [
    eq
    for (eq, meta) in dsym.iter_equations_with_metadata()
    if "transition" in meta["tags"]
]
f_eq_t = [
    eq
    for (eq, meta) in dsym.iter_equations_with_metadata()
    if "arbitrage" in meta["tags"]
]

from copy import deepcopy

# construct the context for the transition equations
context_t = deepcopy(dsym.context)

from dyno.dynsym.analyze import EquationsEvaluator


def transition(s, x, e):

    context_t = deepcopy(dsym.context)

    for k, v in s.items():
        context_t["variables"][k] = {0: v, 1: v}
    for k, v in x.items():
        context_t["variables"][k] = {0: v, 1: v}
    for k, v in e.items():
        context_t["variables"][k] = {0: v}

    evaluator_t = EquationsEvaluator(context=context_t)
    evaluator_t.function_table.update(
        {
            "log": jnp.log,
            "exp": jnp.exp,
            "sqrt": jnp.sqrt,
            "abs": jnp.abs,
            "pow": jnp.pow,
        }
    )

    g_res_t = [evaluator_t.visit(eq) for eq in f_eq_t]
    return g_res_t


def arbitrage(s, x, S, X, e):
    context_t = deepcopy(dsym.context)

    for k, v in s.items():
        context_t["variables"][k] = {0: v, 1: v}
    for k, v in x.items():
        context_t["variables"][k] = {0: v, 1: v}
    for k, v in e.items():
        context_t["variables"][k] = {0: v}

    evaluator_f = EquationsEvaluator(context=context_t)
    evaluator_f.function_table.update(
        {
            "log": jnp.log,
            "exp": jnp.exp,
            "sqrt": jnp.sqrt,
            "abs": jnp.abs,
            "pow": jnp.pow,
        }
    )

    f_res_t = [evaluator_f.visit(eq) for eq in f_eq_t]
    return f_res_t


transition(s, x, e)

arbitrage(s, x, s, x, e)

s_ = {k: v[0] for k, v in s.items()}
x_ = {k: v[0] for k, v in x.items()}
e_ = {k: v[0] for k, v in e.items()}

arbitrage(s_, x_, s_, x_, e_)
