# Stochastic & Deterministic Simulation

Dyno provides routines for generating synthetic time-series trajectories through forward simulation of calibrated models.

---

## Stochastic Simulation

Simulate a model over $T$ periods subjected to random draws from its calibrated shock distributions:

```python
from dyno import DynoModel

model = DynoModel("examples/neo.dyno")
solution = model.solve()

# Simulate 200 periods with a random seed
sim = solution.simulate(T=200, rng=42)
df_sim = sim.to_df()

print(df_sim.head())
```

The resulting `DataFrame` contains the generated trajectory for all endogenous variables.

---

## Simulating User-Specified Shock Paths

You can simulate model trajectories under historical or counterfactual shock sequences using `solution.simulate(shocks=...)`:

```python
import numpy as np

T = 50
shocks = {
    # Large temporary productivity boom followed by mean-reversion
    "e_z": {t: float(0.02 * (0.8**t)) for t in range(T)}
}

sim_scenario = solution.simulate(T=T, shocks=shocks, units="level")
df_scenario = sim_scenario.to_df()
df_scenario[["y", "c", "k"]].plot(title="Simulated Policy Scenario")
```

---

## Monte Carlo Analysis

To assess sampling distributions or compute empirical moments, simulate $N$ random draws directly using the `N` argument:

```python
n_simulations = 500
horizon = 100

# Simulate 500 stochastic paths over a 100-period horizon
sim_mc = solution.simulate(T=horizon, N=n_simulations, rng=42)
df_mc = sim_mc.to_df()  # MultiIndex DataFrame (n, t)

# Extract output 'y' across draws (unstack across trajectory draws)
y_draws = df_mc["y"].unstack(level=0)

# Compute 10th, 50th, and 90th percentiles
p10 = y_draws.quantile(0.10, axis=1)
p50 = y_draws.quantile(0.50, axis=1)
p90 = y_draws.quantile(0.90, axis=1)
```

---

## Direct Model Simulation with `model.simulate()`

You can also simulate directly from a model instance without manually calling `model.solve()` first:

```python
sim = model.simulate(T=200, rng=42)
```

`model.simulate()` is type-stable across all model families, always returning a `SimulationResult`.

For stochastic models, `model.simulate()` solves perturbation under the hood and caches the solution. You can control solving behavior with the `solve` parameter:
- `solve=None` (default): reuses the cached solution if already solved, otherwise solves automatically.
- `solve=True`: forces solving the model freshly.
- `solve=solution`: uses an explicitly passed `PerturbationSolution`.
- `solve=False`: requires that the model has already been solved, raising `ValueError` otherwise.

---

## Deterministic Simulation (Perfect Foresight)

For non-linear models without uncertainty, use `model.simulate()` (or the lower-level `deterministic_solve`):

```python
from dyno import DynoModel

model_det = DynoModel("examples/ramst.dyno")
sim_det = model_det.simulate(T=100)
df_det = sim_det.to_df()
df_det[["k", "c"]].plot(title="Deterministic Transition Path")
```

Because deterministic models under perfect foresight solve for a specific trajectory across $t = 0 \dots T$ rather than a state-space policy function, they are simulated directly via `model.simulate()`. Calling `model.solve()` on a deterministic model raises a `SystemStructureError`.

