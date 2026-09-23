# Stochastic & Deterministic Simulation

Dyno provides routines for generating synthetic time-series trajectories through forward simulation of calibrated models.

---

## Stochastic Simulation

Simulate a model over $T$ periods subjected to random draws from its calibrated shock distributions:

```python
from dyno import DynoModel
from dyno.simul import simulate

model = DynoModel("examples/neo.dyno")
solution = model.solve()

# Simulate 200 periods with a random seed
df_sim = simulate(model, solution, T=200, seed=42)

print(df_sim.head())
```

The resulting `DataFrame` contains the generated trajectory for all endogenous variables.

---

## Simulating User-Specified Shock Paths

You can simulate model trajectories under historical or counterfactual shock sequences:

```python
import numpy as np
import pandas as pd

T = 50
shocks = {
    # Large temporary productivity boom followed by mean-reversion
    "e_z": np.array([0.02 * (0.8**t) for t in range(T)])
}

# Forward-propagate decision rule: y_t = X y_{t-1} + Y eps_t
var_names = solution.symbols["endogenous"]
X = solution.X
Y = solution.Y
x0 = solution.x0

trajectory = np.zeros((T, len(var_names)))
y_prev = np.zeros(len(var_names))

for t in range(T):
    eps_t = np.array([shocks["e_z"][t]])
    y_current = X @ y_prev + Y @ eps_t
    trajectory[t, :] = x0 + y_current  # Convert to levels
    y_prev = y_current

df_scenario = pd.DataFrame(trajectory, columns=var_names)
df_scenario[["y", "c", "k"]].plot(title="Simulated Policy Scenario")
```

---

## Monte Carlo Analysis

To assess sampling distributions or compute empirical moments:

```python
n_simulations = 500
horizon = 100

simulations = []
for seed in range(n_simulations):
    df_s = simulate(model, solution, T=horizon, seed=seed)
    simulations.append(df_s["y"])

df_mc = pd.concat(simulations, axis=1)

# Compute 10th, 50th, and 90th percentiles
p10 = df_mc.quantile(0.10, axis=1)
p50 = df_mc.quantile(0.50, axis=1)
p90 = df_mc.quantile(0.90, axis=1)
```

---

## Deterministic Simulation (Perfect Foresight)

For non-linear models without uncertainty, use `deterministic_solve`:

```python
from dyno.solver import deterministic_solve

df_det = deterministic_solve(model, T=100)
df_det.plot()
```
