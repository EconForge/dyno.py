# Dyno

Dyno is a Python library for specifying, solving, and simulating dynamic macroeconomic models, including Dynamic Stochastic General Equilibrium (DSGE) and deterministic transition models. It combines a human-readable modeling language (`.dyno`) with native support for Dynare `.mod` files and standard numerical solvers for macroeconomic analysis.

---

## Features

<div class="grid cards" markdown>

-   ### Model Specification
    Write models in clean `.dyno` files with explicit time indexing (`[t]`, `[t-1]`, `[t+1]`, `[~]`), embed specifications in YAML configurations, or load existing Dynare `.mod` files directly.

-   ### Perturbation Solvers
    Compute steady states and solve linearized rational expectations models using first-order perturbation, supporting generalized Schur (QZ) decomposition and time iteration.

-   ### Deterministic Simulation
    Solve non-linear perfect foresight transition dynamics over finite horizons using exact automatic differentiation and sparse block-tridiagonal Newton solvers.

-   ### Diagnostics & Analysis
    Evaluate steady-state equation residuals, verify Blanchard-Kahn rank and order conditions, compute asymptotic variance-covariance moments via Lyapunov equations, and generate impulse response functions.

-   ### Visualization & Tooling
    Export simulation trajectories to Pandas DataFrames, render interactive charts with Plotly or Altair, or explore models interactively in JupyterLab using the Dyno Lab extension.

</div>

---

## Quickstart

### 1. Write a Model (`neo.dyno`)

A `.dyno` file separates calibrated constants, steady-state declarations, dynamic equations, and shock processes:

```text
# Calibrated parameters
α <- 0.36
β <- 0.99
δ <- 0.025
ρ <- 0.95
γ <- 2.0

# Steady-state definitions
z[~] <- 0.0
k[~] <- ((1/β - (1-δ)) / α)**(1 / (α-1))
y[~] <- k[~]^α
i[~] <- δ * k[~]
c[~] <- y[~] - i[~]

# Dynamic equations
z[t] = ρ * z[t-1] + e_z[t]
y[t] = exp(z[t]) * k[t-1]^α
k[t] = (1-δ) * k[t-1] + i[t]
c[t] = y[t] - i[t]
β * (c[t+1]/c[t])^(-γ) * (α * y[t+1]/k[t] + 1 - δ) = 1

# Exogenous shocks (standard deviation 0.01)
e_z[t] <- N(0.01)
```

### 2. Solve and Inspect in Python

```python
from dyno import DynoModel

# Load the model
model = DynoModel("neo.dyno")

# Verify that steady-state declarations satisfy all equations
print("Steady-state residuals:", model.residuals)
model.check()

# Solve the model using first-order perturbation (QZ decomposition)
solution = model.solve()

# Inspect the state-space decision rules: y_t = y_ss + X * (y_{t-1} - y_ss) + Y * eps_t
print("Transition matrix X:\n", solution.X)
print("Impact matrix Y:\n", solution.Y)

# Compute impulse response functions (deviations from steady state over 40 periods)
irfs = solution.irfs(type="deviation", T=40)
print(irfs["e_z"][["y", "c", "k", "i"]].head())

# Render an interactive Plotly chart
fig = solution.plot(type="deviation")
fig.show()
```

---

## Documentation Overview

| Section | Description |
|---|---|
| [**Getting Started**](getting_started/installation.md) | Installation with Pixi, environment options, and introductory walkthroughs for `.dyno` and `.mod` files. |
| [**Model Specification**](model_specification/syntax.md) | Syntax reference for declarations (`<-`), equations (`=`), time indices, metadata tags, and YAML wrappers. |
| [**Solvers & Theory**](solvers/steady_state.md) | Numerical root-finding for steady states, first-order perturbation methods, and deterministic stacked-time Newton solvers. |
| [**Analysis & Simulation**](analysis/irfs.md) | Impulse response functions, theoretical moments, stochastic simulation, and automated reporting. |
| [**Subpackages & Architecture**](subpackages/index.md) | Architectural layout of Dyno and its companion modules (`dynspec` and `dyno.dynare`). |
| [**Dyno Lab (GUI)**](dyno_lab/index.md) | Interactive JupyterLab extension providing live model evaluation, error tracking, and visual diagnostics. |
| [**Tutorials**](tutorials/rbc_model.md) | Step-by-step guides for neoclassical RBC modeling, policy experiments, and interactive workflows. |
| [**API Reference**](api/models.md) | Python class and function signatures across the package. |