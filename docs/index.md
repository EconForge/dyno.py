# Dyno 🦖

<p align="left">
  <span class="badge badge-success">Python 3.10+</span>
  <span class="badge">DSGE Modeling</span>
  <span class="badge">Perturbation Solvers</span>
  <span class="badge">Deterministic Stacked-Time</span>
  <span class="badge badge-warning">Dynare Preprocessor Compatible</span>
</p>

**Dyno** is a modern, high-performance Python framework for Dynamic Stochastic General Equilibrium (DSGE) and macroeconomic modeling. Developed by the EconForge community, Dyno combines the simplicity and elegance of native Python workflows with the numerical power required by macroeconomic research, policy analysis, and quantitative modeling.

---

## Key Highlights

<div class="grid cards" markdown>

-   ### ✍️ Expressive DSL & Native Syntax
    Define models cleanly in human-readable `.dyno` files, embed them in YAML pipelines, or import legacy Dynare `.mod` files directly.

-   ### ⚡ State-of-the-Art Solvers
    Compute steady states, solve stochastic models via 1st-order perturbation (QZ Schur decomposition & Time Iteration), and solve large stacked-time deterministic models with exact automatic differentiation and sparse Jacobians.

-   ### 🔍 Diagnostic Rigor
    Inspect Blanchad-Kahn rank and order conditions, generalized eigenvalues, equation-level residuals, and theoretical asymptotic moments automatically.

-   ### 📊 Rich Visualization & Reporting
    Produce interactive Plotly impulse response charts, export tabular simulations as Pandas DataFrames, render dynamic reports, or explore models through an interactive Solara GUI dashboard.

</div>

---

## Quickstart in 60 Seconds

### 1. Define a Model (`neo.dyno`)

Dyno files clearly separate calibrated constants, steady-state expressions, dynamic equations, and shock distributions:

```text
# Parameters
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

# Exogenous shocks
e_z[t] <- N(0, 0.01^2)
```

### 2. Load, Check, and Solve

```python
from dyno import DynoModel

# Load the model
model = DynoModel("neo.dyno")

# Verify steady-state residuals (should evaluate to ~0)
print(model.residuals)
model.check()

# Solve using first-order perturbation (QZ decomposition)
solution = model.solve()

# Inspect decision rule: y_t = y_bar + X * (y_{t-1} - y_bar) + Y * eps_t
print("Transition matrix X:\n", solution.X)
print("Shock impact matrix Y:\n", solution.Y)

# Generate impulse responses (% log-deviation over 40 periods)
irfs = solution.irfs(type="log-deviation", T=40)
print(irfs["e_z"][["y", "c", "k", "i"]].head())

# Render interactive Plotly chart
fig = solution.plot(type="log-deviation")
fig.show()
```

---

## Navigating the Documentation

To get the most out of Dyno, explore the guided sections:

| Section | Focus |
|---|---|
| [**Getting Started**](getting_started/installation.md) | Pixi installation, environment configuration, and quickstarts for `.dyno` and Dynare `.mod` files. |
| [**Model Specification**](model_specification/syntax.md) | Syntax reference (`<-`, `=`, `[t]`, `[~]`, `forall`), equation blocks, metadata, and YAML wrappers. |
| [**Solvers & Theory**](solvers/steady_state.md) | Steady-state solvers, first-order perturbation (QZ / Time Iteration), and deterministic stacked-time Newton solver. |
| [**Analysis & Simulation**](analysis/irfs.md) | IRFs, conditional/unconditional variance-covariance moments, stochastic simulations, and automated reporting. |
| [**Tutorials**](tutorials/rbc_model.md) | Step-by-step guides for the canonical RBC model, deterministic transition paths, and the Solara GUI. |
| [**API Reference**](api/models.md) | Comprehensive class and function references for `DynoModel`, `DynareModel`, `solve`, and more. |