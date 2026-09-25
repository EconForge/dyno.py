# Quickstart: Writing a Dyno Model

This guide demonstrates how to build, inspect, verify, and solve a macroeconomic model from scratch using Dyno's native `.dyno` format.

---

## The Four Parts of a `.dyno` Model

A `.dyno` file is structured into four intuitive sections:

1. **Parameters & Calibrated Constants**: Assigned using `<-` or `:=`.
2. **Steady-State Values**: Designated with the `[~]` time index.
3. **Dynamic Equations**: Equilibrium conditions written with `=`.
4. **Exogenous Shocks**: Stochastic distributions declared via `N(std)` or `N(mean, std)`.

---

## Step 1: Writing the Model File

Create a file named `neo.dyno`:

```text
# 1. Calibrated Parameters
α <- 0.387
β <- 0.960
γ <- 4.000
δ <- 0.100
ρ <- 0.900

# 2. Steady-State Declarations
z[~] <- 0.0
k[~] <- ((1/β - (1-δ)) / α)**(1 / (α-1))
y[~] <- k[~]^α
i[~] <- δ * k[~]
c[~] <- y[~] - i[~]

# 3. Dynamic Equations
z[t] = ρ * z[t-1] + e_z[t]
y[t] = exp(z[t]) * k[t-1]^α
k[t] = k[t-1] * (1-δ) + i[t]
c[t] = exp(z[t]) * k[t-1]^α - i[t]
β * (c[t+1]/c[t])^(-γ) * (1 - δ + α * y[t+1]/k[t]) = 1

# 4. Exogenous Shocks
e_z[t] <- N(0.002)
```

### Syntax Cheat Sheet

| Notation | Meaning | Example |
|---|---|---|
| `α <- 0.387` | Parameter assignment | Sets constant value |
| `k[~] <- expr` | Steady-state expression | Solved before dynamic system |
| `k[t]` | Current period variable | Contemporaneous state or control |
| `k[t-1]` | Lagged variable | Predetermined capital stock |
| `c[t+1]` | Forward-looking lead | Expected future consumption |
| `e_z[t] <- N(σ)` | Gaussian shock | Standard deviation specification |
| `lhs = rhs` | Dynamic equilibrium condition | Model equation |

---

## Step 2: Loading and Inspecting the Model

Instantiate a `DynoModel` by pointing to your file:

```python
from dyno import DynoModel

model = DynoModel("neo.dyno")

# Inspect classified symbols
print("Endogenous:", model.symbols["endogenous"])
# ['z', 'k', 'y', 'i', 'c']

print("Exogenous:", model.symbols["exogenous"])
# ['e_z']

print("Parameters:", model.symbols["parameters"])
# ['α', 'β', 'γ', 'δ', 'ρ']

# Inspect evaluated steady-state values
print("Steady state:", model.steady_state)
```

---

## Step 3: Checking Steady-State Residuals

Before solving the model, always verify that your steady-state declarations satisfy all equations:

```python
# Evaluates LHS - RHS of each equation at steady state
print("Equation residuals:", model.residuals)

# Raises SteadyStateError if any equation residual > 1e-6
model.check()
```

> [!TIP]
> If you don't know the exact analytical steady state, provide reasonable initial guesses in `[~]` and call `model = model.steady()`. Dyno will run a numerical root finder (`hybr` / Powell) to solve for the steady state automatically!

---

## Step 4: Solving the Model

Solve the model using first-order perturbation (QZ generalized Schur decomposition):

```python
solution = model.solve()
```

The resulting `PerturbationSolution` represents the policy rule:

$$y_t = \bar{y} + X (y_{t-1} - \bar{y}) + Y \varepsilon_t$$

Where:

- $\bar{y}$ is the steady state (`solution.x0`)
- $X$ is the transition matrix (`solution.X`)
- $Y$ is the shock impact matrix (`solution.Y`)

```python
# View decision rule as formatted DataFrames
ss_df, coeffs_df = solution.coefficients_as_df()
print("Steady State:\n", ss_df)
print("\nTransition & Impact Matrices:\n", coeffs_df)
```

---

## Step 5: Impulse Response Functions (IRFs)

Compute impulse responses representing percentage deviations from steady state:

```python
# T=40 periods horizon
irfs_dict = solution.irfs(type="log-deviation", T=40)

# irfs_dict maps each shock name to a Pandas DataFrame
df = irfs_dict["e_z"]
print(df[["y", "c", "k", "i"]].head(10))
```

Plot the IRFs interactively with Plotly:

```python
fig = solution.plot(type="log-deviation")
fig.show()
```

---

## Step 6: Recalibration On the Fly

To perform comparative statics or sensitivity analysis, use `model.recalibrate()`:

```python
# Modify risk aversion γ and discount factor β
model_high_risk = model.recalibrate(γ=8.0, β=0.98)

# Re-solve and compare
sol_high_risk = model_high_risk.solve()
sol_high_risk.plot().show()
```

---

## Defining Models Inline from Strings

You can also pass model code directly as a multi-line Python string:

```python
txt = """
ρ <- 0.95
x[~] <- 0.0
e[t] <- N(0.01)
x[t] = ρ * x[t-1] + e[t]
"""

inline_model = DynoModel(txt=txt)
sol = inline_model.solve()
print("Autoregressive coefficient:", sol.X[0, 0])
```
