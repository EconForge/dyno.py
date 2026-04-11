# Getting Started: Writing a Dyno Model

This guide walks through writing a model from scratch in the native `.dyno` format, loading it, and solving it.

## The `.dyno` Format

A `.dyno` file has four parts:

1. **Parameters** — calibrated constants
2. **Steady-state values** — initial guesses or analytical steady states
3. **Dynamic equations** — the model
4. **Exogenous processes** — shock distributions

## A Neoclassical Growth Model

Create a file `neo.dyno`:

```text
# Parameters
α <- 0.387
β <- 0.96
γ <- 4.0
δ <- 0.1
ρ <- 0.9

# Steady-state values
z[~] <- 0.0
k[~] <- ((1/β-(1-δ))/α)**(1/(α-1))
y[~] <- k[~]^α
i[~] <- δ*k[~]
c[~] <- y[~] - i[~]

# Dynamic equations
z[t] = ρ*z[t-1] + e_z[t] + e_y[t]
y[t] = exp(z[t])*k[t-1]^α
k[t] = k[t-1]*(1-δ) + i[t]
c[t] = exp(z[t])*k[t-1]^α - i[t]
β*(c[t+1]/c[t])^(-γ)*(1-δ+α*y[t+1]/k[t]) = 1

# Exogenous shocks
e_z[t] <- N(0, 0.002^2)
e_y[t] <- N(0, 0.001^2)
```

### Syntax Highlights

| Syntax | Meaning |
|---|---|
| `α <- 0.387` | Parameter assignment |
| `k[~] <- expr` | Steady-state value for variable `k` |
| `k[t]`, `k[t-1]`, `c[t+1]` | Current, lagged, and lead variables |
| `e_z[t] <- N(0, σ²)` | Exogenous shock (normal distribution) |
| `lhs = rhs` | Dynamic equation (equilibrium condition) |

Steady-state expressions can reference parameters and previously defined steady states (e.g. `i[~] <- δ*k[~]`).

## Loading and Inspecting

```python
from dyno import DynoModel

model = DynoModel("neo.dyno")

# Model structure
print(model.symbols["endogenous"])   # ['z', 'k', 'y', 'i', 'c']
print(model.symbols["exogenous"])    # ['e_z', 'e_y']
print(model.symbols["parameters"])   # ['α', 'β', 'γ', 'δ', 'ρ']

# Steady-state values
print(model.steady_state)

# Check residuals at steady state
print(model.residuals)
model.check()
```

## Solving

```python
solution = model.solve()

# Impulse response functions (% deviation from steady state)
irfs_dict = solution.irfs(type="log-deviation", T=40)

for shock, df in irfs_dict.items():
    print(f"--- {shock} ---")
    print(df.head(10))

# Interactive plot
fig = solution.plot(type="log-deviation")
fig.show()
```

## Creating a Model from a String

You can also define a model inline without a file:

```python
from dyno import DynoModel

txt = """
ρ <- 0.9
x[~] <- 0.0
e[t] <- N(0, 0.01^2)
x[t] = ρ*x[t-1] + e[t]
"""

model = DynoModel(txt=txt)
solution = model.solve()
solution.plot().show()
```

## Numerical Steady State

If you cannot provide an analytical steady state, give initial guesses and let Dyno solve for it numerically:

```python
model = DynoModel("neo.dyno")

# Solve for the steady state numerically
model = model.steady()

# Verify
print(model.residuals)  # should be ~0
```

## Recalibration

Change parameters and re-solve without rewriting the file:

```python
model2 = model.recalibrate(β=0.99, δ=0.05)
solution2 = model2.solve()
solution2.plot().show()
```

## Run Directives

`.dyno` files can include `@run:` directives that execute a pipeline when using `model.run()`:

```text
@run: steady
@run: check
@run: solve
@run: simul: {T: 20}
```

```python
model = DynoModel("neo.dyno")
results = model.run()

results.model       # model (after steady-state solve)
results.solution    # PerturbationSolution
results.simulation  # simulation DataFrame
```

## Deterministic Models

If no exogenous shocks are declared, the model is deterministic. Use initial values and deterministic paths instead:

```text
# Parameters
alpha <- 0.36
rho   <- 0.95
beta  <- 1/1.01
delta <- 0.025

T <- 20

# Steady state
y[~] <- 1.08
c[~] <- 0.80
k[~] <- 11.08
a[~] <- 0

# Exogenous path (not a shock — no N(...))
e[t] <- N(0, 0.002)

# Dynamic equations
y[t] = exp(a[t])*(k[t-1]^alpha)
k[t] = y[t] - c[t] + (1-delta)*k[t-1]
a[t] = rho*a[t-1] + e[t]

# Initial condition override
k[0] <- k[~]*1.01

# Deterministic shock path
∀ t, 0 <= t < 10 : e[t] <- 0.99/(t+1)
```

For deterministic models, `model.solve()` returns a `DataFrame` with the perfect-foresight solution path:

```python
model = DynoModel("deterministic.dyno")
df = model.solve()
print(df)
```
