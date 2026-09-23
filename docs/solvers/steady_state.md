# Steady State Computation & Verification

In dynamic economic modeling, the deterministic steady state represents the long-run stationary equilibrium where all time indices collapse ($y_{t-1} = y_t = y_{t+1} = \bar{y}$) and all stochastic shocks are set to zero ($\varepsilon_t = 0$).

Solving for the steady state is a necessary prerequisite for first-order perturbation around the steady-state point.

---

## Analytical Steady States (`[~]`)

Whenever possible, defining an exact closed-form analytical steady state is fastest and most reliable:

```text
# Analytical steady state for Neoclassical Growth Model
z[~] <- 0.0
k[~] <- ((1/β - (1-δ)) / α)**(1 / (α-1))
y[~] <- k[~]^α
i[~] <- δ * k[~]
c[~] <- y[~] - i[~]
```

Steady-state definitions:
- Can reference model parameters (`α`, `β`, `δ`).
- Can reference previously defined steady-state values (e.g., `y[~]` references `k[~]`).
- Are evaluated in topological order during model loading.

---

## Verifying Steady States

Dyno provides tools to verify that declared steady states satisfy the dynamic system:

### 1. `model.residuals`
Evaluates the absolute difference $|LHS - RHS|$ for each equation at the current steady-state point:

```python
model = DynoModel("my_model.dyno")

for i, res in enumerate(model.residuals, start=1):
    print(f"Eq {i} residual: {res:.2e}")
```

### 2. `model.check()`
Checks whether any residual exceeds the numerical tolerance (default: $10^{-6}$). If so, it raises a `SteadyStateError`:

```python
from dyno.errors import SteadyStateError

try:
    model.check()
    print("Steady state is verified!")
except SteadyStateError as e:
    print("Residual mismatch detected:")
    print(e)
```

---

## Numerical Steady State Finding (`model.steady()`)

For complex models where an analytical steady state is intractable, Dyno provides an automated numerical root finder.

### How it Works

1. Specify initial guesses in your `.dyno` file using `[~]`:
   ```text
   # Initial guesses
   k[~] <- 5.0
   c[~] <- 1.0
   y[~] <- 1.0
   ```
2. Call `model.steady()` in Python:
   ```python
   # Solves f(y_bar, y_bar, y_bar) = 0 numerically
   model = model.steady()

   # Verify convergence
   model.check()
   print("Computed Steady State:", model.steady_state)
   ```

`model.steady()` uses SciPy's non-linear root-finding algorithms (`hybr` / Powell hybrid method) to solve the stationary system of equations simultaneously.

---

## Steady State Diagnostics and Error Handling

If the numerical solver fails to converge, Dyno provides diagnostic information:

- **Unmatched variables**: Variables that appear in dynamic equations but lack initial guesses.
- **Equation residuals**: Which equations are furthest from equilibrium.
- **Bounds or scale issues**: When variables take invalid values (e.g., negative capital or zero consumption inside a logarithm).
