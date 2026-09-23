# Quickstart: Solving Dynare Models

Dyno provides native compatibility with Dynare `.mod` files, allowing macroeconomists to leverage existing model codebases while benefiting from Python's rich ecosystem of data analysis and visualization tools.

---

## Two Ways to Run Dynare Files in Dyno

Dyno offers two distinct approaches for loading `.mod` files:

1. **`DynoModel` (Default, Lark-based)**:
   - Uses an internal, pure-Python Lark parser.
   - Requires **no external C++ libraries** or MATLAB runtime.
   - Ideal for standard DSGE models, quick scripting, and cloud/CI environments.

2. **`DynareModel` (Official Preprocessor)**:
   - Uses the official Dynare preprocessor C++ library (`dynare-preprocessor-pylib`).
   - Ensures 100% adherence to Dynare preprocessing syntax and rules.
   - Requires the optional `dynare` feature in Pixi.

Both classes expose the exact same high-level Python API: `model.solve()`, `model.residuals`, `model.steady_state`, `model.recalibrate()`, and `model.run()`.

---

## Example: A Real Business Cycle (RBC) Model

Consider a standard Hansen Real Business Cycle model saved as `RBC.mod`:

```text
var c, r, w, n, k, i, y, a;
varexo epsilon, leta;
parameters beta, delta, khi, eta, alpha, rho, nss;

beta = 0.985;
delta = 0.025;
nss = 0.33;
eta = 1.0;
alpha = 0.33;
rho = 0.95;
khi = (1-alpha)*(1-nss)^eta/nss*(1/beta-1+delta)/(1/beta-1+delta-delta*alpha);

model;
1/c = beta*(r(+1)+1-delta)/c(+1);
w = khi*c/(1-n)^eta;
k = (1-delta)*k(-1)+i;
y = a*k(-1)^alpha*n^(1-alpha);
log(a) = rho*log(a(-1))+epsilon;
w = (1-alpha)*y/n;
r = alpha*y/k(-1);
y = c+i;
end;

steady_state_model;
a = 1;
r = 1/beta-1+delta;
n = nss;
k = (alpha/r)^(1/(1-alpha))*n;
y = k^alpha*n^(1-alpha);
w = (1-alpha)*y/n;
i = delta*k;
c = y-i;
end;

shocks;
var epsilon; stderr 0.009;
var leta; stderr 0.001;
end;
```

---

## Loading and Inspecting with `DynoModel`

```python
from dyno import DynoModel

# Load the .mod file directly
model = DynoModel("examples/modfiles/RBC.mod")

# Inspect symbols
print("Endogenous:", model.symbols["endogenous"])
# ['c', 'r', 'w', 'n', 'k', 'i', 'y', 'a']

print("Exogenous shocks:", model.symbols["exogenous"])
# ['epsilon', 'leta']

# Inspect parameters and steady states
print("Parameters:", model.context["constants"])
print("Steady State:", model.steady_state)
```

---

## Verifying Steady State and Diagnostics

Verify that the `steady_state_model` block satisfies the dynamic equilibrium conditions:

```python
# Check equation residuals
print("Residuals:", model.residuals)

# Strict validation (raises SteadyStateError on mismatch)
model.check()
```

---

## Solving and Simulating

Solve the model using generalized Schur (QZ) decomposition:

```python
solution = model.solve()

# Inspect decision rule components
print("Steady state vector x0:\n", solution.x0)
print("Transition matrix X:\n", solution.X)
print("Impact matrix Y:\n", solution.Y)
```

### Computing Impulse Responses

```python
# Generate log-deviation IRFs for 40 periods
irfs = solution.irfs(type="log-deviation", T=40)

# Extract response to technology shock 'epsilon'
df_tech = irfs["epsilon"]
print(df_tech[["y", "c", "i", "n", "w"]].head())

# Interactive visualization
fig = solution.plot(type="log-deviation")
fig.show()
```

---

## Recalibrating Parameters

You can modify parameter calibrations without editing the underlying `.mod` file:

```python
# Increase persistence rho and capital share alpha
model_alt = model.recalibrate(rho=0.98, alpha=0.36)

solution_alt = model_alt.solve()
solution_alt.plot().show()
```

---

## Using `DynareModel` (Dynare Preprocessor)

If your model uses advanced Dynare preprocessing features (such as intricate macro-processor loops, external steady-state functions, or complex block structures), use `DynareModel`:

```python
from dyno import DynareModel

# Uses dynare-preprocessor-pylib under the hood
model_official = DynareModel("examples/modfiles/RBC.mod")

solution_official = model_official.solve()
fig = solution_official.plot()
fig.show()
```

> [!NOTE]
> `DynareModel` requires `dynare-preprocessor-pylib`. If not already installed, activate it via the `dynare` environment:
> ```bash
> pixi run -e dev-dynare python my_script.py
> ```
