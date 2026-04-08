# Getting Started: Solving a Dynare Model

This guide shows how to load an existing Dynare `.mod` file with Dyno, inspect and solve the model, and plot impulse response functions.

## Loading a `.mod` File

Dyno can read standard Dynare `.mod` files using the `DynoModel` class (Lark-based parser) or the `DynareModel` class (Dynare preprocessor).

```python
from dyno import DynoModel

model = DynoModel("examples/modfiles/RBC.mod")
```

The `.mod` file used here is a standard Real Business Cycle model:

```text
var c, r, w, n, k, i, y, a;
varexo epsilon, leta;
parameters beta, delta, khi, eta, alpha, rho, nss;

beta=0.985;
delta=0.025;
nss=0.33;
eta=1.0;
alpha=.33;
rho=.95;
khi=(1-alpha)*(1-nss)^eta/nss*(1/beta-1+delta)/(1/beta-1+delta-delta*alpha);

model;
1/c = beta*(r(+1)+1-delta)/c(1);
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
var epsilon; stderr .009;
var leta; stderr .001;
end;
```

## Inspecting the Model

Once loaded, you can inspect the model structure:

```python
# Variables and parameters
print(model.symbols["endogenous"])
# ['c', 'r', 'w', 'n', 'k', 'i', 'y', 'a']

print(model.symbols["exogenous"])
# ['epsilon', 'leta']

print(model.symbols["parameters"])
# ['beta', 'delta', 'khi', 'eta', 'alpha', 'rho', 'nss']

# Equations (as strings)
for eq in model.equations:
    print(eq)

# Steady-state values
print(model.steady_state)
```

## Checking the Model

Verify that the steady state satisfies the model equations:

```python
# Residuals at steady state (should be close to zero)
print(model.residuals)

# Raises SteadyStateError if any residual exceeds tolerance
model.check()
```

## Solving the Model

Solve using first-order perturbation (QZ decomposition):

```python
solution = model.solve()
```

The returned `PerturbationSolution` contains a decision rule of the form:

$$y_t = \bar{y} + X (y_{t-1} - \bar{y}) + Y \varepsilon_t$$

where $\bar{y}$ is the steady state, $X$ is the transition matrix, and $Y$ maps shocks to endogenous variables.

```python
# Steady-state vector
print(solution.x0)

# Transition matrix and shock impact matrix
print(solution.X)
print(solution.Y)

# Steady state and coefficients as DataFrames
ss_df, coeffs_df = solution.coefficients_as_df()
print(ss_df)
print(coeffs_df)
```

## Impulse Response Functions

Compute IRFs to each shock:

```python
irfs_dict = solution.irfs(type="log-deviation", T=40)

# irfs_dict is a dict: {shock_name: DataFrame}
for shock, df in irfs_dict.items():
    print(f"--- Response to {shock} ---")
    print(df[["y", "c", "k"]].head(10))
```

Plot the IRFs interactively (requires Plotly):

```python
fig = solution.plot(type="log-deviation")
fig.show()
```

## Recalibration

Create a variant of the model with different parameter values:

```python
model2 = model.recalibrate(beta=0.99, alpha=0.35)
solution2 = model2.solve()
solution2.plot().show()
```

## Using `DynareModel` Instead

If you need higher fidelity to Dynare's own preprocessor (e.g. for complex `.mod` features), use `DynareModel`:

```python
from dyno import DynareModel

model = DynareModel("examples/modfiles/RBC.mod")
solution = model.solve()
```

The API (`symbols`, `solve`, `residuals`, `jacobians`, etc.) is the same.
