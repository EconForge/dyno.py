# Asymptotic Moments & Covariance Analysis

In business cycle analysis, comparing model-implied theoretical moments against empirical data (such as relative volatilities, contemporaneous correlations, and autocorrelations with GDP) is central to evaluating model performance.

---

## Theoretical Derivation

Consider the stationary first-order decision rule:

$$y_t = X y_{t-1} + Y \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d. } \mathcal{N}(0, \Sigma)$$

where $y_t$ denotes deviations from steady state.

### 1. Conditional Covariance Matrix ($\Gamma_0$)

The covariance of the contemporaneous shock impact is:

$$\Gamma_0 = \text{Cov}(Y \varepsilon_t) = Y \Sigma Y^T$$

### 2. Unconditional Covariance Matrix ($\Gamma$)

Applying the covariance operator to both sides of the VAR(1) system:

$$\text{Cov}(y_t) = X \, \text{Cov}(y_{t-1}) \, X^T + Y \, \text{Cov}(\varepsilon_t) \, Y^T$$

Under covariance stationarity, $\text{Cov}(y_t) = \text{Cov}(y_{t-1}) = \Gamma$. Thus, $\Gamma$ satisfies the **discrete Lyapunov equation**:

$$\Gamma = X \Gamma X^T + \Gamma_0$$

Dyno solves this discrete Lyapunov equation efficiently using doubled Schur/Kronecker algebraic methods.

---

## Computing Moments in Dyno

Call `solution.moments()` directly on a `PerturbationSolution`:

```python
from dyno import DynoModel

model = DynoModel("examples/modfiles/RBC.mod")
solution = model.solve()

# Returns (Gamma_0, Gamma)
Gamma_0, Gamma = solution.moments()
```

### Inspecting Standard Deviations & Correlations

Convert the unconditional covariance matrix $\Gamma$ into standard deviations and correlation matrices:

```python
import numpy as np
import pandas as pd

var_names = solution.symbols["endogenous"]

# Variance is the diagonal of Gamma
variances = np.diag(Gamma)
std_devs = np.sqrt(variances)

# Standard deviations DataFrame
df_volatility = pd.DataFrame({
    "Variable": var_names,
    "Std Dev (%)": std_devs * 100
}).set_index("Variable")

print("Theoretical Volatilities:")
print(df_volatility)

# Correlation matrix
diag_inv = np.diag(1.0 / std_devs)
corr_matrix = diag_inv @ Gamma @ diag_inv

df_corr = pd.DataFrame(corr_matrix, index=var_names, columns=var_names)
print("\nCorrelation Matrix:")
print(df_corr[["y", "c", "i", "n", "w"]].loc[["y", "c", "i", "n", "w"]])
```

---

## Standalone `moments` Function

You can also compute moments directly given arbitrary state-space matrices $(X, Y, \Sigma)$:

```python
from dyno.solver import moments

gamma_0, gamma = moments(X, Y, Sigma)
```
