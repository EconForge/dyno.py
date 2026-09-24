# First-Order Perturbation & Stochastic Solvers

Perturbation is the standard numerical method for solving Dynamic Stochastic General Equilibrium (DSGE) models. Dyno solves linearized models around their deterministic steady state using first-order perturbation.

---

## Theoretical Framework

Consider a non-linear DSGE model consisting of $n$ equations:

$$\mathbb{E}_t \left[ f(y_{t+1}, y_t, y_{t-1}, \varepsilon_t) \right] = 0$$

where:

- $y_t \in \mathbb{R}^n$ is the vector of endogenous variables.
- $\varepsilon_t \in \mathbb{R}^m$ is the vector of i.i.d. exogenous shocks with covariance matrix $\Sigma = \mathbb{E}[\varepsilon_t \varepsilon_t^T]$.
- $\bar{y}$ is the deterministic steady state satisfying $f(\bar{y}, \bar{y}, \bar{y}, 0) = 0$.

### Linearization

Taking a first-order Taylor expansion around $(\bar{y}, \bar{y}, \bar{y}, 0)$ yields:

$$A \, \mathbb{E}_t[\hat{y}_{t+1}] + B \, \hat{y}_t + C \, \hat{y}_{t-1} + D \, \varepsilon_t = 0$$

where $\hat{y}_t = y_t - \bar{y}$ denotes deviation from steady state, and the Jacobian matrices are:

$$A = \frac{\partial f}{\partial y_{t+1}}, \quad B = \frac{\partial f}{\partial y_t}, \quad C = \frac{\partial f}{\partial y_{t-1}}, \quad D = \frac{\partial f}{\partial \varepsilon_t}$$

### The Recursive Decision Rule

The unique stable solution takes the recursive Vector Autoregressive form:

$$\hat{y}_t = X \hat{y}_{t-1} + Y \varepsilon_t$$

Substituting this policy function into the linearized dynamic equation gives:

$$(A X^2 + B X + C) \hat{y}_{t-1} + (A X Y + B Y + D) \varepsilon_t = 0$$

This yields two fundamental matrix equations:

1. **The Quadratic Matrix Equation** for the transition matrix $X$:
   $$A X^2 + B X + C = 0$$

2. **The Shock Transmission Equation** for the impact matrix $Y$:
   $$(A X + B) Y + D = 0 \implies Y = -(A X + B)^{-1} D$$

---

## Solvers in Dyno

Dyno provides two algorithms to solve the quadratic matrix equation $A X^2 + B X + C = 0$:

### 1. Generalized Schur / QZ Decomposition (`method="qz"`, Default)

The QZ decomposition transforms the matrix pencil into upper triangular forms:

- Constructs the block state-space representation.
- Computes generalized eigenvalues $\lambda_i = \alpha_i / \beta_i$.
- Reorders the decomposition so stable eigenvalues ($|\lambda_i| < 1$) appear first.
- Computes $X$ directly using the stable subspace.

```python
solution = model.solve(method="qz")
```

### 2. Time Iteration (`method="ti"`)

Time iteration is a fixed-point iteration algorithm that solves for $X$:

$$X_{k+1} = -(B + A X_k)^{-1} C$$

Starting from an initial guess $X_0 = 0$, iteration continues until $\|X_{k+1} - X_k\|_\infty < \text{tol}$.

```python
solution = model.solve(method="ti", options={"maxiter": 1000, "tol": 1e-10})
```

---

## Blanchard-Kahn Conditions

For a rational expectations equilibrium to be unique and stable (saddle-path stable), the **Blanchard-Kahn (1980)** conditions must hold:

1. **Order Condition**: The number of generalized eigenvalues outside the unit circle ($|\lambda_i| > 1$) must equal the number of forward-looking (non-predetermined) variables.
2. **Rank Condition**: The eigenvectors associated with the unstable roots must span the space of forward-looking variables.

### Diagnostic Outcomes

- **Determinacy (Unique Stable Path)**: Exactly matches $\implies$ solution found.
- **Indeterminacy**: Too few explosive eigenvalues $\implies$ multiplicity of equilibria (sunspot equilibria).
- **No Stable Solution (Explosive)**: Too many explosive eigenvalues $\implies$ no non-explosive path exists.

If Blanchard-Kahn conditions fail, Dyno raises `BlanchardKahnError`:

```python
from dyno.errors import BlanchardKahnError

try:
    solution = model.solve()
except BlanchardKahnError as e:
    print("Blanchard-Kahn failure:", e)
```

---

## Working with `PerturbationSolution`

The object returned by `model.solve()` is a `PerturbationSolution` wrapping a `RecursiveDecisionRule`:

```python
sol = model.solve()

# Key attributes:
sol.x0          # Steady-state vector y_bar
sol.X           # Transition matrix (n x n)
sol.Y           # Shock impact matrix (n x m)
sol.Σ           # Shock covariance matrix (m x m)
sol.evs         # Sorted generalized eigenvalues

# Format as Pandas DataFrames:
ss_df, coeffs_df = sol.coefficients_as_df()
```

### Eigenvalue Inspection

```python
import numpy as np

eigenvalues = sol.evs
moduli = np.abs(eigenvalues)

print(f"Number of eigenvalues: {len(eigenvalues)}")
print(f"Stable roots (|λ| < 1): {np.sum(moduli < 1.0)}")
print(f"Unit roots (|λ| ≈ 1): {np.sum(np.isclose(moduli, 1.0, atol=1e-6))}")
print(f"Explosive roots (|λ| > 1): {np.sum(moduli > 1.0)}")
```
