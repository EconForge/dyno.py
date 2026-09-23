# Deterministic & Perfect Foresight Solvers

When analyzing non-linear transition dynamics, anticipated policy reforms, or deterministic shocks over a finite time horizon, first-order perturbation is insufficient. Dyno implements a high-performance **stacked-time Newton-Raphson solver** with exact automatic differentiation and sparse block-tridiagonal matrix factorization.

---

## Overview

The `deterministic_solve` method solves the non-linear perfect foresight system:

```python
from dyno.solver import deterministic_solve

df_trajectory = deterministic_solve(model, T=100)
```

**Parameters:**
- `model`: A `DynoModel` instance containing equations, calibration, and boundary values.
- `x0` *(optional)*: Initial guess matrix for the solution path across all periods. Defaults to steady state.
- `T` *(optional)*: Time horizon (number of periods). Inferred from model context if omitted.
- `verbose` *(bool)*: Print iteration progress and residual norms.

**Returns:**
- A `pandas.DataFrame` where rows correspond to time periods $t = 0, 1, \dots, T$ and columns represent model variables.

---

## Mathematical Formulation

### Variables and Stacking

Let:
- $p$ be the total number of variables (endogenous + exogenous).
- $q$ be the number of dynamic equations ($q \le p$).
- $T$ be the simulation horizon.
- $v_t \in \mathbb{R}^p$ be the vector of all variables at time $t$.
- $V = (v_0, v_1, \ldots, v_T) \in \mathbb{R}^{(T+1) \times p}$ be the stacked trajectory matrix.

The model equations are expressed in zero-form:

$$f_i(v_{t-1}, v_t, v_{t+1}) = 0, \quad i = 1, \ldots, q$$

### Boundary Conditions

To close the $(T+1) \times p$ dimensional system, three sets of boundary conditions are applied:

1. **Initial Condition ($t = 0$)**:
   State variables at date 0 are pinned to predetermined historical or policy values:
   $$v_0 = \bar{v}_0$$

2. **Exogenous Path Pinning ($t = 0, \ldots, T$)**:
   The $p - q$ exogenous variables (e.g., tax rates, technology paths, government spending) are pinned to their deterministic profiles:
   $$v_t^{\text{exo}} = \bar{v}_t^{\text{exo}}$$

3. **Terminal Condition ($t = T$)**:
   At the horizon $T$, forward-looking variables are assumed to have converged to a stationary steady state. Dyno collapses forward indices:
   $$f(v_T, v_T, v_T) = 0$$

### The Complete Stacked System

The full non-linear system $F(V) = 0$ is defined as:

$$
F(V) = \begin{bmatrix}
F_0(V) \\
F_1(V) \\
\vdots \\
F_{T-1}(V) \\
F_T(V)
\end{bmatrix} = \begin{bmatrix}
v_0 - \bar{v}_0 \\
f(v_0, v_1, v_2) \\
\vdots \\
f(v_{T-2}, v_{T-1}, v_T) \\
f(v_T, v_T, v_T)
\end{bmatrix} = \mathbf{0}
$$

---

## Sparse Block-Tridiagonal Jacobian Structure

Because equations at time $t$ depend only on $v_{t-1}$, $v_t$, and $v_{t+1}$, the global Jacobian $J = \frac{\partial F}{\partial V}$ exhibits a sparse block-tridiagonal structure:

$$
J = \begin{bmatrix}
I_p & 0 & 0 & \cdots & 0 & 0\\
D_1^{(-1)} & D_1^{(0)} & D_1^{(1)} & \cdots & 0 & 0\\
0 & D_2^{(-1)} & D_2^{(0)} & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & D_{T-1}^{(0)} & D_{T-1}^{(1)} \\
0 & 0 & 0 & \cdots & 0 & D_T^{(-1)} + D_T^{(0)} + D_T^{(1)}
\end{bmatrix}
$$

where:
- $D_t^{(-1)} = \frac{\partial F_t}{\partial v_{t-1}}$ (derivatives with respect to lags)
- $D_t^{(0)} = \frac{\partial F_t}{\partial v_t}$ (derivatives with respect to current variables)
- $D_t^{(1)} = \frac{\partial F_t}{\partial v_{t+1}}$ (derivatives with respect to leads)

### Computational Efficiency

- **Automatic Differentiation**: Dyno uses a forward-mode automatic differentiation engine (`DNumber`) to evaluate exact machine-precision Jacobians without symbolic explosion.
- **Sparse CSR Format**: Storing $J$ in SciPy's Compressed Sparse Row (`csr_matrix`) format reduces memory footprint from $\mathcal{O}(T^2 p^2)$ to $\mathcal{O}(T p^2)$.
- **Sparse Direct Factorization**: Linear Newton steps $J \Delta V = -F(V)$ are solved in $\mathcal{O}(T)$ operations using sparse LU factorization (`scipy.sparse.linalg.spsolve`).

---

## Solution Procedure

```mermaid
graph TD
    A[Initial Guess V_0 from Steady State] --> B[Evaluate Residuals F V_k and Jacobian J V_k]
    B --> C{Check Convergence ||F|| < tol?}
    C -->|Yes| D[Return Solution Trajectory DataFrame]
    C -->|No| E[Solve Sparse Linear System J ΔV = -F]
    E --> F[Update Trajectory: V_k+1 = V_k + ΔV]
    F --> B
```

---

## Practical Example: A Permanent Shock

Consider a model with a permanent technology improvement:

```text
# neo_det.dyno
alpha <- 0.36
beta  <- 0.99
delta <- 0.025
rho   <- 0.95

# Horizon
T <- 50

# Steady states
k[~] <- 10.0
c[~] <- 0.80
y[~] <- 1.05
a[~] <- 0.0

# Dynamic equations
y[t] = exp(a[t]) * k[t-1]^alpha
k[t] = y[t] - c[t] + (1-delta)*k[t-1]
1/c[t] = beta * (1/c[t+1]) * (alpha*y[t+1]/k[t] + 1 - delta)
a[t] = rho*a[t-1] + e[t]

# Initial capital perturbed 5% below steady state
k[0] <- k[~] * 0.95

# Permanent shock path
e[0] <- 0.05
forall t, 1 <= t < 50 : e[t] <- 0.0
```

Solve and plot the transition trajectory:

```python
from dyno import DynoModel

model = DynoModel("neo_det.dyno")
trajectory = model.solve()  # Automatically detects deterministic model

print(trajectory[["k", "c", "y", "a"]].head(10))

# Plot transition
import matplotlib.pyplot as plt
trajectory[["k", "c", "y"]].plot(title="Deterministic Transition Path")
plt.xlabel("Period t")
plt.grid(True)
plt.show()
```
