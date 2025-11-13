# Solution Methods

## Deterministic Solution

The `deterministic_solve` method computes the perfect foresight solution (deterministic simulation) of a dynamic stochastic general equilibrium (DSGE) model over a finite time horizon.

### Method Signature

```python
def deterministic_solve(model, x0=None, T=None, method="hybr", verbose=True)
```

**Parameters:**

- `model`: A dyno model instance containing the equations and calibration
- `x0` (optional): Initial guess for the solution path. If `None`, uses `model.deterministic_guess(T=T)`
- `T` (optional): Time horizon (number of periods). If `None`, extracted from the initial guess or model context
- `method`: Solution method (currently uses Newton's method regardless of this parameter)
- `verbose`: Whether to print iteration information during solving

**Returns:**

- A pandas DataFrame containing the solution path for all variables over time periods `t = 0, 1, ..., T`

### Algorithm Overview

The method solves a large stacked-time nonlinear system using Newton's method with sparse Jacobian computations. The system stacks all variables across all time periods into a single vector and enforces that the model equations hold at each time period.

### The Stacked-Time System

#### Variables and Notation

Let:
- $p$ = number of variables (endogenous + exogenous)
- $q$ = number of dynamic equations
- $T$ = time horizon
- $v_t \in \mathbb{R}^p$ = vector of all variables at time $t$
- $V = (v_0, v_1, \ldots, v_T) \in \mathbb{R}^{(T+1) \times p}$ = stacked variables over all time periods

The model consists of $q$ dynamic equations of the form:

$$f_i(v_{t-1}, v_t, v_{t+1}) = 0, \quad i = 1, \ldots, q$$

where the equations can depend on lagged, current, and future values of the variables.

#### Boundary Conditions

The system requires boundary conditions:

1. **Initial condition** ($t=0$): All variables are fixed to their initial values:
   $$v_0 = \bar{v}_0$$
   where $\bar{v}_0$ is determined from the model's initial conditions (typically from the context's `values` dictionary).

2. **Terminal condition** ($t = T$): For the last period, we use a modified terminal condition that effectively assumes the system converges to a steady state:
   - At $t=T$: $f(v_T, v_T, v_T)$ (assuming all time indices collapse to $v_T$)

3. **Exogenous variables**: The remaining $n_{exo} = p - q$ variables (typically exogenous shocks) are pinned to their specified paths:
   $$v_t^{exo} = \bar{v}_t^{exo}, \quad t = 0, 1, \ldots, T$$

#### Stacked System Formulation

The complete stacked system $F(V) = 0$ where $F: \mathbb{R}^{(T+1) \times p} \to \mathbb{R}^{(T+1) \times p}$ is defined as:

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
\end{bmatrix} = 0
$$

where:
- $F_0(V) = v_0 - \bar{v}_0$ enforces the initial condition
- For $t = 1, \ldots, T-1$: $F_t(V) = \begin{bmatrix} f(v_{t-1}, v_t, v_{t+1}) \\ v_t^{exo} - \bar{v}_t^{exo} \end{bmatrix}$ (standard form)
- For $t = T$: $F_T(V) = \begin{bmatrix} f(v_T, v_T, v_T) \\ v_T^{exo} - \bar{v}_T^{exo} \end{bmatrix}$ (all time indices collapse to $v_T$)

More precisely, for periods $t = 1, \ldots, T-1$:

$$
F_t(V) = \begin{bmatrix}
f_1(v_{t-1}, v_t, v_{t+1}) \\
\vdots \\
f_q(v_{t-1}, v_t, v_{t+1}) \\
v_{t,q+1} - \bar{v}_{t,q+1} \\
\vdots \\
v_{t,p} - \bar{v}_{t,p}
\end{bmatrix}
$$

And for the terminal period $t = T$:

$$
F_T(V) = \begin{bmatrix}
f_1(v_T, v_T, v_T) \\
\vdots \\
f_q(v_T, v_T, v_T) \\
v_{T,q+1} - \bar{v}_{T,q+1} \\
\vdots \\
v_{T,p} - \bar{v}_{T,p}
\end{bmatrix}
$$

#### Jacobian Structure

The Jacobian $J = \frac{\partial F}{\partial V} \in \mathbb{R}^{(T+1)p \times (T+1)p}$ has a sparse block-tridiagonal structure:

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
- $I_p$ is the $p \times p$ identity matrix for the initial condition
- $D_t^{(-1)} = \frac{\partial F_t}{\partial v_{t-1}} \in \mathbb{R}^{p \times p}$ captures effects of lagged variables
- $D_t^{(0)} = \frac{\partial F_t}{\partial v_t} \in \mathbb{R}^{p \times p}$ captures effects of current variables
- $D_t^{(1)} = \frac{\partial F_t}{\partial v_{t+1}} \in \mathbb{R}^{p \times p}$ captures effects of future variables

For the terminal period ($t=T$), $D_T^{(-1)} + D_T^{(0)} + D_T^{(1)}$ combines all derivatives since all time indices in $f(v_T, v_T, v_T)$ point to $v_T$.

The derivative blocks for $t = 1, \ldots, T$ have the structure:

$$
D_t^{(k)} = \begin{bmatrix}
\frac{\partial f_1}{\partial v_{t+k}} \\
\vdots \\
\frac{\partial f_q}{\partial v_{t+k}} \\
\mathbf{0}_{(p-q) \times p} \text{ if } k \neq 0 \\
I_{p-q} \text{ if } k = 0
\end{bmatrix}
$$

for $k \in \{-1, 0, 1\}$, where the bottom rows correspond to the exogenous variables being pinned to their values.

### Solution Procedure

1. **Initialize**: Obtain initial guess $V^{(0)}$ either from user input or `model.deterministic_guess(T)`

2. **Newton Iterations**: For iteration $k = 0, 1, 2, \ldots$:
   - Compute residuals $F(V^{(k)})$ and sparse Jacobian $J(V^{(k)})$
   - Solve the linear system: $J(V^{(k)}) \Delta V = -F(V^{(k)})$
   - Update: $V^{(k+1)} = V^{(k)} + \Delta V$
   - Check convergence: if $\|F(V^{(k+1)})\| < \epsilon$, stop

3. **Return**: Convert the solution $V^*$ to a pandas DataFrame with columns for each variable and rows for each time period

### Implementation Notes

- The method uses **automatic differentiation** (via `DNumber` class) to compute exact Jacobians
- The Jacobian is stored in **sparse CSR format** to exploit the block-tridiagonal structure
- The sparse linear system is solved efficiently using scipy's sparse solvers
- The terminal condition $v_{T+1} = v_T$ is handled by setting the forward values equal to the current values in the last period

### Example Usage

```python
import dyno

# Load a model
model = dyno.load("path/to/model.dyno")

# Solve for 100 periods with default initial guess
solution = model.deterministic_solve(T=100)

# Plot results
import matplotlib.pyplot as plt
solution.plot(x='t', y=['consumption', 'capital'])
plt.show()
```

### Related Methods

- `deterministic_guess(model, T)`: Generates an initial guess for the solution path
- `deterministic_residuals(model, v)`: Computes residuals $F(V)$ for a given path
- `deterministic_residuals_with_jacobian(model, v)`: Computes both residuals and Jacobian
