# Function Generation & Compilation

Once an equation group has been validated against a model recipe, DynSpec can compile the symbolic AST directly into high-performance, executable numerical functions using `dyno.dynspec.funcgen`.

---

## 1. Compiling Equation Groups

The `compile_equation_group()` function compiles a list of AST equation nodes into a unified, vectorized Python callable:

```python
from dyno.dynspec.funcgen import compile_equation_group
from dyno.dynspec.recipe import DTCC_RECIPE

spec = DTCC_RECIPE.get_group_spec("transition")

conformity, func = compile_equation_group(
    transition_equations,
    variables=variables,
    spec=spec,
    constants=model.context["constants"]
)

assert conformity.ok
assert callable(func)
```

---

## 2. Function Execution Semantics

The signature and behavior of the compiled function depend on the equation group's role in the recipe:

### A. Recursive Groups (e.g., Transition Functions)

For state transition equations ($s_t = g(m_{t-1}, s_{t-1}, x_{t-1}, m_t)$):
- **Inputs**: NumPy arrays ordered according to the recipe's `allowed` groups:
  `func(exo_prev, states_prev, controls_prev, exo_curr)`
- **Output**: A 1-D NumPy array containing the updated states at period $t$.
- **Evaluation**: Evaluated in topological DAG order, enabling auxiliary calculations to feed subsequent state equations.

```python
import numpy as np

exo_m1 = np.array([0.0])
states_m1 = np.array([0.0, 9.35])      # [z_{t-1}, k_{t-1}]
controls_m1 = np.array([0.33, 0.23])   # [n_{t-1}, i_{t-1}]
exo_0 = np.array([0.01])               # Innovation at date t

# Execute compiled transition function:
new_states = func(exo_m1, states_m1, controls_m1, exo_0)
print("Updated states [z_t, k_t]:", new_states)
```

### B. Residual Groups (e.g., Arbitrage & Equilibrium)

For simultaneous equilibrium equations ($0 = \mathbb{E}_t [ h(\dots) ]$):
- **Inputs**: Current and lead variable arrays:
  `func(exo_0, states_0, controls_0, exo_1, states_1, controls_1)`
- **Output**: A 1-D NumPy array of residuals ($LHS - RHS$).
- **Use Case**: Fed directly into non-linear root finders (e.g. Newton-Krylov, Powell, or projection collocation).

```python
arbitrage_spec = DTCC_RECIPE.get_group_spec("arbitrage")
_, arb_func = compile_equation_group(
    arbitrage_equations,
    variables=variables,
    spec=arbitrage_spec,
    constants=constants
)

residuals = arb_func(exo_0, states_0, controls_0, exo_1, states_1, controls_1)
print("Arbitrage equation residuals:", residuals)
```

---

## 3. Vectorization and Solver Interoperability

Because the compiled functions operate on standard NumPy arrays:
- They can be integrated into custom Monte Carlo simulators without symbolic overhead.
- They can be called millions of times inside optimization loops or value function iteration routines.
- They provide a direct, clean bridge to external nonlinear solvers (such as Dolo, JAX, or SciPy).
