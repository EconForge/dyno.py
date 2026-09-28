# Evaluation & Automatic Differentiation

DynSpec combines an AST interpreter with a forward-mode automatic differentiation engine to evaluate dynamic equations and compute analytical Jacobians without symbolic swelling.

---

## 1. Formula Evaluation (`FormulaEvaluator`)

The `FormulaEvaluator` class in `dyno.dynspec.analyze` is a Lark `Interpreter` that traverses expression trees to evaluate mathematical operations.

```python
from dyno.dynspec.analyze import FormulaEvaluator
from dyno.dynspec.grammar import parser

tree = parser.parse("exp(z) * k^alpha", start="formula")

context = {
    "constants": {"alpha": 0.33, "z": 0.0, "k": 10.0},
}

evaluator = FormulaEvaluator(context=context)
result = evaluator.visit(tree)
print("Evaluated value:", result)  # 10.0^0.33 ≈ 2.138
```

### Evaluation Modes

- **Steady-State Evaluation (`steady_state=True`)**:
  All time shifts are collapsed; $y[t-1]$, $y[t]$, and $y[t+1]$ all evaluate to the steady-state scalar $\bar{y}$.

- **Dynamic Evaluation (`steady_state=False`)**:
  Variables are resolved with time-shift sensitivity using dated contexts.

- **Fail-Fast NaN Diagnostics (`raise_on_nan=True`)**:
  Rather than allowing `NaN` to propagate silently across the entire model, DynSpec can raise a `DefinitionError` directly at the subtree where an invalid operation occurs (e.g. division by zero or log of a non-positive value).

---

## 2. Forward-Mode Automatic Differentiation (`DNumber`)

Symbolic differentiation of large equation systems can lead to expression swelling, while finite differencing introduces numerical truncation errors.

DynSpec addresses this using **dual numbers** implemented in `dyno.dynspec.autodiff.DNumber`.

### The `DNumber` Data Structure

A `DNumber` carries:

- `value` (`float` or `ndarray`): The primal scalar or array evaluation.
- `derivatives` (`dict[Any, float]`): A dictionary mapping variable identifiers to partial derivatives.

```python
from dyno.dynspec.autodiff import DNumber

# Define a dual number with unit derivative with respect to ('k', -1)
k_prev = DNumber(10.0, {("k", -1): 1.0})

# Arithmetic operations propagate derivatives via the chain rule
y = k_prev ** 0.33

print("Value:", y.value)                            # 10.0^0.33 ≈ 2.138
print("Derivative dy/dk:", y.derivatives[("k", -1)])  # 0.33 * 10.0^(0.33 - 1) ≈ 0.0705
```

### Supported Operations

`DNumber` implements the standard algebraic and elementary operations:

- **Arithmetic**: `+`, `-`, `*`, `/`, `**`
- **Elementary Functions**: `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tan`
- **Broadcasting**: Operates on both scalar floats and NumPy arrays.

---

## 3. Computing Model Jacobians ($A, B, C, D$)

Dyno seeds dual numbers with unit gradients across lead, contemporaneous, and lagged variables. Evaluating the system yields the residuals and the four Jacobian matrices required by perturbation and simulation solvers:

$$A = \frac{\partial f}{\partial y_{t+1}}, \quad B = \frac{\partial f}{\partial y_t}, \quad C = \frac{\partial f}{\partial y_{t-1}}, \quad D = \frac{\partial f}{\partial \varepsilon_t}$$

### Accessing Jacobians via `DynoModel`

In user workflows, compute and inspect Jacobians via the `model.jacobians` property:

```python
from dyno import DynoModel

model = DynoModel("examples/neo.dyno")

# Evaluates equations using dual numbers in a single forward pass
residuals, A, B, C, D = model.jacobians

print("Residuals shape:", residuals.shape)
print("A (Leads Jacobian):", A.shape)
print("B (Contemporaneous Jacobian):", B.shape)
print("C (Lags Jacobian):", C.shape)
print("D (Shocks Jacobian):", D.shape)
```

For date-specific or point-specific evaluations, `model.compute_jacobians(y_lead, y_curr, y_lag, eps)` evaluates the Jacobians at arbitrary points in state space.
