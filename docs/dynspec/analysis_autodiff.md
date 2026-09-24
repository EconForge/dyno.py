# Evaluation & Automatic Differentiation

DynSpec combines an AST interpreter with a forward-mode automatic differentiation engine to evaluate dynamic equations and compute exact analytical Jacobians without symbolic explosion.

---

## 1. Formula Evaluation (`FormulaEvaluator`)

The `FormulaEvaluator` class in `dyno.dynspec.analyze` is a Lark `Interpreter` that traverses formula trees and evaluates mathematical expressions.

```python
from dyno.dynspec.analyze import FormulaEvaluator
from dyno.dynspec.grammar import parser

tree = parser.parse("exp(z) * k^alpha", start="formula")

context = {
    "constants": {"alpha": 0.33},
    "variables": {"z": 0.0, "k": 10.0}
}

evaluator = FormulaEvaluator(context=context)
result = evaluator.visit(tree)
print("Evaluated value:", result)  # 10.0^0.33 ≈ 2.138
```

### Evaluation Modes

- **Steady-State Evaluation (`steady_state=True`)**:
  All time shifts are ignored; $y[t-1]$, $y[t]$, and $y[t+1]$ all evaluate to the same steady-state scalar $\bar{y}$.
- **Dynamic Evaluation (`steady_state=False`)**:
  Variables are resolved with full time index sensitivity using dated contexts.
- **Fail-Fast NaN Diagnostics (`raise_on_nan=True`)**:
  Rather than allowing `NaN` to propagate silently across the entire model, DynSpec raises a `DefinitionError` immediately at the exact subtree where the invalid calculation occurs:
  ```python
  evaluator = FormulaEvaluator(context=bad_context, raise_on_nan=True)
  # Raises DefinitionError: (14, 5): log of non-positive number
  ```

---

## 2. Forward-Mode Automatic Differentiation (`DNumber`)

Symbolic differentiation engines (such as SymPy) often suffer from exponential expression swelling when differentiating large systems of equations. Finite differencing, on the other hand, introduces truncation and roundoff errors.

DynSpec solves this using **dual numbers** implemented in `dyno.dynspec.autodiff.DNumber`.

### The `DNumber` Data Structure

A `DNumber` consists of two components:
- `value` (`float` or `ndarray`): The primal scalar or array evaluation.
- `derivatives` (`dict[str, float]`): A sparse dictionary mapping active variable identifiers (e.g., `'k[t-1]'`, `'c[t]'`, `'c[t+1]'`) to their partial derivatives.

```python
from dyno.dynspec.autodiff import DNumber

# Define a dual number with unit derivative with respect to 'k[t-1]'
k_prev = DNumber(10.0, {"k[t-1]": 1.0})

# Arithmetic operations propagate derivatives via the chain rule
y = k_prev ** 0.33

print("Value:", y.value)                      # 10.0^0.33 ≈ 2.138
print("Derivative dy/dk:", y.derivatives["k[t-1]"])  # 0.33 * 10.0^(0.33 - 1) ≈ 0.0705
```

### Supported Operations

`DNumber` overloads all basic and transcendental operators:
- **Arithmetic**: `+`, `-`, `*`, `/`, `**`
- **Elementary Functions**: `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tan`
- **Broadcasting**: Operates seamlessly with NumPy arrays and scalars.

---

## 3. Computing Model Jacobians ($A, B, C, D$)

DynSpec evaluates equations using dual numbers seeded with unit gradients for each variable appearance. This directly populates the four fundamental Jacobian matrices required by first-order perturbation solvers:

$$A = \frac{\partial f}{\partial y_{t+1}}, \quad B = \frac{\partial f}{\partial y_t}, \quad C = \frac{\partial f}{\partial y_{t-1}}, \quad D = \frac{\partial f}{\partial \varepsilon_t}$$

### Example

```python
from dyno.dynspec import read_model

# Parse and evaluate equations with exact autodiff
residuals, A, B, C, D = read_model("neo.dyno", diff=True)

print("A (Leads Jacobian):", A.shape)
print("B (Contemporaneous Jacobian):", B.shape)
print("C (Lags Jacobian):", C.shape)
print("D (Shocks Jacobian):", D.shape)
```

Each derivative is evaluated in machine precision in a single forward pass, providing the exact numerical inputs needed by the QZ decomposition and stacked-time solvers.
