# Language Syntax & Semantics

The `.dyno` domain-specific language (DSL) is designed for concise, mathematically natural economic modeling. This page provides a complete syntax specification for `.dyno` files.

---

## 1. Assignments vs. Equations

The most fundamental distinction in Dyno is between **declarations/assignments** and **dynamic equations**:

| Construction | Operator | Purpose | Example |
|---|---|---|---|
| **Assignment** | `<-` or `:=` | Assigns parameter, steady-state, or shock values to `model.context` | `alpha <- 0.36` |
| **Equation** | `=` | Dynamic equilibrium conditions evaluated by solvers | `y[t] = c[t] + i[t]` |
| **Residual (Zero-form)** | *bare formula* | Expression assumed equal to 0 ($f(x)=0$) | `1/c[t] - beta*(1/c[t+1])` |

---

## 2. Identifiers and Symbols

Variable and parameter names can be ASCII or Unicode characters:

```text
alpha <- 0.36
α     <- 0.36      # Unicode characters are fully supported
c_t_1 <- 1.0       # Underscores and numbers allowed
```

---

## 3. Time Indexing

Time indices are enclosed in brackets `[...]`:

```text
x[t]      # Contemporaneous variable at time t
x[t-1]    # Lagged variable (previous period)
x[t+1]    # Forward lead (future expectation)
x[~]      # Steady-state value
x[0]      # Pinned initial condition at date 0
x[1]      # Pinned value at date 1
```

### Usage Contexts

- **Equations**: Dynamic equations use `[t]`, `[t-1]`, and `[t+1]`.
- **Steady State**: Steady-state declarations use `[~]`. Expressions can reference parameters and previously defined steady states:
  ```text
  k[~] <- 10.0
  y[~] <- k[~]^alpha
  i[~] <- delta * k[~]
  c[~] <- y[~] - i[~]
  ```
- **Deterministic Bounds**: Initial values or boundary values use integer dates `[0]`, `[1]`, etc.

---

## 4. Exogenous Shocks & Stochastic Processes

Stochastic shock processes are declared using the Gaussian distribution operator `N(...)`:

```text
# Specify standard deviation (zero-mean default):
e[t] <- N(std)

# Or specify both mean and standard deviation:
e[t] <- N(mean, std)
```

> [!NOTE]
> The scale parameter of `N(...)` is **standard deviation** ($\sigma$), not variance. For a 1% standard deviation shock, simply write `N(0.01)`. If only one argument is provided, the mean defaults to zero.

Examples:

```text
# Zero-mean productivity shock with 1% standard deviation
e_a[t] <- N(0.01)

# Demand shock with explicit mean and 0.5% standard deviation
e_d[t] <- N(0.0, 0.005)
```

Behind the scenes:

- Dyno registers `e_a` and `e_d` as exogenous variables in `model.symbols["exogenous"]`.
- The shock covariance matrix $\Sigma$ is automatically constructed for perturbation and simulation solvers (with diagonal entries $\sigma^2$).
- The mean (`0.0` by default) is automatically populated into `model.context["steady_states"]`.

### Note on Correlated Shocks

Currently, shocks declared via `N(...)` are treated as mutually independent. This is **without loss of generality**: any correlated multivariate Gaussian shock system can be represented structurally as a linear combination of independent orthogonal innovations (e.g., via a Cholesky decomposition or common factor structure):

```text
# Orthogonal structural innovations
u_1[t] <- N(1.0)
u_2[t] <- N(1.0)

# Correlated shocks via factor loadings
e_a[t] <- sigma_a * u_1[t]
e_b[t] <- sigma_b * (rho * u_1[t] + sqrt(1 - rho^2) * u_2[t])
```

---

## 5. Deterministic Paths & Quantified Assignments

For deterministic models or policy experiments, exogenous paths or shocks can be pinned across specific time intervals using `forall` (or the Unicode symbol `∀`):

```text
# Pin shock at initial dates
e[0] <- 0.05
e[1] <- 0.025

# Quantified assignment over a bounded interval:
forall t, 2 <= t < 10 : e[t] <- 0.025 / (t - 1)

# Using Unicode symbol:
∀ t, 10 <= t < 20 : e[t] <- 0.0
```

> [!NOTE]
> Quantified assignments must specify explicit bounds of the form `a <= t < b`. The loop expands deterministically into `model.context["values"][var][date]`.

---

## 6. Mathematical Operators & Built-in Functions

Dyno equations and assignments support standard mathematical expressions:

### Arithmetic & Power

- Addition & Subtraction: `+`, `-`
- Multiplication & Division: `*`, `/`
- Exponentiation: `^` or `**` (e.g., `k[t-1]^alpha` or `k[t-1]**alpha`)
- Grouping: `( ... )`

### Built-in Functions

- Exponential: `exp(x)`
- Natural Logarithm: `log(x)`
- Square Root: `sqrt(x)`
- Trigonometric: `sin(x)`, `cos(x)`, `tan(x)`

Example:

```text
y[t] = exp(a[t]) * (k[t-1]^alpha) * (n[t]^(1-alpha))
utility[t] = log(c[t]) - psi * (n[t]^(1+eta)) / (1+eta)
```

---

## 7. Comments

Comments begin with `#` and can appear as standalone lines or inline at the end of a line:

```text
# This is a full-line comment
beta <- 0.99  # Household discount factor
```

---

## 8. Missing and Undefined Parameters

When building a model, parameters may be referenced before being defined, or omitted entirely:

```text
k[t] = alpha * k[t-1]  # 'alpha' is referenced in equations but never assigned
```

Dyno handles missing parameters through a dual diagnostic model:

### Permissive Mode (Default: `strict=False`)

By default, Dyno accommodates incremental model authoring and interactive exploration:

- Unassigned parameters are registered in `model.symbols["parameters"]` and initialized to `NaN` in `model.context["constants"]`.
- Dyno emits an immediate **`UndefinedSymbolWarning`** reporting which parameters are unassigned.
- In Jupyter notebooks and terminal displays, uninitialized parameters and variables are flagged with an orange caret (`^`):
  ```text
  constants: alpha^
  ^ uninitialized (steady-state) value: defaults to nan
  ```
- Unassigned parameters can be supplied later using `recalibrate()`:
  ```python
  model = DynoModel("model.dyno")  # Emits UndefinedSymbolWarning
  model = model.recalibrate(alpha=0.35)  # Now fully calibrated
  ```

### Strict Mode (`strict=True`)

For automated pipelines and continuous integration, pass `strict=True` to reject incomplete models immediately:

```python
model = DynoModel("model.dyno", strict=True)
# Raises dyno.errors.UndefinedSymbolError:
# Undefined parameter(s) used in equation definitions: alpha
```

### Fast-Fail Validation in `check()` and `solve()`

Even in permissive mode, a model cannot be checked or solved while parameters remain `NaN`:

- **`model.check()`** inspects all parameters and steady states. If any are missing, it raises `UndefinedSymbolError` with an explicit list:
  ```text
  UndefinedSymbolError: Cannot check model due to uninitialized symbols (parameters without values: alpha).
  ```
  This prevents silent `NaN` residual propagation and uninformative solver errors.

- **`model.solve()`** verifies that the system is complete and square ($N_{eq} = N_{endo}$), raising `SystemStructureError` if the model is under- or overdetermined.

