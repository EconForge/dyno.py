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

Stochastic shock processes are declared using the Gaussian distribution operator:

```text
e[t] <- N(mean, variance)
```

> [!WARNING]
> The second parameter of `N(...)` is **variance** ($\sigma^2$), not standard deviation. For a standard deviation of 0.01, write `0.01^2`.

Examples:

```text
# Zero-mean productivity shock with 1% standard deviation
e_a[t] <- N(0, 0.01^2)

# Demand shock with 0.5% standard deviation
e_d[t] <- N(0, 0.005^2)
```

Behind the scenes:
- Dyno registers `e_a` and `e_d` as exogenous variables in `model.symbols["exogenous"]`.
- The shock covariance matrix $\Sigma$ is automatically constructed for perturbation and simulation solvers.
- The mean (0) is automatically populated into `model.context["steady_states"]`.

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
