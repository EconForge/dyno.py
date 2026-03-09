# `.dyno` Syntax By Example (`model.context` View)

This page is a syntax-first companion to `model.md`.

## One Mental Model

- Declarations (`<-` or `:=`) populate `model.context`.
- Equations (`=`) go to `model.symbolic.equations` (string view: `model.equations`).
- Symbols used in equations are still tracked in `model.context["variables"]`.

`model.context` keys:

- `constants`
- `steady_states`
- `values`
- `processes`
- `variables`

## Part 1: Declarations

Declarations are assignment-like lines that define parameters, steady states, shocks, and deterministic paths.

### 1) Constants

```text
alpha <- 0.36
beta := 0.99
rho <- 0.9
```

```python
model.context["constants"] == {
    "alpha": 0.36,
    "beta": 0.99,
    "rho": 0.9,
}
```

### 2) Steady-State Declarations (`[~]`)

```text
k[~] <- 10.0
c[~] <- 0.8
```

```python
model.context["steady_states"].get("k") == 10.0
model.context["steady_states"].get("c") == 0.8
```

### 3) Date-Specific Values (`[0]`, `[1]`, ...)

```text
e[0] <- 0.02
e[1] <- 0.01
```

```python
model.context["values"]["e"] == {
    0: 0.02,
    1: 0.01,
}
```

### 4) Stochastic Process Declarations (`[t] <- N(...)`)

```text
e[t] <- N(0, 0.01^2)
u[t] <- N(0, 0.002^2)
```

```python
model.context["processes"].keys()   # contains ("e",), ("u",)
model.context["steady_states"]["e"] == 0.0
model.context["steady_states"]["u"] == 0.0
```

Notes:

- `N(mean, variance)` expects variance.
- The mean is copied into `steady_states`.

### 5) Quantified Path Declarations (`forall` / `∀`)

```text
forall t, 0 <= t < 4 : e[t] <- 1/(t+1)
```

```python
model.context["values"]["e"] == {
    0: 1.0,
    1: 0.5,
    2: 1.0/3.0,
    3: 0.25,
}
```

Use explicit bounds `a <= t < b`.

## Part 2: Equations

Equations define model dynamics and are stored separately from `model.context`.

### 1) Single Dynamic Equation

```text
x[t] = rho*x[t-1] + e[t]
```

Representation:

```python
model.equations
model.symbolic.equations
```

What each one is:

```python
# Human-readable rendering
type(model.equations[0])        # str

# Raw symbolic object from parser
type(model.symbolic.equations[0])   # lark.tree.Tree
```

- `model.equations` is a string representation generated from the symbolic expression tree.
- `model.symbolic.equations` is the underlying Lark symbolic tree used internally for evaluation and differentiation.

Side effect in context:

```python
"x" in model.context["variables"]
"e" in model.context["variables"]
```

### 2) Equation With Lead

```text
1/c[t] = beta*(1/c[t+1])*(r[t+1] + 1 - delta)
```

This is still an equation entry (not a declaration), so it appears in `model.equations` and contributes symbols to `model.context["variables"]`.

### 3) Equation-Only Mini Example

```text
rho <- 0.9
e[t] <- N(0, 1)
x[t] = rho*x[t-1] + e[t]
```

After parsing:

- `model.context["constants"]` contains `rho`.
- `model.context["processes"]` contains process for `e`.
- `model.equations` contains one equation for `x[t]`.

## Inspecting Everything Quickly

```python
from dyno import DynoModel

model = DynoModel("my_model.dyno")

print(model.context.keys())
print(model.context["constants"])
print(model.context["steady_states"])
print(model.context["values"])
print(model.context["processes"])
print(model.context["variables"])
print(model.equations)
```

## Quick Checklist

- Use `<-` or `:=` for declarations.
- Use `=` for equations.
- Use `[~]` for steady-state declarations.
- Use `[t-1]`, `[t]`, `[t+1]` inside equations.
- Use `N(mean, variance)` for shocks.
- Use `forall t, a <= t < b : ...` for bounded paths.