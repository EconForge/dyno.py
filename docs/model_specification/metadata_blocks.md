# Blocks, Metadata & YAML Wrappers

Dyno includes advanced capabilities for organizing equations into semantic groups, tagging statements, embedding models inside YAML pipelines, and declaring automated execution directives.

---

## 1. Equation Blocks (`[tag] { ... }`)

In larger models, equations can be grouped logically using block syntax:

```text
[definition] :: {
    w_r[t] = (1-α)/α * K[t]/l[t]
    k_C[t] = ξ[t] * K[t]
    l_C[t] = ξ[t] * l[t]
    Y_c[t] = A_c[t] * (k_C[t]^α) * (l_C[t]^(1-α))
    C[t] = Y_c[t]
}

[transition] :: {
    K[t] = Y_K[t-1]
    A_k[t] = ψ_k[t-1] * A_k[t-1]
    A_c[t] = ψ_c[t-1] * A_c[t-1]
}

[arbitrage] :: {
    Q[t]/r[t+1] = β/(1+τ) * C[t]/C[t+1] + φ * C[t]/(Q[t] * Y_K[t])
}
```

The double-colon `::` is optional; `[tag] { ... }` is also valid. Every equation defined inside the block inherits the enclosing block's tag in its metadata.

---

## 2. Statement Annotations & Inline Tags

You can tag individual equations directly:

```text
# Inline bracket tag
1/c[t] = beta*(1/c[t+1])*(r[t+1] + 1 - delta)  [euler_equation]

# Colon-colon notation
k[t] = (1-delta)*k[t-1] + i[t]  :: capital_accumulation

# Key-value metadata
y[t] = exp(a[t])*k[t-1]^alpha  [type=production, sector=goods]
```

### Accessing Equation Tags in Python

```python
model = DynoModel("my_model.dyno")

for i, eq_tree in enumerate(model.symbolic.equations):
    meta = getattr(eq_tree.meta, "statement_metadata", {})
    print(f"Equation {i}: {model.equations[i]}")
    print(f"Metadata: {meta}\n")
```

---

## 3. Top-Level Model Metadata (`@key: value`)

Metadata describing the model can be placed at the top level using `@`:

```text
@name: RBC Baseline
@description: Standard Hansen Real Business Cycle model with technology shocks
@version: 1.0.0
@author: EconForge Team

alpha <- 0.36
...
```

Access model metadata via `model.metadata`:

```python
print(model.metadata["name"])        # 'RBC Baseline'
print(model.metadata["description"]) # 'Standard Hansen...'
```

---

## 4. YAML Wrappers (`.dyno.yaml`)

Dyno supports wrapping `.dyno` models in standard YAML files. This is particularly useful for configuration-driven research pipelines and benchmark suites.

### Structure

In a YAML file, all top-level keys except `model` become entries in `model.metadata`. The `model` key contains the raw `.dyno` syntax:

```yaml
name: RBC Baseline
tags: [dsge, baseline, closed-economy]
author: Pablo Winant
date: 2026-09-24

model: |
  # Parameters
  alpha <- 0.36
  beta <- 0.99
  delta <- 0.025
  rho <- 0.95

  # Steady state
  k[~] <- 10.0
  y[~] <- 1.0
  c[~] <- 0.75
  i[~] <- 0.25
  a[~] <- 0.0

  # Equations
  y[t] = exp(a[t]) * k[t-1]^alpha
  k[t] = (1-delta)*k[t-1] + i[t]
  c[t] + i[t] = y[t]
  1/c[t] = beta*(1/c[t+1])*(alpha*y[t+1]/k[t] + 1 - delta)
  a[t] = rho*a[t-1] + e[t]

  # Shocks
  e[t] <- N(0, 0.01^2)
```

### Loading YAML Models

```python
# From a .yaml file
model = DynoModel("model.yaml")

# Or directly from a YAML string
model = DynoModel(yaml=yaml_content)

print(model.metadata["tags"])  # ['dsge', 'baseline', 'closed-economy']
```

### Precedence Rules

If both the top-level YAML and an `@key: value` directive inside the `model` block define the same attribute, the in-model `@...` value takes precedence:

```yaml
name: TopLevelName
model: |
  @name: InModelName
  ...
```

`model.metadata["name"]` will evaluate to `"InModelName"`.

---

## 5. Automated Run Directives (`@run:`)

You can define automated execution steps inside the model file using `@run:` directives:

```text
@name: StochasticGrowth
@run: steady
@run: check
@run: solve
@run: simul: {T: 40}

alpha <- 0.36
...
```

When you call `results = model.run()`, Dyno executes each command in sequence and packages the outputs into a `RunResults` container:

```python
model = DynoModel("model.dyno")
results = model.run()

# Access pipeline products
print("Steady-state residuals:", results.residuals)
print("Perturbation solution:", results.solution)
print("Simulated IRFs / paths:", results.simulation)
```
