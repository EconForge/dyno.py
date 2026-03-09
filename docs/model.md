# Writing A `.dyno` Model

This page explains how to write a model file that Dyno can parse and solve.

## Quick Start

A `.dyno` model usually has four parts:

1. Parameters/constants
2. Steady-state values
3. Exogenous process declarations (optional if deterministic)
4. Dynamic equations

Minimal stochastic example:

```text
# 1) Parameters
rho <- 0.9

# 2) Steady state
x[~] <- 0.0

# 3) Exogenous process (white noise)
e[t] <- N(0, 0.01^2)

# 4) Dynamic equation
x[t] = rho*x[t-1] + e[t]
```

## Core Syntax

### Assignments

Use `<-` (or `:=`) to assign values.

```text
alpha <- 0.36
beta := 0.99
```

### Equations

Use `=` for model equations.

```text
y[t] = c[t] + i[t]
```

### Time Indexing

- `x[t]`: variable at time `t`
- `x[t-1]`: lag
- `x[t+1]`: lead
- `x[~]`: steady-state value
- `x[0]`, `x[1]`, ...: explicit values at fixed dates

Examples:

```text
k[~] <- 10.0
k[0] <- 10.2

c[t] = y[t] - i[t]
k[t] = (1-delta)*k[t-1] + i[t]
```

### Comments

Use `#` for comments:

```text
# full-line comment
rho <- 0.95  # end-of-line comment
```

### Operators And Functions

- Arithmetic: `+`, `-`, `*`, `/`
- Powers: `^` or `**`
- Parentheses: `( ... )`
- Functions: `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, etc.

Example:

```text
y[t] = exp(a[t]) * k[t-1]^alpha
```

## Exogenous Shocks And Paths

### Stochastic Process

Define scalar Gaussian shocks with:

```text
e[t] <- N(mean, variance)
```

Example:

```text
e_z[t] <- N(0, 0.002^2)
```

### Deterministic Shock Values

Pin values at specific dates:

```text
e[0] <- 0.01
e[1] <- 0.005
```

Or over a range with `forall` (or `∀`):

```text
forall t, 0 <= t < 10 : e[t] <- 0.99/(t+1)
```

Note: in current parser behavior, `forall` assignments should include explicit bounds like `0 <= t < 10`.

## Recommended Authoring Pattern

Use this order to keep files readable:

```text
# Parameters
...

# Steady state
...

# Exogenous process declarations
...

# Dynamic equations
...

# Optional: deterministic initial conditions and paths
...
```

## Full Template

```text
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

# Exogenous process
e[t] <- N(0, 0.01^2)

# Dynamic equations
y[t] = exp(a[t]) * k[t-1]^alpha
k[t] = (1-delta)*k[t-1] + i[t]
c[t] + i[t] = y[t]
1/c[t] = beta*(1/c[t+1])*(alpha*y[t+1]/k[t] + 1 - delta)
a[t] = rho*a[t-1] + e[t]

# Optional deterministic path overrides
e[0] <- 0.02
forall t, 1 <= t < 8 : e[t] <- 0.02/t
```

## Loading And Solving A Model

## YAML Wrapper Equivalence

You can wrap a `.dyno` model in YAML using a `model: |` block.

In that form:

- Every top-level YAML key except `model` goes into `model.metadata`.
- `@key: value` lines inside the `model` block write to the same metadata dictionary.
- If both define the same key, the in-model `@...` value takes precedence.

<table>
	<tr>
		<th>Top-Level YAML Metadata</th>
		<th>Metadata Inside <code>model</code> Block</th>
	</tr>
	<tr>
		<td>
			<pre><code class="language-yaml">name: RBC
tags: [baseline, dsge]
model: |
	e[t] &lt;- N(0,1)
	x[t] = 0.9 * x[t-1]</code></pre>
		</td>
		<td>
			<pre><code class="language-yaml">model: |
	@name: RBC
	@tags: [baseline, dsge]

	e[t] &lt;- N(0,1)
	x[t] = 0.9 * x[t-1]</code></pre>
		</td>
	</tr>
	<tr>
		<td>
			<pre><code class="language-yaml">name: TopLevelName
model: |
	@name: InModelName
	x[t] = 0.9 * x[t-1]</code></pre>
		</td>
		<td>
			<pre><code class="language-python"># Result
model.metadata["name"] == "InModelName"</code></pre>
		</td>
	</tr>
</table>

Both forms produce equivalent metadata:

```python
model.metadata == {"name": "RBC", "tags": ["baseline", "dsge"]}
```

```python
from dyno import DynoModel

model = DynoModel("examples/neo.dyno")
solution = model.solve(method="qz")
```

You can also pass model text directly:

```python
from dyno import DynoModel

txt = """
rho <- 0.9
x[~] <- 0.0
e[t] <- N(0, 0.01^2)
x[t] = rho*x[t-1] + e[t]
"""

model = DynoModel(txt=txt)
solution = model.solve()
```

## Common Mistakes

- Using `<-` inside equations. Use `=` for equations and `<-`/`:=` for assignments.
- Forgetting steady-state values (`x[~]`) for endogenous variables.
- Mixing fixed-date syntax (`x[0]`) with dynamic syntax (`x[t]`) unintentionally.
- Writing `forall` without bounds.
- Passing standard deviations to `N(...)` instead of variances.

## Tips

- Start from `examples/neo.dyno` or `examples/rbc.dyno` and modify incrementally.
- Keep variable names consistent and short.
- Add comments above blocks so future edits stay clear.
- If parsing fails, check brackets/indexes first (`[t]`, `[t-1]`, `[~]`).
