# Automated Pipelines & Reports

Dyno includes an automated execution and reporting engine that compiles model verification, perturbation solving, moment evaluation, and simulation into reproducible output reports.

---

## The `@run:` Directive

Declare automated pipeline steps directly within your model file:

```text
# model.dyno
@name: RBC_Report_Example
@run: steady
@run: check
@run: solve
@run: simul: {T: 40}

# Parameters & equations follow...
alpha <- 0.36
...
```

The `@run:` command format is stable: models written with it will keep working in future releases.

Commands are executed in the order they appear. Each `@run:` line holds one command, written as:

| Form | Example |
|---|---|
| Bare command name | `@run: steady` |
| Command with options (a YAML mapping) | `@run: simul: {T: 40}` |
| Command with null options | `@run: steady: null` |

Appending `;` to the command name (e.g. `@run: solve;`) mutes that command's output in reports. In YAML models, the same commands are listed under a top-level `run:` key, and the explicit form `{command: simul, options: {T: 40}}` is also accepted.

Supported `@run:` commands:

| Command | Action |
|---|---|
| `steady` | Solves for steady state numerically if needed (`model.steady(**options)`) |
| `resid` | Computes steady-state residuals |
| `check` | Verifies residuals and checks Blanchard-Kahn eigenvalues |
| `solve` / `perturb` | Computes first-order perturbation policy function |
| `simul` / `simulate` | Generates stochastic simulations, or a perfect-foresight path for deterministic models (options such as `T`, `N`, `mode`, `units`) |
| `analyze` / `stoch_simul` | Solves the model, computes moments and impulse responses (options `T`/`irf`, `type`/`units`, `variables`, `nograph`) |
| `plot` | Plots the current simulation (options such as `variables`, `engine`) |
| `variants` | Runs the remaining commands on each calibration variant (e.g. `@run: variants: {beta: [0.95, 0.99]}`) |

---

## Executing Pipelines with `model.run()`

In Python, execute the pipeline with one call:

```python
from dyno import DynoModel

model = DynoModel("model.dyno")
results = model.run()
```

### Inspecting `RunResults`

The returned `RunResults` object encapsulates the complete execution state:

```python
# Access components:
results.model           # Model after steady-state evaluation
results.residuals       # Equation residual vector
results.solution        # PerturbationSolution decision rule
results.eigenvalues     # Generalized eigenvalues array
results.bk_check        # True if Blanchard-Kahn conditions hold
results.moments         # Asymptotic covariance matrices
results.simulation      # Dictionary of IRFs or simulation DataFrame
```

---

## Rich Display in Notebooks & Terminal

`RunResults` and `Report` automatically render clean, formatted summaries across all environments:

### Terminal (Plain Text)

Printing the result outputs a formatted diagnostic summary:

```python
print(results)
```

```text
RunResults
==========
Model
-----
name: RBC_Report_Example
filename: model.dyno
symbols: variables=8, endogenous=8, exogenous=1, parameters=7

Steady State
------------
residuals: max|res|=0.000e+00, mean|res|=0.000e+00

Solution
--------
method: qz
Eigenvalues: computed (n=8, |lambda|>1: 4, |lambda|≈1: 0)
Blanchard-Kahn conditions: met

Simulation
----------
Simulation: computed (IRFs, shocks=1, horizon=40)
  shocks: e_a

Simulation Plots
----------------
[ASCII / terminal graphs rendered via plotext]
```

You can also customize or extract terminal graphs directly:
```python
# Print or retrieve text plots with plotext:
print(results.plot_text(cols=2, width=80))

# Control graph inclusion in text reports:
print(results.to_text(graphs=False))
```

### Jupyter Notebooks (HTML & Markdown)

In JupyterLab or VSCode Notebooks, simply returning `results` renders interactive HTML tables, equation renderings, and embedded charts:

```python
# In a Jupyter cell:
results
```

---

## Programmatic Report Generation

You can build and export reports explicitly:

```python
from dyno.report import Report

report = Report(model)
html_content = report.to_html()
markdown_content = report.to_markdown()

# Save report to disk
with open("model_report.html", "w") as f:
    f.write(html_content)
```
