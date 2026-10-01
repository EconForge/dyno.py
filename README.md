# dyno.py

Dyno is a Python library for specifying, solving and simulating dynamic
macroeconomic models: stochastic DSGE models solved by first-order
perturbation, and deterministic perfect-foresight (transition) models solved
with a stacked-time Newton solver. Models are written in the human-readable
`.dyno` language (explicit time indices `[t]`, `[t-1]`, `[t+1]`, steady-state
values `[~]`), or imported directly from Dynare `.mod` files. Dyno checks
steady states and Blanchard-Kahn conditions, computes impulse responses,
moments and simulations, renders model reports, and comes with **Dyno Lab**, a
JupyterLab extension for editing and solving models interactively.

Documentation: <https://econforge.github.io/dyno.py/>

## Install

Dyno is distributed as a conda package on the `econforge` channel on
prefix.dev. With [pixi](https://pixi.sh), in a new or existing project:

```console
pixi init my-project && cd my-project            # skip for an existing project
pixi workspace channel add --prepend https://prefix.dev/econforge
pixi add dyno
```

`--prepend` gives the `econforge` channel priority over `conda-forge`, so you
get the latest Dyno release. For the graphical interface (Dyno Lab), also add
JupyterLab and the extension:

```console
pixi add jupyterlab jupyterlab-dyno
pixi run jupyter lab
```

See the [installation guide](https://econforge.github.io/dyno.py/getting_started/installation/)
for optional components (such as the Dynare preprocessor).

## Example

Write a model in `neo.dyno`:

```text
α <- 0.36
β <- 0.99
δ <- 0.025
ρ <- 0.95
γ <- 2.0

z[~] <- 0.0
k[~] <- ((1/β - (1-δ)) / α)**(1 / (α-1))
y[~] <- k[~]^α
i[~] <- δ * k[~]
c[~] <- y[~] - i[~]

z[t] = ρ * z[t-1] + e_z[t]
y[t] = exp(z[t]) * k[t-1]^α
k[t] = (1-δ) * k[t-1] + i[t]
c[t] = y[t] - i[t]
β * (c[t+1]/c[t])^(-γ) * (α * y[t+1]/k[t] + 1 - δ) = 1

e_z[t] <- N(0.01)
```

Then solve it and compute impulse responses:

```python
from dyno import DynoModel

model = DynoModel("neo.dyno")
model.check()                          # raises if the steady state is wrong
solution = model.solve()               # first-order perturbation
irfs = solution.irfs(type="deviation", T=40)
print(irfs["e_z"][["y", "c", "k", "i"]].head())
solution.plot(type="deviation")        # interactive chart in Jupyter
```

Dynare files work the same way: `DynoModel("model.mod")`.

## Contributing

Contributors work from a clone of this repository. The sections below, and the
[Develop Dyno](https://econforge.github.io/dyno.py/getting_started/development/)
page of the documentation, describe the development setup.

## Setup

### Using pixi (production and development environment)

`pixi` can be installed on a *nix system using the following command:
```console
curl -fsSL https://pixi.sh/install.sh | sh
```

For interactive modeling, Dyno features a JupyterLab extension (`jupyterlab_dyno` / Dyno Lab):
```console
pixi run -e dev jupyter lab
```

For development (including documentation and unit tests), `pixi` provides a set of pre-configured tasks.

### Model representation explorer (dev tool)

To visualize how the library currently imports and renders every example
model (across backends, strict options, and text/HTML/Markdown output), run:
```console
pixi run -e solara explorer
```
This is a maintainer/contributor tool for tracking the library's progress
across its example models — it is **not** meant for developing or
calibrating your own models; use Dyno Lab (above) for that. See the
[documentation](https://econforge.github.io/dyno.py/subpackages/model_explorer/)
for details.

To build and serve documentation locally with `zensical`:
```console
pixi run -e dev docs
```

To run unit and coverage tests:
```console
pixi run test
pixi run cov
```

Finally, types can be checked with `mypy`:
```console
pixi run typecheck
```
