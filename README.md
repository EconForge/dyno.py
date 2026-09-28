# dyno.py

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