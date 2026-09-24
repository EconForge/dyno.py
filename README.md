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