# dyno.py

## Setup

### Using pixi (production and development environment)

`pixi` can be installed on a *nix system using the following command:
```console
curl -fsSL https://pixi.sh/install.sh | sh
```

We can then run Dyno's web server using:
```console
pixi run solara
```

For development (including documentation and unit tests), `pixi` provides a set of pre-configured tasks.

To run `mkdocs`:
```console
pixi run docs
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