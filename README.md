# dyno.py

## Setup

### Using pixi (production environment)

`pixi` can be installed on a *nix system using the following command:
```console
curl -fsSL https://pixi.sh/install.sh | sh
```

We can then run Dyno's web server using:
```console
pixi run solara run t_dyno.py
```

### Using poetry (development environment)

In order to run the development environment (including but not limited to documentation and unit tests), one must first install `poetry` with the following command:
```console
curl -sSL https://install.python-poetry.org | python3 -
```

The installation script creates a `poetry` binary which must be added to `PATH` as shown in the script output.

We can then install Dyno's dependencies using:
```console
poetry install
```

Our `poetry` configuration doesn't allow us to run Dyno's web server as before, but we can run `mkdocs`:
```console
poetry run mkdocs serve
```
as well as unit and coverage tests:
```console
poetry run pytest --cov=dyno tests/
```