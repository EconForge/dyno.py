# Install Dyno

This page explains how to use Dyno in your own project. If you want to work on Dyno itself (clone the repository, run the tests, build the documentation), see [Develop Dyno](development.md).

Dyno is distributed as a conda package on the **`econforge`** channel on prefix.dev (`https://prefix.dev/econforge`). We recommend installing it with [Pixi](https://pixi.sh).

---

## Install Pixi

If you do not have Pixi installed on your system:

=== "Linux & macOS"
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

=== "Windows (PowerShell)"
    ```powershell
    iwr -useb https://pixi.sh/install.ps1 | iex
    ```

---

## Add Dyno to a project

Create a project (skip this step to add Dyno to an existing Pixi project), add the `econforge` channel, then add the `dyno` package:

```bash
pixi init my-project
cd my-project
pixi workspace channel add --prepend https://prefix.dev/econforge
pixi add dyno
```

`--prepend` puts the `econforge` channel before `conda-forge`, so Pixi installs the latest Dyno release from `econforge`. Your `pixi.toml` then contains:

```toml
[workspace]
channels = ["https://prefix.dev/econforge", "conda-forge"]

[dependencies]
dyno = ">=0.1.11,<0.2"
```

(the exact version constraint depends on the current release).

### Optional components

**Dyno Lab** (JupyterLab extension). The `jupyterlab-dyno` package provides the extension; add JupyterLab alongside it, then launch it:

```bash
pixi add jupyterlab jupyterlab-dyno
pixi run jupyter lab
```

See the [Dyno Lab documentation](../dyno_lab/index.md) for details.

**Dynare preprocessor.** Dyno reads Dynare `.mod` files with its own parser, which needs no extra package. To use `DynareModel`, which relies on the official Dynare preprocessor, add `dynare-preprocessor-pylib`:

```bash
pixi add dynare-preprocessor-pylib
```

See the [Dynare quickstart](dynare_quickstart.md) for the differences between the two.

---

## Verify the installation

```bash
pixi run python -c "from dyno import DynoModel; print('Dyno is installed')"
```

Then follow the [Dyno quickstart](dyno_quickstart.md) to write and solve your first model.

---

## Supported platforms

`dyno` and `jupyterlab-dyno` are pure-Python (`noarch`) packages and require Python 3.12 or 3.13, so they install wherever those Python versions are available from conda-forge. `dynare-preprocessor-pylib` contains compiled code and is only published for some platforms (currently `linux-64`, `osx-64` and `win-64`; not `osx-arm64`).

Dyno is developed and tested on `linux-64` only: the development workspace in the repository (`platforms` in `pixi.toml`) is restricted to that platform.
