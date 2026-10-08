# Install Dyno

This page explains how to use Dyno in your own project. If you want to work on Dyno itself (clone the repository, run the tests, build the documentation), see [Develop Dyno](development.md).

Dyno is available on PyPI and as conda packages from both [conda-forge](https://conda-forge.org/) and the **`econforge`** channel on [prefix.dev](https://prefix.dev/econforge). Conda packages are also built for WebAssembly (WASM) and are available inside [notebook.link](https://notebook.link/). Any conda-compatible tool can install the conda packages, including `conda` and `micromamba`; we recommend [Pixi](https://pixi.sh) for a reproducible project environment.

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
dyno = ">=0.1.14,<0.2"
```

(the exact version constraint depends on the current release).

If you prefer a different installer, the regular Python package can also be installed from PyPI:

```bash
python -m pip install dynopy
```

The optional components described below are distributed only as conda packages and are not available on PyPI. Install them with a conda-compatible tool such as Pixi, `conda`, or `micromamba`.

### Optional components

**Dyno Lab** (JupyterLab extension). The `jupyterlab-dyno` package provides the extension; add JupyterLab alongside it, then launch it:

```bash
pixi add jupyterlab jupyterlab-dyno
pixi run jupyter lab
```

See the [Dyno Lab documentation](../dyno_lab/index.md) for details.

**Dynare preprocessor.** Dyno reads Dynare `.mod` files with its own parser, which needs no extra package. To use `DynareModel`, which relies on the official Dynare preprocessor, add the native `dynare-preprocessor-pylib` package:

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
