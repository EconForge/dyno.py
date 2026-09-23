# Installation & Setup

This guide explains how to install and configure Dyno in your development or production environment using **Pixi**.

---

## Package Manager: Pixi

Dyno is managed using [Pixi](https://pixi.sh), a modern cross-platform package and workflow manager built on top of the Conda ecosystem. Pixi ensures deterministic, reproducible environments with lockfile guarantees.

> [!TIP]
> We recommend using Pixi for all Dyno development tasks. Avoid running `pip` or `poetry` directly in the project repository to prevent dependency divergence.

### Installing Pixi

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

## Cloning and Environment Setup

Clone the repository and install all dependencies:

```bash
git clone https://github.com/EconForge/dyno.py.git
cd dyno.py

# Install dependencies and setup default dev environment
pixi install -e dev
```

Pixi creates a virtual environment under `.pixi/envs/dev` with Python 3.12, PyTest, NumPy, SciPy, SymPy, Pandas, Plotly, Lark, and development tooling.

---

## Pixi Environments and Features

Dyno defines modular features and environments in `pixi.toml` to support different use cases:

| Environment | Purpose | Key Packages |
|---|---|---|
| `dev` *(default)* | Full development environment | `pytest`, `mypy`, `black`, `jupyterlab`, `zensical` |
| `test` | Lean CI test runner | `pytest`, `coverage`, `pytest-cov` |
| `dynare` | Full Dynare preprocessor integration | `dynare-preprocessor-pylib` |
| `solara` | Interactive web dashboard | `solara`, `ipyvuetify`, `anywidget` |
| `prod` | Production / runtime notebook environment | `jupyter`, `numpy`, `scipy`, `pandas` |

### Running Commands in Environments

To run commands or tasks within an environment, use the `-e` flag:

```bash
# Run test suite
pixi run -e dev test

# Run static type checking
pixi run -e dev typecheck

# Code formatting check
pixi run -e dev black

# Serve documentation locally
pixi run -e dev docs
```

---

## Optional: Dynare Preprocessor

Dyno includes its own native Lark-based parser for `.mod` files that does not require any external C++ binaries. However, if you wish to use the official Dynare preprocessor via `DynareModel` for maximum fidelity with Dynare C++ syntax:

```bash
# Activate the dynare feature
pixi run -e dynare pytest tests/dynare
```

The preprocessor is packaged on the `econforge` prefix channel as `dynare-preprocessor-pylib`.

---

## Verifying the Installation

To verify that your installation is working correctly, run a quick Python check:

```bash
pixi run -e dev python -c "import dyno; print('Dyno loaded successfully!')"
```

You are now ready to write and solve your first economic model!
