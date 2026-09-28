# Subpackages & Architecture

The Dyno repository is structured modularly. In addition to the main `dyno` modeling interface and solvers, it includes specialized subpackages and developer tooling:

---

## Subpackages Overview

### DynSpec (`dyno.dynspec`)

**DynSpec** is the model specification and symbolic processing engine. It provides:

- **Grammar & Parsing**: Lark-based grammar for `.dyno` and `.mod` files.
- **AST Analysis**: Formula evaluation and symbol classification across time leads, lags, and steady states.
- **Automatic Differentiation**: Forward-mode dual numbers (`DNumber`) for machine-precision Jacobian computation.
- **Model Recipes**: Structural typing and DAG topological sorting for model equations.
- **Compilation & LaTeX**: Vectorized NumPy function generation and publication-ready LaTeX export.

For full technical documentation, see the [DynSpec Specification Engine Guide](../dynspec/index.md).

---

### Dynare Integration (`dyno.dynare`)

**`dyno.dynare`** provides Python support for Dynare `.mod` files. Centered around `DynareModel`, it interfaces with the C++ Dynare preprocessor (`dynare-preprocessor-pylib`) available on conda-forge and prefix.dev. This allows running legacy Dynare models directly within the Python scientific stack.

For usage, configuration, and preprocessor setup, see the [Dynare in Python Guide](../dynare/index.md).

---

## Developer Tooling

### Model Representation Explorer

The repository includes an interactive developer tool built with Solara (`pixi run -e solara explorer`) to inspect and verify model parsing and rendering across all supported formats (plain-text, Markdown, and HTML).

For details on running and contributing to the explorer, see the [Model Explorer Documentation](model_explorer.md).


