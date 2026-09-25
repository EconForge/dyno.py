# Subpackages & Roadmap

Dyno is a fresh take on a modeling language for dynamic economic models. During its early development, the repository also temporarily houses two specialized subsystems intended to become standalone packages once they mature: **DynSpec** (the specification and AST engine) and **`dyno.dynare`** (the Python implementation of Dynare).

Developing these modules with strict boundary separation allows rapid experimentation alongside Dyno's own language features while preparing them for independent lifecycles on PyPI and conda-forge.

---

## Incubated Subpackages

### DynSpec (`dyno.dynspec`)

**DynSpec** is a language-independent, solver-agnostic specification and AST engine for dynamic economic models. It handles declarative EBNF grammar parsing, AST transformations, topological dependency sorting, formal model recipes, forward-mode automatic differentiation (`DNumber`), and JIT NumPy compilation. Currently developed inside `dyno.dynspec`, it will graduate into an autonomous `dynspec` package usable across diverse macroeconomic frameworks.

Detailed architecture, AST grammar, recipes, and autodiff documentation are available in the [DynSpec Specification Engine Guide](../dynspec/index.md).

---

### Dynare in Python (`dyno.dynare`)

**`dyno.dynare`** brings first-class, Pythonic Dynare modeling to modern scientific workflows. Built around `DynareModel`, it interfaces directly with a co-developed C++ preprocessor Python library (`dynare-preprocessor-pylib`, distributed on conda-forge and prefix.dev) to ensure exact syntax and macro-processing parity with official Dynare `.mod` files. In the future, `dyno.dynare` will graduate into an independent `dynare` package featuring a standalone CLI, full command emulation, and pure-Python parsing fallbacks.

For usage workflows, lifecycle details, and migration examples, explore the [Dynare in Python Guide](../dynare/index.md).

---

## Development Roadmap

The roadmap toward independent packages follows a straightforward three-phase path:

**Phase 1: Modular Encapsulation (Current)** establishes clean internal interfaces (`DynareModel`, `FormulaEvaluator`, `Recipe`) and comprehensive test coverage while incubating inside the `dyno` repository.

**Phase 2: Decoupled Extraction** will spin off `dynspec` and `dynare` into standalone repositories or workspace packages with dedicated command-line tools and standalone test suites.

**Phase 3: Independent Releases** will publish `dynspec` and `dynare` as independent packages on PyPI and conda-forge for the wider scientific community, while `dyno` continues to focus on its primary vision as a modern, expressive modeling language.

