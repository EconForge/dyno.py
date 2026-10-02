# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.12] - 2026-10-02

### Added

- `@name:` annotation in model files to name models and display their name in
  `model.name` and generated reports.
- Solara model representation explorer GUI supports browsing model files in
  nested subdirectories with breadcrumb/path headers and relative labels.
- New example models: `new_keynesian_fg.dyno` (forward guidance with
  deterministic solver), `soe_mendoza.dyno` (small open economy), and
  `examples/dolo/rbc_dolo.dyno`.
- Multi-OS CI workflows for Linux, macOS, and Windows.
- `RedefinitionWarning`, emitted when a constant is assigned more than once
  (previously printed to stdout).
- `rng` argument (a `numpy.random.Generator` or a seed) on
  `model.simulate()`, `solution.simulate()`, `dyno.simul.simulate()` and the
  `simul` run command, for reproducible random simulations.
- A CI job running the `DynareModel` tests against the Dynare preprocessor.
- Perfect-foresight `.mod` files are read by `DynoModel`: deterministic
  shocks (`periods`/`values`), `endval`, `histval` (date 0), and the
  `perfect_foresight_setup`/`perfect_foresight_solver`, `simul` and `rplot`
  commands map onto the deterministic solver. A `.mod` file with no shock
  variances and no `stoch_simul` is now a deterministic model, and `varexo`
  variables are exogenous even without a shock process. `steady_state_model`
  blocks are evaluated after `initval`/`endval`, and exogenous steady states
  come from `initval` instead of being forced to zero.
- `DynareModel` (preprocessor backend) runs perfect-foresight files too:
  `AbstractModel` gained a generic stacked-time
  `deterministic_residuals_with_jacobian` built on a per-date
  `_dynamic_point` evaluation, and `DynareModel.run()` understands the
  `simul` and `plot` commands produced from `perfect_foresight_setup`,
  `perfect_foresight_solver`, `simul` and `rplot`.

### Changed

- `model.steady()` and `model.simulate()` now raise `SystemStructureError`
  on a non-square system, like `model.solve()` already did, instead of
  failing inside SciPy.
- The license is BSD-3-Clause; the duplicate `BSD-3-Clause.txt` file was
  merged into `LICENSE`.
- `model.solve()` on a model whose steady state is still undefined raises
  `UndefinedSymbolError` naming the variables, instead of a SciPy error.
- `solve_ti` starts from a deterministic initial guess instead of a random
  matrix.
- `dynare-preprocessor-pylib` is no longer restricted to Linux in `pixi.toml`.
- Relaxed dependency floors to what dyno actually needs on Python 3.12:
  `numpy>=1.26.0`, `pandas>=2.1.1`, `scipy>=1.11.3`, `rich>=13.0.0`,
  `altair>=5.0.0`, `lark>=1.1.7`, `typing_extensions>=4.6.0`,
  `pyyaml>=6.0.1` and, for the `experimental` extra, `sympy>=1.12`. A
  `min-deps` pixi environment pins every runtime dependency to its floor and
  CI runs the test suite in it.
- `pyproject.toml` is the single source of truth for the version, Python floor
  and runtime dependencies: pixi-build-python reads them from there, so
  `pixi.toml [package]` no longer repeats the version or run dependencies.
  The workspace Python floor is now `>=3.12`, like the package.
  `ci/check_deps.py` (`pixi run check-deps`, run in CI) fails when the
  dependency lists in `pixi.toml` drift from `pyproject.toml`.
- The top-level `dyno` namespace now declares `__all__` (models, result,
  solution, simulation and variants classes, errors and warnings,
  `examples_path`) and no longer re-exports `dyno.solver` / `dyno.simul`
  with `import *`. Low-level functions such as `solve_qz`, `solve_ti`,
  `moments`, `deterministic_solve`, `irfs`, `simulate` and `sim_to_nsim`
  must be imported from `dyno.solver` or `dyno.simul`.

### Removed

- Backward-compatibility aliases: the `dyno.dynare_model` module (use
  `dyno.dynare.DynareModel`), `model.data` (use `model.symbolic`),
  `dyno.Report`, `dyno.DynoRunResults`, `dyno.DynareRunResults` (use
  `dyno.RunResults`) and `dyno.dynare_cli` (use `dyno.cli.dynare`).

- The unused conda recipe `recipe/recipe.yaml` (stale at version 0.1.7);
  `pixi build` / `pixi publish` builds the conda package.

- Tracked build and scratch artifacts (`coverage.xml`, `mkdocs.yml`,
  `import_tests.yaml`, `TODO`). The root `rbc.mod` used by the tests moved
  to `examples/modfiles/rbc_simple.mod`.

## [0.1.11] and earlier

Breaking changes to the `.dyno` language made before this changelog existed:

- Shock declarations `N(...)` take a standard deviation, not a variance:
  `e[t] <- N(0.01)` or `N(mean, std)`.
- The `::` separator precedes every statement or block annotation
  (`eq :: [tags]`, `[tags] :: { ... }`); the bare `eq [tags]` form is no
  longer accepted (EconForge/dyno.py#17).
