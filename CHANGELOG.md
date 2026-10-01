# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Removed

- Backward-compatibility aliases: the `dyno.dynare_model` module (use
  `dyno.dynare.DynareModel`), `model.data` (use `model.symbolic`),
  `dyno.Report`, `dyno.DynoRunResults`, `dyno.DynareRunResults` (use
  `dyno.RunResults`) and `dyno.dynare_cli` (use `dyno.cli.dynare`).

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
