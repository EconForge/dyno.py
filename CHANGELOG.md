# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `RedefinitionWarning`, emitted when a constant is assigned more than once
  (previously printed to stdout).

### Changed

- `model.steady()` and `model.simulate()` now raise `SystemStructureError`
  on a non-square system, like `model.solve()` already did, instead of
  failing inside SciPy.
- The license is BSD-3-Clause; the duplicate `BSD-3-Clause.txt` file was
  merged into `LICENSE`.

### Removed

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
