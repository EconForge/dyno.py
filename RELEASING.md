# Release Guide for Dyno

This document provides step-by-step instructions for releasing **Dyno** across its distribution channels: **PyPI**, **prefix-dev (`econforge` channel)**, and **conda-forge**.

---

## 1. Naming & Pre-Release Checklist

### Package Name Convention

- **Repository & Conda / Pixi Package Name:** `dyno` (used in `pyproject.toml`, `pixi.toml`, `prefix.dev`, and `conda-forge`).
- **PyPI Package Name:** `dynopy` (the package on PyPI is published as `dynopy`, but installs the importable module `dyno`).

### Pre-Release Verification

Before tagging a new release, run the complete quality suite locally:

```bash
# 1. Verify runtime dependencies sync between pyproject.toml and pixi.toml
pixi run -e dev check-deps

# 2. Run static type checking
pixi run -e dev typecheck

# 3. Run the full test suite
pixi run -e dev test

# 4. Verify local documentation build
pixi run -e dev build-docs
```

### Version Bump & Changelog

1. Update the version number in `pyproject.toml`:
   ```toml
   [project]
   version = "X.Y.Z"
   ```
2. Document new features, bug fixes, and breaking changes in `CHANGELOG.md`.
3. Commit the version bump:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: release vX.Y.Z"
   ```
4. Create and push a git tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```

---

## 2. Publishing to PyPI (`dynopy`)

Dyno uses **PyPA Trusted Publishing (OIDC)** via GitHub Actions.

### Automated Release via GitHub Actions (Recommended)

When a release tag (`v*`) is pushed to GitHub, `.github/workflows/publish-to-pypi.yml` runs automatically:

1. It executes `python ci/build_pypi_dist.py --out-dir dist/`.
2. This script temporarily sets the distribution package name to `dynopy` during wheel and sdist generation.
3. It uploads `dist/dynopy-X.Y.Z-py3-none-any.whl` and `dist/dynopy-X.Y.Z.tar.gz` to PyPI.
4. It creates a GitHub Release with the build artifacts attached.

### Manual / Local Release to PyPI

If you need to build and publish manually:

```bash
# Build PyPI distribution files (dynopy package name)
pixi run -e dev build-dist

# Verify distribution archives with twine
pixi run -e dev check-dist

# Test upload to TestPyPI (optional)
pixi run -e dev publish-testpypi

# Publish to PyPI
pixi run -e dev publish-pypi
```

---

## 3. Publishing to prefix.dev (`econforge` channel)

The `econforge` channel on [prefix.dev](https://prefix.dev/econforge) hosts the official Conda package for `dyno`.

### Package Build with Pixi / Rattler-Build

`pixi.toml` configures `pixi-build-python` for generating Conda package artifacts:

```bash
# Build the conda package artifact (.conda)
pixi build
```

This generates package archives under `output/bld/noarch/dyno-X.Y.Z-*.conda`.

### Uploading to prefix.dev

Upload the `.conda` package artifact to the `econforge` channel using `rattler-build` or `pixi`:

```bash
# Set your prefix.dev API token (if required by environment)
export PREFIX_API_KEY="your-prefix-dev-token"

# Upload to the econforge channel
rattler-build upload prefix -c econforge output/bld/noarch/dyno-X.Y.Z-*.conda
```

### Verification

Verify that the new version is available on `prefix.dev`:

```bash
pixi search dyno --channel https://prefix.dev/econforge
```

---

## 4. Publishing to conda-forge (`dyno-feedstock`)

The [conda-forge](https://conda-forge.org/) package is maintained via the `dyno-feedstock` repository on GitHub.

### Automatic Updates (regro-cf-autotick-bot)

- When a new version is released on PyPI (`dynopy`), the `conda-forge` auto-tick bot automatically creates a Pull Request on `conda-forge/dyno-feedstock`.
- Review the automated PR, check dependency pins in `recipe/meta.yaml`, and merge it once CI checks pass.

### Manual Feedstock Update

If an automated PR is delayed or manual changes are required:

1. Fork and clone `https://github.com/conda-forge/dyno-feedstock`.
2. Edit `recipe/meta.yaml`:
   - Update version string: `{% set version = "X.Y.Z" %}`.
   - Update SHA-256 checksum for the PyPI source archive (`dynopy-X.Y.Z.tar.gz`).
   - Confirm import check is `imports: - dyno`.
3. Re-render the feedstock:
   ```bash
   conda-smithy rerender
   ```
4. Commit, push to your fork, and submit a PR to `conda-forge/dyno-feedstock`.
5. Once Azure Pipelines checks pass, merge the PR.
