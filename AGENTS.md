# AI Agent Guidelines for Dyno.py

This file provides instructions and context for AI coding agents working on this repository.

## Development Environment & Tooling

We use the `pixi` package manager for this project for both development and production environments. 
Please DO NOT use `poetry` or `pip` directly to install dependencies or run tasks.

By default, run Pixi tasks in the `dev` environment to avoid interactive environment selection prompts:
```bash
pixi run -e dev <task-or-command>
```

## Testing & Quality Assurance

When making changes to the codebase, please ensure that you:
1. Run unit tests using the predefined `pixi` task:
   ```bash
   pixi run -e dev test
   ```
   Or for coverage:
   ```bash
   pixi run -e dev cov
   ```
2. Verify type correctness using the `typecheck` task:
   ```bash
   pixi run -e dev typecheck
   ```
3. Update tests corresponding to any new features or changed behaviors.

## Documentation

The project uses `mkdocs` for documentation. 
If developing new features or modifying public APIs, consider updating the corresponding documentation and serve it to verify locally:
```bash
pixi run -e dev docs
```

## General Guidelines

- Write clean, type-hinted code and prioritize robust solutions.
- Adhere to the style of the surrounding code (you can run `pixi run -e dev black` to format).
- Provide clear and concise commit messages or changelogs.
- Ask questions if requirements are ambiguous or if a large architectural change is needed.
