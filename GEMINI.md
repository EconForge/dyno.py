# Gemini Agent Entry

Use AGENTS.md in this repository root as the canonical instruction source.

Project execution policy:

- Use Pixi tasks for all checks and runs.
- Default environment: `dev`.
- Main validation commands:
  - `pixi run -e dev test`
  - `pixi run -e dev typecheck`
  - `pixi run -e dev black`

When there is ambiguity, follow AGENTS.md.
