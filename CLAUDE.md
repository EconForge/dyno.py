# Claude Agent Entry

Use AGENTS.md in this repository root as the canonical instruction source.

Operational rules for this project:

- Use Pixi, not pip/poetry, for project tasks.
- Prefer `pixi run -e dev <task>` for deterministic execution.
- Run tests and type checks after changes:
  - `pixi run -e dev test`
  - `pixi run -e dev typecheck`
- If touching public behavior or APIs, update docs as needed.

If any instruction in this file conflicts with AGENTS.md, AGENTS.md takes precedence.
