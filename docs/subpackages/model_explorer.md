# Model Representation Explorer (Dev Tool)

!!! note "This is not a modeling tool"
    The Model Representation Explorer is a **maintainer/contributor tool** for
    visualizing the current state of the library across every example model.
    It is **not** meant for developing, calibrating, or simulating your own
    models — for that, use [Dyno Lab](../dyno_lab/index.md) or the Python API
    directly.

## What it's for

As `dyno` grows, it needs to keep importing and rendering a large, varied set
of example models (`.dyno`, `.mod`) correctly across several code paths:
two backends (`DynoModel`, `DynareModel`), strict/non-strict import options,
and several output representations (plain-text, HTML, Markdown). Checking all
of that by hand after a change to the parser, the model classes, or the
report templates is slow and easy to get wrong.

The explorer gives a single-glance view of that surface: pick any file under
`examples/`, see how it imports and renders right now, and compare backends
or options side by side. It's a way to *see the library's progress and catch
regressions*, not a place to author models.

## Launching it

```bash
pixi run -e solara explorer
```

This starts a local Solara server (by default at `http://localhost:8765`)
and opens the app in your browser.

## What you can do

- **Browse every example model** under `examples/` from the sidebar list,
  filterable by name, with `↑`/`↓` to jump between results without leaving
  the filter box.
- **Edit the source live** in the main panel (with line numbers) and watch
  every panel recompute as you type.
- **Choose what to display**: the model's *representation* (`repr()`,
  `_repr_html_()`, `_markdown_()`) or the *report* produced by
  `model.run(default_pipeline=True)`.
- **Compare frontends side by side**: text, HTML, and Markdown can all be
  shown at once, stacked per import variant and aligned across variants.
  Markdown is rendered with MyST-flavored syntax (admonitions, dropdowns,
  math), matching the style used by `RunResults`' own report template.
- **Compare import options**: toggle `DynoModel` vs. `DynareModel` (when a
  `.mod` file supports both) and a shared `strict=True` switch, to see how
  each combination handles the same source.
- **Spot parse errors at their source line**: when an import or render step
  fails with a line number attached, that line is highlighted in the gutter.

## Implementation notes

The app lives in `src/dyno/gui/explorer.py` (the Solara UI) and
`src/dyno/gui/explorer_core.py` (entry point: `scripts/model_representation.py`).
`explorer_core` holds all the backend-agnostic logic (discovering models,
building them under a given import variant, rendering a given
content/format combination) and has no dependency on `solara`, so it's
covered by the regular test suite (`tests/test_explorer_core.py`) even
outside the `solara` pixi environment.
