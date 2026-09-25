"""Render model representations for a collection of Dyno and Dynare files."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Callable

from dyno import DynoModel
from dyno.model_render import ansi_to_html as _ansi_to_html
from dyno.model_render import markdown_to_html as _markdown_to_html

OUTPUT_PATH = Path("tests/output/model_representations.html")


def _source_to_html(source: str) -> str:
    lines = source.splitlines()
    return "\n".join(
        f'<span class="source-line"><span class="line-number">{index}</span>'
        f"{html.escape(line)}</span>"
        for index, line in enumerate(lines, start=1)
    )


def _backend_classes(path: Path) -> dict[str, Callable[..., Any]]:
    backends: dict[str, Callable[..., Any]] = {"DynoModel": DynoModel}
    if path.suffix.lower() == ".mod":
        try:
            from dyno.dynare_model import DynareModel
        except ModuleNotFoundError as error:
            if error.name != "dynare_preprocessor":
                raise
        else:
            backends["DynareModel"] = DynareModel
    return backends


def _render_backend_section(name: str, model: Any) -> str:
    return f"""
<section class="backend">
  <h3>{html.escape(name)}</h3>
  <h4>repr()</h4>
  {_ansi_to_html(repr(model))}
  <h4>_repr_html_()</h4>
  {model._repr_html_()}
  <h4>_markdown_()</h4>
  {_markdown_to_html(model._markdown_())}
</section>
"""


def _render_model_section(path: Path, index: int) -> tuple[str, bool]:
    sections = []
    has_error = False
    try:
        backends = _backend_classes(path)
    except Exception as error:
        return _error_section(path, error), True

    for name, backend_cls in backends.items():
        try:
            model = backend_cls(str(path))
            sections.append(_render_backend_section(name, model))
        except Exception as error:
            sections.append(_error_section(path, error, name))
            has_error = True

    model_id = f"model-{index}"
    hidden = "" if index == 0 else " hidden"
    source = _source_to_html(path.read_text())
    return f"""
<article class="model" id="{model_id}"{hidden}>
  <h2>{html.escape(str(path))}</h2>
    <div class="panels">
        <section class="panel source-panel">
            <h3>Source</h3>
            <pre><code>{source}</code></pre>
        </section>
        {"".join(sections).replace('<section class="backend">', '<section class="panel backend">').replace('<section class="backend error">', '<section class="panel backend error">')}
    </div>
</article>
""", has_error


def _error_section(path: Path, error: Exception, backend: str | None = None) -> str:
    label = f" ({backend})" if backend else ""
    message = html.escape(f"{type(error).__name__}: {error}")
    return f"""
<section class="backend error">
  <h3>{html.escape(str(path))}{html.escape(label)}</h3>
  <pre>{message}</pre>
</section>
"""


def discover_models(inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("*.mod"))
            paths.update(path.rglob("*.dyno"))
        elif any(char in value for char in "*?[]"):
            paths.update(Path().glob(value))
        elif path.suffix.lower() in {".mod", ".dyno"}:
            paths.add(path)
    return sorted(path for path in paths if path.is_file())


def render_report(paths: list[Path], output_path: Path = OUTPUT_PATH) -> Path:
    rendered_models = [
        _render_model_section(path, index) for index, path in enumerate(paths)
    ]
    model_sections = [section for section, _ in rendered_models]
    model_index = "\n".join(
        f'<button class="model-link{" has-error" if has_error else ""}" '
        f'data-model="model-{index}" type="button"'
        f'{" title=\"Backend rendering failed\"" if has_error else ""}>'
        f"{html.escape(str(path))}</button>"
        for index, (path, (_, has_error)) in enumerate(zip(paths, rendered_models))
    )
    document = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dyno model representations</title>
<script>
  window.MathJax = {{ tex: {{ displayMath: [['$$', '$$']] }} }};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    body {{ font-family: sans-serif; margin: 1rem; }}
    .workspace {{ display: grid; grid-template-columns: minmax(14rem, 22rem) minmax(0, 1fr); gap: 1rem; align-items: start; }}
    .index {{ position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow: auto; border: 1px solid #ccc; padding: .75rem; }}
    .index input {{ box-sizing: border-box; width: 100%; margin-bottom: .75rem; padding: .5rem; }}
    .model-link {{ display: block; width: 100%; border: 0; border-left: 3px solid transparent; background: transparent; padding: .5rem; text-align: left; overflow-wrap: anywhere; cursor: pointer; }}
    .model-link:hover, .model-link.active {{ border-left-color: #1677b8; background: #eaf4fb; }}
        .model-link.has-error {{ border-left-color: #d97706; color: #b45309; }}
        .model-link.has-error:hover, .model-link.has-error.active {{ border-left-color: #b45309; background: #fff7ed; }}
    .model {{ min-width: 0; }}
    .source {{ margin-bottom: 1rem; }}
    .source summary {{ cursor: pointer; font-weight: 600; }}
    .source pre {{ max-height: 28rem; background: #f7f7f7; padding: .75rem; }}
    .panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(24rem, 1fr)); gap: 1rem; align-items: start; }}
    .panel {{ min-width: 0; overflow: hidden; border: 1px solid #ccc; padding: 1rem; }}
    .panel pre {{ max-width: 100%; max-height: 70vh; overflow: auto; }}
    .panel table {{ max-width: 100%; overflow-wrap: anywhere; }}
    .source-panel pre {{ margin: 0; }}
    .source-line {{ display: block; min-width: max-content; }}
    .line-number {{ display: inline-block; width: 3.5rem; margin-right: .75rem; color: #888; user-select: none; text-align: right; }}
  .markdown-render {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    .panel mjx-container {{ max-width: 100%; overflow-x: auto; overflow-y: hidden; }}
  .error {{ border-color: #c00; color: #900; }}
    @media (max-width: 54rem) {{
        .workspace {{ grid-template-columns: 1fr; }}
        .index {{ position: static; max-height: 16rem; }}
    }}
</style>
</head>
<body>
<h1>Dyno model representations</h1>
<p>{len(paths)} model file(s)</p>
<div class="workspace">
    <nav class="index" aria-label="Model index">
        <label for="model-search">Choose a model</label>
        <input id="model-search" type="search" placeholder="Filter models...">
        <div id="model-links">
            {model_index}
        </div>
    </nav>
    <main id="model-content">
        {"".join(model_sections)}
    </main>
</div>
<script>
    const links = [...document.querySelectorAll('.model-link')];
    const models = [...document.querySelectorAll('.model')];
    const search = document.querySelector('#model-search');

    function selectModel(id) {{
        models.forEach(model => {{ model.hidden = model.id !== id; }});
        links.forEach(link => {{ link.classList.toggle('active', link.dataset.model === id); }});
    }}

    links.forEach(link => link.addEventListener('click', () => selectModel(link.dataset.model)));
    search.addEventListener('input', () => {{
        const query = search.value.toLowerCase();
        links.forEach(link => {{
            link.hidden = !link.textContent.toLowerCase().includes(query);
        }});
    }});
    selectModel('model-0');
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["examples"],
        help="files, directories, or globs; defaults to all models under examples",
    )
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    paths = discover_models(args.paths)
    if not paths:
        parser.error("no .mod or .dyno files found")
    print(f"Model representations report written to {render_report(paths, args.output)}")


if __name__ == "__main__":
    main()