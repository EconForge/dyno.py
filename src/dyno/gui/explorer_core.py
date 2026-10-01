"""Backend-agnostic logic for the interactive model representation explorer.

This module deliberately avoids importing `solara`, so it can be exercised by
the regular (non-GUI) test suite. The actual Solara UI lives in
`dyno.gui.explorer` and only calls into the functions defined here.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from dyno import DynoModel
from dyno.model_render import ansi_to_html
from dyno.myst import render_markdown_myst

ContentType = Literal["representation", "report"]
OutputFormat = Literal["text", "html", "markdown"]
# "myst" content is pre-rendered HTML that still needs client-side math
# typesetting (see `dyno.gui.explorer.MystHtml`); "html" content is ready to
# display as-is.
RenderKind = Literal["html", "myst"]

CONTENT_TYPES: tuple[ContentType, ...] = ("representation", "report")
OUTPUT_FORMATS: tuple[OutputFormat, ...] = ("text", "html", "markdown")


def _dynare_model_class() -> type | None:
    try:
        from dyno.dynare import DynareModel
    except ModuleNotFoundError as error:
        if error.name != "dynare_preprocessor":
            raise
        return None
    return DynareModel


def available_backends(path: Path) -> dict[str, type]:
    """Backend classes that can plausibly import `path`, keyed by name."""
    backends: dict[str, type] = {"DynoModel": DynoModel}
    if path.suffix.lower() == ".mod":
        dynare_cls = _dynare_model_class()
        if dynare_cls is not None:
            backends["DynareModel"] = dynare_cls
    return backends


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def discover_models(directory: str | Path = "examples") -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    paths = {
        p
        for pattern in ("*.dyno", "*.mod")
        for p in root.rglob(pattern)
        if not _is_hidden(p, root)
    }
    return sorted(p for p in paths if p.is_file())


@dataclass(frozen=True)
class ImportVariant:
    """One (backend, strict) combination to import a model with."""

    backend: str
    strict: bool

    @property
    def key(self) -> str:
        return f"{self.backend}{'-strict' if self.strict else ''}"

    @property
    def label(self) -> str:
        return f"{self.backend} (strict={self.strict})"


def variants_for(path: Path) -> list[ImportVariant]:
    """All (backend, strict) combinations worth offering for `path`."""
    variants: list[ImportVariant] = []
    for backend in available_backends(path):
        variants.append(ImportVariant(backend, False))
        variants.append(ImportVariant(backend, True))
    return variants


def default_variant_keys(path: Path) -> list[str]:
    """Sensible default selection: each available backend, non-strict."""
    return [ImportVariant(backend, False).key for backend in available_backends(path)]


def default_backends(path: Path) -> list[str]:
    """Sensible default backend selection: every backend available for `path`."""
    return list(available_backends(path))


def build_model(path: Path, source_text: str, variant: ImportVariant) -> Any:
    backends = available_backends(path)
    backend_cls = backends.get(variant.backend)
    if backend_cls is None:
        raise ValueError(f"Backend {variant.backend!r} is not available for {path}")
    return backend_cls(filename=str(path), txt=source_text, strict=variant.strict)


def render_representation(model: Any, format: OutputFormat) -> tuple[RenderKind, str]:
    if format == "text":
        return "html", ansi_to_html(repr(model))
    if format == "html":
        return "html", model._repr_html_()
    return "myst", render_markdown_myst(model._markdown_())


def _render_results(results: Any, format: OutputFormat) -> tuple[RenderKind, str]:
    if format == "text":
        return "html", ansi_to_html(str(results))
    if format == "html":
        rendered_html = results._repr_html_()
        return "html", (
            rendered_html if rendered_html is not None else ansi_to_html(str(results))
        )
    markdown_text = results._repr_markdown_()
    return "myst", render_markdown_myst(
        markdown_text if markdown_text else str(results)
    )


def render_report(model: Any, format: OutputFormat) -> tuple[RenderKind, str]:
    results = model.run(default_pipeline=True)
    return _render_results(results, format)


def error_fragment(error: Exception) -> str:
    message = html.escape(f"{type(error).__name__}: {error}")
    return f'<pre style="color:#b91c1c; white-space:pre-wrap">{message}</pre>'


def error_line(error: Exception) -> int | None:
    """Best-effort source line number for `error`, or None if it has none.

    Uses the `line` attribute set by dyno's parser errors (`ParserError` and
    subclasses) when present, falling back to scanning the message for a
    "line N" mention (the same convention `RunResults.add_error` uses).
    """
    line = getattr(error, "line", None)
    if isinstance(line, int):
        return line
    match = re.search(r"\blines?\s+(\d+)", str(error), flags=re.IGNORECASE)
    return int(match.group(1)) if match is not None else None


RenderedFormat = tuple[bool, RenderKind, str, "int | None"]


def render_variant_multi(
    path: Path,
    source_text: str,
    variant: ImportVariant,
    content_type: ContentType,
    formats: Sequence[OutputFormat],
) -> dict[OutputFormat, RenderedFormat]:
    """Like `render_variant`, but renders several formats from one model build.

    Building a model (and, for reports, running its default pipeline) is done
    once and reused across `formats`, instead of once per format.
    """
    try:
        model = build_model(path, source_text, variant)
    except Exception as error:
        fragment = error_fragment(error)
        line = error_line(error)
        return {fmt: (False, "html", fragment, line) for fmt in formats}

    results = None
    if content_type == "report":
        try:
            results = model.run(default_pipeline=True)
        except Exception as error:
            fragment = error_fragment(error)
            line = error_line(error)
            return {fmt: (False, "html", fragment, line) for fmt in formats}

    rendered: dict[OutputFormat, RenderedFormat] = {}
    for fmt in formats:
        try:
            if content_type == "representation":
                kind, content = render_representation(model, fmt)
            else:
                assert results is not None
                kind, content = _render_results(results, fmt)
            rendered[fmt] = (True, kind, content, None)
        except Exception as error:
            rendered[fmt] = (False, "html", error_fragment(error), error_line(error))
    return rendered


def render_variant(
    path: Path,
    source_text: str,
    variant: ImportVariant,
    content_type: ContentType,
    format: OutputFormat,
) -> RenderedFormat:
    """Build a model under `variant` and render the requested (content_type, format).

    Returns `(ok, kind, content, error_line)`. On failure, `ok` is False,
    `content` is an HTML fragment describing the error, and `error_line` is
    the offending source line when one could be determined.
    """
    return render_variant_multi(path, source_text, variant, content_type, [format])[
        format
    ]
