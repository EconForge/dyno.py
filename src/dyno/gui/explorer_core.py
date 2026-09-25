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

ContentType = Literal["representation", "report"]
OutputFormat = Literal["text", "html", "markdown"]
# "myst" content is pre-rendered HTML that still needs client-side math
# typesetting (see `dyno.gui.explorer.MystHtml`); "html" content is ready to
# display as-is.
RenderKind = Literal["html", "myst"]

CONTENT_TYPES: tuple[ContentType, ...] = ("representation", "report")
OUTPUT_FORMATS: tuple[OutputFormat, ...] = ("text", "html", "markdown")

# MyST-flavored markdown-it-py plugins, minus `dollarmath` (its default
# renderer replaces "$$...$$" with a `<div class="math">`, which would stop
# the client-side KaTeX auto-render script -- borrowed from `solara.Markdown`
# -- from finding anything to typeset). Plain "$$...$$" passthrough already
# works with that auto-render step, so it's left alone. `colon_fence` is
# included, but needs the custom directive rendering below to be useful --
# on its own it just turns `:::{tip} ...` into an unstyled code block.
_MYST_PLUGIN_MODULES = (
    ("mdit_py_plugins.front_matter", "front_matter_plugin"),
    ("mdit_py_plugins.myst_blocks", "myst_block_plugin"),
    ("mdit_py_plugins.myst_role", "myst_role_plugin"),
    ("mdit_py_plugins.deflist", "deflist_plugin"),
    ("mdit_py_plugins.footnote", "footnote_plugin"),
    ("mdit_py_plugins.tasklists", "tasklists_plugin"),
    ("mdit_py_plugins.attrs", "attrs_plugin"),
    ("mdit_py_plugins.colon_fence", "colon_fence_plugin"),
)

# name -> (default title, accent color, background color)
_ADMONITION_STYLES: dict[str, tuple[str, str, str]] = {
    "note": ("Note", "#3b82f6", "#eff6ff"),
    "tip": ("Tip", "#10b981", "#ecfdf5"),
    "hint": ("Hint", "#10b981", "#ecfdf5"),
    "seealso": ("See also", "#10b981", "#ecfdf5"),
    "warning": ("Warning", "#f59e0b", "#fffbeb"),
    "caution": ("Caution", "#f59e0b", "#fffbeb"),
    "attention": ("Attention", "#f59e0b", "#fffbeb"),
    "error": ("Error", "#ef4444", "#fef2f2"),
    "danger": ("Danger", "#ef4444", "#fef2f2"),
    "important": ("Important", "#8b5cf6", "#f5f3ff"),
    "admonition": ("Note", "#6b7280", "#f9fafb"),
}

# Matches a colon-fence/directive-fence info string like "{tip} some title".
_DIRECTIVE_INFO_RE = re.compile(r"^\{([\w-]+)\}\s*(.*)$")
# Matches a MyST directive option line, e.g. ":class: dropdown".
_DIRECTIVE_OPTION_RE = re.compile(r"^:([\w-]+):\s*(.*)$")


def _split_directive_options(content: str) -> tuple[dict[str, str], str]:
    """Split leading MyST `:key: value` option lines off a directive body."""
    lines = content.splitlines()
    options: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = _DIRECTIVE_OPTION_RE.match(lines[i])
        if match is None:
            break
        options[match.group(1)] = match.group(2).strip()
        i += 1
    if i < len(lines) and lines[i].strip() == "":
        i += 1
    return options, "\n".join(lines[i:])


def _details_html(*, title: str, body_html: str, style: str, open: bool = False) -> str:
    open_attr = " open" if open else ""
    return (
        f'<details{open_attr} style="{style}">'
        f'<summary style="font-weight:600; cursor:pointer;">{html.escape(title)}</summary>'
        f'<div style="margin-top:0.5em;">{body_html}</div></details>'
    )


def _render_directive(md: Any, name: str, title_arg: str, content: str) -> str:
    """Render one MyST colon-fence directive's body to an HTML fragment."""
    options, body = _split_directive_options(content)
    body_html = md.render(body)

    if name in _ADMONITION_STYLES:
        default_title, accent, background = _ADMONITION_STYLES[name]
        title = title_arg or default_title
        box_style = (
            f"border-left:4px solid {accent}; background:{background}; "
            "padding:0.75em 1em; margin:1em 0; border-radius:4px;"
        )
        if options.get("class") == "dropdown":
            return _details_html(title=title, body_html=body_html, style=box_style)
        title_html = (
            f'<p style="font-weight:700; margin:0 0 0.5em 0; color:{accent};">'
            f"{html.escape(title)}</p>"
        )
        return f'<div style="{box_style}">{title_html}{body_html}</div>'

    plain_style = (
        "border:1px solid #e2e8f0; padding:0.5em 1em; margin:1em 0; border-radius:4px;"
    )
    if name == "dropdown":
        return _details_html(
            title=title_arg or "Details", body_html=body_html, style=plain_style
        )
    if name == "tab-item":
        return _details_html(
            title=title_arg or "Tab", body_html=body_html, style=plain_style, open=True
        )
    if name == "tab-set":
        # No interactive tab strip (that needs real client-side JS); each
        # `tab-item` already renders as its own open, labeled section.
        return f'<div class="dyno-tab-set">{body_html}</div>'

    # Unrecognized directive: show its name/title rather than raw `:::` syntax.
    label = html.escape(name) + (f": {html.escape(title_arg)}" if title_arg else "")
    return (
        '<div style="border:1px dashed #cbd5e1; padding:0.5em 1em; margin:1em 0;">'
        f'<p style="font-weight:600; margin:0 0 0.5em 0;">{label}</p>{body_html}</div>'
    )


def render_markdown_myst(markdown_text: str) -> str:
    """Render Markdown to HTML using MyST-flavored syntax extensions.

    This layers the same markdown-it-py plugins MyST itself is built on
    (front matter, MyST block/role syntax, definition lists, footnotes, task
    lists, attribute lists, colon-fence directives) on top of the
    CommonMark+GFM-table preset, without pulling in the full Sphinx/docutils
    toolchain. Directives get a hand-rolled renderer covering what this app's
    own Markdown output actually uses (admonitions, dropdowns, tab items, and
    `` ```{code} lang `` fences) -- not arbitrary Sphinx directives.
    """
    import importlib

    from markdown_it import MarkdownIt

    md = MarkdownIt("js-default", {"html": True, "typographer": True})
    for module_name, plugin_name in _MYST_PLUGIN_MODULES:
        plugin = getattr(importlib.import_module(module_name), plugin_name)
        md = md.use(plugin)

    default_fence_rule = md.renderer.rules["fence"]  # type: ignore[attr-defined]

    def render_colon_fence(
        self: Any, tokens: Any, idx: Any, options: Any, env: Any
    ) -> str:
        token = tokens[idx]
        match = _DIRECTIVE_INFO_RE.match(token.info.strip())
        if match is None:
            return f"<pre><code>{html.escape(token.content)}</code></pre>\n"
        return _render_directive(
            md, match.group(1), match.group(2).strip(), token.content
        )

    def render_fence(self: Any, tokens: Any, idx: Any, options: Any, env: Any) -> str:
        token = tokens[idx]
        match = _DIRECTIVE_INFO_RE.match(token.info.strip())
        if match is not None and match.group(1) == "code":
            token.info = match.group(2).strip()
        return default_fence_rule(tokens, idx, options, env)

    md.add_render_rule("colon_fence", render_colon_fence)
    md.add_render_rule("fence", render_fence)

    return md.render(markdown_text)


def _dynare_model_class() -> type | None:
    try:
        from dyno.dynare_model import DynareModel
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


def discover_models(directory: str | Path = "examples") -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    paths = {p for pattern in ("*.dyno", "*.mod") for p in root.rglob(pattern)}
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
