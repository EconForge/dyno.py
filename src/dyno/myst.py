"""Render MyST-flavored Markdown to self-contained HTML.

Used by the model explorer and by the `output_type="myst"` report mode, so
MyST reports display correctly in any frontend that can show HTML (including
JupyterLite/wasm, where `jupyterlab-myst` cannot be installed), without a
frontend MyST renderer.
"""

from __future__ import annotations

import html
import re
from typing import Any

# MyST-flavored markdown-it-py plugins. `dollarmath` is only used to keep
# markdown syntax out of math (so `_`, `*` or `\\` inside "$...$" are not
# turned into emphasis or escapes); its math tokens are rendered back as the
# original "$...$" / "$$...$$" text (see `_render_math_*`), which frontends
# then typeset: KaTeX auto-render in the explorer, MathJax in JupyterLab.
# `colon_fence` needs the custom directive rendering below to be useful -- on
# its own it just turns `:::{tip} ...` into an unstyled code block.
_MYST_PLUGIN_MODULES = (
    ("mdit_py_plugins.front_matter", "front_matter_plugin"),
    ("mdit_py_plugins.myst_blocks", "myst_block_plugin"),
    ("mdit_py_plugins.myst_role", "myst_role_plugin"),
    ("mdit_py_plugins.deflist", "deflist_plugin"),
    ("mdit_py_plugins.footnote", "footnote_plugin"),
    ("mdit_py_plugins.tasklists", "tasklists_plugin"),
    ("mdit_py_plugins.attrs", "attrs_plugin"),
    ("mdit_py_plugins.colon_fence", "colon_fence_plugin"),
    ("mdit_py_plugins.dollarmath", "dollarmath_plugin"),
)

# name -> (default title, accent color, background color). Backgrounds are
# translucent tints of the accent, so they work on light and dark themes.
_ADMONITION_STYLES: dict[str, tuple[str, str, str]] = {
    "note": ("Note", "#3b82f6", "#3b82f61a"),
    "tip": ("Tip", "#10b981", "#10b9811a"),
    "hint": ("Hint", "#10b981", "#10b9811a"),
    "seealso": ("See also", "#10b981", "#10b9811a"),
    "warning": ("Warning", "#f59e0b", "#f59e0b1a"),
    "caution": ("Caution", "#f59e0b", "#f59e0b1a"),
    "attention": ("Attention", "#f59e0b", "#f59e0b1a"),
    "error": ("Error", "#ef4444", "#ef44441a"),
    "danger": ("Danger", "#ef4444", "#ef44441a"),
    "important": ("Important", "#8b5cf6", "#8b5cf61a"),
    "admonition": ("Note", "#6b7280", "#6b72801a"),
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


def _render_math_inline(
    self: Any, tokens: Any, idx: Any, options: Any, env: Any
) -> str:
    return f"${html.escape(tokens[idx].content)}$"


def _render_math_inline_double(
    self: Any, tokens: Any, idx: Any, options: Any, env: Any
) -> str:
    return f"$${html.escape(tokens[idx].content)}$$"


def _render_math_block(self: Any, tokens: Any, idx: Any, options: Any, env: Any) -> str:
    return f"<p>$${html.escape(tokens[idx].content)}$$</p>\n"


_MATH_RENDER_RULES = {
    "math_inline": _render_math_inline,
    "math_inline_double": _render_math_inline_double,
    "math_block": _render_math_block,
    "math_block_label": _render_math_block,
}


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
        if plugin_name == "dollarmath_plugin":
            md = md.use(plugin, double_inline=True)
        else:
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
    for rule_name, rule in _MATH_RENDER_RULES.items():
        md.add_render_rule(rule_name, rule)

    return md.render(markdown_text)
