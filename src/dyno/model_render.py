from __future__ import annotations

import html
from math import nan
from typing import Any

import numpy as np

_ANSI_FRAGMENT_FORMAT = (
    "<pre style=\"font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace; "
    'white-space:pre !important; overflow-x:auto !important; word-break:normal !important; word-wrap:normal !important;">'
    '<code style="font-family:inherit; white-space:pre !important; word-break:normal !important;">{code}</code></pre>'
)
_MARKDOWN_FRAGMENT_FORMAT = '<div class="markdown-render">{code}</div>'


def ansi_to_html(ansi_text: str) -> str:
    """Render text that may contain ANSI escape codes (e.g. from `rich`) as HTML."""
    from rich.console import Console
    from rich.text import Text

    compact_text = "\n".join(line.rstrip() for line in ansi_text.splitlines())
    console = Console(record=True, force_terminal=True, width=1000)
    console.print(Text.from_ansi(compact_text), soft_wrap=True)
    return console.export_html(inline_styles=True, code_format=_ANSI_FRAGMENT_FORMAT)


def markdown_to_html(markdown_text: str) -> str:
    """Render a Markdown string to a standalone HTML fragment via `rich`."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console(record=True, width=90)
    console.print(Markdown(markdown_text))
    return console.export_html(
        inline_styles=True, code_format=_MARKDOWN_FRAGMENT_FORMAT
    )


def model_repr_data(model: Any) -> dict[str, Any]:
    name = model.name if model.name is not None else "Unnamed"
    filename = getattr(model, "filename", None)
    constants = model.context.get("constants", {})
    steady_states = model.context.get("steady_states", {})
    equations_count = len(getattr(model.symbolic, "equations", []))

    def _is_nan(value: Any) -> bool:
        return isinstance(value, float) and np.isnan(value)

    endogenous = [
        (v, _is_nan(steady_states.get(v, nan))) for v in model.symbols["endogenous"]
    ]
    exogenous = [
        (v, _is_nan(steady_states.get(v, nan))) for v in model.symbols["exogenous"]
    ]
    parameters = [
        (p, _is_nan(constants.get(p, nan))) for p in model.symbols["parameters"]
    ]

    has_uninitialized = any(flag for _, flag in endogenous + exogenous + parameters)

    latex_equations: str | None = None
    equations_table: str | None = None
    if hasattr(model, "latex_equations"):
        try:
            rendered = model.latex_equations()
            if isinstance(rendered, str) and rendered.strip() != "":
                latex_equations = rendered
        except Exception:
            latex_equations = None

    if hasattr(model, "symbolic") and hasattr(
        model.symbolic, "equations_table_markdown"
    ):
        try:
            table_rendered = model.symbolic.equations_table_markdown()
            if isinstance(table_rendered, str) and table_rendered.strip() != "":
                equations_table = table_rendered
        except Exception:
            equations_table = None

    return {
        "name": name,
        "filename": filename,
        "equations_count": equations_count,
        "endogenous": endogenous,
        "exogenous": exogenous,
        "parameters": parameters,
        "has_uninitialized": has_uninitialized,
        "latex_equations": latex_equations,
        "equations_table": equations_table,
    }


def render_model_text(data: dict[str, Any]) -> str:
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        orange_style = "orange3"

        def _styled_list(items: list[tuple[str, bool]]) -> Text:
            if len(items) == 0:
                return Text("<none>")
            t = Text()
            for i, (item, is_uninitialized) in enumerate(items):
                if i > 0:
                    t.append(", ")
                if is_uninitialized:
                    t.append(item)
                    t.append("^", style=orange_style)
                else:
                    t.append(item)
            return t

        table = Table(
            title=f"[bold white on dark_blue] MODEL: {data['name']} [/bold white on dark_blue]",
            box=box.HEAVY,
            show_header=False,
        )
        table.add_column("", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Names")
        table.add_row("equations", str(data["equations_count"]), "")
        table.add_row(
            "variables",
            str(len(data["endogenous"]) + len(data["exogenous"])),
            "",
        )
        table.add_row(
            "  endogenous",
            str(len(data["endogenous"])),
            _styled_list(data["endogenous"]),
        )
        table.add_row(
            "  exogenous",
            str(len(data["exogenous"])),
            _styled_list(data["exogenous"]),
        )
        table.add_row(
            "constants",
            str(len(data["parameters"])),
            _styled_list(data["parameters"]),
        )

        console = Console(force_terminal=True, color_system="truecolor", width=120)
        with console.capture() as capture:
            console.print(table)
            if data["has_uninitialized"]:
                console.print(
                    "[orange3]^[/orange3] uninitialized (steady-state) value: defaults to nan"
                )
        return capture.get().rstrip()
    except Exception:

        def _fallback_join_with_mark(items: list[tuple[str, bool]]) -> str:
            if len(items) == 0:
                return "<none>"
            out: list[str] = []
            for item, is_uninitialized in items:
                if is_uninitialized:
                    out.append(f"{item}^")
                else:
                    out.append(item)
            return ", ".join(out)

        endogenous = _fallback_join_with_mark(data["endogenous"])
        exogenous = _fallback_join_with_mark(data["exogenous"])
        parameters = _fallback_join_with_mark(data["parameters"])
        base = "\n".join(
            [
                f"* Model: {data['name']}",
                f"  equations: {data['equations_count']}",
                f"  variables: {len(data['endogenous']) + len(data['exogenous'])}",
                f"    endogenous: {endogenous}",
                f"    exogenous: {exogenous}",
                f"  constants: {parameters}",
            ]
        )
        if data["has_uninitialized"]:
            return base + "\n^ uninitialized (steady-state) value: defaults to nan"
        return base


def render_model_html(
    data: dict[str, Any],
    filename: str | None = None,
    *,
    variants: list[str] | None = None,
) -> str:
    resolved_filename = filename or data.get("filename")

    def _html_list(items: list[tuple[str, bool]]) -> str:
        if len(items) == 0:
            return "&lt;none&gt;"
        formatted: list[str] = []
        for name, is_uninitialized in items:
            badge = f"<code>{html.escape(name)}</code>"
            if is_uninitialized:
                badge += '<sup style="color:#d97706">^</sup>'
            formatted.append(badge)
        return ", ".join(formatted)

    file_line = (
        f'<p style="margin:4px 0 10px 0; color:#475569; font-size:13px;"><strong>File:</strong> <code>{html.escape(resolved_filename)}</code></p>'
        if resolved_filename
        else ""
    )

    footnote = (
        '<p style="margin-top:6px; font-size:12px; color:#64748b;"><span style="color:#d97706">^</span> uninitialized (steady-state) value: defaults to nan</p>'
        if data.get("has_uninitialized")
        else ""
    )

    cell_style = "padding:6px 10px; border:1px solid #e2e8f0;"
    header_style = "padding:6px 10px; border:1px solid #e2e8f0; background:#f8fafc;"

    variants_row = ""
    if variants:
        v_badges = ", ".join(f"<code>{html.escape(v)}</code>" for v in variants)
        variants_row = f"""    <tr>
      <td style="{cell_style}"><strong>Variants</strong></td>
      <td style="{cell_style} text-align:right;">{len(variants)}</td>
      <td style="{cell_style}">{v_badges}</td>
    </tr>\n"""

    return f"""
<h3>Model: {html.escape(data['name'])}</h3>
{file_line}<table style="border-collapse:collapse; border:1px solid #e2e8f0; font-size:13px; margin:8px 0;">
  <thead>
    <tr>
      <th style="{header_style} text-align:left;">Component</th>
      <th style="{header_style} text-align:right;">Count</th>
      <th style="{header_style} text-align:left;">Symbols</th>
    </tr>
  </thead>
  <tbody>
{variants_row}    <tr>
      <td style="{cell_style}"><strong>Equations</strong></td>
      <td style="{cell_style} text-align:right;">{data.get('equations_count', 0)}</td>
      <td style="{cell_style}"></td>
    </tr>
    <tr>
      <td style="{cell_style}"><strong>Endogenous</strong></td>
      <td style="{cell_style} text-align:right;">{len(data.get('endogenous', []))}</td>
      <td style="{cell_style}">{_html_list(data.get('endogenous', []))}</td>
    </tr>
    <tr>
      <td style="{cell_style}"><strong>Exogenous</strong></td>
      <td style="{cell_style} text-align:right;">{len(data.get('exogenous', []))}</td>
      <td style="{cell_style}">{_html_list(data.get('exogenous', []))}</td>
    </tr>
    <tr>
      <td style="{cell_style}"><strong>Parameters</strong></td>
      <td style="{cell_style} text-align:right;">{len(data.get('parameters', []))}</td>
      <td style="{cell_style}">{_html_list(data.get('parameters', []))}</td>
    </tr>
  </tbody>
</table>
{footnote}
"""


def render_model_overview_markdown(
    data: dict[str, Any],
    filename: str | None = None,
    *,
    variants: list[str] | None = None,
) -> str:
    """Render a structured overview card with model components, dimensions, and symbols."""

    def _format_symbols(items: list[Any]) -> str:
        parts: list[str] = []
        for item in items:
            if isinstance(item, tuple):
                name, is_uninit = item
                parts.append(f"`{name}^`" if is_uninit else f"`{name}`")
            else:
                parts.append(f"`{item}`")
        return ", ".join(parts)

    lines = [":::{note} Model Overview"]
    model_name = data.get("name")
    resolved_filename = filename or data.get("filename")

    if model_name and resolved_filename:
        lines.append(f"**Model:** {model_name}  ")
        lines.append(f"**File:** `{resolved_filename}`")
    elif model_name:
        lines.append(f"**Model:** {model_name}")
    elif resolved_filename:
        lines.append(f"**File:** `{resolved_filename}`")

    endo_str = _format_symbols(data.get("endogenous", []))
    exo_str = _format_symbols(data.get("exogenous", []))
    params_str = _format_symbols(data.get("parameters", []))

    lines.extend(
        [
            "",
            "| Component | Count | Symbols |",
            "|:---|---:|:---|",
        ]
    )
    if variants:
        v_list = ", ".join(f"`{v}`" for v in variants)
        lines.append(f"| **Variants** | {len(variants)} | {v_list} |")

    lines.extend(
        [
            f"| **Equations** | {data.get('equations_count', 0)} | |",
            f"| **Endogenous** | {len(data.get('endogenous', []))} | {endo_str} |",
            f"| **Exogenous** | {len(data.get('exogenous', []))} | {exo_str} |",
            f"| **Parameters** | {len(data.get('parameters', []))} | {params_str} |",
            ":::",
        ]
    )
    if data.get("has_uninitialized"):
        lines.extend(["", "`^` uninitialized (steady-state) value: defaults to `nan`"])
    return "\n".join(lines)


def render_model_markdown(data: dict[str, Any], filename: str | None = None) -> str:
    overview = render_model_overview_markdown(data, filename)
    lines = [overview]

    equations_table = data.get("equations_table")
    if isinstance(equations_table, str) and equations_table.strip() != "":
        lines.extend(["", "## Equations", "", equations_table])
    else:
        latex_equations = data.get("latex_equations")
        if isinstance(latex_equations, str) and latex_equations.strip() != "":
            lines.extend(["", "## Equations", "", latex_equations])

    return "\n".join(lines)
