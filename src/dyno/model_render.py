from __future__ import annotations

from math import nan
from typing import Any

import numpy as np


def model_repr_data(model: Any) -> dict[str, Any]:
    name = model.name if model.name is not None else "Unnamed"
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

    return {
        "name": name,
        "equations_count": equations_count,
        "endogenous": endogenous,
        "exogenous": exogenous,
        "parameters": parameters,
        "has_uninitialized": has_uninitialized,
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


def render_model_html(data: dict[str, Any]) -> str:
    def _html_list(items: list[tuple[str, bool]]) -> str:
        if len(items) == 0:
            return "&lt;none&gt;"
        formatted: list[str] = []
        for name, is_uninitialized in items:
            if is_uninitialized:
                formatted.append(f'<span style="color:#d97706">{name}<sup>^</sup></span>')
            else:
                formatted.append(name)
        return ", ".join(formatted)

    footnote = (
        '<p><span style="color:#d97706">^</span> uninitialized (steady-state) value: defaults to nan</p>'
        if data["has_uninitialized"]
        else ""
    )

    return f"""
<h3>Model: {data['name']}</h3>
<table>
  <tbody>
    <tr><td>equations</td><td>{data['equations_count']}</td><td></td></tr>
    <tr><td>variables</td><td>{len(data['endogenous']) + len(data['exogenous'])}</td><td></td></tr>
    <tr><td>&nbsp;&nbsp;endogenous</td><td>{len(data['endogenous'])}</td><td>{_html_list(data['endogenous'])}</td></tr>
    <tr><td>&nbsp;&nbsp;exogenous</td><td>{len(data['exogenous'])}</td><td>{_html_list(data['exogenous'])}</td></tr>
    <tr><td>constants</td><td>{len(data['parameters'])}</td><td>{_html_list(data['parameters'])}</td></tr>
  </tbody>
</table>
{footnote}
"""


def render_model_markdown(data: dict[str, Any], filename: str) -> str:
    def _md_list(items: list[tuple[str, bool]]) -> str:
        if len(items) == 0:
            return "<none>"
        out: list[str] = []
        for name, is_uninitialized in items:
            if is_uninitialized:
                out.append(f"{name}^")
            else:
                out.append(name)
        return ", ".join(out)

    variables_count = len(data["endogenous"]) + len(data["exogenous"])

    lines = [
        f"# Model: {data['name']}",
        f"- *filename*: {filename}",
        "",
        "|  | Count | Names |",
        "|---|---:|---|",
        f"| equations | {data['equations_count']} |  |",
        f"| variables | {variables_count} |  |",
        f"|   endogenous | {len(data['endogenous'])} | {_md_list(data['endogenous'])} |",
        f"|   exogenous | {len(data['exogenous'])} | {_md_list(data['exogenous'])} |",
        f"| constants | {len(data['parameters'])} | {_md_list(data['parameters'])} |",
    ]

    if data["has_uninitialized"]:
        lines.extend(["", "`^` uninitialized (steady-state) value: defaults to `nan`"])

    return "\n".join(lines)
