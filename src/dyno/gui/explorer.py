"""Interactive Solara app for browsing example models and comparing how
different import backends/options render them.

The heavy lifting (discovering files, building models, rendering output) is
delegated to `dyno.gui.explorer_core`, which has no dependency on `solara` and
is covered by the regular test suite. This module only wires that logic into
a reactive Solara UI: editing the source text, or toggling any option,
recomputes the representation panels automatically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .explorer_core import (
    CONTENT_TYPES,
    OUTPUT_FORMATS,
    ImportVariant,
    available_backends,
    default_backends,
    discover_models,
    render_variant_multi,
)

_FONT_SIZE_PX = 13
_LINE_HEIGHT_PX = 20
_GUTTER_PADDING_PX = 12
# Empirical correction: the gutter's numbers were landing exactly half a line
# below where they should (each line of text fell between two numbers rather
# than next to its own), so shift the gutter down by half a line-height.
_GUTTER_OFFSET_PX = _LINE_HEIGHT_PX // 2


def model_representation_gui(directory: str | Path = "examples"):
    import ipyvuetify as v
    import solara
    from reacton import ipyvue
    from solara.alias import rv
    from solara.components.markdown import _markdown_template

    @solara.component
    def MystHtml(html_content: str):
        # Reuses Solara's own Markdown-rendering shell (`_markdown_template`)
        # so MyST-rendered HTML gets the same client-side KaTeX math
        # typesetting and Jupyter-like styling as `solara.Markdown`, instead
        # of reimplementing that from scratch. This reaches into a private
        # Solara helper (`solara.components.markdown._markdown_template`),
        # so it may need adjusting if a future Solara release changes it.
        content_hash = hashlib.sha256(html_content.encode("utf-8")).hexdigest()
        return v.VuetifyTemplate.element(template=_markdown_template(html_content)).key(
            content_hash
        )

    files = discover_models(directory)
    if not files:
        raise FileNotFoundError(f"No .dyno or .mod files found under {directory!r}")

    file_labels = [str(path) for path in files]

    selected_file = solara.reactive(file_labels[0])
    source_text = solara.reactive(files[0].read_text())
    content_type = solara.reactive(CONTENT_TYPES[0])
    selected_formats: solara.Reactive[list[str]] = solara.reactive(["html"])
    show_source = solara.reactive(True)
    selected_backends: solara.Reactive[list[str]] = solara.reactive(
        default_backends(files[0])
    )
    strict_mode = solara.reactive(False)
    search = solara.reactive("")

    def select_file(label: str) -> None:
        path = Path(label)
        selected_file.value = label
        source_text.value = path.read_text()
        selected_backends.value = default_backends(path)

    def visible_labels_for(query: str) -> list[str]:
        query = query.strip().lower()
        return [label for label in file_labels if not query or query in label.lower()]

    def move_selection(delta: int) -> None:
        visible = visible_labels_for(search.value)
        if not visible:
            return
        try:
            index = visible.index(selected_file.value)
        except ValueError:
            index = -1 if delta > 0 else 0
        new_index = min(max(index + delta, 0), len(visible) - 1)
        select_file(visible[new_index])

    @solara.component
    def ModelList():
        visible_labels = visible_labels_for(search.value)

        with solara.Column(gap="2px"):
            search_field = rv.TextField(
                v_model=search.value,
                on_v_model=search.set,
                label="Filter models",
                dense=True,
                hide_details=True,
                clearable=True,
            )
            ipyvue.use_event(search_field, "keydown.down", lambda *_: move_selection(1))
            ipyvue.use_event(search_field, "keydown.up", lambda *_: move_selection(-1))
            solara.Text(
                "↑/↓ to jump between models",
                style="color:#94a3b8; font-size:0.75rem;",
            )
            for label in visible_labels:
                active = label == selected_file.value
                solara.Button(
                    label,
                    text=True,
                    outlined=active,
                    on_click=lambda label=label: select_file(label),
                    style="justify-content:flex-start; width:100%; text-transform:none;",
                )

    @solara.component
    def Controls(path: Path):
        with solara.Column(gap="8px"):
            solara.Select(
                label="Content", value=content_type, values=list(CONTENT_TYPES)
            )

            solara.Text("Frontends to show")
            with solara.Row(gap="12px"):
                for fmt in OUTPUT_FORMATS:
                    checked = fmt in selected_formats.value

                    def toggle_format(value: bool, fmt: str = fmt) -> None:
                        formats = set(selected_formats.value)
                        if value:
                            formats.add(fmt)
                        else:
                            formats.discard(fmt)
                        selected_formats.value = [
                            f for f in OUTPUT_FORMATS if f in formats
                        ]

                    solara.Checkbox(label=fmt, value=checked, on_value=toggle_format)

            with solara.Row(gap="12px"):
                solara.Checkbox(label="Show source", value=show_source)
                solara.Checkbox(label="strict=True", value=strict_mode)

            solara.Text("Compare imports")
            for backend in available_backends(path):
                backend_checked = backend in selected_backends.value

                def toggle_backend(value: bool, backend: str = backend) -> None:
                    backends = set(selected_backends.value)
                    if value:
                        backends.add(backend)
                    else:
                        backends.discard(backend)
                    selected_backends.value = sorted(backends)

                solara.Checkbox(
                    label=backend, value=backend_checked, on_value=toggle_backend
                )

    @solara.component
    def SourcePanel(error_lines: set[int]):
        text = source_text.value
        n_lines = text.count("\n") + 1

        def _gutter_line(i: int) -> str:
            if i not in error_lines:
                return str(i)
            return (
                '<span style="display:inline-block; width:100%; '
                "background:#fecaca; color:#7f1d1d; font-weight:700; "
                'border-radius:3px;">'
                f"{i}</span>"
            )

        gutter_text = "\n".join(_gutter_line(i) for i in range(1, n_lines + 1))
        content_height_px = n_lines * _LINE_HEIGHT_PX + 2 * _GUTTER_PADDING_PX
        with solara.Card(title="Source"):
            with solara.Row(
                gap="0px",
                style=(
                    f"font-family:monospace; font-size:{_FONT_SIZE_PX}px; "
                    f"line-height:{_LINE_HEIGHT_PX}px;"
                ),
            ):
                solara.HTML(
                    tag="pre",
                    unsafe_innerHTML=gutter_text,
                    style=(
                        f"margin:0; padding:{_GUTTER_PADDING_PX + _GUTTER_OFFSET_PX}px 8px "
                        f"{_GUTTER_PADDING_PX}px 12px; text-align:right; "
                        f"font-family:monospace; font-size:{_FONT_SIZE_PX}px; "
                        f"line-height:{_LINE_HEIGHT_PX}px !important; letter-spacing:normal; "
                        "color:#94a3b8; user-select:none; background:#f8fafc; "
                        "border-right:1px solid #e2e8f0; box-sizing:border-box;"
                    ),
                )
                rv.Textarea(
                    v_model=text,
                    on_v_model=source_text.set,
                    auto_grow=False,
                    hide_details=True,
                    solo=True,
                    flat=True,
                    class_="dyno-source-textarea",
                )
            solara.HTML(
                tag="style",
                unsafe_innerHTML=(
                    ".dyno-source-textarea textarea {"
                    f"font-family:monospace !important; font-size:{_FONT_SIZE_PX}px !important; "
                    f"line-height:{_LINE_HEIGHT_PX}px !important; white-space:pre !important; "
                    f"overflow-x:auto !important; overflow-y:hidden !important; "
                    "box-sizing:border-box !important; border:0 !important; "
                    f"padding-top:{_GUTTER_PADDING_PX}px !important; "
                    f"padding-bottom:{_GUTTER_PADDING_PX}px !important; "
                    f"height:{content_height_px}px !important; "
                    f"min-height:{content_height_px}px !important;"
                    "}"
                ),
            )

    @solara.component
    def VariantColumn(variant: ImportVariant, rendered: dict):
        with solara.Column():
            solara.Text(variant.label, style="font-weight:600;")
            for fmt, (ok, kind, content, _line) in rendered.items():
                with solara.Card(title=fmt):
                    if not ok:
                        solara.Error(label="Import or rendering failed")
                    if kind == "myst":
                        MystHtml(content)
                    else:
                        solara.HTML(tag="div", unsafe_innerHTML=content)

    @solara.component
    def Page():
        path = Path(selected_file.value)

        with solara.Head():
            solara.Title(f"Dyno: {path}")

        with solara.Sidebar():
            solara.HTML(
                tag="style",
                unsafe_innerHTML=(
                    ".v-navigation-drawer {"
                    "min-width:260px !important; max-width:300px !important;"
                    "}"
                ),
            )
            # `solara.Column` is itself a flex column (`display:flex; flex-direction:column`),
            # so nesting one bounded to the sidebar's own height turns the inner
            # "model list" column into a proper flex item that can scroll on its
            # own, independently of the (fixed) controls above it.
            with solara.Column(style="height:100%; overflow:hidden;", gap="0px"):
                with solara.Column(style="flex:0 0 auto;"):
                    Controls(path)
                    solara.Markdown("---")
                with solara.Column(
                    style="flex:1 1 auto; min-height:0; overflow-y:auto;"
                ):
                    ModelList()

        solara.Markdown(f"## {path}")

        active_variants = [
            ImportVariant(backend, strict_mode.value)
            for backend in available_backends(path)
            if backend in selected_backends.value
        ]
        panel_count = (1 if show_source.value else 0) + len(active_variants)
        if panel_count == 0:
            solara.Warning(
                label="Select at least one import option, or show the source, in the sidebar."
            )
        elif not selected_formats.value and active_variants:
            solara.Warning(
                label="Select at least one frontend to show, in the sidebar."
            )
        else:
            formats = [fmt for fmt in OUTPUT_FORMATS if fmt in selected_formats.value]
            variant_results = {
                variant.key: render_variant_multi(
                    path, source_text.value, variant, content_type.value, formats
                )
                for variant in active_variants
            }
            error_lines = {
                line
                for rendered in variant_results.values()
                for ok, _kind, _content, line in rendered.values()
                if not ok and line is not None
            }
            with solara.Columns([1] * panel_count):
                if show_source.value:
                    SourcePanel(error_lines)
                for variant in active_variants:
                    VariantColumn(variant, variant_results[variant.key])

    return Page()
