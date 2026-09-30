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
    render_variant,
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
        template = _markdown_template(html_content)
        # Add v-pre to prevent Vue from parsing raw mustache tags (e.g. in math or code)
        template = template.replace(
            '<div class="solara-markdown rendered_html jp-RenderedHTMLCommon"',
            '<div v-pre class="solara-markdown rendered_html jp-RenderedHTMLCommon"',
            1,
        )
        return v.VuetifyTemplate.element(template=template).key(content_hash)

    files = discover_models(directory)
    if not files:
        raise FileNotFoundError(f"No .dyno or .mod files found under {directory!r}")

    file_labels = [str(path) for path in files]

    selected_file = solara.reactive(file_labels[0])
    source_text = solara.reactive(files[0].read_text())
    content_type = solara.reactive(CONTENT_TYPES[0])
    selected_format = solara.reactive("html")
    show_source = solara.reactive(True)
    initial_backends = default_backends(files[0])
    selected_backend = solara.reactive(initial_backends[0])
    strict_mode = solara.reactive(False)
    search = solara.reactive("")

    def select_file(label: str) -> None:
        path = Path(label)
        selected_file.value = label
        source_text.value = path.read_text()
        available = list(available_backends(path))
        if selected_backend.value not in available:
            selected_backend.value = available[0]

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

        with solara.Column(gap="4px", style="height:100%;"):
            search_field = rv.TextField(
                v_model=search.value,
                on_v_model=search.set,
                label="Filter models",
                dense=True,
                outlined=True,
                hide_details=True,
                clearable=True,
                class_="mb-1",
            )
            ipyvue.use_event(search_field, "keydown.down", lambda *_: move_selection(1))
            ipyvue.use_event(search_field, "keydown.up", lambda *_: move_selection(-1))
            solara.Text(
                "↑/↓ to jump between models",
                style="color:#94a3b8; font-size:0.75rem; margin-bottom:4px;",
            )
            with solara.Column(
                style="flex:1 1 auto; min-height:0; overflow-y:auto;", gap="2px"
            ):
                for label in visible_labels:
                    active = label == selected_file.value
                    solara.Button(
                        label,
                        text=not active,
                        outlined=active,
                        color="primary" if active else None,
                        on_click=lambda label=label: select_file(label),
                        style="justify-content:flex-start; width:100%; text-transform:none; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;",
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
        with rv.Card(
            elevation=1,
            outlined=True,
            style_="height:calc(100vh - 84px); display:flex; flex-direction:column; overflow:hidden;",
        ):
            with rv.Toolbar(
                dense=True,
                flat=True,
                style_="background:#f8fafc; border-bottom:1px solid #e2e8f0; height:48px; min-height:48px; flex:0 0 48px; padding-left:12px; padding-right:12px;",
            ):
                rv.ToolbarTitle(
                    children=["Source"],
                    class_="subtitle-2 font-weight-bold grey--text text--darken-3",
                )
                rv.Spacer()
                rv.Chip(
                    small=True,
                    outlined=True,
                    children=[f"{n_lines} lines"],
                    class_="mr-1",
                )
                rv.Btn(
                    icon=True,
                    small=True,
                    on_click=lambda: source_text.set(
                        Path(selected_file.value).read_text()
                    ),
                    children=[rv.Icon(small=True, children=["mdi-refresh"])],
                    title="Reset source to original file content",
                )
            with rv.CardText(
                class_="pa-0",
                style_="flex:1 1 auto; min-height:0; overflow-y:auto; overflow-x:auto;",
            ):
                with solara.Row(
                    gap="0px",
                    style=(
                        f"font-family:monospace; font-size:{_FONT_SIZE_PX}px; "
                        f"line-height:{_LINE_HEIGHT_PX}px; min-height:100%;"
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
                        f"min-height:100% !important;"
                        "}"
                    ),
                )

    @solara.component
    def PreviewPanel(
        path: Path,
        variant: ImportVariant,
        rendered_result: tuple[bool, str, str, int | None],
    ):
        ok, kind, content, _error_line = rendered_result
        available = list(available_backends(path))

        with rv.Card(
            elevation=1,
            outlined=True,
            style_="height:calc(100vh - 84px); display:flex; flex-direction:column; overflow:hidden;",
        ):
            with rv.Toolbar(
                dense=True,
                flat=True,
                style_="background:#f8fafc; border-bottom:1px solid #e2e8f0; height:48px; min-height:48px; flex:0 0 48px; padding-left:8px; padding-right:8px;",
            ):
                with solara.Row(
                    style="align-items:center; gap:6px; width:100%; overflow-x:auto; flex-wrap:nowrap;",
                ):
                    solara.Button(
                        "Source",
                        icon_name="mdi-code-tags",
                        dense=True,
                        outlined=not show_source.value,
                        color="primary" if show_source.value else None,
                        on_click=lambda: show_source.set(not show_source.value),
                        style="text-transform:none;",
                    )
                    rv.Divider(vertical=True, class_="mx-1 my-1")
                    with solara.ToggleButtonsSingle(value=content_type, dense=True):
                        solara.Button(
                            "Representation",
                            value="representation",
                            text=True,
                            style="text-transform:none;",
                        )
                        solara.Button(
                            "Report",
                            value="report",
                            text=True,
                            style="text-transform:none;",
                        )
                    rv.Divider(vertical=True, class_="mx-1 my-1")
                    with solara.ToggleButtonsSingle(value=selected_format, dense=True):
                        solara.Button(
                            "HTML",
                            value="html",
                            text=True,
                            style="text-transform:none;",
                        )
                        solara.Button(
                            "Markdown",
                            value="markdown",
                            text=True,
                            style="text-transform:none;",
                        )
                        solara.Button(
                            "Text",
                            value="text",
                            text=True,
                            style="text-transform:none;",
                        )
                    rv.Divider(vertical=True, class_="mx-1 my-1")
                    if len(available) > 1:
                        with solara.ToggleButtonsSingle(
                            value=selected_backend, dense=True
                        ):
                            for b in available:
                                solara.Button(
                                    b, value=b, text=True, style="text-transform:none;"
                                )
                    else:
                        rv.Chip(
                            small=True,
                            outlined=True,
                            children=[available[0]],
                            class_="my-auto",
                        )
                    rv.Divider(vertical=True, class_="mx-1 my-1")
                    solara.Button(
                        "strict",
                        dense=True,
                        outlined=not strict_mode.value,
                        color="primary" if strict_mode.value else None,
                        on_click=lambda: strict_mode.set(not strict_mode.value),
                        style="text-transform:none;",
                    )

            with rv.CardText(
                style_="flex:1 1 auto; min-height:0; overflow-y:auto; overflow-x:auto; padding:16px;",
            ):
                if not ok:
                    solara.Error(
                        label=f"Rendering failed ({variant.backend}, strict={variant.strict})"
                    )
                if kind == "myst":
                    MystHtml(content)
                else:
                    with solara.Div(style="overflow-x:auto; max-width:100%;"):
                        solara.HTML(tag="div", unsafe_innerHTML=content)

    @solara.component
    def Page():
        path = Path(selected_file.value)

        with solara.Head():
            solara.Title(f"Dyno: {path}")
        with solara.AppBar():
            solara.AppBarTitle(str(path))

        solara.Style("""
            .v-navigation-drawer {
                width: 280px !important;
            }
            .solara-content-main > div {
                padding: 6px 12px !important;
            }
            .v-card pre, .v-card code {
                white-space: pre !important;
                overflow-x: auto !important;
                word-break: normal !important;
                word-wrap: normal !important;
            }
            .v-card {
                overflow-x: auto !important;
                max-width: 100% !important;
            }
        """)

        with solara.Sidebar():
            ModelList()

        variant = ImportVariant(selected_backend.value, strict_mode.value)
        ok, kind, content, line = render_variant(
            path, source_text.value, variant, content_type.value, selected_format.value
        )
        error_lines = {line} if (not ok and line is not None) else set()

        with solara.Columns([1, 1] if show_source.value else [1], gutters_dense=True):
            if show_source.value:
                SourcePanel(error_lines)
            PreviewPanel(path, variant, (ok, kind, content, line))

    return Page()
