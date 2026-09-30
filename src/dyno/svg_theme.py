"""Make dyno's hand-drawn SVG charts follow the JupyterLab theme.

The chart generators draw with fixed light-theme colors. `theme_svg` keeps
those as fallbacks and layers theme-aware colors on top:

- neutral colors (text, frames, background) read JupyterLab's CSS theme
  variables, falling back to the original color when they're undefined;
- series colors switch to brighter shades when JupyterLab uses a dark theme.

Both only take effect when the SVG is inlined in the page: inside an `<img>`
the page's CSS doesn't reach it, and the chart keeps its light colors.
"""

from __future__ import annotations

import re

# Neutral light-theme color -> JupyterLab theme variable replacing it.
_NEUTRALS: dict[str, str] = {
    "white": "--jp-layout-color0",
    "#0f172a": "--jp-content-font-color1",
    "#334155": "--jp-content-font-color1",
    "#64748b": "--jp-content-font-color2",
    "#cbd5e1": "--jp-border-color1",
}

# Series color -> brighter shade for dark themes.
_DARK_SERIES: dict[str, str] = {
    "#0f766e": "#2dd4bf",
    "#dc2626": "#f87171",
    "#2563eb": "#60a5fa",
    "#ca8a04": "#facc15",
    "#7c3aed": "#a78bfa",
    "#ea580c": "#fb923c",
    "#059669": "#34d399",
    "#d97706": "#fbbf24",
    "#0891b2": "#22d3ee",
    "#db2777": "#f472b6",
    "#4f46e5": "#818cf8",
}

_TAG_RE = re.compile(r"<[a-z]+\b[^>]*>")
_PAINT_RE = re.compile(r'\b(fill|stroke)="([^"]+)"')


def _series_class(color: str) -> str:
    return "dyno-c-" + color.lstrip("#")


def theme_svg(svg: str) -> str:
    """Return `svg` with theme-aware colors (see module docstring)."""
    used_series: set[str] = set()

    def repaint(tag_match: re.Match[str]) -> str:
        tag = tag_match.group(0)
        styles: list[str] = []
        classes: list[str] = []
        for prop, color in _PAINT_RE.findall(tag):
            key = color.lower()
            if key in _NEUTRALS:
                # An inline style beats the presentation attribute, which
                # stays as the fallback for renderers without CSS variables.
                styles.append(f"{prop}:var({_NEUTRALS[key]}, {color})")
            elif key in _DARK_SERIES:
                used_series.add(key)
                classes.append(_series_class(key))
        extra = ""
        if styles:
            extra += f' style="{"; ".join(styles)}"'
        if classes:
            extra += f' class="{" ".join(classes)}"'
        if not extra:
            return tag
        end = -2 if tag.endswith("/>") else -1
        return tag[:end] + extra + tag[end:]

    svg = _TAG_RE.sub(repaint, svg)

    rules = "".join(
        f'[data-jp-theme-light="false"] .{_series_class(c)}'
        f"{{stroke:{_DARK_SERIES[c]}}}"
        for c in sorted(used_series)
    )
    root_end = svg.index(">") + 1
    root = svg[:root_end].replace(
        "<svg ", '<svg style="max-width:100%; height:auto;" ', 1
    )
    style = f"<style>{rules}</style>" if rules else ""
    return root + style + svg[root_end:]
