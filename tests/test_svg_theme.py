from dyno.svg_theme import theme_svg


def test_theme_svg_uses_theme_variables_with_original_fallback():
    svg = theme_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text fill="#64748b">t</text></svg>'
    )
    assert 'style="fill:var(--jp-layout-color0, white)"' in svg
    assert 'fill="#64748b" style="fill:var(--jp-content-font-color2, #64748b)">' in svg
    assert svg.startswith('<svg style="max-width:100%; height:auto;" ')


def test_theme_svg_brightens_series_on_dark_theme():
    svg = theme_svg(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<polyline fill="none" stroke="#2563eb" points="0,0 1,1"/></svg>'
    )
    assert 'points="0,0 1,1" class="dyno-c-2563eb"/>' in svg
    assert (
        '<style>[data-jp-theme-light="false"] .dyno-c-2563eb{stroke:#60a5fa}</style>'
        in svg
    )


def test_theme_svg_merges_several_paints_into_one_style_attribute():
    svg = theme_svg(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<rect fill="white" stroke="#cbd5e1"/></svg>'
    )
    assert svg.count("style=") == 2  # root + rect
    assert (
        'style="fill:var(--jp-layout-color0, white); '
        'stroke:var(--jp-border-color1, #cbd5e1)"' in svg
    )
