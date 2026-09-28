"""Solara entry point: interactive model representation explorer.

Run with:

    pixi run -e solara solara run scripts/model_representation.py
"""

from dyno.gui.explorer import model_representation_gui

Page = model_representation_gui("examples")
