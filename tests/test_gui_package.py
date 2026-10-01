import importlib

import pytest


def test_gui_package_imports_without_solara():
    gui = importlib.import_module("dyno.gui")

    assert callable(gui.model_representation_gui)
    assert not hasattr(gui, "dyno_gui")


@pytest.mark.parametrize("name", ["dyno.gui.dynare", "dyno.gui.components"])
def test_removed_gui_modules_are_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
