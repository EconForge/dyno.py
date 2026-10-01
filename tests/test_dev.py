import pytest

from dyno.dyno_model import DynoModel

# Dynare example files that are expected to import through the Lark backend.
# Files needing a steady-state function or an external steady-state file
# (``example3.mod``, ``Gali_2015.mod``) are covered by ``test_errors.py``.
MODFILES = ["example1.mod", "example2.mod", "NK_baseline.mod"]


@pytest.mark.parametrize("name", MODFILES)
def test_modfile_imports_and_renders(name):
    model = DynoModel("examples/modfiles/" + name)

    assert len(model.symbolic.equations) == len(model.symbols["endogenous"]) > 0
    assert model.symbols["parameters"]

    text = repr(model)
    assert model.name in text

    markdown = model._markdown_()
    assert isinstance(markdown, str) and markdown.strip()
