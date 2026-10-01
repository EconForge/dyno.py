"""Guard against regressions in ``import dyno`` time (issue #36).

Heavy dependencies must be imported lazily, inside the functions that need
them. The check runs in a fresh interpreter so other tests don't pollute
``sys.modules``.
"""

import subprocess
import sys

import pytest

HEAVY_MODULES = [
    "scipy.stats",
    "scipy.linalg",
    "scipy.optimize",
    "pandas",
    "altair",
    "plotext",
    "rich",
    "markdown_it",
    "sympy",
]


@pytest.mark.parametrize("module", HEAVY_MODULES)
def test_import_dyno_does_not_load_heavy_module(module):
    code = f"import sys, dyno; sys.exit({module!r} in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code])
    assert result.returncode == 0, f"`import dyno` eagerly imports {module}"


def test_grammar_parser_is_built_lazily():
    code = (
        "import sys, dyno\n"
        "from dyno.dynspec.grammar import get_parser\n"
        "sys.exit(get_parser.cache_info().currsize)\n"
    )
    result = subprocess.run([sys.executable, "-c", code])
    assert result.returncode == 0, "`import dyno` builds the Lark grammar parser"
