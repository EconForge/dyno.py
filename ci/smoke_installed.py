"""Smoke test for an installed ``dyno`` package (issue #35).

Run it with the interpreter of a clean environment where only the built wheel
is installed, from a directory outside the source tree:

    cd "$(mktemp -d)" && /path/to/venv/bin/python /path/to/ci/smoke_installed.py

It checks that the package is not imported from the checkout, that package
data (grammars, examples) is shipped, and that a ``.dyno`` and a ``.mod``
example go through steady state, solution, simulation and ``run()``.
"""

import shutil
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path

import dyno
from dyno import DynoModel, examples_path

package_dir = Path(dyno.__file__).resolve().parent
assert "site-packages" in package_dir.parts, f"dyno imported from {package_dir}"
try:
    pkg_ver = version("dynopy")
except Exception:
    pkg_ver = version("dyno")
print(f"dyno {pkg_ver} imported from {package_dir}")

# Package data.
grammars = sorted(p.name for p in (package_dir / "dynspec" / "grammars").glob("*.lark"))
assert grammars == ["grammar.lark", "modfile_grammar.lark"], grammars
assert (package_dir / "py.typed").is_file()
assert examples_path().is_dir(), f"examples_path() -> {examples_path()}"
assert examples_path().is_relative_to(package_dir), examples_path()

# Load examples from copies, so nothing depends on the package directory.
workdir = Path(tempfile.mkdtemp())
for source in [examples_path("neo.dyno"), examples_path("modfiles", "example1.mod")]:
    filename = shutil.copy(source, workdir)
    model = DynoModel(filename)

    model.steady()
    solution = model.solve()
    simulation = model.simulate(T=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = model.run()

    assert solution is not None and simulation is not None and results is not None
    print(f"ok: {source.name}")

print("smoke test passed")
