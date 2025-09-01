from .compat import fix_solara

from .dynare import dyno_gui

import sys

if sys.platform == "emscripten":
    fix_solara()
