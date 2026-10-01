from .compat import fix_solara

from .explorer import model_representation_gui

import sys

if sys.platform == "emscripten":
    fix_solara()
