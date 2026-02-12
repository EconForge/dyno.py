from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Callable, Any

TVector = NDArray[Any]
TMatrix = NDArray[Any]

Solver = Literal["ti", "qz"]
IRFType = Literal["level", "log-deviation", "deviation"]

DynamicFunction = Callable[[TVector, TVector, TVector, TVector, TVector, TVector], None]
