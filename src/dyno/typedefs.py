from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Callable, Any

TVector = NDArray[np.floating[Any]]  # 1-D intent
TMatrix = NDArray[np.floating[Any]]  # 2-D intent

Solver = Literal["ti", "qz"]
IRFType = Literal["level", "log-deviation", "deviation"]

DynamicFunction = Callable[[TVector, TVector, TVector, TVector, TVector, TVector], None]

from typing import TypedDict


class ModelContext(TypedDict, total=False):
    constants: dict[str, float]
    variables: dict[str, dict]
    steady_states: dict[str, float]
    processes: dict[tuple[str, ...], Any]
    values: dict[str, dict[int, float]]
    metadata: dict[str, Any]
