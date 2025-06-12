import numpy as np
from typing import Literal, TypeVar, Callable

DType = TypeVar("DType", bound=np.generic)
Vector = np.ndarray[tuple[int], DType]
Matrix = np.ndarray[tuple[int, int], DType]

Solver = Literal["ti", "qz"]
SymbolType = Literal["endogenous", "exogenous", "parameters"]
IRFType = Literal["level", "log-deviation", "deviation"]

DynamicFunction = Callable[
    [Vector, Vector, Vector, Vector, Vector, Vector],
    Vector | tuple[Vector, Matrix, Matrix, Matrix, Matrix],
]
