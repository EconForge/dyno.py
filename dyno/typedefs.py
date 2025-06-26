import numpy as np
from typing import Literal, TypeVar, Callable

DType = TypeVar("DType", bound=np.generic)
TVector = np.ndarray[tuple[int], DType]
TMatrix = np.ndarray[tuple[int, int], DType]

Solver = Literal["ti", "qz"]
SymbolType = Literal["endogenous", "exogenous", "parameters"]
IRFType = Literal["level", "log-deviation", "deviation"]

DynamicFunction = Callable[[TVector, TVector, TVector, TVector, TVector, TVector], None]
