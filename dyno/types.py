import numpy as np
from typing import Literal, TypeVar

DType = TypeVar("DType", bound=np.generic)
Vector =  np.ndarray[tuple[int], DType]
Matrix = np.ndarray[tuple[int,int], DType]

Solver = Literal["ti", "qz"]

IRFType = Literal["level", "log-deviation", "deviation"]