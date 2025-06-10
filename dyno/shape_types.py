from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

Vector = Annotated[npt.NDArray[DType], Literal["N"]]
SquareMatrix = Annotated[npt.NDArray[DType], Literal["N", "N"]]
