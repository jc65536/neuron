from typing import TypeVar
import time

import numpy as np
from numpy import float32 as R, complexfloating as C
from numpy.typing import NDArray

from neuronxcc import nki
import neuronxcc.nki.language as nl

from .test import MODE


T = TypeVar("T", bound=np.generic)


@nki.jit(mode=MODE)
def matmul_C(AT: NDArray[T], B: NDArray[T]) -> NDArray[T]:
    pass
