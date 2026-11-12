"""Aliases de tipos compartilhados pelo pacote."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AnyArray = NDArray[Any]
