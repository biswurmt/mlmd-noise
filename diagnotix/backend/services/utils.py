"""utils.py
===========
Shared utilities for the Diagnotix backend services.
"""

import math
from typing import Any


def _clean(v: Any) -> Any:
    """Recursively convert non-JSON-safe values to safe equivalents."""
    if v is None:
        return None

    if isinstance(v, float) and math.isnan(v):
        return None

    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]

    try:
        import numpy as np  # noqa: F401
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            val = float(v)
            return None if math.isnan(val) else val
        if isinstance(v, np.ndarray):
            return [_clean(x) for x in v.tolist()]
        if isinstance(v, np.bool_):
            return bool(v)
    except ImportError:
        pass

    return v
