from typing import Any

import numpy as np


def serialize_params(obj: Any) -> Any:
    """Serialize the given object into a JSON-serializable format."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize_params(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_params(value) for key, value in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj):
        return obj.__name__
    return str(obj)
