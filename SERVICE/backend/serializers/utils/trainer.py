from slugify import slugify
from typing import Any

import numpy as np


def serialize_params(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [serialize_params(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_params(value) for key, value in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, type):
        return obj.__name__
    elif callable(obj):
        return obj.__name__
    else:
        return str(obj)
