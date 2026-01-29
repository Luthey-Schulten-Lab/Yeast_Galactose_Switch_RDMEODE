"""Custom JSON encoder for numpy arrays"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)
