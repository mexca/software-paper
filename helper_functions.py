"""Helper functions.
"""

from typing import Dict, List, Union

import numpy as np


def calc_face_height(bbox: Union[List, float]) -> float:
    """Calculates the height of a face bounding box."""
    if isinstance(bbox, list):
        return bbox[3] - bbox[1]
    return np.nan


def sub_labels(label: Union[float, int, str], mapping: Dict) -> str:
    """Subsitutes a label for its mapped value."""
    if np.isnan(label):
        return np.nan
    return mapping[str(int(label))].lower()
