from typing import Tuple, List

def calculate_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate the center coordinate of a bounding box.
    
    Args:
        bbox: A list or tuple of [x1, y1, x2, y2]
        
    Returns:
        A tuple of (cx, cy)
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 coordinates [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
