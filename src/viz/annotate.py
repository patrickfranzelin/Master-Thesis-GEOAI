import cv2, numpy as np
from typing import List, Tuple

def draw_points(rgb: np.ndarray, inside: List[Tuple[int,int]], outside: List[Tuple[int,int]]) -> np.ndarray:
    img = rgb.copy()
    for (x,y) in inside:  cv2.circle(img, (x,y), 6, (0,255,0), -1)
    for (x,y) in outside: cv2.circle(img, (x,y), 6, (0,0,255), -1)
    return img
