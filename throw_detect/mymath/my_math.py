import math
import numpy as np
from mymath.mask import create_mask_cy
from mymath.array_centroid import get_rgb_array_cnt_cy

def get_angle(vec1, vec2):
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.degrees(math.acos(cos))

def get_rgb_array_centroid(rgb_frame):
    return get_rgb_array_cnt_cy(rgb_frame)

def create_mask(rgb_frame, threshold):
    return create_mask_cy(rgb_frame, threshold)