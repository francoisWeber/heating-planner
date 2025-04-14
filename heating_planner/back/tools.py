import numpy as np

def minmax_scale(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())