import numpy as np


def neutral_delta_normalized(map: np.ndarray, lower, upper):
    delta_map = np.where(np.isnan(map), np.nan, 0)
    delta_map += np.where(map < lower, map - lower, 0)
    delta_map += np.where(map > upper, -(map - upper), 0)
    delta_map /= np.nanstd(map)
    return delta_map


def lower_better_delta_normalized(map, lower, upper, decay=0.1):
    delta_map = np.where(np.isnan(map), np.nan, 0)
    delta_map += np.where(map < lower, decay * (lower - map), 0)
    delta_map += np.where(map > upper, -(map - upper), 0)
    delta_map /= np.nanstd(map)
    return delta_map


def upper_better_delta_normalized(map, lower, upper, decay=0.1):
    delta_map = np.where(np.isnan(map), np.nan, 0)
    delta_map += np.where(map < lower, map - lower, 0)
    delta_map += np.where(map > upper, decay * (map - upper), 0)
    delta_map /= np.nanstd(map)
    return delta_map
