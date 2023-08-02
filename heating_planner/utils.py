import hashlib

import numpy as np


def sha(*args):
    sha1_creator = hashlib.sha256()
    for item in args:
        sha1_creator.update(str(item).encode())
    return sha1_creator.digest().hex()


def softdict_assignation(
    query, keys: np.ndarray, values: np.ndarray, factor: float = 10.0
) -> float:
    """Soft dict from hashed colors to measured values"""
    v = factor / np.abs(keys - query)
    return np.sum(np.exp(v) / np.exp(v).sum() * values)
