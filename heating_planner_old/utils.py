import hashlib
from PIL import Image
import numpy as np
import requests
import io
from os import path as osp


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


def load_np_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


def load_np_from_anywhere(uri: str) -> np.ndarray:
    if osp.isfile(uri):
        return np.load(uri)
    elif uri.startswith("http"):
        return load_np_from_url(uri)
    else:
        raise NotImplementedError(f"URI not handled: {uri}")


def load_pil_from_anywhere(uri: str) -> Image:
    if osp.isfile(uri):
        return Image.open(uri)
    elif uri.startswith("http"):
        response = requests.get(uri)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        raise NotImplementedError(f"URI not handled: {uri}")


def minmax_bounding(arr: np.ndarray, new_min=0.0, new_max=1.0) -> np.ndarray:
    arr = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    arr *= new_max - new_min
    arr += new_min
    return arr
