from os import path as osp
from pathlib import Path
from typing import Any
from geopy.geocoders import Nominatim
from geopy.location import Location
import pickle as pkl
from enum import StrEnum

from loguru import logger

NOMINATIM_AGENT = "fweber"
GEOCODE_TIMEOUT = 5
CACHE_DIR = Path(osp.expanduser("~/.heating_planner"))
GEOCODE_CACHE_PREFIX = "geocode"
REVERSE_CACHE_PREFIX = "reverse"

SEP = "-"


class GeoCodingOperation(StrEnum):
    GEOCODE = "geocode"
    REVERSE = "reverse"


class CachedNominatim(Nominatim):
    def __init__(self, **kwargs):
        if "user_agent" not in kwargs:
            kwargs["user_agent"] = NOMINATIM_AGENT
        super().__init__(**kwargs)
        self.cache = {operation: {} for operation in GeoCodingOperation}
        self._ensure_cache()
        self._load_cache()

    def _ensure_cache(self):
        if not osp.exists(CACHE_DIR):
            CACHE_DIR.mkdir()

    def _cache_entry(self, query: Any, obtained_location: Location, operation: GeoCodingOperation):
        # RAM cache
        self.cache[operation][query] = obtained_location
        # disk cache
        to_cache = (operation, query, obtained_location)
        cache_fname = self._to_cache_fname(operation, query)
        with open(cache_fname, "wb") as f:
            pkl.dump(to_cache, f)

    def _load_cache(self):
        for file in CACHE_DIR.iterdir():
            if file.suffix != ".pkl":
                continue
            try:
                with open(file, "rb") as f:
                    operation, query, obtained_location = pkl.load(f)
                self.cache[operation][query] = obtained_location
            except EOFError:
                logger.warning(f"{file} corrupted")

    def _to_cache_fname(self, operation: GeoCodingOperation, query: Any) -> str:
        _id = hex(hash((operation, query)))
        return CACHE_DIR / f"cache{_id}.pkl"

    @staticmethod
    def _minimal_kwargs(**kwargs):
        if "language" not in kwargs:
            kwargs["language"] = "fr"
        if "timeout" not in kwargs:
            kwargs["timeout"] = GEOCODE_TIMEOUT
        return kwargs

    def geocode(self, query: Any, **kwargs) -> Location:
        if query in self.cache[GeoCodingOperation.GEOCODE]:
            return self.cache[GeoCodingOperation.GEOCODE][query]
        kwargs = self._minimal_kwargs(**kwargs)
        location = super().geocode(query, **kwargs)
        self._cache_entry(query, location, GeoCodingOperation.GEOCODE)
        return location

    def reverse(self, query: Any, **kwargs) -> Location:
        if query in self.cache[GeoCodingOperation.REVERSE]:
            return self.cache[GeoCodingOperation.REVERSE][query]
        kwargs = self._minimal_kwargs(**kwargs)
        location = super().reverse(query, **kwargs)
        self._cache_entry(query, location, GeoCodingOperation.REVERSE)
        return location


geocoding = CachedNominatim()
