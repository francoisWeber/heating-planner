import numpy as np
from geopy.geocoders import Nominatim
from geopy.location import Location

NOMINATIM_AGENT = "fweber"
REVERSE_GEOCODE_TIMEOUT = 5

DEFAULT_GRID_SIZE = 200


class _GeoTool:
    def __init__(self, nominatim: Nominatim | None = None):
        self.geolocator = nominatim if nominatim else Nominatim(user_agent=NOMINATIM_AGENT)
        self.cache_geocode = {}
        self.cache_reverse = {}

    def geocode(self, name: str) -> Location:
        if name in self.cache_geocode:
            return self.cache_geocode[name]
        location = self.geolocator.geocode(name, language="fr")
        self.cache_geocode[name] = location
        return location

    def reverse_geocode(self, coords: tuple) -> Location:
        if coords in self.cache_reverse:
            return self.cache_reverse[coords]
        location = self.geolocator.reverse(coords, language="fr", timeout=REVERSE_GEOCODE_TIMEOUT)
        self.cache_reverse[coords] = location
        return location


geo_tool = _GeoTool()
