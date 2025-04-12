from geopy.geocoders import Nominatim
from geopy.location import Location
import numpy as np

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


def mercator_projection(latitude, longitude, R=1):
    """
    Convert spherical coordinates (latitude, longitude) to Cartesian coordinates (X, Y)
    using the Mercator projection.

    Parameters:
    latitude (float or np.array): Latitude in degrees.
    longitude (float or np.array): Longitude in degrees.
    R (float): Radius of the sphere. Default is 1.

    Returns:
    tuple: X, Y coordinates.
    """
    # Convert degrees to radians
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    # Calculate Mercator projection
    X = R * lon_rad
    Y = R * np.log(np.tan((np.pi / 4) + (lat_rad / 2)))

    return X, Y


def project_to_grid(latitude: np.ndarray, longitude: np.ndarray, grid_size=DEFAULT_GRID_SIZE):
    """
    Project spherical coordinates onto a grid with integer indices using Mercator projection.

    Parameters:
    latitude (float or np.array): Latitude in degrees.
    longitude (float or np.array): Longitude in degrees.
    grid_size (int): Size of the grid. Default is 100.

    Returns:
    tuple: Grid indices (X_grid, Y_grid).
    """
    # Calculate Mercator projection
    x_proj, y_proj = mercator_projection(latitude, longitude)

    # Normalize coordinates to range [0, 1]
    x_normalized = (x_proj - x_proj.min()) / (x_proj.max() - x_proj.min())
    y_normalized = (y_proj - y_proj.min()) / (y_proj.max() - y_proj.min())

    # Scale to grid size and convert to integer indices
    x_grid = np.floor(x_normalized * (grid_size - 1)).astype(int)
    y_grid = np.floor(y_normalized * (grid_size - 1)).astype(int)

    return x_grid, y_grid
