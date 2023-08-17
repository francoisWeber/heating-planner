from functools import partial
from geopy.geocoders import Nominatim
from geopy.location import Location
import numpy as np

EARTH_RADIUS = 6378
DEFAULT_EXTREM_GEOLOC = {
    "east": (42.6993979, 9.4509187, 0.0),
    "south": (41.3878259, 9.1606179, 0.0),
    "west": (48.3605295, -4.7709059, 0.0),
    "north": (51.071002, 2.5245134, 0.0),
}
DEFAULT_EXTREM_MAP = {"west": 13, "east": 1164, "north": 15, "south": 1142}

geolocator = Nominatim(user_agent="fweber")
geocode = partial(geolocator.geocode, language="fr")
reverse_geocode = partial(geolocator.reverse, language="fr")


class Loc:
    def __init__(self, name: str = None, coords: tuple = None):
        self.name = name
        self.coords = coords
        self.cartesian = None
        self.location = None
        if name:
            self.location: Location = geocode(name)
            self.coords = (
                self.location.latitude,
                self.location.longitude,
                self.location.altitude,
            )
        elif coords is not None:
            self.location: Location = reverse_geocode(coords)
            self.name = self.location.address


def point_extrem_on_geoloc(use_prcomputed=True):
    if use_prcomputed:
        return DEFAULT_EXTREM_GEOLOC
    else:
        return {
            "east": Loc("Bastia").coords,
            "south": Loc("Bonifacio").coords,
            "west": Loc("Le Conquet").coords,
            "north": Loc("Bray-Dunes").coords,
        }


def point_extrem_on_map(use_prcomputed=True, map: np.ndarray = None):
    if use_prcomputed:
        xtrems_map = DEFAULT_EXTREM_MAP
    elif map:
        map_having_info = np.where(np.isnan(map), 0, 1)
        xtrems_map = {
            "west": np.where(map_having_info.sum(axis=0) > 0)[0][0],
            "east": np.where(map_having_info.sum(axis=0) > 0)[0][-1],
            "north": np.where(map_having_info.sum(axis=1) > 0)[0][0],
            "south": np.where(map_having_info.sum(axis=1) > 0)[0][-1],
        }
    else:
        raise ValueError("must input a map as np array !")
    return xtrems_map


def to_relative_coord(coord, lowerbound, upperbound):
    return (coord - lowerbound) / (upperbound - lowerbound)


def to_absolute_coord(coord, lowerbound, upperbound):
    return coord * (upperbound - lowerbound) + lowerbound


def get_xy_map_coords_of_place(
    place: str,
    xtrems_map=DEFAULT_EXTREM_MAP,
    xtrems_geo=DEFAULT_EXTREM_GEOLOC,
):
    coords = Loc(place).coords
    # get relative coords
    rel_v = to_relative_coord(coords[0], xtrems_geo["south"][0], xtrems_geo["north"][0])
    rel_h = to_relative_coord(coords[1], xtrems_geo["west"][1], xtrems_geo["east"][1])
    # take back absolute coords on the map
    x = to_absolute_coord(rel_v, xtrems_map["south"], xtrems_map["north"])
    y = to_absolute_coord(rel_h, xtrems_map["west"], xtrems_map["east"])
    return (x, y)
