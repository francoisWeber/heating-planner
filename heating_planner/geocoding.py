from functools import partial
from geopy.geocoders import Nominatim
from geopy.location import Location
import numpy as np

CALIBRATION_DATA = [
    {
        "loc": "Bestrée",
        "geo_x": 48.0363826,
        "geo_y": -4.716446,
        "map_x": 390,
        "map_y": 22,
    },
    {
        "loc": "Coustouges",
        "geo_x": 42.3675578,
        "geo_y": 2.6502443,
        "map_x": 1035,
        "map_y": 609,
    },
    {
        "loc": "Menton",
        "geo_x": 43.7753495,
        "geo_y": 7.5029213,
        "map_x": 882,
        "map_y": 995,
    },
    {
        "loc": "Ferney-Voltaire",
        "geo_x": 46.2555744,
        "geo_y": 6.1076131,
        "map_x": 588,
        "map_y": 889,
    },
    {
        "loc": "Le verdon-sur-mer",
        "geo_x": 45.547543,
        "geo_y": -1.0622993,
        "map_x": 681,
        "map_y": 310,
    },
    {
        "loc": "Porto-vecchio",
        "geo_x": 41.5911382,
        "geo_y": 9.2794469,
        "map_x": 1119,
        "map_y": 1141,
    },
    {
        "loc": "Bray-dunes",
        "geo_x": 51.071002,
        "geo_y": 2.5245134,
        "map_x": 15,
        "map_y": 601,
    },
]

EARTH_RADIUS = 6378
DEFAULT_EXTREM_GEOLOC = {
    "east": (42.6993979, 9.4509187, 0.0),
    "south": (41.5911382, 9.2794469, 0.0),
    "west": (48.3605295, -4.7709059, 0.0),
    "north": (51.071002, 2.5245134, 0.0),
}
DEFAULT_EXTREM_MAP = {
    "west": 13 + 3,
    "east": 1164 - 9,
    "north": 15,
    "south": 1150,
}

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
            "south": Loc("porto-vecchio").coords,
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
    return get_xy_map_coords_from_geo_coords(coords, xtrems_map, xtrems_geo)


def get_xy_map_coords_from_geo_coords(
    coords: tuple,
    xtrems_map=DEFAULT_EXTREM_MAP,
    xtrems_geo=DEFAULT_EXTREM_GEOLOC,
):
    # get relative coords
    rel_v = to_relative_coord(coords[0], xtrems_geo["south"][0], xtrems_geo["north"][0])
    rel_h = to_relative_coord(coords[1], xtrems_geo["west"][1], xtrems_geo["east"][1])
    # take back absolute coords on the map
    x = to_absolute_coord(rel_v, xtrems_map["south"], xtrems_map["north"])
    y = to_absolute_coord(rel_h, xtrems_map["west"], xtrems_map["east"])
    return (x, y)


def from_numpy_to_geo_coords(x, y):
    x = to_relative_coord(x, DEFAULT_EXTREM_MAP["north"], DEFAULT_EXTREM_MAP["south"])
    y = to_relative_coord(y, DEFAULT_EXTREM_MAP["west"], DEFAULT_EXTREM_MAP["east"])
    x = to_absolute_coord(
        x, DEFAULT_EXTREM_GEOLOC["north"][0], DEFAULT_EXTREM_GEOLOC["south"][0]
    )
    y = to_absolute_coord(
        y, DEFAULT_EXTREM_GEOLOC["west"][1], DEFAULT_EXTREM_GEOLOC["east"][1]
    )
    return (x, y, 0)


def coords_geo2xy(x, y, z=0):
    xx = (
        5887.144399089925
        - 114.419181 * x
        # + (-9.377498800377365 + 0.022830 * x - 0.016616 * y)
    )
    yy = (
        398.29186938014215
        + 79.616294 * y
        # + (-1.9804381703042186 - 0.000802 * x + 0.005395 * y)
    )
    return int(xx), int(yy)


def coords_xy2geo(x, y):
    xx = 51.44936058639635 - 0.008736 * x
    yy = -5.002142821293705 + 0.012559 * y
    return xx, yy, 0.0
