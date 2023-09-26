from functools import partial
from geopy.geocoders import Nominatim
from geopy.location import Location
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

CALIBRATION_DATA = {
    "Paris": [589, 293],
    "Belfort": [950, 439],
    "Rouen": [487, 219],
    "Rennes": [267, 379],
    "Le Mans": [414, 395],
    "Troyes": [725, 359],
    "Poitiers": [427, 561],
    "Limoges": [501, 649],
    "Villeurbanne": [790, 657],
    "Grenoble": [861, 725],
    "Bordeaux": [353, 761],
    "Toulouse": [517, 901],
    "Montpellier": [710, 901],
    "Avignon": [785, 864],
    "Perpignan": [632, 999],
}
TWO_PI_IN_XY = 5751.1  # based on a calibration map


geolocator = Nominatim(user_agent="fweber")
geocode = partial(geolocator.geocode, language="fr")
reverse_geocode = partial(geolocator.reverse, language="fr")


def clean_geocoded_address(address: str, rm_zip=True):
    loc_el = []
    for el in address.split(", "):
        # remove zip
        if rm_zip:
            try:
                int(el)
            except ValueError:
                loc_el.append(el)
    return loc_el


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
            self.name = self.location.address
        elif coords is not None:
            self.location: Location = reverse_geocode(coords)
            self.name = self.location.address


class CoordsConverter:
    def __init__(self):
        self.lm_x = None
        self.lm_y = None
        self.lm_lat = None
        self.lm_long = None
        self.TWO_PI_IN_XY = TWO_PI_IN_XY
        self.factor = 5.0

    def calibrate(self, city2coords: dict, factor=5.0):
        """
        Calibrate the converter with examples of named cities with their (x, y)
        coords on a map
        """
        # prepare dataframe
        data_factor = {
            k: {"xy": np.array(v) / factor, "coords": Loc(k).coords}
            for k, v in city2coords.items()
        }
        df = pd.DataFrame.from_dict(data_factor, orient="index")
        # expand features
        xy_expanded = df.xy.apply(expand_xy)
        coords_expanded = df.coords.apply(expand_coords)
        df["lat"] = df.coords.apply(lambda c: c[0])
        df["long"] = df.coords.apply(lambda c: c[1])
        df["x"] = df.xy.apply(lambda c: c[0])
        df["y"] = df.xy.apply(lambda c: c[1])

        # fit models
        # predict lat from xy
        lm_lat = LinearRegression()
        lm_lat.fit(np.asarray(xy_expanded.tolist()), df[["lat"]])
        # predict long from xy
        lm_long = LinearRegression()
        lm_long.fit(np.asarray(xy_expanded.tolist()), df[["long"]])
        # symmetric
        lm_x = LinearRegression()
        lm_x.fit(np.asarray(coords_expanded.tolist()), df[["x"]])
        lm_y = LinearRegression()
        lm_y.fit(np.asarray(coords_expanded.tolist()), df[["y"]])

        self.lm_x = lm_x
        self.lm_y = lm_y
        self.lm_long = lm_long
        self.lm_lat = lm_lat
        self.factor = factor

    def from_xy(self, xy):
        coords_estim = [
            self.lm_lat.predict(np.array(self.expand_xy(xy)).reshape(1, -1))[0, 0],
            self.lm_long.predict(np.array(self.expand_xy(xy)).reshape(1, -1))[0, 0],
            0.0,
        ]
        return coords_estim

    def from_geo(self, coords):
        coords_estim = [
            self.lm_x.predict(np.array(expand_coords(coords[:2])).reshape(1, -1))[0, 0],
            self.lm_y.predict(np.array(expand_coords(coords[:2])).reshape(1, -1))[0, 0],
        ]
        return coords_estim

    def dump(self, path):
        with open(path, "wb") as f:
            joblib.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return joblib.load(f)


def expand_coords(coords):
    expansion = [
        coords[0],
        coords[1],
        np.cos(np.deg2rad(coords[0])),
        np.sin(np.deg2rad(coords[0])),
        np.cos(np.deg2rad(coords[1])),
        np.sin(np.deg2rad(coords[1])),
    ]
    return expansion


def expand_xy(xy, two_pi_in_map=TWO_PI_IN_XY):
    expansion = [
        *xy,
        np.arccos(np.pi * (xy[0] - 4) / two_pi_in_map),
        np.arcsin(np.pi * (xy[0] - 4) / two_pi_in_map),
        np.arccos(np.pi * (xy[1] - 4) / two_pi_in_map),
        np.arcsin(np.pi * (xy[1] - 4) / two_pi_in_map),
    ]
    return expansion


def convert_geo_to_xy(coords):
    coords = np.array(expand_coords(coords))
    intercept_x = -3957.51741644
    intercept_y = -3703.95874956
    coefs_x = np.array(
        [
            [
                9.33911301e01,
                -9.72425268e-01,
                3.78752172e03,
                -3.78626630e03,
                -1.65949274e02,
                9.67554593e02,
            ]
        ]
    )
    coefs_y = np.array(
        [
            [
                48.52856456,
                28.05531954,
                3964.33317017,
                -1804.75821396,
                141.90046119,
                -1599.31416722,
            ]
        ]
    )
    x = int(intercept_x + np.sum(coords * coefs_x))
    y = int(intercept_y + np.sum(coords * coefs_y))
    return y, x


def convert_xy_to_geo(xy):
    xy = expand_xy(xy)
    intercept_lat = -1473.70409818
    intercept_long = -453.8265402
    coefs_lat = np.array(
        [
            [
                7.04078129e-02,
                9.46665670e-01,
                6.41179296e01,
                -6.41179296e01,
                9.03983156e02,
                -9.03983156e02,
            ]
        ]
    )
    coefs_long = np.array(
        [
            [
                2.87792139e-01,
                8.65329824e-02,
                2.05822332e02,
                -2.05822332e02,
                7.90947093e01,
                -7.90947093e01,
            ]
        ]
    )
    lat = intercept_lat + np.sum(xy * coefs_lat)
    long = intercept_long + np.sum(xy * coefs_long)
    return (lat, long, 0.0)
