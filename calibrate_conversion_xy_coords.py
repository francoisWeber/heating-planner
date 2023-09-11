import pandas as pd
import numpy as np
import json
from heating_planner.geocoding import Loc
from sklearn.linear_model import LinearRegression

with open("/Users/f.weber/tmp-fweber/heating/aux_maps/ref_carte_name2xy.json") as f:
    data = json.load(f)

# remember we need a factor 5 resizing
data_f5 = {
    k: {"xy": np.array(v) / 5.0, "coords": Loc(k).coords} for k, v in data.items()
}

df = pd.DataFrame.from_dict(data_f5, orient="index")

# equivalent of 2Pi in the resized image context
TWO_PI_IN_XY = 5751.1


def expand_coords(coords):
    expansion = [
        *coords[:2],
        np.cos(np.deg2rad(coords[0])),
        np.sin(np.deg2rad(coords[0])),
        np.cos(np.deg2rad(coords[1])),
        np.sin(np.deg2rad(coords[1])),
    ]
    return expansion


def expand_xy(xy):
    expansion = [
        *xy,
        np.arccos(np.pi * (xy[0] - 4) / TWO_PI_IN_XY),
        np.arcsin(np.pi * (xy[0] - 4) / TWO_PI_IN_XY),
        np.arccos(np.pi * (xy[1] - 4) / TWO_PI_IN_XY),
        np.arcsin(np.pi * (xy[1] - 4) / TWO_PI_IN_XY),
    ]
    return expansion


xy_expanded = df.xy.apply(expand_xy)
coords_expanded = df.coords.apply(expand_coords)

# obtain predicitve models for xy2coords
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


def convert_xy_to_coords(xy):
    coords_estim = [
        lm_lat.predict(np.array(expand_xy(xy)).reshape(1, -1))[0, 0],
        lm_long.predict(np.array(expand_xy(xy)).reshape(1, -1))[0, 0],
        0.0,
    ]
    return coords_estim


def convert_coords_to_xy(coords):
    coords_estim = [
        lm_x.predict(np.array(expand_coords(coords[:2])).reshape(1, -1))[0, 0],
        lm_y.predict(np.array(expand_coords(coords[:2])).reshape(1, -1))[0, 0],
    ]
    return coords_estim
