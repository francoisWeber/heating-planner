import pandas as pd
import numpy as np
import seaborn as sns
import pygal
from matplotlib import pyplot as plt
from functools import partial

from heating_planner.geocoding import get_xy_map_coords_of_place
from heating_planner import value_comparison as compare
from heating_planner.colorscale import convert


input_path = "/Users/f.weber/tmp-fweber/heating/metadata.json"
term = "near"
mask_path = "/Users/f.weber/tmp-fweber/heating/france_mask.npy"

df = pd.read_json(input_path)
term_df = df[(df.term == "near")]
mask = np.load(mask_path)


index2deltamaps = {}
for index, metadata in term_df.iterrows():
    map = np.load(metadata.fpath_array)
    ref_metadata = df[
        (df.term == "ref")
        & (df.variable == metadata.variable)
        & (df.season == metadata.season)
    ].iloc[0]
    if ref_metadata.optimal_direction == "neutral":
        delta_map = compare.neutral_delta_normalized(map, *ref_metadata.optimal_range)
    elif ref_metadata.optimal_direction == "less is better":
        delta_map = compare.lower_better_delta_normalized(
            map, *ref_metadata.optimal_range
        )
    elif ref_metadata.optimal_direction == "more is better":
        delta_map = compare.upper_better_delta_normalized(
            map, *ref_metadata.optimal_range
        )
    # mask by security
    delta_map = np.where(mask, np.nan, delta_map)
    index2deltamaps[index] = delta_map

delta_map = np.concatenate(
    [np.expand_dims(m, -1) for m in index2deltamaps.values()], axis=-1
)

mmax = np.nanmax(delta_map)
mmin = np.nanmin(delta_map)

# get every comparison slices
n_rows = len(term_df.variable.unique())
ordered_slice_name = []
n_col = 4
fig, axs = plt.subplots(n_rows, n_col, figsize=(n_col * 3, n_rows * 3))
for row_id, variable in enumerate(sorted(term_df.variable.unique())):
    _df = term_df[term_df.variable == variable]
    _df = _df.assign(
        k=_df.season.apply(
            lambda s: {"autumn": 3, "spring": 1, "summer": 2, "winter": 4, "anno": 0}[s]
        )
    ).sort_values(by="k")
    for j, (index, metadata) in enumerate(_df.iterrows()):
        ordered_slice_name.append(f"{metadata.variable} - {metadata.season}")
        slice = index2deltamaps[index]
        slice[0, :] = mmin
        slice[1, :] = mmax
        axs[row_id, j].imshow(-slice, cmap="jet")
        if j == 0:
            axs[row_id, j].set_ylabel(metadata.variable)
        if row_id == 0:
            axs[row_id, j].set_title(metadata.season)

score_map = np.sum(delta_map, axis=-1)

plt.imshow(np.nanmin(delta_map, axis=-1), cmap="jet")
plt.colorbar()
plt.title("Meilleure adéquation toute variable confondue par km2")

plt.imshow(score_map / 33, cmap="jet")
plt.colorbar()
plt.title("Score par km2, agrégation de 33 métriques")

# quantile map
quantiles = np.arange(1, 1 / 100, step=-1 / 100)
score_quantiles = np.nanquantile(score_map.ravel(), q=quantiles)
score_bestof = np.sum(
    np.concatenate(
        [np.expand_dims((score_map >= q), -1) for q in score_quantiles],
        dtype=np.float32,
        axis=-1,
    ),
    axis=-1,
)
score_bestof = np.where(mask, np.nan, score_bestof)
plt.imshow(score_bestof, cmap="jet")
plt.colorbar()
plt.title("Quantiles par écarts de 1% du score agrégé")


# Compare places
from heating_planner.geocoding import Loc, coords_geo2xy

loc_names = ["St Brieuc", "Nice", "Caen", "Strasbourg"]

locs = [Loc(loc_name) for loc_name in loc_names]

slices = []
w = 2
for loc in locs:
    x, y = coords_geo2xy(*loc.coords)
    _slices = np.nanmean(delta_map[x - w : x + w, y - w : y + w, :], axis=(0, 1))
    slices.append(_slices)

slices = np.array(slices).T

prev_h = 0
for i, (e, name) in enumerate(zip(slices, ordered_slice_name)):
    if np.all(e == 0):
        continue
    plt.bar(x=range(len(e)), height=e, bottom=prev_h, label=name)
    prev_h += e
plt.legend()
plt.xticks(ticks=range(len(loc_names)), labels=loc_names)
plt.grid()
