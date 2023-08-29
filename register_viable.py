from matplotlib import pyplot as plt
import streamlit as st
from glob import glob
from os import path as osp
import pandas as pd
import numpy as np
from heating_planner.geocoding import get_xy_map_coords_of_place
from heating_planner.map_handling import get_np_subgrid_from_xy_coords

st.title("Give your run a label !")


INPUT_PATH = "/Users/f.weber/tmp-fweber/heating/metadata.json"
OUTPUT_PATH = "/Users/f.weber/tmp-fweber/heating/viable_ranges.json"


def gather(questions, st_questions):
    return {question["id"]: answer for question, answer in zip(questions, st_questions)}


def list_files(metadata_path):
    sorting_key = lambda d: d["index"]
    df = pd.read_json(metadata_path)
    # restrict to ref data only
    df = df[df.term == "ref"]
    ls = sorted(df.reset_index().to_dict(orient="records"), key=sorting_key)
    if st.session_state.previous_data is None:
        return ls
    else:
        already_done = {entry["index"] for entry in st.session_state.previous_data}
        st.session_state.idx = len(already_done)
        remaining_index = set([entry["index"] for entry in ls]).difference(already_done)
        return list(already_done) + [l for l in ls if l["index"] in remaining_index]


def maybe_load_previous_data(path) -> list:
    try:
        return pd.read_json(path).to_dict(orient="records")
    except:
        return None


if "idx" not in st.session_state:
    st.session_state.idx = 0

if "previous_data" not in st.session_state:
    st.session_state.previous_data = maybe_load_previous_data(OUTPUT_PATH)

if "files_paths" not in st.session_state:
    st.session_state.maps_data = list_files(INPUT_PATH)

if "current_map_data" not in st.session_state:
    st.session_state.current_map_data = None

if "max_id" not in st.session_state:
    st.session_state.max_id = len(st.session_state.maps_data)

if "data_infos" not in st.session_state:
    st.session_state.data_infos = []

if "map_min_value" not in st.session_state:
    st.session_state.map_min_value = 0

if "map_max_value" not in st.session_state:
    st.session_state.map_max_value = 0

if "scale_ticks" not in st.session_state:
    st.session_state.scale_ticks = ""

if "current_map" not in st.session_state:
    st.session_state.current_map = 0


if "ref_locations" not in st.session_state:
    st.session_state.ref_locations = None


# constants
ref_cities = ["Strasbourg, Chaise-Dieu-du-Theil"]

ref_locations = {
    "Strasbourg": {"xy": (304.3843143057161, 1026.3990127117725)},
    "chaise-dieu-du-theil": {"xy": (283.0154188458886, 460.6224280954779)},
}


def craft_title(term, variable, season, **kwargs):
    return f"{term}-term map of {variable} during {season}"


def save():
    df = pd.DataFrame(data=st.session_state.data_infos)
    df.to_json(OUTPUT_PATH)


def push():
    st.session_state.data_infos.append(
        {
            "optimal_range": st.session_state.slider_values,
            "optimal_direction": st.session_state.general_choice,
            "index": st.session_state.current_map_data["index"],
        }
    )


def get_next_item():
    if st.session_state.idx < st.session_state.max_id - 1:
        st.session_state.idx += 1
    else:
        st.write("This was the last")
        save()
        st.stop()


def on_click():
    push()
    get_next_item()


def extract_values_from_ref_and_mask(map, ref_loc_xy):
    this_variables_values = {}
    mask_xx = []
    mask_yy = []
    for loc, data in ref_loc_xy.items():
        xx, yy = get_np_subgrid_from_xy_coords(*data["xy"], w=3)
        this_variables_values[loc] = map[xx, yy].ravel()
        # keep this mask
        mask_xx += xx
        mask_yy += yy

    map_masked = map.copy()
    map_masked[mask_xx, mask_yy] = np.nan
    return this_variables_values, map_masked


bar_context, figure_context = st.columns([2, 1])
with bar_context:
    st.progress(st.session_state.idx / st.session_state.max_id)
with figure_context:
    st.text(
        f"{st.session_state.idx + 1}e element en cours sur {st.session_state.max_id}"
    )


image_context, form_context = st.columns([3, 1])

with st.form("form", clear_on_submit=True):
    st.session_state.current_map_data = st.session_state.maps_data[st.session_state.idx]
    st.session_state.current_map = np.load(
        st.session_state.current_map_data["fpath_array"]
    )
    with image_context:
        this_variables_values, map_masked = extract_values_from_ref_and_mask(
            st.session_state.current_map, ref_locations
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        alpha = 0.75
        log = True
        bin_h, bins, _ = ax.hist(
            map_masked.ravel(), label="global", density=False, alpha=alpha, log=log
        )
        for loc, values in this_variables_values.items():
            aera_n_points = len(values)
            ax.hist(
                np.repeat(values, int(max(bin_h) / aera_n_points)),
                label=loc,
                density=False,
                alpha=1,
                log=log,
            )
        plt.legend()
        plt.grid()
        plt.title(craft_title(**st.session_state.current_map_data))

        st.pyplot(fig)
        st.session_state.map_min_value = np.nanmin(st.session_state.current_map)
        st.session_state.map_max_value = np.nanmax(st.session_state.current_map)
    with form_context:
        st.radio(
            "sided choice",
            ["less is better", "neutral", "more is better"],
            index=0,
            key="general_choice",
        )
        st.slider(
            "range of acceptable values",
            st.session_state.map_min_value,
            st.session_state.map_max_value,
            (st.session_state.map_min_value, st.session_state.map_max_value),
            key="slider_values",
        )

    submitted = st.form_submit_button("go", on_click=on_click)
save_btn = st.button("save", on_click=save)
