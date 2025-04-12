from typing import List
from matplotlib import pyplot as plt
import streamlit as st
from glob import glob
from os import path as osp
import pandas as pd
import numpy as np


def gather(questions, st_questions):
    return {question["id"]: answer for question, answer in zip(questions, st_questions)}


def maybe_load_previous_data(path) -> list:
    try:
        return pd.read_json(path).to_dict(orient="records")
    except:
        return None


def craft_title(term, variable, season, **kwargs):
    return f"map of {variable} during {season}"


def load_metadata(input_path) -> List[dict]:
    df = pd.read_json(input_path)
    metadata = df[df.term == "ref"].to_dict(orient="records")
    return metadata


def prepare_data_to_analyse(input_path, previous_data: List[dict] = None):
    metadata = load_metadata(input_path)
    if previous_data is None:
        return metadata
    # else
    parsed_index = set([m["index"] for m in previous_data])
    metadata = [m for m in metadata if m["index"] not in parsed_index]
    return sorted(metadata, key=lambda m: m["variable"] + "-" + m["season"])


def save(output_path):
    df = pd.DataFrame(data=st.session_state.registered_infos)
    df.to_json(output_path)


def push():
    st.session_state.registered_infos.append(
        {
            "optimal_range": st.session_state.slider_values,
            "optimal_direction": st.session_state.general_choice,
            **st.session_state.current_metadata,
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


def init(input_path, output_path):
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    if "previous_data" not in st.session_state:
        st.session_state.previous_data = maybe_load_previous_data(output_path)

    if "metadata" not in st.session_state:
        st.session_state.metadata = prepare_data_to_analyse(
            input_path, st.session_state.previous_data
        )

    if "registered_infos" not in st.session_state:
        st.session_state.registered_infos = []

    if "current_metadata" not in st.session_state:
        st.session_state.current_metadata = None

    if "max_id" not in st.session_state:
        st.session_state.max_id = len(st.session_state.metadata)

    if "map_min_value" not in st.session_state:
        st.session_state.map_min_value = 0

    if "map_max_value" not in st.session_state:
        st.session_state.map_max_value = 0

    if "scale_ticks" not in st.session_state:
        st.session_state.scale_ticks = ""

    if "current_map" not in st.session_state:
        st.session_state.current_map = 0


# constants
ref_locations = {
    "Strasbourg": {"xy": (304.3843143057161, 1026.3990127117725)},
    "chaise-dieu-du-theil": {"xy": (283.0154188458886, 460.6224280954779)},
}


def register(input_path, output_path):
    init(input_path, output_path)
    bar_context, figure_context = st.columns([2, 1])
    with bar_context:
        st.progress(st.session_state.idx / st.session_state.max_id)
    with figure_context:
        st.text(
            f"{st.session_state.idx + 1}e element en cours sur {st.session_state.max_id}"
        )

    image_context, form_context = st.columns([3, 1])

    with st.form("form", clear_on_submit=True):
        st.session_state.current_metadata = st.session_state.metadata[
            st.session_state.idx
        ]
        st.session_state.current_map = np.load(
            st.session_state.current_metadata["fpath_array"]
        )
        with image_context:
            fig, ax = plt.subplots(figsize=(10, 7))
            bin_h, bin_v, bars = ax.hist(
                st.session_state.current_map.ravel(), log=True, bins=15
            )
            bins_to_highlight = []
            for city, loc in ref_locations.items():
                h, v = loc["xy"]
                h = int(h) // 5
                v = int(v) // 5
                city_val = st.session_state.current_map[h, v]
                id = np.where(bin_v <= city_val)[0][-1]
                bins_to_highlight.append(id)
            bins_to_highlight = set(bins_to_highlight)
            if len(bins_to_highlight) == 2:
                bars[min(bins_to_highlight)].set_facecolor("green")
                bars[max(bins_to_highlight)].set_facecolor("red")
            else:
                bars[max(bins_to_highlight)].set_facecolor("orange")
            plt.legend()
            plt.grid()
            plt.title(craft_title(**st.session_state.current_metadata))

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
                step=0.5,
            )

        submitted = st.form_submit_button("go", on_click=on_click)
    save_btn = st.button("save", on_click=save)
