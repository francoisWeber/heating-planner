from typing import List
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from heating_planner import value_comparison
from heating_planner.streamlit import io
from heating_planner import utils
from heating_planner.utils import load_np_from_anywhere
from enum import Enum

COMPARISON_TYPE_OPTIONS = ["map", "point"]
WEIGHT_PREFIX = "weight!"
REF_POINT_LOCATION_KEY = "map_clicked_xy"
TERM_SELECTOR_KEY = "term_selector_key"
REAL_ESTATE_TOGGLE_KEY = "real_estate_toggle_key"
REF_POINT_TOGGLE_KEY = "ref_point_toggle_key"
COMPARISON_TYPE_SELECTOR_KEY = "comparison_kind_key"
INTELLIGENT_COMPARISON_TOGGLE_KEY = "intelligent_comparison"
ComparisonType = Enum("Compare wrt", COMPARISON_TYPE_OPTIONS)


## Comparators
def comparison_neutral_delta(value_minus_reference, normalize=True):
    delta_compared = -np.abs(value_minus_reference)
    if normalize:
        delta_compared /= np.nanstd(delta_compared)
    return delta_compared


def comparison_upper_better_delta(value_minus_reference, decay=0.1, normalize=True):
    delta_compared = np.where(
        value_minus_reference > 0, value_minus_reference * decay, value_minus_reference
    )
    if normalize:
        delta_compared /= np.nanstd(delta_compared)
    return delta_compared


def comparison_lower_better_delta(value_minus_reference, decay=0.1, normalize=True):
    return comparison_upper_better_delta(
        -value_minus_reference, decay=decay, normalize=normalize
    )


## ST stuff and other


def maybe_add_to_session_state(key, value, force=False):
    if key not in st.session_state or force:
        st.session_state[key] = value


def metadata2featurename(variable, season, prefix=None):
    infos = [prefix] if prefix else []
    infos += [variable, season]
    return "_".join(infos)


def serie2featurename(meta: pd.Series, prefix=None):
    return metadata2featurename(meta.variable, meta.season, prefix=prefix)


class Selector:
    def __init__(
        self,
        key,
        label,
        values: List | bool | None = None,
        default=None,
        is_toggle=False,
    ) -> None:
        self.label = label
        if values is None and not is_toggle:
            raise ValueError("Values must be set")
        self.values = values
        self.default = default if default else 0
        self.key = key
        self.is_toggle = is_toggle
        maybe_add_to_session_state(self.key, self.default)

    def get_st_object(self):
        if self.is_toggle:
            st.toggle(label=self.label, value=self.default, key=self.key)
        else:
            try:
                st.radio(
                    label=self.label,
                    options=self.values,
                    index=0,
                    key=self.key,
                )
            except:
                st.text(self.label)


SELECTORS = [
    Selector(
        key=TERM_SELECTOR_KEY,
        label="Term to consider ?",
        values=["near", "medium"],
        default="near",
    ),
    Selector(
        key=COMPARISON_TYPE_SELECTOR_KEY,
        label="Method of comparison ?",
        values=COMPARISON_TYPE_OPTIONS,
        default="map",
    ),
    Selector(key=REAL_ESTATE_TOGGLE_KEY, label="Include real estate ?", is_toggle=True),
    Selector(
        key=INTELLIGENT_COMPARISON_TOGGLE_KEY,
        label="Do intelligent comparison ?",
        is_toggle=True,
        default=True,
    ),
]


@st.cache_data
def load_json(df_path, **kwargs):
    return pd.read_json(df_path, **kwargs)


@st.cache_data
def load_csv(df_path, **kwargs):
    return pd.read_csv(df_path, **kwargs)


@st.cache_data
def load_numpy(npx_path):
    return load_np_from_anywhere(npx_path)


@st.cache_data
def load_hypercubes(hypercubes_path):
    hypercubes = load_np_from_anywhere(hypercubes_path)
    return hypercubes["map"], hypercubes["aux"]


def init_weights(df: pd.DataFrame, value=1, force=False):
    features_names = (
        df.apply(lambda s: serie2featurename(s, prefix=WEIGHT_PREFIX), axis=1)
        .unique()
        .tolist()
    )
    for feature_name in features_names:
        maybe_add_to_session_state(feature_name, value)


def init(metadata_path, metadata_aux_path, viable_path, hypercube_path):
    df = load_json(metadata_path)
    df_aux = load_json(metadata_aux_path)
    df_viable = load_csv(viable_path, index_col=0)
    hypercube, hypercube_aux = load_hypercubes(hypercube_path)
    df_concat = pd.concat([df, df_aux])
    init_weights(df_concat)
    return df, df_aux, df_viable, hypercube, hypercube_aux


direction2function = {
    "neutral": comparison_neutral_delta,
    "less is better": comparison_lower_better_delta,
    "more is better": comparison_upper_better_delta,
}


def gather_weights():
    w = {k: v for k, v in st.session_state.items() if WEIGHT_PREFIX in k}
    w = {k: v**2 for k, v in w.items()}
    norm = sum(list(w.values()))
    w = {k: v / norm for k, v in w.items()}
    return w


def prepare_img(
    term: str,
    comparison: str,
    df: pd.DataFrame,
    viable_ranges: pd.DataFrame,
    hypercube: np.ndarray,
):
    ## prepare
    df = df.reset_index(drop=True).drop(columns=[c for c in df.columns if "fpath" in c])
    df = pd.merge(left=df.reset_index(), right=viable_ranges, sort=False)
    # separate data
    df_ref = df[df.term == "ref"]
    hypercube_ref = hypercube[:, :, df_ref.index.tolist()]
    df_term = df[df.term == term]
    hypercube_term = hypercube[:, :, df_term.index.tolist()]
    if comparison == "map":
        reference = hypercube_ref
    elif comparison == "point" and (yx := st.session_state.map_clicked_xy) is not None:
        xy = [yx["y"] // 5, yx["x"] // 5]
        reference = hypercube_ref[*xy, :]
    else:
        st.toast(f"No comparison like {comparison}")
        return np.zeros_like(hypercube[:, :, 0])
    # prepare slice-wise comparison
    term_minus_ref = np.moveaxis(hypercube_term - reference, -1, 0)
    deltas = []
    weights_dict = gather_weights()
    weights = []
    ordered_features = []
    for delta, (i, metadata) in zip(term_minus_ref, df_ref.iterrows()):
        ordered_features.append(serie2featurename(metadata))
        weight_key = serie2featurename(metadata, prefix=WEIGHT_PREFIX)
        weights.append(weights_dict[weight_key])
        comparator = direction2function[metadata.optimal_direction]
        delta_compared = comparator(delta)
        deltas.append(delta_compared)
    delta = np.stack(deltas, axis=-1)
    return np.average(delta, axis=-1, weights=weights)


def text_center(txt):
    return st.markdown(
        f'<div style="text-align: center;">{txt}</div>', unsafe_allow_html=True
    )


def render_weights_setters(df: pd.DataFrame, df_aux: pd.DataFrame, value=1):
    _df = pd.concat([df.assign(source="primary"), df_aux.assign(source="aux")])
    init_weights(_df, value=value, force=True)
    _df = _df[(_df.term == "ref") | (_df.source == "aux")]
    _df["key"] = _df.apply(lambda s: serie2featurename(s, prefix=WEIGHT_PREFIX), axis=1)
    _df["label"] = _df.apply(lambda s: f"{s.variable} during {s.season}", axis=1)

    # gather info about variables anno vs seasons

    df_non_seasonal = _df[(_df.season == "anno")]
    df_season = _df[_df.season != "anno"]
    four_seasons = ["winter", "spring", "summer", "autumn"]
    season_variables = [None] + sorted(df_season.variable.unique().tolist())
    # annual variables
    anno_cols = st.columns([1, len(season_variables)])
    with anno_cols[0]:
        text_center("annual")
    with anno_cols[1]:
        columns_anno_feats = st.columns(len(df_non_seasonal))
        for i, col in enumerate(columns_anno_feats):
            metadata = df_non_seasonal.iloc[i - 1]
            with col:
                st.slider(
                    metadata.label,
                    min_value=0,
                    max_value=5,
                    step=1,
                    value=value,
                    key=metadata.key,
                )
    # for col, (_, metadata) in zip(columns_anno_feats, df_anno.iterrows()):
    for i, season in enumerate(four_seasons):
        with st.container():
            columns = st.columns(len(season_variables))
            for j, (col, variable) in enumerate(zip(columns, season_variables)):
                with col:
                    if j == 0:
                        text_center(season)
                    else:
                        ddf = df_season[
                            (df_season.variable == variable)
                            & (df_season.season == season)
                        ]
                        metadata = ddf.iloc[0]
                        st.slider(
                            metadata.variable,
                            min_value=0,
                            max_value=5,
                            step=1,
                            value=value,
                            key=metadata.key,
                            label_visibility="visible" if i == 0 else "hidden",
                        )


def render(
    metadata_path,
    metadata_aux_path,
    viable_path,
    mask_path,
    hypercube_path,
):
    df, df_aux, df_viable, hypercube, hypercube_aux = init(
        metadata_path, metadata_aux_path, viable_path, hypercube_path
    )
    col_img, col_selectors = st.columns([5, 2])
    with col_img:
        img = prepare_img(
            st.session_state[TERM_SELECTOR_KEY],
            st.session_state[COMPARISON_TYPE_SELECTOR_KEY],
            df,
            df_viable,
            hypercube,
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.imshow(img, cmap="jet")
        plt.colorbar()
        plt.grid(which="both", alpha=0.5)
        _ = plt.xticks(ticks=np.arange(0, img.shape[1], step=20))
        _ = plt.yticks(ticks=np.arange(0, img.shape[0], step=20))
        st.pyplot(fig)
    with col_selectors:
        for selector in SELECTORS:
            selector.get_st_object()

    # weights
    render_weights_setters(df, df_aux, value=1)


def craft_key(metadata, prefix=None):
    el = [prefix] if prefix is not None else []
    el += [metadata.variable, metadata.season]
    return "_".join(el)


def craft_weight_key(metadata):
    return craft_key(metadata=metadata, prefix="weight")
