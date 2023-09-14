from typing import List
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from heating_planner import value_comparison
from heating_planner.streamlit.tools.selectors import Selector
from heating_planner.streamlit.tools.utils import (
    maybe_add_to_session_state,
    load_json,
    text_center,
)
from heating_planner.utils import load_np_from_anywhere, minmax_bounding
from textwrap import wrap


COMPARISON_TYPE_OPTIONS = ["map", "point", "optimal ranges"]
WEIGHT_PREFIX = "weight!"
TERM_SELECTOR_KEY = "term_selector_key"
REAL_ESTATE_TOGGLE_KEY = "real_estate_toggle_key"
SEALEVEL_TOGGLE_KEY = "sea_level_toggle_key"
REF_POINT_TOGGLE_KEY = "ref_point_toggle_key"
COMPARISON_TYPE_SELECTOR_KEY = "comparison_kind_key"
SMART_COMPARISON_TOGGLE_KEY = "intelligent_comparison"
MAP_CLICKED_POSITION_KEY = "map_clicked_xy"


## ST stuff and other


def serie2featurename(meta: pd.Series, prefix=None):
    infos = [prefix] if prefix else []
    infos += [meta.variable, meta.season]
    return "_".join(infos)


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
    Selector(
        key=SMART_COMPARISON_TOGGLE_KEY,
        label="Do smart comparison ?",
        is_toggle=True,
        default=True,
    ),
    Selector(
        key=REAL_ESTATE_TOGGLE_KEY,
        refers_to_variable="realEstate",
        label="Include real estate ?",
        is_toggle=True,
    ),
    Selector(
        key=SEALEVEL_TOGGLE_KEY,
        refers_to_variable="seaLevelElevation",
        label="Include sea elevation ?",
        is_toggle=True,
    ),
]


@st.cache_resource(show_spinner="loading maps ...")
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
    df_viable = load_json(viable_path)
    hypercube, hypercube_aux = load_hypercubes(hypercube_path)
    df_concat = pd.concat([df, df_aux])
    init_weights(df_concat)
    return df, df_aux, df_viable, hypercube, hypercube_aux


direction2function = {
    "neutral": value_comparison.comparison_neutral_delta,
    "less is better": value_comparison.comparison_lower_better_delta,
    "more is better": value_comparison.comparison_upper_better_delta,
}


def gather_weights():
    w = {k: v for k, v in st.session_state.items() if WEIGHT_PREFIX in k}
    w = {k: v**2 for k, v in w.items()}
    norm = sum(list(w.values()))
    w = {k: v / norm for k, v in w.items()}
    return w


def get_delta_hypercube(
    hypercube_term: np.ndarray,
    reference: np.ndarray,
    comparators_str: list,
) -> np.ndarray:
    comparators = [direction2function[c] for c in comparators_str]
    term_minus_ref = np.moveaxis(hypercube_term - reference, -1, 0)
    deltas = []
    for delta, comparator in zip(term_minus_ref, comparators):
        delta_compared = comparator(delta)
        deltas.append(delta_compared)
    delta = np.stack(deltas, axis=-1)
    return delta


@st.cache_data
def get_ordered_weights(ordered_keys: list) -> np.ndarray:
    w = np.array([st.session_state[k] for k in ordered_keys], dtype=np.float32)
    w = np.square(w)
    w /= w.sum()
    return w


@st.cache_data
def get_delta_hypercube_cached(hypercube_term, hypercube_ref, comparators_str):
    return get_delta_hypercube(hypercube_term, hypercube_ref, comparators_str)


@st.cache_data
def precompute_helpers_for_climate_score(
    term: str,
    df: pd.DataFrame,
    viable_ranges: pd.DataFrame,
    hypercube: np.ndarray,
    smart_comparison: bool,
):
    ## prepare
    df = df.reset_index(drop=True).drop(columns=[c for c in df.columns if "fpath" in c])
    df = pd.merge(left=df.reset_index(), right=viable_ranges, sort=False)
    df_ref = df[df.term == "ref"]
    # external info
    ordered_weights_keys = df_ref.apply(
        lambda s: serie2featurename(s, WEIGHT_PREFIX), axis=1
    ).tolist()
    weights = get_ordered_weights(ordered_weights_keys)
    if smart_comparison:
        comparators_str = df_ref.optimal_direction.tolist()
    else:
        comparators_str = ["neutral"] * len(df_ref)
    # separate data
    hypercube_ref = hypercube[:, :, df_ref.index.tolist()]
    df_term = df[df.term == term]
    hypercube_term = hypercube[:, :, df_term.index.tolist()]
    return df_ref, hypercube_ref, hypercube_term, comparators_str, weights


def build_climate_score(
    term: str,
    comparison: str,
    df: pd.DataFrame,
    viable_ranges: pd.DataFrame,
    hypercube: np.ndarray,
    smart_comparison: bool,
):
    ## prepare
    (
        df_ref,
        cube_ref,
        cube_term,
        comparators_str,
        weights,
    ) = precompute_helpers_for_climate_score(
        term, df, viable_ranges, hypercube, smart_comparison
    )
    # proceed to comparison
    if comparison == "map":
        delta = get_delta_hypercube_cached(cube_term, cube_ref, comparators_str)
    elif comparison == "point":
        if (yx := st.session_state[MAP_CLICKED_POSITION_KEY]) is not None:
            xy = [yx["y"] // 5, yx["x"] // 5]
            reference = cube_ref[*xy, :]
            delta = get_delta_hypercube(cube_term, reference, comparators_str)
        else:
            st.toast(":warning: No reference point selected !")
            delta = cube_ref
    elif comparison == "optimal ranges":
        optimal_ranges = np.array(df_ref.optimal_range.tolist()).T
        delta1 = get_delta_hypercube_cached(
            cube_term, optimal_ranges[0], comparators_str
        )
        delta2 = get_delta_hypercube_cached(
            cube_term, optimal_ranges[1], comparators_str
        )
        delta = np.max(np.stack([delta1, delta2], axis=-1), -1)
    else:
        st.toast(f"No comparison like {comparison}")
        return np.zeros_like(hypercube[:, :, 0])
    # prepare slice-wise comparison
    return np.average(delta, axis=-1, weights=weights)


def pimp_score_with_auxiliary_data(
    score: np.ndarray, df_aux: pd.DataFrame, hypercube_aux: np.ndarray
):
    # make sure everything is positive
    score -= np.nanmin(score)
    df_aux = df_aux.reset_index(drop=True)
    for selector in [s for s in SELECTORS if s.refers_to_variable is not None]:
        variable = selector.refers_to_variable
        if selector.is_toggle and st.session_state[selector.key] is True:
            df = df_aux[df_aux.variable == variable]
            score_aux = hypercube_aux[:, :, df.index[0]]
            if variable == "seaLevelElevation":
                score += np.where(score_aux < 0, -score, 0)
            elif variable == "realEstate":
                score = score / np.sqrt(score_aux)

    score = minmax_bounding(score, 0, 1)
    return score


def render_weights_setters(df: pd.DataFrame, value=1):
    init_weights(df, value=value, force=True)
    df = df[(df.term == "ref")]
    df["key"] = df.apply(lambda s: serie2featurename(s, prefix=WEIGHT_PREFIX), axis=1)
    df["label"] = df.apply(lambda s: f"{s.variable} during {s.season}", axis=1)

    # gather info about variables anno vs seasons

    df_non_seasonal = df[(df.season == "anno")]
    df_season = df[df.season != "anno"]
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


def create_title(term: str, comparison: str, real_estate: bool) -> str:
    unit = "unit is score/(€/m2)" if real_estate else "no unit"
    title_el = [
        f"Climate-score map (higher the better)",
        f"for {term}-term projection",
        f"comparing to the reference {comparison}",
        f"({unit})",
    ]
    return "\n".join(wrap(" ".join(title_el), width=70))


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
    _, col_img, col_selectors, _ = st.columns([1, 8, 4, 1])
    with col_img:
        score = build_climate_score(
            st.session_state[TERM_SELECTOR_KEY],
            st.session_state[COMPARISON_TYPE_SELECTOR_KEY],
            df,
            df_viable,
            hypercube,
            smart_comparison=st.session_state[SMART_COMPARISON_TOGGLE_KEY],
        )
        score = pimp_score_with_auxiliary_data(score, df_aux, hypercube_aux)
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(score, cmap="jet")
        plt.colorbar()
        plt.grid(which="both", alpha=0.5)
        _ = plt.xticks(ticks=np.arange(0, score.shape[1], step=20))
        _ = plt.yticks(ticks=np.arange(0, score.shape[0], step=20))
        title = create_title(
            st.session_state[TERM_SELECTOR_KEY],
            st.session_state[COMPARISON_TYPE_SELECTOR_KEY],
            st.session_state[REAL_ESTATE_TOGGLE_KEY],
        )
        st.caption(title)
        st.pyplot(fig)
    with col_selectors:
        for selector in SELECTORS:
            selector.get_st_object()

    # weights
    render_weights_setters(df, value=1)
