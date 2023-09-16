import io
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
from heating_planner.geocoding import (
    Loc,
    convert_xy_to_geo,
    convert_geo_to_xy,
    clean_geocoded_address,
)


@st.cache_data
def convert_xy_to_geo_cached(xy):
    return convert_xy_to_geo(xy)


def get_clicked_loc_from_key(st_key, factor=None):
    if (xy := st.session_state[st_key]) is not None:
        xy = np.array([xy["x"], xy["y"]])
        if factor:
            xy = xy // factor
        coords = convert_xy_to_geo_cached(xy)
        loc = Loc(coords=coords)
        address_elements = clean_geocoded_address(loc.name)
        if len(address_elements) <= 5:
            place = address_elements[0:3]
        else:
            place = address_elements[-6:-3]
        return ", ".join(place)


# MISC
RELOADING_WARNING_MSG = ":warning: wrong init but it'll be OK soon ; just hit R"

# Key from outside
MAP_CLICKED_POSITION_KEY = "map_clicked_xy"

# Local keys
WEIGHT_PREFIX = "weight!"
TERM_SELECTOR_KEY = "term_selector_key"
REAL_ESTATE_TOGGLE_KEY = "real_estate_toggle_key"
SEALEVEL_TOGGLE_KEY = "sea_level_toggle_key"
REF_POINT_TOGGLE_KEY = "ref_point_toggle_key"
POSITIVE_SCORES_BONUS_TOGGLE_KEY = "positive_score_bonus_key"
SMART_COMPARISON_TOGGLE_KEY = "intelligent_comparison"
COMPARISON_TYPE_SELECTOR_KEY = "comparison_kind_key"
SCORE_CLICKED_POSITION_KEY = "scoremap_clicked_xy"
COMPARISON_TYPE_OPTIONS = ["map", "point (if set)", "optimal ranges"]
CITIES_TO_ANALYZE_KEY = "cities_to_analyze_key"
INCREASE_CONTRAST_TOGGLE_KEY = "increased_contrast_toggle_key"


## ST stuff and other


@st.cache_resource
def loc_cached(city):
    return Loc(city)


def serie2featurename(meta: pd.Series, prefix=None):
    infos = [prefix] if prefix else []
    infos += [meta.variable, meta.season]
    return "_".join(infos)


def init_selectors(is_demo: bool):
    selectors = [
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
            key=POSITIVE_SCORES_BONUS_TOGGLE_KEY,
            label="apply a bonus for positive scores ?",
            is_toggle=True,
            default=True,
        ),
        Selector(
            key=INCREASE_CONTRAST_TOGGLE_KEY,
            label="Increase contrast ?",
            is_toggle=True,
        ),
    ]
    bonus_selectors = [
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
    if not is_demo:
        selectors += bonus_selectors
    if "selectors" not in st.session_state:
        st.session_state.selectors = selectors
    for selector in selectors + bonus_selectors:
        maybe_add_to_session_state(selector.key, selector.values)


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


def init(
    metadata_path, metadata_aux_path, viable_path, hypercube_path, is_demo: bool = False
):
    df = load_json(metadata_path)
    df_aux = load_json(metadata_aux_path)
    df_viable = load_json(viable_path)
    hypercube, hypercube_aux = load_hypercubes(hypercube_path)
    df_concat = pd.concat([df, df_aux])
    init_weights(df_concat)
    maybe_add_to_session_state(SCORE_CLICKED_POSITION_KEY, None)
    maybe_add_to_session_state(CITIES_TO_ANALYZE_KEY, None)
    init_selectors(is_demo)
    for selector in st.session_state.selectors:
        maybe_add_to_session_state(selector.key, selector.default)
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
    ordered_features = df_ref.apply(lambda s: serie2featurename(s), axis=1).tolist()
    if smart_comparison:
        comparators_str = df_ref.optimal_direction.tolist()
    else:
        comparators_str = ["neutral"] * len(df_ref)
    # separate data
    hypercube_ref = hypercube[:, :, df_ref.index.tolist()]
    df_term = df[df.term == term]
    hypercube_term = hypercube[:, :, df_term.index.tolist()]
    return (
        df_ref,
        hypercube_ref,
        hypercube_term,
        comparators_str,
        ordered_weights_keys,
        ordered_features,
    )


def build_weighted_hypercube(
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
        ordered_weights_keys,
        ordered_features,
    ) = precompute_helpers_for_climate_score(
        term, df, viable_ranges, hypercube, smart_comparison
    )
    weights = get_ordered_weights(ordered_weights_keys)
    # proceed to comparison
    if comparison == "map":
        delta = get_delta_hypercube_cached(cube_term, cube_ref, comparators_str)
    elif comparison == "point (if set)":
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
    return delta * weights, ordered_features


def pimp_score_with_auxiliary_data(
    score: np.ndarray, df_aux: pd.DataFrame, hypercube_aux: np.ndarray
):
    df_aux = df_aux.reset_index(drop=True)

    if st.session_state[POSITIVE_SCORES_BONUS_TOGGLE_KEY]:
        score = np.where(score > 0, 2 * score, score)

    # score = minmax_bounding(score)

    if st.session_state[SEALEVEL_TOGGLE_KEY]:
        df = df_aux[df_aux.variable == "seaLevelElevation"]
        score_aux = hypercube_aux[:, :, df.index[0]]
        score = np.where(score_aux < 0, np.nanmin(score), score)

    if st.session_state[INCREASE_CONTRAST_TOGGLE_KEY]:
        score = minmax_bounding(score)
        score = np.sign(score) * np.power(np.abs(score), 3)

    if st.session_state[REAL_ESTATE_TOGGLE_KEY]:
        df = df_aux[df_aux.variable == "realEstate"]
        score_aux = hypercube_aux[:, :, df.index[0]]
        score = minmax_bounding(score) / np.log(score_aux)
        score = minmax_bounding(score)

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


def optimize_if_nan(xy, hypercube):
    if np.any(np.isnan(hypercube[*xy, :])):
        x, y = xy  # target point
        xx, yy, _ = hypercube.shape
        xx = xx // 2  # middle of the map
        yy = yy // 2  # middle of the map
        # where is the xy point wrt center ?
        delta_x = x - xx
        delta_y = y - yy
        # try to come closer to the center
        x += -1 if delta_x > 0 else 1
        y += -1 if delta_y > 0 else 1
        return optimize_if_nan((x, y), hypercube)
    else:
        return xy


@st.cache_data
def get_quantile_cached(score: np.ndarray, q: list):
    values = np.nanquantile(score.ravel(), q=q)
    return {qq: vv for qq, vv in zip(q, values)}


def display_cities_slices(cities: str | None, hypercube, ordered_features):
    if cities:
        cities = [c.strip() for c in cities.split(",")]
        xys = np.array(
            [convert_geo_to_xy(loc_cached(c).coords) for c in cities], dtype=np.int16
        )
        xys = np.array([optimize_if_nan(xy, hypercube) for xy in xys])
        cities_values = hypercube[*xys.T, :]
        df = pd.DataFrame(data=cities_values, index=cities, columns=ordered_features)
        st.bar_chart(df)


def search_and_display_tops(
    top,
    score,
):
    top_score = score.copy()
    xys_of_tops = []
    score_of_tops = []
    window_size = 7
    for i in range(top):
        id_of_max = np.nanargmax(top_score)
        xy = np.unravel_index(id_of_max, score.shape)
        xy_geo = [xy[1], xy[0]]
        xys_of_tops.append(xy_geo)
        score_of_tops.append(score[*xy])
        # clear area to avoid to pick it again
        top_score[
            xy[0] - window_size : xy[0] + window_size,
            xy[1] - window_size : xy[1] + window_size,
        ] = np.nan
    coords_of_top = [convert_xy_to_geo_cached(xy) for xy in xys_of_tops]

    loc_of_top = [Loc(coords=coords) for coords in coords_of_top]
    name_of_top = [
        ", ".join(clean_geocoded_address(loc.name)[-5:]) for loc in loc_of_top
    ]
    str_of_top = [f"**TOP {i+1}** : {name}" for i, name in enumerate(name_of_top)]
    st.markdown("\n\n".join(str_of_top))


def draw_fig(score, term, comparison, real_estate, size=10):
    fig, _ = plt.subplots(figsize=(size, size))
    plt.imshow(score, cmap="jet")
    plt.colorbar()
    plt.grid(which="both", alpha=0.5)
    _ = plt.xticks(ticks=np.arange(0, score.shape[1], step=20))
    _ = plt.yticks(ticks=np.arange(0, score.shape[0], step=20))
    title = create_title(term, comparison, real_estate)
    plt.title(title)
    return fig


def extract_params():
    weights = {k: v for k, v in st.session_state.items() if WEIGHT_PREFIX in k}
    options = {s.key: st.session_state[s.key] for s in st.session_state.selectors}
    return {"weights": weights, "options": options}


def save_fig():
    buffer = io.BytesIO()
    st.session_state.fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer


def render(metadata_path, metadata_aux_path, viable_path, hypercube_path, is_demo):
    df, df_aux, df_viable, hypercube, hypercube_aux = init(
        metadata_path, metadata_aux_path, viable_path, hypercube_path, is_demo
    )
    col_img, col_selectors = st.columns(2)
    with col_img:
        hypercube, ordered_features = build_weighted_hypercube(
            st.session_state[TERM_SELECTOR_KEY],
            st.session_state[COMPARISON_TYPE_SELECTOR_KEY],
            df,
            df_viable,
            hypercube,
            smart_comparison=st.session_state[SMART_COMPARISON_TOGGLE_KEY],
        )
        score = hypercube.mean(axis=-1)
        score = pimp_score_with_auxiliary_data(score, df_aux, hypercube_aux)
        try:
            fig = draw_fig(
                score,
                st.session_state[TERM_SELECTOR_KEY],
                st.session_state[COMPARISON_TYPE_SELECTOR_KEY],
                st.session_state[REAL_ESTATE_TOGGLE_KEY],
                size=10,
            )
            st.pyplot(fig)
            st.session_state["fig"] = fig
        except ValueError:
            st.markdown(RELOADING_WARNING_MSG)
    with col_selectors:
        for selector in st.session_state.selectors:
            selector.get_st_object()
        if loc := get_clicked_loc_from_key(MAP_CLICKED_POSITION_KEY, factor=5):
            st.markdown("---")
            st.markdown("Picked **reference** location: " + loc)
    if not is_demo:
        col_analysis, col_top = st.columns(2)
        with col_analysis:
            try:
                st.text_input(label="cities to analyze ...", key=CITIES_TO_ANALYZE_KEY)
                display_cities_slices(
                    st.session_state[CITIES_TO_ANALYZE_KEY], hypercube, ordered_features
                )
            except TypeError:
                st.markdown(RELOADING_WARNING_MSG)
        with col_top:
            with st.spinner("Getting best places wrt your parameters ..."):
                search_and_display_tops(3, score)

    with st.expander(label="Weight setting", expanded=True):
        render_weights_setters(df, value=1)
