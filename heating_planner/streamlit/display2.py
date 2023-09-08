from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from heating_planner import value_comparison
from heating_planner.streamlit import io
from heating_planner import utils
from heating_planner.data import Datum

data: Datum = st.session_state.data


def craft_key(metadata, prefix=None):
    el = [prefix] if prefix is not None else []
    el += [metadata.variable, metadata.season]
    return "_".join(el)


def craft_weight_key(metadata):
    return craft_key(metadata=metadata, prefix="weight")


def init():
    # init as many things as possible, maybe too much
    init_key_values = {}

    # init weights
    df = pd.concat([data.metadata])
    weights_names = df.apply(craft_weight_key, axis=1).tolist()
    init_key_values.update({name: 1.0 for name in weights_names})

    # init other stuff
    init_key_values.update({"term": "near"})

    # proceed
    for key, value in init_key_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def comparison_neutral(delta_map: np.ndarray) -> np.ndarray:
    return np.abs(delta_map)


def comparison_less_is_better(delta_map: np.ndarray) -> np.ndarray:
    map = np.where(np.isnan(delta_map), np.nan, 0)
    map += np.where(delta_map > 0, 0.1 * delta_map, 0)
    map += np.where(delta_map < 0, delta_map, 0)
    return map


def comparison_more_is_better(delta_map: np.ndarray) -> np.ndarray:
    map = np.where(np.isnan(delta_map), np.nan, 0)
    map += np.where(delta_map < 0, -0.1 * delta_map, 0)
    map += np.where(delta_map > 0, -delta_map, 0)
    return map


@st.cache_data
def compute_scores_with_refpoint(ref_xy, term):
    df = data.metadata
    df_term = df[df.term == term]
    df_ref = df[df.term == "ref"]
    # keep track of mananged features
    ordered_feats = df_ref.apply(craft_weight_key, axis=1).tolist()
    # extract sub-cubes
    hypercube_term = data.hypercube[:, :, df_term.index.tolist()]
    hypercube_ref = data.hypercube[:, :, df_ref.index.tolist()]
    # compare
    if ref_xy is None:
        # then just assert which area will change the most
        hypercube_delta = hypercube_ref - hypercube_term
    else:
        xy = np.array([ref_xy["y"], ref_xy["x"]]) // 5
        ref_values = np.expand_dims(hypercube_ref[*xy, :].reshape(1, -1), 0)
        hypercube_delta = ref_values - hypercube_term
    # now apply directions
    slices_scored = []
    for i in range(hypercube_delta.shape[-1]):
        _slice = hypercube_delta[:, :, i]
        opt_direction = df.loc[i].optimal_direction
        if opt_direction == "neutral":
            slices_scored.append(comparison_neutral(_slice))
        elif opt_direction == "less_is_better":
            slices_scored.append(comparison_less_is_better(_slice))
        elif opt_direction == "more_is_better":
            slices_scored.append(comparison_more_is_better(_slice))

    return -np.stack(slices_scored, axis=-1), ordered_feats


@st.cache_data
def compute_aux_scores():
    df = data.metadata_aux.reset_index()
    # keep track of mananged features
    ordered_feats = df.apply(craft_weight_key, axis=1).tolist()
    # extract sub cubes
    hypercube = data.hypercube_aux
    std = np.nanstd(
        hypercube.reshape(np.prod(hypercube.shape[:2]), hypercube.shape[-1]), axis=0
    )
    hypercube = hypercube / std
    # know how to tweak the aux data
    higher_better_dict = {"seaLevelElevation": -1, "realEstate": -1}
    sign_op = np.expand_dims(
        df.variable.apply(lambda k: higher_better_dict[k]).to_numpy().reshape(1, -1), 0
    )
    hypercube = hypercube * sign_op
    return hypercube, ordered_feats


def get_ordered_weights(feat_order):
    w = np.array(
        [st.session_state[featname] for featname in feat_order],
        dtype=np.float32,
    )
    # w = np.square(w)
    w /= w.sum()
    return w


## STREAMLIT DISPLAY PART
def render():
    init()
    with st.container():
        image_ctx, weight_ctx = st.columns([7, 4])

        with image_ctx:
            options_columns = st.columns(2)
            with options_columns[0]:
                st.radio(
                    "Looking for near or medium term?",
                    options=["near", "medium"],
                    key="term",
                )
            with options_columns[1]:
                st.toggle(
                    "Use reference point if provided ?",
                    value=False,
                    key="use_ref_point",
                )
            # not plot image
            xy = (
                st.session_state.map_clicked_xy
                if st.session_state.use_ref_point
                else None
            )
            scores, ordered_feats = compute_scores_with_refpoint(
                xy, st.session_state.term
            )
            # aux_scores, ordered_feats_aux = compute_aux_scores()
            # scores = np.concatenate([scores, aux_scores], axis=-1)
            weights = get_ordered_weights(ordered_feats)
            # weights
            scores_agg = np.average(scores, axis=-1, weights=weights)
            scores_agg = utils.minmax_bounding(scores_agg)
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.imshow(scores_agg, cmap="jet")
            plt.colorbar()
            plt.grid(which="both", alpha=0.5)
            _ = plt.xticks(ticks=np.arange(0, scores.shape[1], step=20))
            _ = plt.yticks(ticks=np.arange(0, scores.shape[0], step=20))
            st.caption("Global weighted score map")
            st.pyplot(fig)

        with weight_ctx:
            # anno variables
            df = data.metadata[
                (data.metadata.season == "anno") & (data.metadata.term == "ref")
            ]
            for _, meta in df.iterrows():
                key = craft_weight_key(meta)
                st.slider(meta.variable, 0, 5, 1, step=1, key=key)
            # seasonal variables
            season_cols = st.columns(4)
            for i, season in enumerate(["winter", "spring", "summer", "autumn"]):
                df, _ = data.get_term(term=st.session_state.term)
                df = df[df.season == season].sort_values("variable")
                is_first = True
                with season_cols[i]:
                    for _, meta in df.iterrows():
                        if is_first:
                            st.write(season)
                            is_first = False
                        st.slider(
                            meta.variable,
                            0,
                            5,
                            1,
                            step=1,
                            key=craft_weight_key(meta),
                        )
