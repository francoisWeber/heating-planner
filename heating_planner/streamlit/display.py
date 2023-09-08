from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from heating_planner import value_comparison
from heating_planner.streamlit import io
import os
import re
from heating_planner.utils import load_np_from_url


def craft_featname(variable, season):
    return f"{variable} during {season}"


@st.cache_data
def init(
    term,
    df_metadata_path,
    metadata_aux_path,
    df_viable_path,
    mask_path,
    hypercube_path,
    map_clicked_xy,
):
    df_metadata = pd.read_json(df_metadata_path).reset_index()
    df_metadata_aux = pd.read_json(metadata_aux_path).reset_index()
    df_viable = pd.read_json(df_viable_path)

    df_term = df_metadata[df_metadata.term == term]
    df_term_aux = df_metadata_aux[df_metadata_aux.term == term]

    direction2delta_fn = {
        "neutral": value_comparison.neutral_delta_normalized,
        "less is better": value_comparison.lower_better_delta_normalized,
        "more is better": value_comparison.upper_better_delta_normalized,
    }

    # special tool to load npy from URL
    if os.path.isfile(hypercube_path):
        hypercubes = np.load(hypercube_path)
    elif hypercube_path.startswith("http"):
        hypercubes = load_np_from_url(hypercube_path)
    else:
        raise ValueError(f"No valid data for {hypercube_path=}")
    hypercube = hypercubes["map"]
    hypercube_aux = hypercubes["aux"]

    if os.path.isfile(mask_path):
        mask = np.load(mask_path)
    elif mask_path.startswith("http"):
        mask = load_np_from_url(mask_path)
    else:
        raise ValueError(f"No valid data for {mask_path=}")

    # build the un-weighted scores
    ordered_scores = []
    ordered_features_names = []
    ordered_features = []
    for index, metadata in df_term.iterrows():
        map = hypercube[:, :, index] * mask
        try:
            ref_values = df_viable[
                (df_viable.season == metadata.season)
                & (df_viable.variable == metadata.variable)
            ].iloc[0]
            opt_direction = ref_values.optimal_direction
            opt_ranges = ref_values.optimal_range
        except:
            opt_direction = "neutral"
            opt_ranges = [0.0, 0.0]
        if map_clicked_xy is not None:
            opt_ranges = [map[map_clicked_xy["y"] // 5, map_clicked_xy["x"] // 5]] * 2
        compare_fn = direction2delta_fn[opt_direction]
        score_on_map = compare_fn(map, *opt_ranges)
        ordered_scores.append(score_on_map)
        ordered_features.append([metadata.variable, metadata.season])
        ordered_features_names.append(
            craft_featname(metadata.variable, metadata.season)
        )
    scores = np.stack(ordered_scores, -1)

    # ad aux parts
    map_estate = None
    for i, metadata in df_metadata_aux.iterrows():
        # TODO: register viable ranges and trend for aux variables
        if "seaLevelElevation" == metadata.variable:
            # upper is better, like scores
            map = hypercube_aux[:, :, i]
            # normalize map based on current scores
            map = (map - np.nanmin(map)) / (np.nanmax(map) - np.nanmin(map))
            map = map * np.nanmax(np.abs(scores))
            ordered_scores.append(map)
            ordered_features_names.append(
                craft_featname(metadata.variable, metadata.season)
            )
            ordered_features.append([metadata.variable, metadata.season])

        elif "realEstate" in metadata.variable:
            map_estate = hypercube_aux[:, :, i]
            continue

    scores = np.stack(ordered_scores, -1)
    return (
        scores,
        np.array(ordered_features_names, dtype="<U50"),
        np.array(ordered_features),
        map_estate,
    )


@st.cache_data
def organize_features(ordered_features):
    anno_feats = ordered_features[ordered_features[:, 1] == "anno"]
    season_feats = ordered_features[ordered_features[:, 1] != "anno"]
    # unique season / variable
    variables_unique = np.unique(season_feats.T[0]).astype("str")
    seasons_unique = np.unique(season_feats.T[1]).astype("str")
    seasons_id, variables_id = np.meshgrid(seasons_unique, variables_unique)
    featnames_grid = np.apply_along_axis(
        lambda a: np.array(craft_featname(*a), dtype="<U50"),
        axis=-1,
        arr=np.stack((variables_id, seasons_id), -1),
    )
    anno_featnames_line = np.apply_along_axis(
        lambda a: np.array(craft_featname(*a), dtype="<U50"), -1, anno_feats
    )
    return featnames_grid, anno_featnames_line, variables_unique, seasons_unique


def display(metadata_path, metadata_aux_path, viable_path, mask_path, hypercube_path):
    # parse args
    if metadata_path is None:
        raise ValueError("Provide metadata_path")
    if viable_path is None:
        raise ValueError("Provide viable_path")
    if mask_path is None:
        raise ValueError("Provide mask_path")
    if hypercube_path is None:
        raise ValueError("Provide hypercube_path")

    if "ref_point_comparison" not in st.session_state:
        st.session_state["ref_point_comparison"] = True

    if "term" not in st.session_state:
        st.session_state["term"] = "near"
    # launch the rest
    if st.session_state.ref_point_comparison:
        xy = st.session_state.map_clicked_xy
    else:
        xy = None
    scores, ordered_features_names, ordered_features, map_estate = init(
        st.session_state.term,
        metadata_path,
        metadata_aux_path,
        viable_path,
        mask_path,
        hypercube_path,
        xy,
    )
    (
        featnames_grid,
        anno_featnames_line,
        variables_unique,
        seasons_unique,
    ) = organize_features(ordered_features)

    for feat in ordered_features_names:
        key = "slider-" + feat
        if key not in st.session_state:
            st.session_state[key] = 1

    def get_weighted_scores(scores):
        weights = [
            st.session_state[f"slider-{feat}"] for feat in ordered_features_names
        ]
        weights = np.square(weights, dtype=np.float32)
        weights /= weights.sum()
        return scores * weights

    with st.container():
        img_context, weights_context = st.columns([7, 4])
        with img_context:
            radio_dividers = st.columns(2)
            with radio_dividers[0]:
                st.radio(
                    "Which temporal term to display ?",
                    options=["near", "medium"],
                    key="term",
                )
            with radio_dividers[1]:
                st.toggle(
                    "Activate real estate if possible?",
                    value=True,
                    key="includeReadlEstate",
                )
                st.toggle(
                    "Compare to reference point is selected?",
                    value=True,
                    key="ref_point_comparison",
                )
            fig, ax = plt.subplots(figsize=(8, 8))
            weighted_scores = get_weighted_scores(scores)
            agg_weighted_score = np.mean(weighted_scores, -1)
            title_suffix = ""
            if map_estate is not None and st.session_state.includeReadlEstate:
                agg_weighted_score /= map_estate
                title_suffix = " by estate price"
            agg_weighted_score -= np.nanmin(agg_weighted_score)
            agg_weighted_score /= np.nanmax(agg_weighted_score)
            plt.imshow(agg_weighted_score, cmap="jet")
            plt.colorbar()
            plt.grid(which="both", alpha=0.5)
            _ = plt.xticks(ticks=np.arange(0, scores.shape[1], step=20))
            _ = plt.yticks(ticks=np.arange(0, scores.shape[0], step=20))
            st.caption("Global weighted score map" + title_suffix)
            st.pyplot(fig)

        with weights_context:
            st.text("Set weights for the considered variables")
            for feat in anno_featnames_line:
                st.slider(f"{feat}", 0, 5, 1, step=1, key=f"slider-{feat}")
            # then split in 4 col, one per season
            cols = st.columns(4)
            for i, col in enumerate(cols):
                with col:
                    st.markdown(seasons_unique[i])
                    for feat in featnames_grid[:, i]:
                        variable = feat.split(" ")[0]
                        st.slider(variable, 0, 5, 1, step=1, key=f"slider-{feat}")

    with st.container():
        st.divider()
        coords_str = st.text_input(
            "give some coordinates to explain their score, format (x, y), (z, t)",
        )
        coords_re = re.compile("\(\d+,[ ]*\d+\)")
        coords = coords_re.findall(coords_str)
        if len(coords) > 0:
            coords = [[int(el) for el in coord[1:-1].split(",")] for coord in coords]
            # now explain the score for every locations
            penalties_per_loc = np.array(
                [weighted_scores[*loc_xy, :] for loc_xy in coords]
            )

            feats_nonzeros_id = np.where(np.linalg.norm(penalties_per_loc, axis=0))[0]
            penalties_per_loc_nz = penalties_per_loc.T[feats_nonzeros_id, :]
            feats_nz = np.array(ordered_features_names)[feats_nonzeros_id]

            fig, ax = plt.subplots(figsize=(15, 6))
            NUM_COLORS = len(feats_nz)
            cm = plt.get_cmap("gist_rainbow")
            ax.set_prop_cycle(
                color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
            )
            prev_h_neg = np.zeros(len(coords))
            ticks = range(len(coords))
            for penalty, feat in zip(penalties_per_loc_nz, feats_nz):
                penalty_neg_part = np.where(penalty < 0, penalty, 0.0)
                if "summer" in feat:
                    hatch = "/"
                elif "winter" in feat:
                    hatch = "//"
                elif "spring" in feat:
                    hatch = "x"
                else:
                    hatch = ":"
                _ = plt.bar(
                    x=ticks,
                    height=penalty_neg_part,
                    label=feat,
                    bottom=prev_h_neg,
                    hatch=hatch,
                )
                prev_h_neg += penalty_neg_part
            _ = plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
            _ = plt.tight_layout()
            _ = plt.xticks(ticks=ticks, labels=[str(coord) for coord in coords])
            _ = plt.grid(axis="y")
            _ = plt.title("Interprétation des scores pondérés")
            st.pyplot(fig)
