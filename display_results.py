from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from heating_planner import value_comparison
import click
import os
import requests
import io
import re


st.set_page_config(layout="wide")


def craft_featname(variable, season):
    return f"{variable} during {season}"


def load_np_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


@st.cache_data
def init(
    term,
    df_metadata_path,
    df_viable_path,
    mask_path,
    hypercube_path,
):
    df_metadata = pd.read_json(df_metadata_path).reset_index()
    df_viable = pd.read_json(df_viable_path)

    df_term = df_metadata[df_metadata.term == term]

    direction2delta_fn = {
        "neutral": value_comparison.neutral_delta_normalized,
        "less is better": value_comparison.lower_better_delta_normalized,
        "more is better": value_comparison.upper_better_delta_normalized,
    }

    # special tool to load npy from URL
    if os.path.isfile(hypercube_path):
        hypercube = np.load(hypercube_path)
    elif hypercube_path.startswith("http"):
        hypercube = load_np_from_url(hypercube_path)
    else:
        raise ValueError(f"No valid data for {hypercube_path=}")

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
        ref_values = df_viable[
            (df_viable.season == metadata.season)
            & (df_viable.variable == metadata.variable)
        ].iloc[0]
        compare_fn = direction2delta_fn[ref_values.optimal_direction]
        score_on_map = compare_fn(map, *ref_values.optimal_range)
        ordered_scores.append(score_on_map)
        ordered_features.append([metadata.variable, metadata.season])
        ordered_features_names.append(
            craft_featname(metadata.variable, metadata.season)
        )

    scores = np.stack(ordered_scores, -1)
    return scores, np.array(ordered_features_names), np.array(ordered_features)


@st.cache_data
def organize_features(ordered_features):
    anno_feats = ordered_features[ordered_features[:, 1] == "anno"]
    season_feats = ordered_features[ordered_features[:, 1] != "anno"]
    # unique season / variable
    variables_unique = np.unique(season_feats.T[0]).astype("str")
    seasons_unique = np.unique(season_feats.T[1]).astype("str")
    seasons_id, variables_id = np.meshgrid(seasons_unique, variables_unique)
    featnames_grid = np.apply_along_axis(
        lambda a: np.array(craft_featname(*a), dtype="<U28"),
        axis=-1,
        arr=np.stack((variables_id, seasons_id), -1),
    )
    anno_featnames_line = np.apply_along_axis(
        lambda a: craft_featname(*a), -1, anno_feats
    )
    return featnames_grid, anno_featnames_line, variables_unique, seasons_unique


df_metadata_path = "/Users/f.weber/tmp-fweber/heating/metadata.json"
df_viable_path = "/Users/f.weber/tmp-fweber/heating/viable_ranges2.json"
mask_path = "/Users/f.weber/tmp-fweber/heating/france_mask_factor5.npy"
term = "near"


@click.command()
@click.option("--metadata-path", type=str, default=None)
@click.option("--viable-path", type=str, default=None)
@click.option("--mask-path", type=str, default=None)
@click.option("--hypercube-path", type=str, default=None)
def go(metadata_path, viable_path, mask_path, hypercube_path):
    # parse args
    if metadata_path is None:
        raise ValueError("Provide metadata_path")
    if viable_path is None:
        raise ValueError("Provide viable_path")
    if mask_path is None:
        raise ValueError("Provide mask_path")
    if hypercube_path is None:
        raise ValueError("Provide hypercube_path")

    if "term" not in st.session_state:
        st.session_state["term"] = "near"
    # launch the rest
    scores, ordered_features_names, ordered_features = init(
        st.session_state.term, metadata_path, viable_path, mask_path, hypercube_path
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
            st.radio(
                "Which temporal term to display ?",
                options=["near", "medium"],
                key="term",
            )
            fig, ax = plt.subplots(figsize=(8, 8))
            weighted_scores = get_weighted_scores(scores)
            agg_weighted_score = np.mean(weighted_scores, -1)
            plt.imshow(agg_weighted_score, cmap="jet")
            plt.colorbar()
            plt.grid(which="both", alpha=0.5)
            _ = plt.xticks(ticks=np.arange(0, scores.shape[1], step=20))
            _ = plt.yticks(ticks=np.arange(0, scores.shape[0], step=20))
            st.caption("Global weighted score map")
            st.pyplot(fig)

        with weights_context:
            st.text("Set eights for the considered variables")
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


if __name__ == "__main__":
    go(standalone_mode=False, auto_envvar_prefix="HEAT")
