from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from heating_planner import value_comparison
import click
import os


def craft_featname(variable, season):
    return f"{variable} during {season}"


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
    mask = np.load(mask_path)

    df_term = df_metadata[df_metadata.term == term]

    direction2delta_fn = {
        "neutral": value_comparison.neutral_delta_normalized,
        "less is better": value_comparison.lower_better_delta_normalized,
        "more is better": value_comparison.upper_better_delta_normalized,
    }

    if os.path.isfile(hypercube_path):
        hypercube = np.load(hypercube_path)
    elif hypercube_path.startswith("http"):
        import requests
        import io

        response = requests.get(hypercube_path)
        response.raise_for_status()
        hypercube = np.load(io.BytesIO(response.content))
    else:
        raise ValueError(f"No valid data for {hypercube_path=}")

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
    term = st.radio("Which temporal term to display ?", options=["near", "medium"])
    # parse args
    if metadata_path is None:
        raise ValueError("Provide metadata_path")
    if viable_path is None:
        raise ValueError("Provide viable_path")
    if mask_path is None:
        raise ValueError("Provide mask_path")
    if hypercube_path is None:
        raise ValueError("Provide hypercube_path")

    # launch the rest
    scores, ordered_features_names, ordered_features = init(
        term, metadata_path, viable_path, mask_path, hypercube_path
    )
    (
        featnames_grid,
        anno_featnames_line,
        variables_unique,
        seasons_unique,
    ) = organize_features(ordered_features)

    # init
    st.title("Find your place !")

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
        img_context, weights_context = st.columns([4, 2])
        with img_context:
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.imshow(np.mean(get_weighted_scores(scores), -1), cmap="jet")
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
        a = st.text_input("give some coordinates to check")


if __name__ == "__main__":
    go(standalone_mode=False, auto_envvar_prefix="HEAT")
